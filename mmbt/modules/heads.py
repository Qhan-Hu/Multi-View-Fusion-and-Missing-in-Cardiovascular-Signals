import torch
import torch.nn as nn
from mmbt.modules.trans.biosig_transformer import Block
from mmbt.modules.position_embedding import CosinePositionEncoding, LearnablePositionalEncoding
from mmbt.modules import objectives
# from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BPEHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class AFDHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class SSCHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

class TransformHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.fc(x)
        return x


class MSMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, config["hidden_size_decoder"]))
        torch.nn.init.normal_(self.masked_embed, std=0.02)
        self.token_type_embeds = nn.Embedding(2, config['hidden_size_decoder'])
        self.token_type_embeds.apply(objectives.init_weights)

        self.enc_to_dec = nn.Linear(config["hidden_size"], config["hidden_size_decoder"]) if config["hidden_size"] != config["hidden_size_decoder"] else nn.Identity()
        self.decoder = nn.ModuleList([
            Block(config["hidden_size_decoder"], config["num_heads_decoder"], config["mlp_ratio"], qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for _ in range(config["num_layers_decoder"])
        ])
        self.ecg_num_patches = config["ecg_size"]//config["ecg_patch_size"]
        self.ppg_num_patches = config["ppg_size"]//config["ppg_patch_size"]
        self.num_masked = int(self.ppg_num_patches * config["msm_prob"])
        if config["pos_embed"] == "CS":
            self.ecg_pos_embed = CosinePositionEncoding(embed_dim=config["hidden_size_decoder"], seq_len=self.ecg_num_patches+1)
            self.ppg_pos_embed = CosinePositionEncoding(embed_dim=config["hidden_size_decoder"], seq_len=self.ppg_num_patches+1)
        if config["pos_embed"] == "LP":
            self.ecg_pos_embed = LearnablePositionalEncoding(embed_dim=config["hidden_size_decoder"], seq_len=self.ecg_num_patches+1)
            self.ppg_pos_embed = LearnablePositionalEncoding(embed_dim=config["hidden_size_decoder"], seq_len=self.ppg_num_patches+1)
        self.decoder_norm = nn.LayerNorm(config["hidden_size_decoder"])
        self.ecg_decoder_pred = TransformHead(config["hidden_size_decoder"], config["ecg_patch_size"])
        self.ppg_decoder_pred = TransformHead(config["hidden_size_decoder"], config["ppg_patch_size"]*3)


    def forward(self, infer):
        ppg_feats, ecg_feats = self.enc_to_dec(infer["ppg_feats"]), self.enc_to_dec(infer["ecg_feats"])  # with cls token
        ppg_ids_res, ecg_ids_res = infer["ppg_ids_res"], infer["ecg_ids_res"]

        # append mask embeds to sequence
        ppg_mask_feats = self.masked_embed.repeat(ppg_feats.shape[0], ppg_ids_res.shape[1] + 1 - ppg_feats.shape[1], 1) if ppg_ids_res is not None else None # no cls token
        ecg_mask_feats = self.masked_embed.repeat(ecg_feats.shape[0], ecg_ids_res.shape[1] + 1 - ecg_feats.shape[1], 1) if ecg_ids_res is not None else None # no cls token
        ppg_feats_ = torch.cat([ppg_feats[:, 1:, :], ppg_mask_feats], dim=1) if ppg_mask_feats is not None else ppg_feats[:, 1:, :] # no cls token
        ecg_feats_ = torch.cat([ecg_feats[:, 1:, :], ecg_mask_feats], dim=1) if ecg_mask_feats is not None else ecg_feats[:, 1:, :] # no cls token

        ppg_feats_ = torch.gather(ppg_feats_, dim=1, index=ppg_ids_res.unsqueeze(-1).repeat(1, 1, ppg_feats_.shape[2])) if ppg_ids_res is not None else ppg_feats_ # unshuffle
        ecg_feats_ = torch.gather(ecg_feats_, dim=1, index=ecg_ids_res.unsqueeze(-1).repeat(1, 1, ecg_feats_.shape[2])) if ecg_ids_res is not None else ecg_feats_ # unshuffle

        ppg_feats = torch.cat([ppg_feats[:, :1, :], ppg_feats_], dim=1) # append cls token
        ecg_feats = torch.cat([ecg_feats[:, :1, :], ecg_feats_], dim=1) # append cls token

        # add position embeds
        ppg_feats = self.ppg_pos_embed(ppg_feats) + self.token_type_embeds(
            torch.zeros_like(ppg_feats[:, :, 0].to(torch.int64)
                             ))
        ecg_feats = self.ecg_pos_embed(ecg_feats) + self.token_type_embeds(
            torch.full_like(ecg_feats[:, :, 0].to(torch.int64), 1)
        )
        co_feats = torch.cat([ppg_feats, ecg_feats], dim=1)
        x = co_feats
        #
        # apply decoder
        for blk in self.decoder:
            x, _attn = blk(x)
        x = self.decoder_norm(x)
        ppg_feats, ecg_feats = (
            x[:, :ppg_feats.shape[1]],
            x[:, ppg_feats.shape[1]:],
        )

        # predictor projection
        ppg_values = self.ppg_decoder_pred(ppg_feats)
        ecg_values = self.ecg_decoder_pred(ecg_feats)

        # remove cls token
        ppg_values = ppg_values[:, 1:, :]
        ecg_values = ecg_values[:, 1:, :]

        return {
            "ppg_preds": ppg_values,
            "ecg_preds": ecg_values,
        }

