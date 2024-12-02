from sacred import Experiment

ex = Experiment("MMBT")


def _loss_names(d):
    """
       msm: mask ecg&ppg signal modelling (msm consist of mpm and mem)
       cep: contrastive ecg&ppg
       bpe: blood pressure estimation
       afd: atrial fibrillation detection
       ssc: sleep stage classification
    """
    ret = {
        "cep": 0,
        "msm": 0,
        "bpe": 0,
        "afd": 0,
        "ssc": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "Pretrain"
    seed = 0
    datasets = ["pretrain_pulsedb"]
    dataset_ratio = 1.0
    loss_names = _loss_names({"msm": 1, "cep": 0.2})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None

    # missing modality config
    missing_ratio = {'train': 0, 'val': 0, 'test': 0}
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'}
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/pretrain_arrows'

    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0,1,2,3,4,5]
    multi_layer_prompt = True

    # PPG & ECG setting
    ppg_size = 1250
    ppg_patch_size = 50
    ecg_size = 1250
    ecg_patch_size = 50
    draw_false_ppg = 0
    draw_false_ecg = 0
    msm_prob = 0.8
    train_transform_key = "pooldrift"
    val_transform_key = None

    # Transformer(encoder) Setting for pretraining and downstream tasks
    bist = "bist_custom_patch50"
    ecg_encoder = "resnet34"
    ppg_encoder = "resnet50"
    shared_encoder = "resnet101"
    hidden_size = 512
    num_layers = 8
    num_heads = 8
    proj_size = 256
    mlp_ratio = 4
    drop_rate = 0.1
    pos_embed = "CS"
    view_flag = "ppg"

    # Decoder Setting for pretraining
    hidden_size_decoder = 256
    num_layers_decoder = 3
    num_heads_decoder = 8

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-3
    weight_decay = 0.01
    decay_power = "cosine"
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = r".../multi-view-biosignal-v1/datasets/pretrain_arrows"
    log_dir = "./results"
    per_gpu_batchsize = 2048  # you should define this manually with per_gpu_batch_size=#
    num_gpus = [0]
    num_nodes = 1
    load_path = ""
    num_workers = 40
    precision = 16


@ex.named_config
def test_pretrain():
    exp_name = "test_pretrain"
    datasets = ["pretrain_pulsedb"]
    data_root = ".../multi-view-biosignal-v1/datasets/pretrain_arrows"
    # datasets = ["pretrain_pulsedb"]
    loss_names = _loss_names({"msm": 1})
    # ppg_size = 3000
    # ecg_size = 3000
    test_only = True
    num_gpus = [1]
    fast_dev_run = 2
    per_gpu_batchsize = 32
    load_path = ""
    precision = 32


@ex.named_config
def finetune_pulsedb():
    exp_name = "ft_pulsedb"
    datasets = ["pulsedb"]
    data_root = r".../multi-view-biosignal-v1/datasets/PulseDB"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PulseDB'
    # missing_ratio = {'train': 0.7, 'val': 0, 'test': 0.7}
    # missing_type = {'train': 'both', 'val': 'both', 'test': 'both'}
    load_path = ""
    dataset_ratio = 1.0
    loss_names = _loss_names({"bpe": 1})
    resume_from = None
    train_transform_key = None
    val_transform_key = None
    finetune_first = True
    # per_gpu_batchsize = 4096
    per_gpu_batchsize = 2048
    max_epoch = 20
    # max_steps = None
    # warmup_steps = 0.1
    # learning_rate = 1e-4
    # val_check_interval = 1.0
    # weight_decay = 2e-2
    # optim_type = "adam"
    # lr_mult = 5
    num_gpus = [0]


@ex.named_config
def fromscratch_pulsedb():
    exp_name = "fs_pulsedb"
    datasets = ["pulsedb"]
    data_root = ".../multi-view-biosignal-v1/datasets/PulseDB"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PulseDB'
    # missing_ratio = {'train': 0.7, 'val': 0, 'test': 0.7}
    # missing_type = {'train': 'both', 'val': 'both', 'test': 'both'}
    train_transform_key = None
    val_transform_key = None
    per_gpu_batchsize = 2048
    hidden_size = 256
    num_layers = 6
    num_heads = 8
    loss_names = _loss_names({"bpe": 1})
    num_gpus = [1]
    resume_from = None


@ex.named_config
def test_pulsedb():
    exp_name = "test_pulsedb"
    datasets = ["pulsedb"]
    data_root = r".../multi-view-biosignal-v1/datasets/PulseDB"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PulseDB'
    # missing_ratio = {'train': 0.1, 'val': 0, 'test': 0.9}
    # missing_type = {'train': 'ecg', 'val': 'both', 'test': 'ecg'}
    prompt_length = 20
    loss_names = _loss_names({"bpe": 1})
    num_gpus = [1]
    test_only = True
    load_path = ""
    precision = 32


@ex.named_config
def fromscratch_performaf():
    exp_name = "fs_performaf"
    datasets = ["performaf"]
    data_root = ".../multi-view-biosignal-v1/datasets/PERFormAF"
    missing_table_root = '.../multi-view-biosignal-v1/datasets/PERFormAF'
    train_transform_key = None
    val_transform_key = None
    per_gpu_batchsize = 128
    loss_names = _loss_names({"afd": 1})
    num_gpus = [1]
    resume_from = None


@ex.named_config
def finetune_performaf():
    exp_name = "ft_performaf"
    datasets = ["performaf"]
    data_root = r".../multi-view-biosignal-v1/datasets/PERFormAF"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PERFormAF'
    missing_ratio = {'train': 0.7, 'val': 0, 'test': 0.7}
    missing_type = {'train': 'ecg', 'val': 'both', 'test': 'ecg'}
    prompt_length = 20
    load_path = ""
    dataset_ratio = 1.0
    loss_names = _loss_names({"afd": 1})
    train_transform_key = None
    val_transform_key = None
    finetune_first = True
    per_gpu_batchsize = 128
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-4
    val_check_interval = 0.11
    weight_decay = 2e-2
    optim_type = "adam"
    lr_mult = 5
    num_gpus = [1]


@ex.named_config
def test_performaf():
    exp_name = "test_performaf"
    datasets = ["performaf"]
    data_root = r".../multi-view-biosignal-v1/datasets/PERFormAF"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PERFormAF'
    missing_ratio = {'train': 0.7, 'val': 0, 'test': 0.7}
    missing_type = {'train': 'ppg', 'val': 'both', 'test': 'ecg'}
    loss_names = _loss_names({"afd": 1})
    prompt_length = 20
    num_gpus = [1]
    test_only = True
    load_path = ""
    precision = 32


@ex.named_config
def pretrain_mesa():
    exp_name = "pretrain_mesa"
    datasets = ["mesa"]
    data_root = ".../multi-view-biosignal-v1/datasets/MESA"
    missing_table_root = '.../multi-view-biosignal-v1/datasets/MESA'
    ppg_size = 3000
    ppg_patch_size = 50
    ecg_size = 3000
    ecg_patch_size = 50
    train_transform_key = None
    val_transform_key = None
    per_gpu_batchsize = 1024
    loss_names = _loss_names({"msm": 1, "cep": 0.2})
    num_gpus = [0]
#    optim_type = "adam"
    resume_from = None


@ex.named_config
def fromscratch_mesa():
    exp_name = "fs_mesa"
    datasets = ["mesa"]
    data_root = ".../multi-view-biosignal-v1/datasets/MESA"
    missing_table_root = '.../multi-view-biosignal-v1/datasets/MESA'
    ppg_size = 3000
    ecg_size = 3000
    train_transform_key = None
    val_transform_key = None
    per_gpu_batchsize = 1024
    loss_names = _loss_names({"ssc": 1})
    num_gpus = [1]
    optim_type = "adam"
    resume_from = None


@ex.named_config
def finetune_mesa():
    exp_name = "ft_mesa"
    datasets = ["mesa"]
    data_root = ".../multi-view-biosignal-v1/datasets/MESA"
    missing_table_root = '.../multi-view-biosignal-v1/datasets/MESA'
    missing_ratio = {'train': 0.7, 'val': 0, 'test': 0.7}
    missing_type = {'train': 'ecg', 'val': 'both', 'test': 'ecg'}
    ppg_size = 3000
    ecg_size = 3000
    prompt_length = 20
    load_path = "./"
    # dataset_ratio = 0.3
    loss_names = _loss_names({"ssc": 1})
    train_transform_key = None
    val_transform_key = None
    finetune_first = True
    per_gpu_batchsize = 1024
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-3
    # val_check_interval = 0.11
    weight_decay = 2e-2
    # optim_type = "adam"
    lr_mult = 5
    num_gpus = [1]


@ex.named_config
def test_mesa():
    exp_name = ""
    datasets = ["mesa"]
    data_root = ".../multi-view-biosignal-v1/datasets/MESA"
    missing_table_root = '.../multi-view-biosignal-v1/datasets/MESA'
    missing_ratio = {'train': 0.1, 'val': 0, 'test': 0.7}
    missing_type = {'train': 'ecg', 'val': 'both', 'test': 'ppg'}
    prompt_length = 20
    ppg_size = 3000
    ecg_size = 3000
    loss_names = _loss_names({"ssc": 1})
    num_gpus = [0]
    test_only = True
    load_path = ""
    precision = 32



@ex.named_config
def finetune_missratio09_ecg():
    exp_name = ""
    datasets = ["performaf"]
    data_root = r".../multi-view-biosignal-v1/datasets/PERFormAF"
    missing_table_root = r'.../multi-view-biosignal-v1/datasets/PERFormAF'
    missing_ratio = {'train': 0.9, 'val': 0, 'test': 0.9}
    missing_type = {'train': 'ecg', 'val': 'both', 'test': 'ecg'}
    load_path = ""
    dataset_ratio = 1.0
    loss_names = _loss_names({"afd": 1})
    train_transform_key = None
    val_transform_key = None
    finetune_first = True
    per_gpu_batchsize = 128
    prompt_length = 20
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1e-4
    val_check_interval = 0.11
    weight_decay = 2e-2
    optim_type = "adam"
    lr_mult = 5
    num_gpus = [1]

