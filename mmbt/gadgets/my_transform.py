from tsaug import Pool, Drift, Reverse
from scipy.interpolate import interp1d
import numpy as np


def Pool_Drift(ts):  # ts: narray, [channel, length]
    ts_aug = Pool(size=2).augment(ts) if np.random.rand() < 0.5 else ts
    ts_aug = Drift(max_drift=0.2, n_drift_points=200).augment(ts_aug) if np.random.rand() < 0.5 else ts_aug

    # import matplotlib.pyplot as plt
    # plt.plot(ts_aug[0])
    # plt.show()

    return ts_aug


def Reverse_Drift(ts):
    ts_aug = Reverse().augment(ts) if np.random.rand() < 0.5 else ts
    ts_aug = Drift(max_drift=0.2, n_drift_points=200).augment(ts_aug) if np.random.rand() < 0.5 else ts_aug

    return ts_aug


def Crop_Resize(ts, crop_ratio=0.5):
    assert crop_ratio <= 1.0
    if np.random.rand() < 0.5:
        crop_length = int(ts.shape[1] * crop_ratio)
        start_idx = np.random.randint(0, ts.shape[1] - crop_length)
        ts_cropped = ts[:, start_idx:start_idx + crop_length]

        raw_idx_ts_cropped = np.arange(ts_cropped.shape[1])
        new_idx_ts_cropped = np.linspace(0, ts_cropped.shape[1] - 1, ts.shape[1])
        interpolator = interp1d(raw_idx_ts_cropped, ts_cropped, axis=1, kind='cubic')
        ts_aug = interpolator(new_idx_ts_cropped)

        return ts_aug
    else:
        return ts


def Identity(ts):
    return ts


def key_to_transform(key):
    _transforms = {
        "default": Identity,
        "pooldrift": Pool_Drift,
        "reversedrifit": Reverse_Drift,
        "cropresize": Crop_Resize,
    }
    return _transforms[key]


# def ts_transform():
#     my_aug = [
#         AddNoise(scale=0.02),
#         Drift(max_drift=0.2, n_drift_points=10)
#     ]
#     return my_aug