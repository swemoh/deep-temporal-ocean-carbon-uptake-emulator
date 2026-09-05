import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def build_xy_from_dicts(feature_dict, target_dict, lookback=6, dtype=np.float32,
                        start_year=1958):
    """
    Returns:
      X_all:    (N_samples, lookback, 8)
      y_all:    (N_samples, 1)
      coord_all:(N_samples, 2)   lat, lon
      time_all: (N_samples, 2)   year, month
    """
    X_list, y_list, coord_list, time_list = [], [], [], []

    for (lat, lon), year_blocks in feature_dict.items():
        if (lat, lon) not in target_dict:
            continue

        # (T, 8)
        X_ts = np.concatenate(year_blocks, axis=0).astype(dtype, copy=False)
        # (T, 1)
        y_ts = np.concatenate(target_dict[(lat, lon)], axis=0).astype(dtype, copy=False)

        T, F = X_ts.shape
        if F != 8:
            raise ValueError(f"Expected 8 features, got {F}")
        if y_ts.shape != (T, 1):
            raise ValueError(f"Target shape mismatch: {y_ts.shape} vs {(T,1)}")

        # pad for early months
        pad = np.zeros((lookback - 1, F), dtype=dtype)
        X_pad = np.vstack([pad, X_ts])  # (T+lookback-1, 8)  (allocates)

        # windows as a view (no copy)
        X_win = sliding_window_view(X_pad, window_shape=(lookback,), axis=0)  # (T, 8, lookback)
        X_win = np.transpose(X_win, (0, 2, 1))  # (T, lookback, 8)

        y_win = y_ts  # (T, 1)

        X_list.append(X_win)
        y_list.append(y_win)

        # coords (T,2)
        coord_list.append(np.tile(np.array([[lat, lon]], dtype=dtype), (T, 1)))

        # time (T,2): year, month
        # t=0 is Jan of start_year
        t = np.arange(T, dtype=np.int32)
        years = start_year + (t // 12)
        months = 1 + (t % 12)
        time_list.append(np.stack([years, months], axis=1).astype(np.int16, copy=False))

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    coord_all = np.concatenate(coord_list, axis=0)
    time_all = np.concatenate(time_list, axis=0)

    return X_all, y_all, coord_all, time_all