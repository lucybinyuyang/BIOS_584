import numpy as np
import matplotlib.pyplot as plt

def produce_trun_mean_cov(input_signal, input_type, E_val):
    sample_size, feature_len = input_signal.shape
    length_per_electrode = int(feature_len / E_val)

    uniq = np.unique(input_type)
    if set(uniq) == {-1, 1}:
        target_label, nontarget_label = 1, -1
    elif set(uniq) == {1, 2}:
        target_label, nontarget_label = 1, 2
    else:
        target_label, nontarget_label = 1, 0

    signal_tar = input_signal[input_type == target_label, :]
    signal_ntar = input_signal[input_type == nontarget_label, :]

    signal_tar_3d = signal_tar.reshape(signal_tar.shape[0], E_val, length_per_electrode)
    signal_ntar_3d = signal_ntar.reshape(signal_ntar.shape[0], E_val, length_per_electrode)
    signal_all_3d = input_signal.reshape(sample_size, E_val, length_per_electrode)

    signal_tar_mean = np.mean(signal_tar_3d, axis=0)
    signal_ntar_mean = np.mean(signal_ntar_3d, axis=0)

    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))

    for e in range(E_val):
        if signal_tar_3d.shape[0] > 1:
            signal_tar_cov[e] = np.cov(signal_tar_3d[:, e, :], rowvar=False)
        if signal_ntar_3d.shape[0] > 1:
            signal_ntar_cov[e] = np.cov(signal_ntar_3d[:, e, :], rowvar=False)
        signal_all_cov[e] = np.cov(signal_all_3d[:, e, :], rowvar=False)

    return signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov


def _grid_order_from_names(electrode_name_ls):
    return np.arange(len(electrode_name_ls))


def plot_trunc_mean(eeg_tar_mean, eeg_ntar_mean, subject_name, time_index,
                    E_val, electrode_name_ls, save_path,
                    y_limit=(-5, 8), fig_size=(12, 12)):
    L = eeg_tar_mean.shape[1]
    order = _grid_order_from_names(electrode_name_ls)
    fig, axes = plt.subplots(4, 4, figsize=fig_size, constrained_layout=True)
    axes = axes.ravel()
    t = time_index if time_index.shape[0] == L else np.arange(L)

    for k, ei in enumerate(order):
        ax = axes[k]
        ax.plot(t, eeg_tar_mean[ei], color="red", label="Target", linewidth=1.8)
        ax.plot(t, eeg_ntar_mean[ei], color="blue", label="Non-Target", linewidth=1.6)
        ax.set_title(electrode_name_ls[ei], fontsize=10)
        ax.set_ylim(*y_limit)
        ax.set_xlim(t[0], t[-1])
        if k == 0:
            ax.legend(fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)

    fig.suptitle(f"Mean ERP — {subject_name}", fontsize=14)
    plt.savefig(save_path)
    plt.close()


def plot_trunc_cov(cov_3d, subject_title, electrode_name_ls, save_path,
                   fig_size=(12, 12), cmap="viridis"):
    E_val, L, _ = cov_3d.shape
    order = _grid_order_from_names(electrode_name_ls)
    vmax = np.nanmax(np.abs(cov_3d))
    vmin = -vmax
    fig, axes = plt.subplots(4, 4, figsize=fig_size, constrained_layout=True)
    axes = axes.ravel()

    for k, ei in enumerate(order):
        ax = axes[k]
        im = ax.imshow(cov_3d[ei], vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(electrode_name_ls[ei], fontsize=10)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Time (samples)")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9, pad=0.01)
    cbar.set_label("Covariance")
    fig.suptitle(subject_title, fontsize=14)
    plt.savefig(save_path)
    plt.close()
