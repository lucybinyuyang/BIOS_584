import os
import numpy as np
import scipy.io as sio
from HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

# === Global settings ===
bp_low, bp_upp = 0.5, 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                     'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
parent_dir = r"C:\Users\lucyy\Documents\GitHub\BIOS_584"
subject_name, session_name = "K114", "001_BCI_TRN"
data_path = os.path.join(parent_dir, "data",
                         f"{subject_name}_{session_name}_Truncated_Data_{bp_low}_{bp_upp}.mat")
time_index = np.linspace(0, 800, 25)

# === Load data ===
mat_data = sio.loadmat(data_path)
input_signal = mat_data['Signal']
input_type = mat_data['Type'].squeeze()
print("Unique values in input_type:", np.unique(input_type))

# === Compute means & covariances ===
signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov = \
    produce_trun_mean_cov(input_signal, input_type, electrode_num)

# === Create save directory ===
save_dir = os.path.join(parent_dir, subject_name)
os.makedirs(save_dir, exist_ok=True)

# === Save all figures ===
plot_trunc_mean(signal_tar_mean, signal_ntar_mean, subject_name,
                time_index, electrode_num, electrode_name_ls,
                save_path=os.path.join(save_dir, "Mean.png"))

plot_trunc_cov(signal_tar_cov, f"{subject_name} - Target", electrode_name_ls,
               save_path=os.path.join(save_dir, "Covariance_Target.png"))
plot_trunc_cov(signal_ntar_cov, f"{subject_name} - Non-Target", electrode_name_ls,
               save_path=os.path.join(save_dir, "Covariance_Non-Target.png"))
plot_trunc_cov(signal_all_cov, f"{subject_name} - All", electrode_name_ls,
               save_path=os.path.join(save_dir, "Covariance_All.png"))
