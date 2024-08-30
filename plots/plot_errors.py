import matplotlib.pyplot as plt
import numpy as np

x = [3, 17, 51]

x_new = [3, 17, 51, 202]

ape_trans_mean_droid_slam = np.array([30.111, 2.556, 0.446, 0.272])
ape_trans_std_droid_slam = np.array([7.674, 1.236, 1.532, 0.167])

ape_rotation_mean_droid_slam = np.array([1.126, 1.122, 0.065, 0.097])
ape_rotation_std_droid_slam = np.array([0.265, 0.432, 0.203, 0.043])

cd_mean_droid_slam = np.array([0.636, 0.401, 0.134, 0.157])
cd_std_droid_slam = np.array([0.344, 0.168, 0.214, 0.167])

time_mean_droid_slam = np.array([1, 12, 33, 68])
time_std_droid_slam = np.array([1, 3, 5, 5])


ape_trans_mean_dust3r = np.array([0.668, 0.536, 0.352])
ape_trans_std_dust3r = np.array([0.197, 0.267, 0.236])

ape_rotation_mean_dust3r = np.array([0.186, 0.087, 0.062])
ape_rotation_std_dust3r = np.array([0.117, 0.038, 0.104])

cd_mean_dust3r = np.array([0.289, 0.238, 0.127])
cd_std_dust3r = np.array([0.124, 0.267, 0.147])

time_mean_dust3r = np.array([35, 260, 660])
time_std_dust3r = np.array([6, 60, 43])

ape_trans_mean_colmap = np.array([0.429])
ape_trans_std_colmap = np.array([0.387])

ape_rotation_mean_colmap = np.array([0.182])
ape_rotation_std_colmap = np.array([0.104])

cd_mean_colmap = np.array([0.228])
cd_std_colmap = np.array([0.237])

time_mean_colmap = np.array([83])
time_std_colmap = np.array([13])



plt.rcParams['font.size'] = 16
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 1 row, 3 columns


axs[0, 0].plot(x_new, ape_trans_mean_droid_slam, label = 'DROID_SLAM', color = 'blue')
axs[0, 0].fill_between(x_new, ape_trans_mean_droid_slam - ape_trans_std_droid_slam, ape_trans_mean_droid_slam + ape_trans_std_droid_slam, alpha=0.2, color = 'blue')
axs[0, 0].plot(x, ape_trans_mean_dust3r, label = 'DUSt3R', color = 'orange')
axs[0, 0].fill_between(x, ape_trans_mean_dust3r - ape_trans_std_dust3r, ape_trans_mean_dust3r + ape_trans_std_dust3r, alpha=0.2, color = 'orange')
axs[0, 0].set_xlabel('Number of Frames')
axs[0, 0].set_ylabel('APE-Translation')

# axs[0, 0].plot(np.array([51]), ape_trans_mean_colmap, "*")
axs[0, 0].errorbar(np.array([51]), ape_trans_mean_colmap, yerr=[ape_trans_std_colmap, ape_trans_std_colmap], 
                   color = 'red',label = 'COLMAP', fmt='o', markersize=6)

axs[0, 0].legend()

# axs[0].savefig('/private/home/wangyu1369/dust3r/error_plots/errors_APE Translation.png', bbox_inches='tight')


axs[0, 1].plot(x_new, ape_rotation_mean_droid_slam, label = 'DROID_SLAM', color = 'blue')
axs[0, 1].fill_between(x_new, ape_rotation_mean_droid_slam - ape_rotation_std_droid_slam, ape_rotation_mean_droid_slam + ape_rotation_std_droid_slam, alpha=0.2, color = 'blue')
axs[0, 1].plot(x, ape_rotation_mean_dust3r, label = 'DUSt3R', color = 'orange')
axs[0, 1].fill_between(x, ape_rotation_mean_dust3r - ape_rotation_std_dust3r, ape_rotation_mean_dust3r + ape_rotation_std_dust3r, alpha=0.2, color = 'orange')
axs[0, 1].set_xlabel('Number of Frames')
axs[0, 1].set_ylabel('APE-Rotation')
axs[0, 1].errorbar(np.array([51]), ape_rotation_mean_colmap, yerr=[ape_rotation_std_colmap, ape_rotation_std_colmap], 
                   color = 'red',label = 'COLMAP', fmt='o', markersize=6)
axs[0, 1].legend()
# plt.savefig('/private/home/wangyu1369/dust3r/error_plots/errors_APE Rotation.png', bbox_inches='tight')


axs[1, 0].plot(x_new, cd_mean_droid_slam, label = 'DROID-SLAM', color= 'blue')
axs[1, 0].fill_between(x_new, cd_mean_droid_slam - cd_std_droid_slam, cd_mean_droid_slam + cd_std_droid_slam, alpha=0.2, color = 'blue')
# axs[2].errorbar(x[0], cd_mean_droid_slam[0], yerr=cd_std_droid_slam[0], fmt='o', label='Point with CI', capsize=5, color = 'blue')
axs[1, 0].plot(x, cd_mean_dust3r, label = 'DUSt3R', color= 'orange')
axs[1, 0].fill_between(x, cd_mean_dust3r - cd_std_dust3r, cd_mean_dust3r + cd_std_dust3r, alpha=0.2, color = 'orange')
axs[1, 0].set_xlabel('Number of Frames')
axs[1, 0].set_ylabel('Chamfer Distance')
axs[1, 0].errorbar(np.array([51]), cd_mean_colmap, yerr=[cd_std_colmap, cd_std_colmap], 
                   color = 'red',label = 'COLMAP', fmt='o', markersize=6)
axs[1, 0].legend()

axs[1, 1].plot(x_new, time_mean_droid_slam, label = 'DROID-SLAM', color= 'blue')
axs[1, 1].fill_between(x_new, time_mean_droid_slam - time_std_droid_slam, time_mean_droid_slam + time_std_droid_slam, alpha=0.2, color = 'blue')
# axs[2].errorbar(x[0], cd_mean_droid_slam[0], yerr=cd_std_droid_slam[0], fmt='o', label='Point with CI', capsize=5, color = 'blue')
axs[1, 1].plot(x, time_mean_dust3r, label = 'DUSt3R', color= 'orange')
axs[1, 1].fill_between(x, time_mean_dust3r - time_std_dust3r, time_mean_dust3r + time_std_dust3r, alpha=0.2, color = 'orange')
axs[1, 1].set_xlabel('Number of Frames')
axs[1, 1].set_ylabel('Time')
axs[1, 1].errorbar(np.array([51]), time_mean_colmap, yerr=[time_std_colmap, time_std_colmap], 
                   color = 'red',label = 'COLMAP', fmt='o', markersize=6)
axs[1, 1].legend()

plt.savefig('/private/home/wangyu1369/dust3r/error_plots/errors.png', bbox_inches='tight')




