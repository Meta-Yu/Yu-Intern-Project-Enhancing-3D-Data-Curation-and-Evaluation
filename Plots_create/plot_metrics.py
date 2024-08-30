import matplotlib.pyplot as plt
import numpy as np

# ape = np.array([0.3985, 0.3266, 0.6361, 0.2476, 0.4807, 0.8558, 0.2894, 0.4775, 0.1772, 1.1133])
cd = np.array([0.0248, 0.2275, 0.0242, 0.0625, 0.0359, 0.0754, 0.045, 0.0513, 0.0267, 0.0987, 0.302, 0.412])
dcd = np.array([0.3746, 0.8483, 0.4556, 0.7143, 0.6456, 0.5991, 0.5574, 0.685, 0.6279, 0.8176, 0.868, 0.901])

dcd_2 = np.array([0.8746, 0.8483, 0.8556, 0.9143, 0.9456, 0.8991, 0.9574, 0.985, 0.9279, 0.9176, 0.968, 0.999])
dcd_3 = np.array([0.9746, 0.9483, 0.9556, 0.9143, 0.9456, 0.9991, 0.9574, 0.985, 0.9279, 0.9176, 0.968, 0.999])

x = np.arange(len(dcd)) + 1

# Sort indices of ape array
sorted_indices = np.argsort(cd)

# sorted_ape = ape[sorted_indices]
# Sort cd and dcd arrays using sorted indices
sorted_cd = cd[sorted_indices]
sorted_dcd = dcd[sorted_indices]


# Plot cd vs. ape
p1 = 1
p2 = 6
selected  = np.index_exp[p1, p2]
plt.scatter([x[1], x[6]], [sorted(cd)[1], sorted(cd)[6]], c='green', marker='o')

plt.scatter([x[1], x[6]], [sorted(dcd)[1], sorted(dcd)[6]], c='blue', marker='o')
plt.plot(x, sorted(cd), label = 'CD', c='green')
plt.plot(x, sorted(dcd), label = r'DCD ($ \alpha=1$)', c='blue')
plt.plot(x, sorted(dcd_2), label = r'DCD ($ \alpha=100$)')
plt.plot(x, sorted(dcd_3), label = r'DCD ($ \alpha=1000$)')
plt.axvline(x=9, color='gray', linestyle='dashed')
plt.xlabel('Noise Level')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/private/home/wangyu1369/dust3r/dcd/cd_and_dcd.png', bbox_inches='tight')