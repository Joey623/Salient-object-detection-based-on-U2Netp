import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

'''
some models' F-measure in DUTS-TE, HKU-IS and ECSSD
'''
name_list1 = ['MDF', 'DCL', 'UCF', 'Amulet', 'SRM', 'DSS', 'PiCANet', 'UNet-Light', 'BASNet', 'ITSD', 'MINet', 'PFS']
name_list2 = ['MDF', 'DCL', 'UCF', 'Amulet', 'SRM', 'DSS', 'PiCANet', 'UNet-Light', 'BASNet']
label = ['DUTS-TE', 'HKU-IS', 'ECSSD']

dut_list = [0.729, 0.782, 0.773, 0.778, 0.827, 0.825, 0.851, 0.830, 0.860, 0.885, 0.884, 0.898]
hku_list = [0.860, 0.892, 0.888, 0.895, 0.906, 0.910, 0.919, 0.914, 0.928, 0.935, 0.935, 0.943]
ecssd_list = [0.832, 0.890, 0.903, 0.915, 0.917, 0.916, 0.931, 0.925, 0.942, 0.946, 0.947, 0.952]

mean_dut_list = [0.673, 0.714, 0.629, 0.676, 0.757, 0.791, 0.755, 0.825, 0.845]
mean_hku_list = [0.784, 0.853, 0.808, 0.839, 0.874, 0.895, 0.870, 0.892, 0.914]
mean_ecssd_list = [0.807, 0.829, 0.840, 0.870, 0.892, 0.901, 0.884, 0.908, 0.931]

dut_list = np.array(dut_list)
hku_list = np.array(hku_list)
ecssd_list = np.array(ecssd_list)
mean_dut_list = np.array(mean_dut_list)
mean_hku_list = np.array(mean_hku_list)
mean_ecssd_list = np.array(mean_ecssd_list)


bar_width = 0.8
# bar_width = 1.0
index1 = np.arange(12)
index2 = np.arange(9)

# sns.barplot
plt.bar(3*index1, dut_list, bar_width, alpha=0.7)
plt.bar(3*index1 + bar_width, hku_list, bar_width, alpha=0.7)
plt.bar(3*index1 + 2*bar_width, ecssd_list, bar_width, alpha=0.7)
plt.xticks(3*index1+(bar_width+bar_width)/2, name_list1, fontsize=6.5)
plt.xlabel("Models")
plt.ylim(0.60, 1.00)
plt.yticks(np.arange(0.60, 1.05, 0.05))
# plt.ylabel("F-measure")
plt.ylabel("F-measure")
plt.legend(label, loc='upper left')
# plt.title('F-measure of different methods on different datasets')
plt.title('MAX F-measure of different methods on different datasets')
plt.savefig("maxF.png")
plt.show()
plt.cla()
plt.bar(3*index2, mean_dut_list, bar_width, alpha=0.7)
plt.bar(3*index2 + bar_width, mean_hku_list, bar_width, alpha=0.7)
plt.bar(3*index2 + 2*bar_width, mean_ecssd_list, bar_width, alpha=0.7)

plt.xticks(3*index2+(bar_width+bar_width)/2, name_list2, fontsize=8)
plt.xlabel("Models")

plt.ylim(0.60, 1.00)
plt.yticks(np.arange(0.60, 1.05, 0.05))
# plt.ylabel("F-measure")
plt.ylabel("F-measure")

plt.legend(label, loc='upper left')
# plt.title('F-measure of different methods on different datasets')
plt.title('Mean F-measure of different methods on different datasets')
plt.savefig("meanF.png")
plt.show()
