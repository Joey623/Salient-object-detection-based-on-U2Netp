import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import cv2 as cv
import seaborn as sns
# sns.set()
sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
# add the noise in the mask
def mask_normalize(mask):
    return mask / (np.amax(mask) + 1e-8)

# compute a image's mae
def compute_mae(mask1, mask2):
    # _, mask1 = cv.threshold(mask1, 125, 255, cv.THRESH_BINARY)
    # _, mask2 = cv.threshold(mask2, 125, 255, cv.THRESH_BINARY)
    if(len(mask1.shape) < 2 or len(mask2.shape) < 2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape) > 2):
        mask1 = mask1[:, :, 0]
    if(len(mask2.shape) > 2):
        mask2 = mask2[:, :, 0]
    if(mask1.shape != mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()

    h, w = mask1.shape[0], mask1.shape[1]
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError / (float(h) * float(w) + 1e-8)

    return maeError

def compute_ave_MAE_of_methods(gt_path, res_path):
    res_list = os.listdir(res_path)
    mae = []
    gt2rs = []
    for i in range(len(res_list)):
        r_name = res_path + res_list[i]   # indicate map
        g_name = gt_path + res_list[i]
        res = cv.imread(r_name)    # red the indicating map
        gt = cv.imread(g_name)
        tmp_map = compute_mae(gt, res)
        mae.append(tmp_map)
        gt2rs.append(1)
    mae_sum = np.sum(mae)
    gt2rs_sum = np.sum(gt2rs)
    ave_maes = mae_sum / (gt2rs_sum + 1e-8)
    return ave_maes

# compute precision and recall
def compute_pre_rec(gt, mask, mybins = np.arange(0, 256)):
    # _, mask = cv.threshold(mask, 125, 255, cv.THRESH_BINARY)
    # _, mask = cv.threshold(mask, 128, 255, cv.THRESH_TOZERO)
    if(len(gt.shape) < 2 or len(mask.shape) < 2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape) > 2):
        gt = gt[:, :, 0]
    if(len(mask.shape) > 2):
        mask = mask[:, :, 0]
    if(gt.shape != mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt > 128].size   # calculate the number of pixel of ground truth's foreground regions
    # print("gtNum: ", gtNum)
    pp = mask[gt > 128]      # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt <= 128]     # mask predicted pixel values in the ground truth background region

    # count pixel numbers with values in each interval [0, 1),[1,2),...,[254,255)
    pp_hist, pp_edges = np.histogram(pp, bins=mybins)
    nn_hist, nn_edges = np.histogram(nn, bins=mybins)

    # reverse the histogram to the following order(255, 254]...(1, 0]
    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)
    # accumulate the pixel in intervals:(255,254],...,(1,0]
    pp_hist_flip_cum = np.cumsum(pp_hist_flip)    # TP
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)    # FP

    # calculate the precision and recall
    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-8)
    # recall = pp_hist_flip_cum / (pp_hist_flip_cum + gtNum + 1e-8)
    recall = pp_hist_flip_cum / (gtNum + 1e-8)

    precision[np.isnan(precision)] = 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision, (len(precision))), np.reshape(recall, (len(recall)))


def compute_PRE_REC_FM_of_methods(gt_path, res_path, beta=0.3):
    mybins = np.arange(0, 256)
    gt_list = os.listdir(gt_path)
    num_gt = len(gt_list)
    if num_gt == 0:
        print("ERROR: The ground truth directory is empty!")
        exit()
    PRE = np.zeros((num_gt, 1, len(mybins)-1))
    # print("PRE's shape: ", PRE.shape)
    REC = np.zeros((num_gt, 1, len(mybins)-1))
    gt2rs = np.zeros((num_gt, 1))
    for i in range(len(gt_list)):
        g_name = gt_path + gt_list[i]
        r_name = res_path + gt_list[i]
        gt = cv.imread(g_name)
        gt = mask_normalize(gt) * 255.0
        res = cv.imread(r_name)
        res = mask_normalize(res) * 255.0
        pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins))
        pre, rec = compute_pre_rec(gt, res, mybins=np.arange(0, 256))
        # PRE[i, 1, :] = pre
        # REC[i, 1, :] = rec
        PRE[i, :, :] = pre
        REC[i, :, :] = rec
        gt2rs[i, :] = 1.0
    gt2rs = np.sum(gt2rs, 0)
    gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1)

    PRE = np.sum(PRE, 0) / (gt2rs + 1e-8)
    REC = np.sum(REC, 0) / (gt2rs + 1e-8)
    FM = (1 + beta) * PRE * REC / (beta * PRE + REC +1e-8)

    return PRE, REC, FM, gt2rs




def plot_save_pr_curves(PRE, REC, xrange=(0.0, 1.0), yrange=(0.0, 1.0)):
    fig1 = plt.figure(1)
    num = PRE.shape[0]
    for i in range(0, num):
        if len(np.array(PRE[i]).shape) != 0:
            plt.plot(REC[i], PRE[i], '--', linewidth=1, alpha=0.7)

    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])

    xyrange1 = np.arange(xrange[0], xrange[1] + 0.01, 0.1)
    xyrange2 = np.arange(yrange[0], yrange[1] + 0.01, 0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1, fontsize=10, fontname='serif')
    plt.yticks(xyrange2, fontsize=10, fontname='serif')
    plt.text((xrange[0] + xrange[1]) / 2.0, yrange[0] + 0.02, 'UNet-Light', horizontalalignment='center', fontsize=10, fontname='serif', fontweight='bold')
    plt.xlabel('Recall', fontsize=10, fontname='serif')
    plt.ylabel('Precision', fontsize=10, fontname='serif')
    font1={'family': 'serif',
           'weight': 'normal',
           'size': 7,
    }
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [len(handles) - x for x in range(1, len(handles) + 1)]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', prop=font1)
    # plt.grid(linstyle='--')
    # plt.show()
    # plt.text((xrange[0] + xrange[1]) / 2.0, yrange[0] + 0.02, dataset_name, horizontalalignment='center', fontsize=20,
    #          fontname='serif', fontweight='bold')



if __name__ == '__main__':
    ecssd_res_path = 'ECSSD/images/'
    ecssd_gt_path = 'ECSSD/gt/'
    hku_res_path = 'HKU-IS/images/'
    hku_gt_path = 'HKU-IS/gt/'
    duts_res_path = 'DUTS-TE/images/'
    duts_gt_path = 'DUTS-TE/gt/'

    # lineSylClr = ['r-', 'b-', 'g-']  # curve style, same with the num of dataset
    print("-----------loading your model:UNet-Light...")
    print("MAE:")
    print("ECSSD:", compute_ave_MAE_of_methods(ecssd_gt_path, ecssd_res_path))
    print("HKU-IS:", compute_ave_MAE_of_methods(hku_gt_path, hku_res_path))
    print("DUTS-TE:", compute_ave_MAE_of_methods(duts_gt_path, duts_res_path))
    print("-"*30)
    ecssd_PRE, ecssd_REC, ecssd_FM, ecssd_gt2rs_fm = compute_PRE_REC_FM_of_methods(ecssd_gt_path, ecssd_res_path)
    hku_PRE, hku_REC, hku_FM, hku_gt2rs_fm = compute_PRE_REC_FM_of_methods(hku_gt_path, hku_res_path)
    duts_PRE, duts_REC, duts_FM, duts_gt2rs_fm = compute_PRE_REC_FM_of_methods(duts_gt_path, duts_res_path)
    for i in range(0, ecssd_PRE.shape[0]):
        print("ECSSD: maxF->%.6f, "%(np.max(ecssd_FM, 1)[i]), "meanF->%.6f" %(np.mean(ecssd_FM, 1)[i]))
    for i in range(0, hku_PRE.shape[0]):
        print("HKU-IS: maxF->%.6f, "%(np.max(hku_FM, 1)[i]), "meanF->%.6f" %(np.mean(hku_FM, 1)[i]))
    for i in range(0, duts_PRE.shape[0]):
        print("DUTS-TE: maxF->%.6f, "%(np.max(duts_FM, 1)[i]), "meanF->%.6f" %(np.mean(duts_FM, 1)[i]))
    plot_save_pr_curves(ecssd_PRE, ecssd_REC, xrange=(0.5, 1.0), yrange=(0.5, 1.0))
    plot_save_pr_curves(hku_PRE, hku_REC, xrange=(0.5, 1.0), yrange=(0.5, 1.0))
    plot_save_pr_curves(duts_PRE, duts_REC, xrange=(0.5, 1.0), yrange=(0.5, 1.0))
    dataset = ["ECSSD", "HKU-IS", "DUTS-TE"]
    plt.legend(dataset, loc='lower left')
    # plt.title("PR-Curve", fontsize=10, fontname='serif')
    plt.title("PR-Curve", fontname='serif')
    plt.savefig("prcurve.png")
    plt.show()
    # plt.savefig("prcurve.png")
