import numpy as np
import matplotlib.pyplot as plt

def get_bbox_lst(res_path):
    file = open(res_path, 'r')
    lines = file.readlines()
    file.close()
    bbox_lst = []
    for line in lines:
        items = line.strip().split()
        items = [float(c) for c in items[1 :]]
        bboxes = [items[i : i + 4] for i in range(0, len(items) - 4, 5)]
        bbox_lst.append(bboxes)
    return bbox_lst

def get_bbox_conf(pred_path):
    file = open(pred_path, 'r')
    lines = file.readlines()
    file.close()
    bbox_conf_lst = []
    for line in lines:
        items = line.strip().split()
        items = [float(c) for c in items[1 :]]
        bbox_conf = [items[i] for i in range(4, len(items), 5)]
        bbox_conf_lst.append(bbox_conf)
    return bbox_conf_lst

def cal_overlap_area(pred_bbox, truth_bbox):
    overlap_x = (pred_bbox[2] - pred_bbox[0]) + (truth_bbox[2] - truth_bbox[0]) - (max(pred_bbox[2], truth_bbox[2]) - min(pred_bbox[0], truth_bbox[0]))
    overlap_y = (pred_bbox[3] - pred_bbox[1]) + (truth_bbox[3] - truth_bbox[1]) - (max(pred_bbox[3], truth_bbox[3]) - min(pred_bbox[1], truth_bbox[1]))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return overlap_x * overlap_y * 1.0


def cal_overlap_ratio(pred_bbox, truth_bbox):
    overlap_area = cal_overlap_area(pred_bbox, truth_bbox)
    pred_area = 1.0 * (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    truth_area = 1.0 * (truth_bbox[2] - truth_bbox[0]) * (truth_bbox[3] - truth_bbox[1])
    union_area = pred_area + truth_area - overlap_area
    ratio = overlap_area / union_area
    return ratio


def eval_single_image(triplet, threshold):
    pred_bbox_lst = triplet[0]
    truth_bbox_lst = triplet[1]
    bbox_conf_lst = triplet[2]
    ratio_matrix = np.array([[cal_overlap_ratio(pred_bbox, truth_bbox) for pred_bbox in pred_bbox_lst] for truth_bbox in truth_bbox_lst])
    pred_flag = np.zeros(len(pred_bbox_lst))
    pred_ratio = np.zeros(len(pred_bbox_lst))
    truth_flag = np.zeros(len(truth_bbox_lst))

    # predict no bbox in the image
    if ratio_matrix.shape[0] == 0 or ratio_matrix.shape[1] == 0:
        result = []
        for thresh in threshold:
            tmp_res = np.zeros(4)
            tmp_res[3] = len(truth_bbox_lst)    # FN
            result.append(tmp_res)
        return result

    # result: [TP, TN, FP, FN]
    result = []
    for thresh in threshold:
        tmp_ratio_matrix = ratio_matrix

        # ignore bbox whose conf < thresh
        for i in range(len(pred_bbox_lst)):
            if bbox_conf_lst[i] < thresh:
                for j in range(len(truth_bbox_lst)):
                    tmp_ratio_matrix[j][i] = 0

        # match pred bbox with ground truth
        while np.max(ratio_matrix) != 0:
            ratio = np.max(tmp_ratio_matrix)
            index = np.where(tmp_ratio_matrix == ratio)
            pred_flag[index[1][0]] = 1
            truth_flag[index[0][0]] = 1
            pred_ratio[index[1][0]] = ratio
            ratio_matrix[:, index[1][0]] = 0
            ratio_matrix[index[0][0], :] = 0

        tmp_res = np.zeros(4)
        for item in zip(pred_ratio, bbox_conf_lst, pred_flag):
            # predicted as positive
            if item[1] >= thresh:
                if item[0] >= 0.5:
                    tmp_res[0] += 1    # TP
                else:
                    tmp_res[2] += 1    # FP
        # FN
        for item in truth_flag:
            if item == 0:
                tmp_res[3] += 1
        result.append(tmp_res)
    return result


def evaluate(pred_path, truth_path):
    pred_bbox_lst = get_bbox_lst(pred_path)
    bbox_conf_lst = get_bbox_conf(pred_path)
    truth__bbox_lst = get_bbox_lst(truth_path)

    conf_thresh = 0.0
    thresh = np.arange(conf_thresh, 1.0, 0.001)
    result = np.zeros((len(thresh), 4))
    for triplet in zip(pred_bbox_lst, truth__bbox_lst, bbox_conf_lst):
        tmp_res = np.array(eval_single_image(triplet, thresh))
        result += tmp_res

    precision = [item[0] * 1.0 / (item[0] + item[2]) for item in result]
    recall = [item[0] * 1.0 / (item[0] + item[3]) for item in result]
    res_triplet = zip(precision, recall)
    res_triplet.append((1.0, 0))
    res_triplet.append((0, 1.0))
    res_triplet = sorted(res_triplet, key=lambda  x : x[1])
    # compute average precision (integration of the precsion/recall curve)
    ap = 0.0
    for i in range(0, len(res_triplet) - 1):
        tuple1 = res_triplet[i]
        tuple2 = res_triplet[i + 1]
        tmp_area = 0.5 * (tuple2[1] - tuple1[1]) * (tuple1[0] + tuple2[0])
        ap += tmp_area
    print "average precision: " + str(ap)

    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()




if __name__ == '__main__':
    pred_path = '/home/mingdongwang/Desktop/UCSD/mx-rcnn/data/cache/results/val.lst'
    truth_path = '/home/mingdongwang/Desktop/UCSD/mx-rcnn/data/custom_0/imglists/val.lst'
    evaluate(pred_path, truth_path)