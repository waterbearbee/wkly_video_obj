## Zhenheng Yang
## 06/29/2018
## --------------------------------------------------------------
## Evaluation of object detection on Charades data for each class
## --------------------------------------------------------------
import numpy as np
import os
import pickle

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def charades_eval(
    detpath, recs, imagesetfile, classname, ovthresh=0.5, use_07_metric=False
):
    """rec, prec, ap = charades_eval(detpath,
                               recs,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])

   detpath: Path to detections
       detpath.format(classname) should produce the detection results file in txt.
       Txt files are written by _write_voc_results_file function
   recs: Dictionary that stores the detection results for each image
   imagesetfile: Text file containing the list of images, one image per line.
   classname: Category name
   [ovthresh]: Overlap threshold (default = 0.5)
   [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default False, meaning use VOC12 metric)
   """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # for charades, recs is already the loaded ground truth
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.split(".")[0].split("/")[1] for x in lines]

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    corloc_imgs = []
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj[1] == classname]
        bbox = np.array([x[0] for x in R])
        difficult = np.array([False for x in R]).astype(np.bool)
        for i, x in enumerate(R):
            if np.min([x[0][2]-x[0][0], x[0][3]-x[0][1]]) < 30: 
                difficult[i] = True
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0].split('.')[0].split('/')[1] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            if image_ids[d] not in imagenames: continue
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R["bbox"].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = (
                    (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.)
                    - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R["difficult"][jmax]:
                    if not R["det"][jmax]:
                        tp[d] = 1.
                        R["det"][jmax] = 1
            else:
                fp[d] = 1.
            if np.all(R["det"]):
                corloc_imgs.append(image_ids[d])


    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    if len(image_ids) == 0:
        corloc = 0
    else:
        corloc = len(list(set(corloc_imgs)))/float(len(list(set(image_ids))))

    return rec, prec, ap, corloc