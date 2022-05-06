""" Evaluation script for the CULane metric on the LLAMAS dataset.
This script will compute the F1, precision and recall metrics as described in the CULane benchmark.
The predictions format is the same one used in the CULane benchmark.
In summary, for every annotation file:
    labels/a/b/c.json
There should be a prediction file:
    predictions/a/b/c.lines.txt
Inside each .lines.txt file each line will contain a sequence of points (x, y) separated by spaces.
For more information, please see https://xingangpan.github.io/projects/CULane.html
This script uses two methods to compute the IoU: one using an image to draw the lanes (named `discrete` here) and
another one that uses shapes with the shapely library (named `continuous` here). The results achieved with the first
method are very close to the official CULane implementation. Although the second should be a more exact method and is
faster to compute, it deviates more from the official implementation. By default, the method closer to the official
metric is used.
"""

import os
import argparse
from functools import partial

import cv2
import numpy as np
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon

import clrnet.utils.llamas_utils as llamas_utils

LLAMAS_IMG_RES = (717, 1276)


def add_ys(xs):
    """For each x in xs, make a tuple with x and its corresponding y."""
    xs = np.array(xs[300:])
    valid = xs >= 0
    xs = xs[valid]
    assert len(xs) > 1
    ys = np.arange(300, 717)[valid]
    return list(zip(xs, ys))


def draw_lane(lane, img=None, img_shape=None, width=30):
    """Draw a lane (a list of points) on an image by drawing a line with width `width` through each
    pair of points i and i+i"""
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1, ), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=LLAMAS_IMG_RES):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys by drawing the lanes on
    an image"""
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # IoU by the definition: sum all intersections (binary and) and divide by the sum of the union (binary or)
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=LLAMAS_IMG_RES):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys using the area between each
    pair of points"""
    h, w = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interpolate_lane(points, n=50):
    """Spline interpolation of a lane. Used on the predictions"""
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., n)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  unofficial=False,
                  img_shape=LLAMAS_IMG_RES):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    """Computes CULane's metric for a single image"""
    if len(pred) == 0:
        return 0, 0, len(anno), _metric
    if len(anno) == 0:
        return 0, len(pred), 0, _metric

    interp_pred = np.array([
        interpolate_lane(pred_lane, n=50) for pred_lane in pred
    ])  # (4, 50, 2)
    anno = np.array([np.array(anno_lane) for anno_lane in anno], dtype=object)

    if unofficial:
        ious = continuous_cross_iou(interp_pred, anno, width=width)
    else:
        ious = discrete_cross_iou(interp_pred,
                                  anno,
                                  width=width,
                                  img_shape=img_shape)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]

    return _metric


def load_prediction(path):
    """Loads an image's predictions
    Returns a list of lanes, where each lane is a list of points (x,y)
    """
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_prediction_list(label_paths, pred_dir):
    return [
        load_prediction(
            os.path.join(pred_dir, path.replace('.json', '.lines.txt')))
        for path in label_paths
    ]


def load_labels(label_dir):
    """Loads the annotations and its paths
    Each annotation is converted to a list of points (x, y)
    """
    label_paths = llamas_utils.get_files_from_folder(label_dir, '.json')
    annos = [
        [
            add_ys(xs) for xs in
            llamas_utils.get_horizontal_values_for_four_lanes(label_path)
            if (np.array(xs) >= 0).sum() > 1
        ]  # lanes annotated with a single point are ignored
        for label_path in label_paths
    ]
    label_paths = [llamas_utils.get_label_base(p) for p in label_paths]
    return np.array(annos, dtype=object), np.array(label_paths, dtype=object)


def eval_predictions(pred_dir,
                     anno_dir,
                     width=30,
                     iou_thresholds=[0.5],
                     unofficial=True,
                     sequential=False):
    """Evaluates the predictions in pred_dir and returns CULane's metrics (precision, recall, F1 and its components)"""
    print(f'Loading annotation data ({anno_dir})...')
    os.makedirs('cache', exist_ok=True)
    annotations_path = 'cache/llamas_annotations.pkl'
    label_path = 'cache/llamas_label_paths.pkl'
    import pickle as pkl
    if os.path.exists(annotations_path) and os.path.exists(label_path):
        with open(annotations_path, 'rb') as cache_file:
            annotations = pkl.load(cache_file)
        with open(label_path, 'rb') as cache_file:
            label_paths = pkl.load(cache_file)
    else:
        annotations, label_paths = load_labels(anno_dir)
        with open(annotations_path, 'wb') as cache_file:
            pkl.dump(annotations, cache_file)
        with open(label_path, 'wb') as cache_file:
            pkl.dump(label_paths, cache_file)
        
    print(f'Loading prediction data ({pred_dir})...')
    predictions = load_prediction_list(label_paths, pred_dir)
    print('Calculating metric {}...'.format(
        'sequentially' if sequential else 'in parallel'))
    if sequential:
        results = map(
            partial(culane_metric,
                    width=width,
                    unofficial=unofficial,
                    img_shape=LLAMAS_IMG_RES), predictions, annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric, zip(predictions, annotations,
                        repeat(width),
                        repeat(iou_thresholds),
                        repeat(unofficial),
                        repeat(LLAMAS_IMG_RES)))

    import logging
    logger = logging.getLogger(__name__)
    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {}, '
                'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(total_tp, total_fp,
            total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure CULane's metric on the LLAMAS dataset")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True)
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True)
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--sequential",
                        action='store_true',
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--unofficial",
                        action='store_true',
                        help="Use a faster but unofficial algorithm")

    return parser.parse_args()


def main():
    args = parse_args()
    results = eval_predictions(args.pred_dir,
                               args.anno_dir,
                               width=args.width,
                               iou_thresholds=[0.5],
                               unofficial=args.unofficial,
                               sequential=args.sequential)

    header = '=' * 20 + ' Results' + '=' * 20
    print(header)
    for metric, value in results.items():
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print('=' * len(header))


if __name__ == '__main__':
    main()
