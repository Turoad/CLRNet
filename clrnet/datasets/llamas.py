import os
import pickle as pkl
import cv2

from .registry import DATASETS
import numpy as np
from tqdm import tqdm
from .base_dataset import BaseDataset

TRAIN_LABELS_DIR = 'labels/train'
TEST_LABELS_DIR = 'labels/valid'
TEST_IMGS_DIR = 'color_images/test'
SPLIT_DIRECTORIES = {'train': 'labels/train', 'val': 'labels/valid'}
from clrnet.utils.llamas_utils import get_horizontal_values_for_four_lanes
import clrnet.utils.llamas_metric as llamas_metric


@DATASETS.register_module
class LLAMAS(BaseDataset):
    def __init__(self, data_root, split='train', processes=None, cfg=None):
        self.split = split
        self.data_root = data_root
        super().__init__(data_root, split, processes, cfg)
        if split != 'test' and split not in SPLIT_DIRECTORIES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        if split != 'test':
            self.labels_dir = os.path.join(self.data_root,
                                           SPLIT_DIRECTORIES[split])

        self.data_infos = []
        self.load_annotations()

    def get_img_heigth(self, _):
        return self.cfg.ori_img_h

    def get_img_width(self, _):
        return self.cfg.ori_img_w

    def get_metrics(self, lanes, _):
        # Placeholders
        return [0] * len(lanes), [0] * len(lanes), [1] * len(lanes), [
            1
        ] * len(lanes)

    def get_img_path(self, json_path):
        # /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext
        base_name = '/'.join(json_path.split('/')[-3:])
        image_path = os.path.join(
            'color_images', base_name.replace('.json', '_color_rect.png'))
        return image_path

    def get_img_name(self, json_path):
        base_name = (json_path.split('/')[-1]).replace('.json',
                                                       '_color_rect.png')
        return base_name

    def get_json_paths(self):
        json_paths = []
        for root, _, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def load_annotations(self):
        # the labels are not public for the test set yet
        if self.split == 'test':
            imgs_dir = os.path.join(self.data_root, TEST_IMGS_DIR)
            self.data_infos = [{
                'img_path':
                os.path.join(root, file),
                'img_name':
                os.path.join(TEST_IMGS_DIR,
                             root.split('/')[-1], file),
                'lanes': [],
                'relative_path':
                os.path.join(root.split('/')[-1], file)
            } for root, _, files in os.walk(imgs_dir) for file in files
                               if file.endswith('.png')]
            self.data_infos = sorted(self.data_infos,
                                     key=lambda x: x['img_path'])
            return

        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/llamas_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in self.data_infos)
                return

        self.max_lanes = 0
        print("Searching annotation files...")
        json_paths = self.get_json_paths()
        print('{} annotations found.'.format(len(json_paths)))

        for json_path in tqdm(json_paths):
            lanes = get_horizontal_values_for_four_lanes(json_path)
            lanes = [[(x, y) for x, y in zip(lane, range(self.cfg.ori_img_h))
                      if x >= 0] for lane in lanes]
            lanes = [lane for lane in lanes if len(lane) > 0]
            lanes = [list(set(lane))
                     for lane in lanes]  # remove duplicated points
            lanes = [lane for lane in lanes
                     if len(lane) > 2]  # remove lanes with less than 2 points

            lanes = [sorted(lane, key=lambda x: x[1])
                     for lane in lanes]  # sort by y
            lanes.sort(key=lambda lane: lane[0][0])
            mask_path = json_path.replace('.json', '.png')

            # generate seg labels
            seg = np.zeros((717, 1276, 3))
            for i, lane in enumerate(lanes):
                for j in range(0, len(lane) - 1):
                    cv2.line(seg, (round(lane[j][0]), lane[j][1]),
                             (round(lane[j + 1][0]), lane[j + 1][1]),
                             (i + 1, i + 1, i + 1),
                             thickness=15)

            cv2.imwrite(mask_path, seg)

            relative_path = self.get_img_path(json_path)
            img_path = os.path.join(self.data_root, relative_path)
            self.max_lanes = max(self.max_lanes, len(lanes))
            self.data_infos.append({
                'img_path': img_path,
                'img_name': relative_path,
                'mask_path': mask_path,
                'lanes': lanes,
                'relative_path': relative_path
            })

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def assign_class_to_lanes(self, lanes):
        return {
            label: value
            for label, value in zip(['l0', 'l1', 'r0', 'r1'], lanes)
        }

    def get_prediction_string(self, pred):
        ys = np.arange(300, 717, 1) / (self.cfg.ori_img_h - 1)
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * (self.cfg.ori_img_w - 1)
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * (self.cfg.ori_img_h - 1)
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(predictions):
            relative_path = self.data_infos[idx]['relative_path']
            output_filename = '/'.join(relative_path.split('/')[-2:]).replace(
                '_color_rect.png', '.lines.txt')
            output_filepath = os.path.join(output_basedir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)
        if self.split == 'test':
            return None
        result = llamas_metric.eval_predictions(output_basedir,
                                                self.labels_dir,
                                                iou_thresholds=np.linspace(0.5, 0.95, 10),
                                                unofficial=False)
        return result[0.5]['F1']
