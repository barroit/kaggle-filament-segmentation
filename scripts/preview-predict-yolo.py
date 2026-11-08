#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from multiprocessing import Pool as alloc_pool, cpu_count
from os import makedirs
from sys import argv

from ultralytics import YOLO as alloc_yolo

from lib.color import *
from lib.coord import *
from lib.draw import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

GT_COLOR = (255, 255, 0, 50)
TP_COLOR = (0, 0, 255, 50)
FP_COLOR = (255, 0, 0, 50)

def load_gt_masks(path):
	masks = list()
	label = open(path)

	for line in label:
		coords = coord_load_label_yolo(line)
		mask = mask_from_coords(coords)

		masks.append(mask)

	return masks

def draw_pred(image, mask, conf, color, alpha):
	bbox = coord_calc_bbox(mask)
	conf_str = str(conf)

	image = draw_blend_area_mask(image, mask, color, alpha)
	image = draw_blend_label(image, bbox, conf_str)

	return image

def draw(src, dst, pred_masks, pred_confs, gt_masks, tp_pairs):
	image = draw_load(src)
	tp_idxs = set()

	gt_color = color_rgba_to_bgra(GT_COLOR)
	gt_alpha = color_alpha_percent(gt_color)

	tp_color = color_rgba_to_bgra(TP_COLOR)
	tp_alpha = color_alpha_percent(tp_color)

	fp_color = color_rgba_to_bgra(FP_COLOR)
	fp_alpha = color_alpha_percent(fp_color)

	for idx, _ in tp_pairs:
		tp_idxs.add(idx)

	for mask in gt_masks:
		image = draw_blend_area_mask(image, mask, gt_color, gt_alpha)

	for idx, mask in enumerate(pred_masks):
		if idx not in tp_idxs:
			mask = pred_masks[idx]
			conf = pred_confs[idx]

			image = draw_pred(image, mask, conf, fp_color, fp_alpha)

	for idx in tp_idxs:
		mask = pred_masks[idx]
		conf = pred_confs[idx]

		image = draw_pred(image, mask, conf, tp_color, tp_alpha)

	draw_dump(image, dst)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)
	jobs = list()

	split = 'test' if len(argv) < 3 else argv[2]
	prefix = f"preview/yolo/{argv[1]}/{split}"
	label_map = dict()

	model = alloc_yolo(f"{CONFIG_PREDICTION_DIR}/{argv[1]}/weights/best.pt")
	stream = model.predict(source=f"{DATASET_YOLO_DIR}/{split}/images",
			       stream=True, save=False, verbose=False,
			       retina_masks=True, device=0,
			       batch=CONFIG_YOLO_PRED_BATCH,
			       conf=CONFIG_YOLO_CONF_THRESHOLD / 100)

	for path in glob(f"{DATASET_YOLO_DIR}/{split}/labels/*.txt"):
		name = path_resolve_stem(path)

		label_map[name] = path

	makedirs(prefix)

	for result in stream:
		result = result.numpy()
		name = path_resolve_stem(result.path)

		src = result.path
		dst = f"{prefix}/{name}.png"

		pred_masks = result.masks.data if result.masks else list()
		pred_confs = result.boxes.conf

		gt_masks = load_gt_masks(label_map[name])
		tp_pairs = mask_find_tps(pred_masks, gt_masks)

		task = (src, dst, pred_masks, pred_confs, gt_masks, tp_pairs)
		job = pool.apply_async(draw, task)

		jobs.append(job)

	pool.close()
	pool.join()

	for job in jobs:
		job.get()
