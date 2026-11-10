#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from multiprocessing import Pool as alloc_pool, cpu_count
from os import makedirs
from sys import argv

from lib.color import *
from lib.coord import *
from lib.draw import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *
from lib.predict import *

GT_COLOR = (255, 255, 0, 90)
TP_COLOR = (0, 0, 255, 90)
FP_COLOR = (255, 0, 0, 90)

LABLE_COLOR = [
	(0, 235, 255, 220),	# Cyan
	(50, 110, 255, 230),	# Electric blue
	(70, 255, 100, 220),	# Lime green
	(255, 235, 40, 230),	# Yellow
	(190, 90, 255, 230),	# Violet
	(255, 255, 255, 210),	# White
	(0, 255, 180, 230),	# Turquoise
	(80, 60, 255, 230),	# Indigo
	(180, 255, 40, 220),	# Chartreuse
	(255, 80, 220, 230),	# Magenta
	(120, 200, 255, 230),	# Sky blue
	(180, 255, 200, 220),	# Mint
]

def load_gt_masks(path):
	masks = list()
	label = open(path)

	for line in label:
		coords = coord_load_label_yolo(line)
		mask = mask_from_coords(coords)

		masks.append(mask)

	return masks

def draw_pred(image, idx, mask, color, alpha):
	bbox = coord_calc_bbox(mask)
	idx_str = str(idx)

	color_next = idx % len(LABLE_COLOR)
	label_color = color_rgba_to_bgra(LABLE_COLOR[color_next])
	label_alpha = color_alpha_percent(label_color)

	image = draw_blend_area_mask(image, mask, color, alpha)
	image = draw_blend_label(image, bbox, idx_str, label_color, label_alpha,
				 label_color)

	return image

def draw(src, dst, res, gt_masks):
	pred = pred_format_result(res)
	image = draw_load(src)

	tp_idxs = set()
	tp_pairs = mask_find_tps(pred.masks, gt_masks)

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

	for idx, mask in enumerate(pred.masks):
		if idx not in tp_idxs:
			image = draw_pred(image, idx + 1, pred.masks[idx],
					  fp_color, fp_alpha)

	for idx in tp_idxs:
		image = draw_pred(image, idx + 1, pred.masks[idx], tp_color,
				  tp_alpha)

	draw_dump(image, dst)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)
	jobs = list()

	set = 'test' if len(argv) < 3 else argv[2]
	prefix = f"preview/yolo/{argv[1]}/{set}"
	label_map = dict()
	stream = pred_start_stream(argv[1], set)

	for path in glob(f"{DATASET_YOLO_DIR}/{set}/labels/*.txt"):
		name = path_resolve_stem(path)

		label_map[name] = path

	makedirs(prefix)

	for result in stream:
		if len(result):
			res = result.numpy()
			name = path_resolve_stem(res.path)

			src = res.path
			dst = f"{prefix}/{name}.png"
			gt_masks = load_gt_masks(label_map[name])

			task = (src, dst, res, gt_masks)
			job = pool.apply_async(draw, task)

			jobs.append(job)

	pool.close()
	pool.join()

	for job in jobs:
		job.get()
