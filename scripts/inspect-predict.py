#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from csv import writer
from multiprocessing import Pool as alloc_pool, cpu_count
from os import makedirs
from sys import argv

from lib.coord import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *
from lib.predict import *

def load_gt_masks(path):
	masks = list()
	label = open(path)

	for line in label:
		coords = coord_load_label_yolo(line)
		mask = mask_from_coords(coords)

		masks.append(mask)

	return masks

def inspect(dst, res, label):
	__tsv = open(dst, 'w', newline='')
	tsv = writer(__tsv, delimiter='\t')

	tsv.writerow(('index', 'tp', 'conf', 'spine', 'area', 'coord'))

	if len(res):
		pred = pred_format_result(res)
		gt_masks = load_gt_masks(label)
		tp_idxs = set()
		tp_pairs = mask_find_tps(pred.masks, gt_masks)

		for idx, _ in tp_pairs:
			tp_idxs.add(idx)

		for idx, mask in enumerate(pred.masks):
			x1, y1, x2, y2 = pred.bboxes[idx]
			area = mask_calc_area(mask)
			length = mask_calc_spine_length_approx(mask)
			tp = 0

			if idx in tp_idxs:
				tp = 1

			tsv.writerow((idx + 1, tp, f"{pred.confs[idx]:.3f}",
				     f"{length:.1f}", area,
				     f"{x1},{y1},{x2},{y2}"))
 
if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)
	jobs = list()

	dataset = 'test' if len(argv) < 3 else argv[2]
	prefix = f"inspect/yolo/{argv[1]}/{dataset}"
	stream = pred_start_stream(argv[1], dataset)

	makedirs(prefix)

	for result in stream:
		res = result.numpy()
		name = path_resolve_stem(res.path)
		dst = f"{prefix}/{name}"
		label = f"{DATASET_YOLO_DIR}/{dataset}/labels/{name}.txt"

		task = (dst, res, label)
		job = pool.apply_async(inspect, task)

		jobs.append(job)

	pool.close()
	pool.join()

	for job in jobs:
		job.get()
