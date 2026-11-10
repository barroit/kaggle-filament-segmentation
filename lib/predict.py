#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from cv2 import CC_STAT_AREA, connectedComponentsWithStats as components, \
		convexHull as convex_hull, fillConvexPoly as fill_convex_poly
from itertools import combinations
from numpy import argwhere, concatenate, delete, insert, nonzero 
from scipy.spatial import cKDTree as kd_tree
from types import SimpleNamespace as simple_namespace
from ultralytics import YOLO as alloc_yolo

from lib.kconfig import *
from lib.mask import *
from lib.path import *

__all__ = [
	'pred_start_stream',
	'pred_format_result',
]

def pred_start_stream(name, dataset='test'):
	model = alloc_yolo(f"{CONFIG_PREDICTION_DIR}/{name}/weights/best.pt")
	stream = model.predict(source=f"{DATASET_YOLO_DIR}/{dataset}/images",
			       stream=True, device=0, save=False, verbose=False,
			       retina_masks=True,
			       batch=CONFIG_PRED_BATCH_SIZE,
			       conf=CONFIG_PRED_CONF_THRESHOLD / 100,
			       agnostic_nms=True)

	return stream

def fill_islands(mask_a, mask_b):
	coords_a = argwhere(mask_a)
	coords_b = argwhere(mask_b)

	tree = kd_tree(coords_a)
	dists, nearest = tree.query(coords_b)
	closest = dists.argmin()

	y_a, x_a = coords_a[nearest[closest]]
	y_b, x_b = coords_b[closest]

	dist_a = (coords_a[:, 0] - y_a) ** 2 + (coords_a[:, 1] - x_a) ** 2
	dist_b = (coords_b[:, 0] - y_b) ** 2 + (coords_b[:, 1] - x_b) ** 2

	near_a = coords_a[dist_a <= CONFIG_PRED_ISLAND_BRIDGE_RADIUS ** 2]
	near_b = coords_b[dist_b <= CONFIG_PRED_ISLAND_BRIDGE_RADIUS ** 2]

	__points = concatenate((near_a, near_b))
	points = __points[:, ::-1].astype('int32')
	hull = convex_hull(points)

	fill_convex_poly(mask_a, hull, 1)

def format_island(pred):
	for idx in range(pred.n - 1, -1, -1):
		groups = list()
		conf = pred.confs[idx]
		pos = idx
		binary = pred.masks[idx].astype('uint8')
		n, labels, st, _ = components(binary, connectivity=8)

		for label in range(1, n):
			if st[label, CC_STAT_AREA] >= \
			   CONFIG_PRED_ISLAND_DROP_THRESHOLD:
				island = (labels == label).astype('uint8')

				groups.append(island)

		groups.sort(key=mask_calc_area, reverse=True)

		while 39:
			merged = 0
			count = len(groups)
			idxs = range(count)
			pairs = combinations(idxs, 2)

			for idx_a, idx_b in pairs:
				dist = mask_calc_distance(groups[idx_a],
							  groups[idx_b])

				if dist < CONFIG_PRED_ISLAND_MERGE_THRESHOLD:
					fill_islands(groups[idx_a],
						     groups[idx_b])
					groups[idx_a] |= groups[idx_b]

					del groups[idx_b]
					merged = 1
					break

			if not merged:
				break

		for group in groups:
			ys, xs = nonzero(group)
			x1 = xs.min()
			y1 = ys.min()
			x2 = xs.max() + 1
			y2 = ys.max() + 1
			bbox = (x1, y1, x2, y2)

			pred.bboxes = insert(pred.bboxes, pos, bbox, axis=0)
			pred.confs = insert(pred.confs, pos, conf)
			pred.masks = insert(pred.masks, pos, group, axis=0)
			pos += 1

		pred.bboxes = delete(pred.bboxes, pos, axis=0)
		pred.confs = delete(pred.confs, pos)
		pred.masks = delete(pred.masks, pos, axis=0)

	pred.n = len(pred.masks)

def pred_format_result(res):
	pred = simple_namespace()

	pred.bboxes = res.boxes.xyxy
	pred.confs = res.boxes.conf
	pred.masks = res.masks.data
	pred.n = len(res)

	format_island(pred)

	return pred
