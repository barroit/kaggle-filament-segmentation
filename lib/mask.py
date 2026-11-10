# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from math import sqrt

from cv2 import DIST_L2, DIST_MASK_PRECISE, \
		distanceTransform as distance_transform, fillPoly as fill_poly
from numpy import arange, bincount, count_nonzero, diff, flatnonzero, nonzero, \
		  uint8, zeros
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

from lib.kconfig import *

__all__ = [
	'mask_from_coords',
	'mask_calc_iou',
	'mask_calc_coverage',
	'mask_calc_spine_length_approx',
	'mask_calc_area',
	'mask_calc_distance',
	'mask_find_tps',
]

def mask_from_coords(coords):
	mask = zeros(CONFIG_IMAGE_DIMENSION[::-1], dtype=uint8)

	fill_poly(mask, [coords], 1)
	return mask

def mask_calc_iou(mask_a, mask_b):
	intersection = count_nonzero(mask_a & mask_b)
	union = count_nonzero(mask_a | mask_b)
	iou = intersection / union

	return iou

def mask_calc_coverage(mask_a, mask_b):
	intersection = count_nonzero(mask_a & mask_b)
	area_a = count_nonzero(mask_a)
	area_b = count_nonzero(mask_b)
	coverage = intersection / min(area_a, area_b)

	return coverage

def skel_build_graph(skel):
	ys, xs = nonzero(skel)
	coords = zip(ys, xs)
	points = list(coords)
	count = len(points)
	nodes = dict()
	rows = list()
	cols = list()
	weights = list()
	diagonal = sqrt(2)
	steps = ((0, 1, 1), (1, -1, diagonal), (1, 0, 1), (1, 1, diagonal))

	for idx, point in enumerate(points):
		nodes[point] = idx

	for idx, (y, x) in enumerate(points):
		for dy, dx, weight in steps:
			next_idx = nodes.get((y + dy, x + dx))

			if next_idx is None:
				continue

			rows.extend((idx, next_idx))
			cols.extend((next_idx, idx))
			weights.extend((weight, weight))

	return csr_matrix((weights, (rows, cols)), shape=(count, count))

def graph_select_largest(graph):
	_, labels = connected_components(graph, directed=False)
	counts = bincount(labels)
	major = counts.argmax()
	selected = flatnonzero(labels == major)

	return graph[selected][:, selected]

def graph_select_sources(graph):
	degrees = diff(graph.indptr)
	endpoints = flatnonzero(degrees == 1)

	if len(endpoints) < 2:
		endpoints = arange(graph.shape[0])

	return endpoints

def graph_calc_span(graph):
	sources = graph_select_sources(graph)
	distances = dijkstra(graph, directed=False, indices=sources)

	return distances[:, sources].max()

def mask_calc_spine_length_approx(mask):
	skel = skeletonize(mask)
	graph = skel_build_graph(skel)
	largest = graph_select_largest(graph)

	return graph_calc_span(largest)

def mask_calc_area(mask):
	return count_nonzero(mask)

def mask_calc_distance(mask_a, mask_b):
	binary = (mask_a == 0).astype('uint8')
	dist = distance_transform(binary, DIST_L2, DIST_MASK_PRECISE)
	dist = dist[mask_b != 0]

	return dist.min()

def mask_find_tps(mask_preds, mask_gts):
	cands = list()
	tps = list()
	seen_pred = set()
	seen_gt = set()

	for idx_pred, mask_pred in enumerate(mask_preds):
		for idx_gt, mask_gt in enumerate(mask_gts):
			iou = mask_calc_iou(mask_pred, mask_gt)

			if iou * 100 >= CONFIG_YOLO_GT_MATCH_THRESHOLD:
				cands.append((iou, idx_pred, idx_gt))

	cands.sort(reverse=True)

	for iou, idx_pred, idx_gt in cands:
		if idx_pred not in seen_pred and idx_gt not in seen_gt:
			tps.append((idx_pred, idx_gt))

			seen_pred.add(idx_pred)
			seen_gt.add(idx_gt)

	return tps
