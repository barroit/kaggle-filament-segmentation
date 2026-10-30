# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from math import sqrt

from cv2 import fillPoly as fill_poly
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
