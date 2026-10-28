# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from itertools import combinations
from json import dump as json_dump, load as json_load
from multiprocessing import Pool as alloc_pool, cpu_count

from lib.coord import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

def drop_overlapped_anno(task):
	name, annos = task

	for anno_a, anno_b in combinations(annos, 2):
		coords_a = coord_pack_int32(anno_a['polygon'])
		coords_b = coord_pack_int32(anno_b['polygon'])

		mask_a = mask_from_coords(coords_a)
		mask_b = mask_from_coords(coords_b)
		iou = mask_calc_iou(mask_a, mask_b)

		if iou * 100 >= CONFIG_ANNO_OVERLAP_THRESHOLD and \
		   anno_a in annos and anno_b in annos:
			if anno_a['area'] > anno_b['area']:
				annos.remove(anno_b)
			else:
				annos.remove(anno_a)

	return name, annos

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)

	src = open(DATASET_SHARED_ANNOTATIONS)
	annotations = json_load(src)

	periods_src = open(DATASET_SHARED_PERIODS)
	periods = json_load(periods_src)
	periods_dst = open(DATASET_SHARED_PERIODS, 'w+')

	tasks = list()
	results = None

	for period in periods:
		for name in period['images']:
			tasks.append((name, annotations[name]))

	results = pool.map(drop_overlapped_anno, tasks)
	annotations = dict(results)

	for period in periods:
		images = len(period['images'])

		for idx in range(images - 1, -1, -1):
			name = period['images'][idx]

			period['annos'][idx] = len(annotations[name])

			if not period['annos'][idx]:
				period['images'].pop(idx)
				period['annos'].pop(idx)

	json_dump(periods, periods_dst, separators=(',', ':'))
