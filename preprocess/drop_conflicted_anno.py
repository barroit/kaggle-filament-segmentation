# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from collections import defaultdict
from itertools import combinations
from json import dump as json_dump, load as json_load
from multiprocessing import Pool as alloc_pool, cpu_count

from lib.coord import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

conflict_coverage_threshold = CONFIG_CONFLICT_COVERAGE_THRESHOLD / 100
winner_length_ratio = CONFIG_CONFLICT_WINNER_LENGTH_RATIO / 100

def drop_conflicted_anno(task):
	name, annos = task
	groups = []

	for anno_a, anno_b in combinations(annos, 2):
		if anno_a['author'] != anno_b['author']:
			group = [anno_a, anno_b]
			coords_a = coord_pack_int32(anno_a['polygon'])
			coords_b = coord_pack_int32(anno_b['polygon'])

			mask_a = mask_from_coords(coords_a)
			mask_b = mask_from_coords(coords_b)
			coverage = mask_calc_coverage(mask_a, mask_b)

			if coverage < conflict_coverage_threshold:
				continue

			for seen in groups[:]:
				if anno_a in seen or anno_b in seen:
					groups.remove(seen)

					for anno in seen:
						if anno not in group:
							group.append(anno)

			groups.append(group)

	for group in groups:
		lengths = {}
		longest = 0

		winner = None
		best = float('inf')

		for anno in group:
			author = anno['author']
			coords = coord_pack_float64(anno['spine'])
			length = coord_calc_spine_length(coords)
			longest = max(longest, length)

			if author not in lengths or length > lengths[author]:
				lengths[author] = length

		for author, length in lengths.items():
			ratio = length / longest
			diff = abs(ratio - winner_length_ratio)

			if diff < best:
				best = diff
				winner = author

		for anno in group:
			if anno['author'] != winner:
				annos.remove(anno)

	return name, annos

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)

	src = open(DATASET_SHARED_METADATA)
	metadata = json_load(src)
	dst = open(DATASET_SHARED_METADATA, 'w+')

	tasks = metadata.items()
	results = pool.map(drop_conflicted_anno, tasks)

	metadata = dict(results)
	json_dump(metadata, dst, separators=(',', ':'))
