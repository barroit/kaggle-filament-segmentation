#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from glob import glob
from itertools import chain, combinations
from multiprocessing import Pool, cpu_count

from cv2 import fillPoly as fill_poly
from numpy import count_nonzero, uint8, zeros

from lib.coord import *
from lib.kconfig import *
from lib.path import *

def find(label_input):
	name = path_resolve_stem(label_input)
	label = open(label_input)
	masks = []
	results = []

	for line in label:
		coords = coord_load_label_yolo(line)
		mask = zeros(CONFIG_IMAGE_DIMENSION[::-1], dtype=uint8)

		fill_poly(mask, [coords], 1)
		masks.append(mask)

	for (idx_a, mask_a), (idx_b, mask_b) in combinations(enumerate(masks),
							     2):
		overlap = count_nonzero(mask_a & mask_b)

		if overlap:
			area_a = count_nonzero(mask_a)
			area_b = count_nonzero(mask_b)

			ratio = overlap / min(area_a, area_b)
			result = (name, idx_a + 1, idx_b + 1, overlap, ratio)

			results.append(result)

	return results

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)

	label_inputs = glob(GLOB_YOLO_LABELS)
	worker_results = pool.map(find, label_inputs)
	results = list(chain.from_iterable(worker_results))

	results.sort(reverse=True, key=lambda result: result[4])
	for name, a, b, overlap, ratio in results:
		print(f"{name}: GT {a} and {b} overlapped in {overlap} pixels (ratio={ratio:.3%})")

	print(f"Overlapping pairs: {len(results)}")
