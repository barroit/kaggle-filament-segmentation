#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from csv import writer
from itertools import combinations
from multiprocessing import Pool as alloc_pool, cpu_count
from numpy import asfortranarray, count_nonzero, uint8
from pycocotools.mask import encode
from sys import argv

from lib.path import *
from lib.predict import *

def find_overlap(name, masks):
	n = len(masks)
	idxs = range(n)
	overlap = list()

	for a, b in combinations(idxs, 2):
		intersection = masks[a] & masks[b]
		count = count_nonzero(intersection)

		if count:
			overlap.append((name, a, b, count))

	return overlap

def format_rle(res):
	pred = pred_format_result(res)
	name = path_resolve_stem(res.path)

	masks = list()
	rows = list()
	overlaps = None

	for idx, mask in enumerate(pred.masks, 1):
		__binary = (mask > 0).astype(uint8)
		binary = asfortranarray(__binary)

		__rle = encode(binary)
		rle = __rle['counts'].decode('utf-8')
		id = f"{name}_{idx}"

		masks.append(binary)
		rows.append((id, rle))

	overlaps = find_overlap(name, masks)
	return rows, overlaps

if __name__ == '__main__':
	cpus = cpu_count()
	pool = alloc_pool(cpus)
	jobs = list()

	src = f"{MAGFILO_IMAGE_DIR}/../../test/test_images"
	stream = pred_start_stream(argv[1], '', src)

	__csv = open(f"submission.csv", 'w', newline='')
	csv = writer(__csv)

	rows = list()
	overlaps = list()

	for result in stream:
		res = result.numpy()

		if len(res):
			job = pool.apply_async(format_rle, [res])

			jobs.append(job)

	pool.close()
	pool.join()

	for job in jobs:
		__rows, __overlaps = job.get()

		rows.extend(__rows)
		overlaps.extend(__overlaps)

	rows.sort()
	overlaps.sort()

	for name, a, b, count in overlaps:
		print(f"{name}: {a + 1} and {b + 1} overlap by {count} pixels")

	csv.writerow(('filament_id', 'segmentation_rle'))
	csv.writerows(rows)
