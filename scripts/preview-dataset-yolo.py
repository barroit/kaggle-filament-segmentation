#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>

from glob import glob
from multiprocessing import Pool, cpu_count
from os import makedirs

from lib.color import *
from lib.coord import *
from lib.draw import *
from lib.kconfig import *
from lib.path import *

prefix = 'preview/yolo'

def draw_image(task):
	image_src, label_src, dst = task
	image = draw_load(image_src)
	label = open(label_src)

	for __idx, line in enumerate(label):
		coords = coord_load_label_yolo(line)
		bbox = coord_calc_bbox(coords)
		idx = str(__idx + 1)

		image = draw_blend_area(image, coords)
		image = draw_blend_border(image, coords)
		image = draw_blend_bbox(image, bbox)
		image = draw_blend_label(image, bbox, idx)

	draw_dump(image, dst)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)

	image_srcs = path_glob_sorted(DATASET_YOLO_IMAGE_GLOB)
	label_srcs = path_glob_sorted(DATASET_YOLO_LABEL_GLOB)

	image_names = map(path_resolve_stem, image_srcs)
	dsts = map(lambda name: f"{prefix}/{name}.png", image_names)

	tasks = zip(image_srcs, label_srcs, dsts)

	makedirs(prefix)
	pool.map(draw_image, tasks)
