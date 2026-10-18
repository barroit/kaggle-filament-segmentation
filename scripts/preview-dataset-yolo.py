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

def glob_and_sort(pattern):
	files = glob(pattern)

	files.sort()
	return files

def draw_image(task):
	image_input, label_input, output = task
	image = draw_load(image_input)
	height, width = image.shape[:2]
	label = open(label_input)

	fill_color = color_rgba_to_bgra(CONFIG_DRAW_SPINE_RGBA)
	fill_alpha = color_alpha_percent(fill_color)

	border_color = color_rgba_to_bgra(CONFIG_DRAW_BORDER_RGBA)
	border_alpha = color_alpha_percent(border_color)

	for line in label:
		coords = coord_load_label(line, width, height)
		overlay = image.copy()

		draw_fill_polygon(overlay, [coords], fill_color)
		image = draw_blend(image, overlay, fill_alpha)

		overlay = image.copy()
		draw_polyline(overlay, coords, border_color,
			      CONFIG_DRAW_BORDER_WIDTH)
		image = draw_blend(image, overlay, border_alpha)

	draw_dump(image, output)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)

	image_inputs = glob_and_sort(GLOB_YOLO_IMAGES)
	label_inputs = glob_and_sort(GLOB_YOLO_LABELS)

	image_names = map(path_resolve_stem, image_inputs)
	outputs = map(lambda name: f"{PREVIEW_YOLO_DIR}/{name}.png",
		      image_names)

	tasks = zip(image_inputs, label_inputs, outputs)

	makedirs(PREVIEW_YOLO_DIR)
	pool.map(draw_image, tasks)
