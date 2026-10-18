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

fill_color = color_rgba_to_bgra(CONFIG_DRAW_SPINE_RGBA)
fill_alpha = color_alpha_percent(fill_color)

border_color = color_rgba_to_bgra(CONFIG_DRAW_BORDER_RGBA)
border_alpha = color_alpha_percent(border_color)

label_color = color_rgba_to_bgra(CONFIG_DRAW_TEXT_RGBA)
label_alpha = color_alpha_percent(label_color)

label_border_color = color_rgba_to_bgra(CONFIG_DRAW_TEXT_BORDER_RGBA)
label_border_alpha = color_alpha_percent(label_border_color)

def glob_and_sort(pattern):
	files = glob(pattern)

	files.sort()
	return files

def draw_image(task):
	image_input, label_input, output = task
	image = draw_load(image_input)
	label = open(label_input)

	for __idx, line in enumerate(label):
		coords = coord_load_label_yolo(line)
		bbox = coord_calc_bbox_yolo(coords)
		overlay = image.copy()
		idx = str(__idx + 1)

		draw_fill_polygon(overlay, [coords], fill_color)
		image = draw_blend(image, overlay, fill_alpha)

		overlay = image.copy()
		draw_polyline(overlay, coords, border_color,
			      CONFIG_DRAW_BORDER_WIDTH)
		image = draw_blend(image, overlay, border_alpha)

		overlay = image.copy()
		draw_text(overlay, bbox, idx, label_color)
		draw_text_border(overlay, bbox, idx, label_border_color)
		image = draw_blend(image, overlay, label_border_alpha)

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
