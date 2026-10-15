#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>

from glob import glob
from multiprocessing import Pool, cpu_count
from os import makedirs
from os.path import basename, splitext

from cv2 import IMREAD_COLOR, addWeighted as add_weighted, \
		fillPoly as fill_poly, imread, imwrite, polylines
from numpy import array, float64, int32, rint

from lib.use_kconfig import *

input = f"{CONFIG_DATASET_PREFIX}/yolo"
output = f"{CONFIG_DATASET_PREFIX}/preview/yolo"

def rgba_to_bgra(rgba):
	return (rgba[2], rgba[1], rgba[0], rgba[3])

def draw_image(task):
	image_path, label_path = task
	image_base = basename(image_path)
	stem = None

	image = imread(image_path, IMREAD_COLOR)
	height, width = image.shape[:2]

	overlay = image.copy()
	border_overlay = None
	preview = None
	polygons = []

	fill_color = rgba_to_bgra(CONFIG_PREVIEW_SPINE_RGBA)
	border_color = rgba_to_bgra(CONFIG_PREVIEW_BORDER_RGBA)
	fill_alpha = None
	border_alpha = None

	with open(label_path) as label:
		for line in label:
			cols = line.split()
			coords = array(cols[1:], dtype=float64)

			pairs = coords.reshape(-1, 2)
			points = rint(pairs * [width, height])
			polygon = points.astype(int32)

			polygons.append(polygon)
			fill_poly(overlay, [polygon], fill_color[:3])

	fill_alpha = fill_color[3] / 255
	preview = add_weighted(image, 1 - fill_alpha, overlay, fill_alpha, 0)
	border_overlay = preview.copy()

	for polygon in polygons:
		polylines(border_overlay, [polygon], True, border_color[:3],
			  CONFIG_PREVIEW_BORDER_WIDTH)

	border_alpha = border_color[3] / 255
	preview = add_weighted(preview, 1 - border_alpha, border_overlay,
			       border_alpha, 0)

	stem = splitext(image_base)[0]
	imwrite(f"{output}/{stem}.png", preview)

if __name__ == '__main__':
	cpus = cpu_count()
	pool = Pool(cpus)
	tasks = None

	images = glob(f"{input}/*/images/*.jpeg")
	labels = glob(f"{input}/*/labels/*.txt")

	images.sort()
	labels.sort()
	tasks = zip(images, labels)

	makedirs(output, exist_ok=True)
	pool.map(draw_image, tasks)
