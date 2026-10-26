#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from csv import writer as csv_writer

from matplotlib.pyplot import close, subplots
from numpy import arange, median, percentile, sort

from lib.coord import *
from lib.kconfig import *
from lib.path import *

anno_dst = open(f"{DATASET_YOLO_DIR}/annotations.csv", 'w', newline='')
image_dst = open(f"{DATASET_YOLO_DIR}/images.csv", 'w', newline='')
summary_dst = open(f"{DATASET_YOLO_DIR}/summary.csv", 'w', newline='')

anno_csv = csv_writer(anno_dst)
image_csv = csv_writer(image_dst)
summary_csv = csv_writer(summary_dst)

fig, axes = subplots(1, 3, figsize=(19.2, 6.4))

image_area = CONFIG_IMAGE_DIMENSION[0] * CONFIG_IMAGE_DIMENSION[1]

anno_csv.writerow(('split', 'image', 'bbox_area', 'bbox_aspect_ratio'))
image_csv.writerow(('split', 'image', 'filament_count'))
summary_csv.writerow(('split', 'metric', 'samples', 'p10', 'median', 'p90'))

for set in ('train', 'validate', 'test'):
	images = path_glob_sorted(f"{DATASET_YOLO_DIR}/{set}/images/*")
	bbox_areas = list()
	bbox_ratios = list()
	anno_counts = list()

	for image in images:
		name = path_resolve_stem(image)
		label_src = f"{DATASET_YOLO_DIR}/{set}/labels/{name}.txt"
		label = open(label_src)

		anno_counts.append(0)
		for line in label:
			coords = coord_load_label_yolo(line)
			x1, y1, x2, y2 = coord_calc_bbox(coords)
			width = x2 - x1
			height = y2 - y1

			area = width * height / image_area
			ratio = max(width / height, height / width)

			bbox_areas.append(area)
			bbox_ratios.append(ratio)
			anno_csv.writerow((set, name, area, ratio))
			anno_counts[-1] += 1

		image_csv.writerow((set, name, anno_counts[-1]))

	metrics = (
		('bbox_area', bbox_areas),
		('bbox_aspect_ratio', bbox_ratios),
		('annotation_count', anno_counts),
	)

	for idx, (metric, values) in enumerate(metrics):
		samples = len(values)
		ordered = sort(values)
		cumulative = arange(1, samples + 1) / samples

		p10 = percentile(values, 10)
		p50 = median(values)
		p90 = percentile(values, 90)

		axes[idx].plot(ordered, cumulative, label=set)
		summary_csv.writerow((set, metric, samples, p10, p50, p90))

axes[0].set_xlabel('Normalized bbox area')
axes[0].set_xscale('log')
axes[1].set_xlabel('BBox aspect ratio (long / short)')
axes[1].set_xscale('log')
axes[2].set_xlabel('Filaments per image')

for axis in axes:
	axis.set_ylabel('Cumulative fraction')
	axis.set_ylim(0, 1.01)
	axis.grid(alpha=0.25)
	axis.legend()

fig.tight_layout()
fig.savefig(f"{DATASET_YOLO_DIR}/distribution.png", dpi=200)
close(fig)
