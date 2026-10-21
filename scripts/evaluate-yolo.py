#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from csv import writer
from glob import glob
from os import makedirs
from sys import argv
from types import SimpleNamespace as simple_namespace

from matplotlib.pyplot import close, subplots
from numpy import count_nonzero
from ultralytics import YOLO as yolo

from lib.coord import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

iou_threshold = CONFIG_YOLO_MATCH_IOU_MIN / 100
cr_threshold = CONFIG_YOLO_CROP_RECALL_COVERAGE_MIN / 100

cr_threshold_str = str(CONFIG_YOLO_CROP_RECALL_COVERAGE_MIN)

prefix = f"{CONFIG_EVALUATION_DIR}/{argv[1]}"

label_inputs = dict()
gt_shapes = dict()

gt_coverages = list()
crop_efficiencies = list()

gt_count = 0
cr_count = 0

dr_count = 0
pred_count = 0

model = yolo(f"{CONFIG_PREDICTION_DIR}/{argv[1]}/weights/best.pt")
stream = model.predict(source=TRAIN_IMAGE_DIR, stream=True, save=False,
		       verbose=False, retina_masks=True, device=0,
		       batch=CONFIG_YOLO_PRED_BATCH)

summary_plot = subplots(2, 4, figsize=(25.6, 12.8))
summary_fig = summary_plot[0]
summary_axes = iter(summary_plot[1].flat[[0, 4, 2, 1, 5, 3, 7, 6]])

def fetch_gt_shapes(file):
	if file not in gt_shapes:
		label = open(file)
		masks = list()
		bboxes = list()

		for line in label:
			coords = coord_load_label_yolo(line)
			bbox = coord_calc_bbox(coords)
			mask = mask_from_coords(coords)

			masks.append(mask)
			bboxes.append(bbox)

		gt_shapes[file] = (masks, bboxes)

	return gt_shapes[file][0], gt_shapes[file][1]

def find_tp_pairs(ctx):
	candidates = list()
	pairs = list()
	selected_pred = set()
	selected_gt = set()

	for pred_idx, pred_mask in enumerate(ctx.pred_masks):
		for gt_idx, gt_mask in enumerate(ctx.gt_masks):
			iou = mask_calc_iou(pred_mask, gt_mask)

			if iou >= iou_threshold:
				candidates.append((iou, pred_idx, gt_idx))

	candidates.sort(reverse=True)

	for iou, pred_idx, gt_idx in candidates:
		if pred_idx not in selected_pred and gt_idx not in selected_gt:
			pairs.append((pred_idx, gt_idx))

			selected_pred.add(pred_idx)
			selected_gt.add(gt_idx)

	return pairs

def eval_gt_coverages(ctx):
	for pred_idx, gt_idx in ctx.tp_pairs:
		gt_mask = ctx.gt_masks[gt_idx]
		bbox = ctx.pred_bboxes[pred_idx]
		x1, y1, x2, y2 = coord_snap_bbox(bbox)

		positive = count_nonzero(gt_mask[y1:y2, x1:x2])
		total = count_nonzero(gt_mask)
		gt_coverage = positive / total

		gt_coverages.append((ctx.name, *bbox, gt_coverage,
				    ctx.pred_confs[pred_idx]))

def eval_crop_recall(ctx):
	cov_len = len(gt_coverages)
	selected = gt_coverages[cov_len - ctx.tp_count:cov_len]

	global cr_count

	for *_, gt_coverage, _ in selected:
		if gt_coverage >= cr_threshold:
			cr_count += 1

def eval_crop_efficiencies(ctx):
	for pred_idx, gt_idx in ctx.tp_pairs:
		pred_bbox = ctx.pred_bboxes[pred_idx]
		gt_bbox = ctx.gt_bboxes[gt_idx]
		x1, y1, x2, y2 = pred_bbox

		intersection = coord_calc_bbox_intersection(pred_bbox, gt_bbox)
		area = (x2 - x1) * (y2 - y1)
		efficiency = intersection / area

		crop_efficiencies.append((ctx.name, *ctx.pred_bboxes[pred_idx],
					 efficiency, ctx.pred_confs[pred_idx]))

def produce_gt_coverages():
	*_, covs, confs = zip(*gt_coverages)

	__csv = open(f"{prefix}/gt_coverage.csv", 'w', newline='')
	csv = writer(__csv)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		*_, bars = axis.hist(covs, bins=20, range=(0, 1))

		axis.bar_label(bars, fmt=lambda n: str(int(n)) if n else '')
		axis.set_xlabel('GT Coverage')
		axis.set_ylabel('Predicted Crops')

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/gt_coverage.png")
	close(plot_fig)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))
	for axis in (plot_axis, next(summary_axes)):
		axis.scatter(confs, covs, s=8, alpha=0.4)
		axis.set_xlabel('Confidence')
		axis.set_ylabel('GT Coverage')
		axis.set_xlim(0, 1)
		axis.set_ylim(0, 1)

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/gt_coverage_confidence.png")
	close(plot_fig)

	csv.writerow(('image', 'x1', 'x1', 'x2', 'y2', 'gt_coverage',
		     'confidence'))

	for gt_coverage in gt_coverages:
		csv.writerow(gt_coverage)

def eval_detection_recall(ctx):
	global dr_count

	dr_count += ctx.tp_count

def eval_detection_precision(ctx):
	pass

def produce_crop_recall():
	recall = cr_count / gt_count
	missed = gt_count - cr_count

	__csv = open(f"{prefix}/crop_recall_{cr_threshold_str}.csv", 'w',
		     newline='')
	csv = writer(__csv)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		bars = axis.bar(('Covered', 'Missed'), (cr_count, missed))

		axis.bar_label(bars, fmt='%d')
		axis.set_ylabel('GT Filaments')
		axis.set_title(f"Crop Recall@{cr_threshold_str} = {recall:.2%}")

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/crop_recall_{cr_threshold_str}.png")
	close(plot_fig)

	csv.writerow(('cr_count', 'gt_count', 'recall'))
	csv.writerow((cr_count, gt_count, recall))

def produce_crop_efficiencies():
	*_, effs, confs = zip(*crop_efficiencies)

	__csv = open(f"{prefix}/crop_efficiency.csv", 'w', newline='')
	csv = writer(__csv)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		*_, bars = axis.hist(effs, bins=20, range=(0, 1))

		axis.bar_label(bars, fmt=lambda n: str(int(n)) if n else '')
		axis.set_xlabel('Crop Efficiency')
		axis.set_ylabel('Predicted TP Crops')

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/crop_efficiency.png")
	close(plot_fig)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))
	for axis in (plot_axis, next(summary_axes)):
		axis.scatter(confs, effs, s=8, alpha=0.4)
		axis.set_xlabel('Confidence')
		axis.set_ylabel('Crop Efficiency')
		axis.set_xlim(0, 1)
		axis.set_ylim(0, 1)

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/crop_efficiency_confidence.png")
	close(plot_fig)

	csv.writerow(('image', 'x1', 'y1', 'x2', 'y2',
		     'crop_efficiency', 'confidence'))

	for crop_efficiency in crop_efficiencies:
		csv.writerow(crop_efficiency)

def produce_detection_recall():
	recall = dr_count / gt_count
	missed = gt_count - dr_count

	__csv = open(f"{prefix}/detection_recall.csv", 'w', newline='')
	csv = writer(__csv)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		bars = axis.bar(('Detected', 'Missed'), (dr_count, missed))

		axis.bar_label(bars, fmt='%d')
		axis.set_ylabel('GT Filaments')
		axis.set_title(f"Detection Recall = {recall:.2%}")

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/detection_recall.png")
	close(plot_fig)

	csv.writerow(('dr_count', 'gt_count', 'recall'))
	csv.writerow((dr_count, gt_count, recall))

def produce_detection_precision():
	precision = dr_count / pred_count
	fp_count = pred_count - dr_count

	__csv = open(f"{prefix}/detection_precision.csv", 'w', newline='')
	csv = writer(__csv)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		bars = axis.bar(('TP', 'FP'), (dr_count, fp_count))

		axis.bar_label(bars, fmt='%d')
		axis.set_ylabel('Predicted Crops')
		axis.set_title(f"Detection Precision = {precision:.2%}")

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/detection_precision.png")
	close(plot_fig)

	csv.writerow(('tp_count', 'pred_count', 'precision'))
	csv.writerow((dr_count, pred_count, precision))

makedirs(prefix)

for path in glob(DATASET_YOLO_LABEL_GLOB):
	name = path_resolve_stem(path)

	label_inputs[name] = path

for result in stream:
	result = result.numpy()
	ctx = simple_namespace()

	ctx.pred_bboxes = result.boxes.xyxy
	ctx.pred_confs = result.boxes.conf
	ctx.pred_masks = result.masks.data if result.masks else list()
	ctx.pred_count = len(result)
	ctx.name = path_resolve_stem(result.path)

	ctx.gt_masks, ctx.gt_bboxes = fetch_gt_shapes(label_inputs[ctx.name])
	ctx.gt_count = len(ctx.gt_masks)

	ctx.tp_pairs = find_tp_pairs(ctx)
	ctx.tp_count = len(ctx.tp_pairs)

	gt_count += ctx.gt_count
	pred_count += ctx.pred_count

	eval_gt_coverages(ctx)
	eval_crop_recall(ctx)
	eval_crop_efficiencies(ctx)
	eval_detection_recall(ctx)
	eval_detection_precision(ctx)

produce_gt_coverages()
produce_crop_recall()
produce_crop_efficiencies()
produce_detection_recall()
produce_detection_precision()

for axis in summary_axes:
	summary_fig.delaxes(axis)

summary_fig.tight_layout()
summary_fig.savefig(f"{prefix}/summary.png", dpi=200)
close(summary_fig)
