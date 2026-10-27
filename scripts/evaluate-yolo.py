#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from csv import writer
from glob import glob
from json import load as json_load
from os import makedirs
from sys import argv
from types import SimpleNamespace as simple_namespace

from matplotlib.pyplot import close, subplots
from numpy import count_nonzero, linspace, quantile, searchsorted
from ultralytics import YOLO as yolo

from lib.coord import *
from lib.kconfig import *
from lib.mask import *
from lib.path import *

prefix = f"{CONFIG_EVALUATION_DIR}/{argv[1]}"

annotations_src = open(DATASET_SHARED_ANNOTATIONS)
annotations = json_load(annotations_src)

periods_src = open(DATASET_SHARED_PERIODS)
periods = json_load(periods_src)

label_inputs = dict()
gt_shapes = dict()

gt_coverages = list()
crop_efficiencies = list()

gt_count = 0
cr_count = 0

dr_count = 0
pred_count = 0

spine_lengths = list()
spine_detected = list()

recall_map = dict()

model = yolo(f"{CONFIG_PREDICTION_DIR}/{argv[1]}/weights/best.pt")
stream = model.predict(source=f"{DATASET_YOLO_DIR}/test/images", stream=True,
		       save=False, verbose=False, retina_masks=True, device=0,
		       batch=CONFIG_YOLO_PRED_BATCH)

summary_plot = subplots(2, 6, figsize=(38.4, 12.8))
summary_fig = summary_plot[0]
summary_layout = [0, 6, 2, 1, 7, 3, 9, 4, 5, 8, 10, 11]
summary_axes = iter(summary_plot[1].flat[summary_layout])

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

			if iou * 100 >= CONFIG_YOLO_MATCH_THRESHOLD:
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
		if gt_coverage * 100 >= CONFIG_YOLO_CROP_RECALL_THRESHOLD:
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

def eval_detection_recall(ctx):
	global dr_count

	dr_count += ctx.tp_count

def eval_detection_precision(ctx):
	pass

def eval_detection_recall_by_spine_length(ctx):
	matched = set()
	annos = annotations[ctx.name]

	for _, idx in ctx.tp_pairs:
		matched.add(idx)

	for idx, anno in enumerate(annos):
		coords = coord_pack_float64(anno['spine'])
		length = coord_calc_spine_length(coords)

		spine_lengths.append(length)
		spine_detected.append(idx in matched)

def eval_detection_recall_by_observation_period(ctx):
	recall_map[ctx.name] = (ctx.gt_count, ctx.tp_count)

def produce_gt_coverages():
	*_, covs, confs = zip(*gt_coverages)

	csv_dst = open(f"{prefix}/gt_coverage.csv", 'w', newline='')
	csv = writer(csv_dst)

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

	csv.writerow(('image', 'x1', 'y1', 'x2', 'y2', 'gt_coverage',
		     'confidence'))

	for gt_coverage in gt_coverages:
		csv.writerow(gt_coverage)

def produce_crop_recall():
	recall = cr_count / gt_count
	missed = gt_count - cr_count
	cr_str = str(CONFIG_YOLO_CROP_RECALL_THRESHOLD)

	csv_dst = open(f"{prefix}/crop_recall_{cr_str}.csv", 'w',
		       newline='')
	csv = writer(csv_dst)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	for axis in (plot_axis, next(summary_axes)):
		bars = axis.bar(('Covered', 'Missed'), (cr_count, missed))

		axis.bar_label(bars, fmt='%d')
		axis.set_ylabel('GT Filaments')
		axis.set_title(f"Crop Recall@{cr_str} = {recall:.2%}")

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/crop_recall_{cr_str}.png")
	close(plot_fig)

	csv.writerow(('cr_count', 'gt_count', 'recall'))
	csv.writerow((cr_count, gt_count, recall))

def produce_crop_efficiencies():
	*_, effs, confs = zip(*crop_efficiencies)

	csv_dst = open(f"{prefix}/crop_efficiency.csv", 'w', newline='')
	csv = writer(csv_dst)

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

	csv_dst = open(f"{prefix}/detection_recall.csv", 'w', newline='')
	csv = writer(csv_dst)

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

	csv_dst = open(f"{prefix}/detection_precision.csv", 'w', newline='')
	csv = writer(csv_dst)

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

def produce_detection_recall_by_spine_length():
	gt_counts = [0] * CONFIG_YOLO_SPINE_BINS
	tp_counts = [0] * CONFIG_YOLO_SPINE_BINS
	recalls = list()

	quantiles = linspace(0, 1, CONFIG_YOLO_SPINE_BINS + 1)
	edges = quantile(spine_lengths, quantiles)
	bins = range(1, CONFIG_YOLO_SPINE_BINS + 1)

	csv_dst = open(f"{prefix}/detection_recall_by_spine_length.csv", 'w',
		       newline='')
	csv = writer(csv_dst)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))

	edges[-1] += 1

	for length, detected in zip(spine_lengths, spine_detected):
		idx = searchsorted(edges[1:-1], length, side='right')

		gt_counts[idx] += 1
		tp_counts[idx] += detected

	csv.writerow(('bin', 'length_min', 'length_max', 'gt_count', 'tp_count',
		     'recall'))

	for idx in range(CONFIG_YOLO_SPINE_BINS):
		recall = None

		if gt_counts[idx]:
			recall = tp_counts[idx] / gt_counts[idx]
			csv.writerow((idx + 1, edges[idx], edges[idx + 1],
				     gt_counts[idx], tp_counts[idx], recall))

		recalls.append(recall)

	for axis in (plot_axis, next(summary_axes)):
		axis.plot(bins, recalls, marker='o')
		axis.set_xticks(bins)
		axis.set_xlabel('Spine Length Quantile')
		axis.set_ylabel('Detection Recall')
		axis.set_ylim(0, 1)

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/detection_recall_by_spine_length.png")
	close(plot_fig)

def produce_detection_recall_by_observation_period():
	indices = list()
	recalls = list()

	csv_dst = open(f"{prefix}/detection_recall_by_observation_period.csv",
		       'w', newline='')
	csv = writer(csv_dst)

	plot_fig, plot_axis = subplots(figsize=(12.8, 9.6))
	summary_axis = next(summary_axes)

	csv.writerow(('period', 'gt_count', 'tp_count', 'recall'))

	for period in periods:
		idx = period['idx']
		gt_count = 0
		tp_count = 0

		if period['set'] != 'test':
			continue

		for name in period['images']:
			gt, tp = recall_map[name]

			gt_count += gt
			tp_count += tp

		if gt_count:
			recall = tp_count / gt_count

			indices.append(idx)
			recalls.append(recall)
			csv.writerow((idx, gt_count, tp_count, recall))

	for axis in (plot_axis, summary_axis):
		axis.scatter(indices, recalls, s=8, alpha=0.4)
		axis.set_xlabel('Observation Period')
		axis.set_ylabel('Detection Recall')
		axis.set_ylim(0, 1)

		for idx, recall in zip(indices, recalls):
			if recall * 100 < CONFIG_YOLO_PERIOD_WARN_THRESHOLD:
				label = str(idx)

				axis.annotate(label, (idx, recall),
					      xytext=(0, 5), color='red',
					      textcoords='offset points',
					      ha='center')

	plot_fig.tight_layout()
	plot_fig.savefig(f"{prefix}/detection_recall_by_observation_period.png")
	close(plot_fig)

makedirs(prefix)

for path in glob(f"{DATASET_YOLO_DIR}/test/labels/*.txt"):
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
	eval_detection_recall_by_spine_length(ctx)
	eval_detection_recall_by_observation_period(ctx)

produce_gt_coverages()
produce_crop_recall()
produce_crop_efficiencies()
produce_detection_recall()
produce_detection_precision()
produce_detection_recall_by_spine_length()
produce_detection_recall_by_observation_period()

for axis in summary_axes:
	summary_fig.delaxes(axis)

summary_fig.tight_layout()
summary_fig.savefig(f"{prefix}/summary.png", dpi=200)
close(summary_fig)
