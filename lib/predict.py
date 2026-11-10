#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright 2026 Jiamu Sun <39@barroit.sh>
#

from types import SimpleNamespace as simple_namespace
from ultralytics import YOLO as alloc_yolo

from lib.kconfig import *
from lib.path import *

__all__ = [
	'pred_start_stream',
	'pred_format_result',
]

def pred_start_stream(name, dataset='test'):
	model = alloc_yolo(f"{CONFIG_PREDICTION_DIR}/{name}/weights/best.pt")
	stream = model.predict(source=f"{DATASET_YOLO_DIR}/{dataset}/images",
			       stream=True, device=0, save=False, verbose=False,
			       retina_masks=True,
			       batch=CONFIG_PRED_BATCH_SIZE,
			       conf=CONFIG_PRED_CONF_THRESHOLD / 100,
			       agnostic_nms=True)

	return stream


def pred_format_result(res):
	pred = simple_namespace()

	pred.bboxes = res.boxes.xyxy
	pred.confs = res.boxes.conf
	pred.masks = res.masks.data
	pred.n = len(res)

	return pred
