#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later

set -e

. .venv/bin/activate

yolo settings weights_dir=$PWD/model
