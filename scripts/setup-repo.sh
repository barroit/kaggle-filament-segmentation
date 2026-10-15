#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later

set -e

script=.venv/bin/activate

. $script

if ! grep -qxF 'export PYTHONPATH=.' $script; then
	printf '\nexport PYTHONPATH=.\n' >>$script
	perl -0pi -e 's/^}/\n    unset PYTHONPATH\n}/m' $script
fi

if ! grep -qxF 'export PYTHONDONTWRITEBYTECODE=1' $script; then
	printf '\nexport PYTHONDONTWRITEBYTECODE=1\n' >>$script
	perl -0pi -e 's/^}/\n    unset PYTHONDONTWRITEBYTECODE\n}/m' $script
fi

yolo settings weights_dir=$PWD/model
