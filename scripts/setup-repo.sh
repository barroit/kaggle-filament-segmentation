#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later

set -e

script=.venv/bin/activate
lib=$(python3 -c 'from site import getsitepackages; \
		  packages = getsitepackages(); print(packages[0])')

. $script

if ! grep -qxF 'export PYTHONPATH=.' $script; then
	printf '\nexport PYTHONPATH=.\n' >>$script
	perl -0pi -e 's/^}/\n    unset PYTHONPATH\n}/m' $script
fi

if ! grep -qxF 'export PYTHONDONTWRITEBYTECODE=1' $script; then
	printf '\nexport PYTHONDONTWRITEBYTECODE=1\n' >>$script
	perl -0pi -e 's/^}/\n    unset PYTHONDONTWRITEBYTECODE\n}/m' $script
fi

yolo settings weights_dir=$PWD/model >/dev/null

cat <<EOF >$lib/silent_interrupt.pth
import sys; from sys import __excepthook__; \
sys.excepthook = lambda type, value, tb: \
__excepthook__(type, value, tb) if type is not KeyboardInterrupt else None
EOF
