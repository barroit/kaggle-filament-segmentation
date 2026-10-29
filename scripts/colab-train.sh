#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later

set -e

session=$1
proxy="colab ssh -s $session --proxy-mode"

if [ -z $1 ]; then
	printf 'missing training name\n' >&2
	exit 1
elif [ -z $2 ]; then
	printf 'missing hyperparameter file\n' >&2
	exit 1
elif [ -d runs/$1 ]; then
	printf 'runs/%s already exists\n' $1 >&2
	exit 1
fi

trap 'colab stop -s $1; rm -f dataset.tar $1.tar' EXIT
trap 'exit 130' INT

sleep 50

colab new -s $1 --gpu ${3:-L4}

tar -cf dataset.tar dataset/yolo
scp -o ProxyCommand="$proxy" dataset.tar google.colab:/content

ssh -T -o ProxyCommand="$proxy" google.colab <<EOF
set -e

cd /content
uv pip install --system ultralytics

tar -xf dataset.tar

yolo segment train data=dataset/yolo/dataset.yaml \\
		   device=0 \\
		   project=\$PWD/runs \\
		   name=$1 \\
$(cat $2)

tar -cf $1.tar runs/$1
EOF

scp -o ProxyCommand="$proxy" google.colab:/content/$1.tar .
tar -xf $1.tar
