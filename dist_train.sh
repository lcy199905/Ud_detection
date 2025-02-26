#!/usr/bin/env bash
rm -r 1-train.log
nohup python -u train.py > 1-train.log 2>&1 &
tail -f 1-train.log