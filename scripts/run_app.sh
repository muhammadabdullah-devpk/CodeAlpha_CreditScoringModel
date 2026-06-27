#!/usr/bin/env bash
set -e
python -m src.train
python app.py
