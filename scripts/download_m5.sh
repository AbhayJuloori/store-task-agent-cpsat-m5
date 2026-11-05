#!/usr/bin/env bash
# Downloads the M5 files into data/raw (Kaggle CLI must be authenticated).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle datasets download aryayadav0513/m5-forecasting-accuracy -p data/raw --unzip
shasum -a 256 data/raw/m5-forecasting-accuracy/{calendar,sales_train_evaluation,sell_prices}.csv
