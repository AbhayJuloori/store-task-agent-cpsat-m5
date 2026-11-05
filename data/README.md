# Data

## Real (public)
M5 Forecasting Accuracy (Walmart, via the University of Nicosia M Open Forecasting Center and Kaggle):
`calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`. 10 stores in CA, TX and WI, 3,049 items,
daily unit sales for 1,941 days (2011-01-29 to 2016-05-22), weekly shelf prices.

The competition download needs the Kaggle rules accepted, so `scripts/download_m5.sh` pulls the same files
from the public mirror `aryayadav0513/m5-forecasting-accuracy`. File sizes match the competition files;
sha256 of the copies used:

| file | sha256 |
|---|---|
| calendar.csv | d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5 |
| sales_train_evaluation.csv | 4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0 |
| sell_prices.csv | 9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef |

## Synthetic (seeded, `configs/assumptions.yaml`)
Store layout (departments to aisles to bays), shelf capacity, task durations, associate shifts and
skills, the planogram reset calendar. None of this exists in M5.
