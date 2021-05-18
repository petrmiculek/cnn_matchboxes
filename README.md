Bachelor Thesis - Counting Crates in Images
Author: Petr Mičulek
Date: 19 May 2021

The scripts provide 3 main use-cases.

1) Generate cutouts datasets from source photos.

2) Train a CNN model

3) Evaluate a trained CNN model


The package requirements can be installed like:

```
pip install -r requirements.txt
```


To run the dataset generation, either:


A) Run generation of a single cutouts dataset (training + validation)

Uses the 128x sample size
```
python3 src_util/generate_dataset.py -f -b -c 128 -p 500 -s 25
python3 src_util/generate_dataset.py -b -c 128 -r -p 500 -s 25
python3 src_util/generate_dataset.py -f -b -c 128 -p 500 -s 25 -v
python3 src_util/generate_dataset.py -b -c 128 -r -p 500 -s 25 -v
```

B) Run generate_all_datasets.py

- Takes a few minutes, creates 18 dataset versions. Not necessary for a single training/evaluation run.


Run evaluation of the enclosed trained model (final model)
