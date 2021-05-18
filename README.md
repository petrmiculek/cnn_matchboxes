Bachelor Thesis - Counting Crates in Images
Author: Petr Mičulek
Date: 19 May 2021

The scripts provide 3 main use-cases

1) Generate cutouts datasets from source photos.

2) Train a CNN model

3) Evaluate a trained CNN model


The package requirements can be installed like:

```
pip install -r requirements.txt
```


###Generate cutouts datasets from source photos

To run the dataset generation, either:

    a) Run generation of a single cutouts dataset (training + validation)

Uses the 128x sample size
```
python3 src_util/generate_dataset.py -f -b -c 128 -p 500 -s 25
python3 src_util/generate_dataset.py -b -c 128 -r -p 500 -s 25
python3 src_util/generate_dataset.py -f -b -c 128 -p 500 -s 25 -v
python3 src_util/generate_dataset.py -b -c 128 -r -p 500 -s 25 -v
```

b) Run generate_all_datasets.py

- Takes a few minutes, creates 18 dataset versions. Not necessary for a single training/evaluation run.

###Evaluate a trained CNN model

The final model is by default set for running evaluation.

The model weights and outputs folder names contain the model training run name:
`64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full`

The code is best run by parts in an interactive environment like ipython


