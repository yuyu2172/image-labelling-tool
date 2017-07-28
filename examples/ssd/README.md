# SSD Example


This example provides following functionalities.

+ (Annotating dataset for detection task. This is the functionality supported by this tool.)
+ A data loader object that wraps annotated data, and conveniently access data using Python.
+ SSD training script that works with the data loader.


### Dependency

+ `chainer`
+ `chainercv>=0.6`


### Process
1. Annotate dataset (please read the README in the top of this repo)

Please store all image data below `DATA_DIR`. The label names should be writtein in a YAML file locating at LABEL_NAME.
The file extension of the images can be selected by EXT (e.g. jpg).

After finish annotating data,
the annotation files are stored umder `DATA_DIR` together with the images.

```bash
$ python flask_app.py --image_dir DATA_DIR --label_names LABEL_FILE --file_ext EXT
```

2. Train SSD

This script assumes data stored in the annotation style of this annotation tool.
This means that you do not need to write any data loader code by yourself.

```bash
python train.py --train DATA_DIR --label_names LABEL_NAMES --gpu GPU
```

```
$ python train.py -h
usage: train.py [-h] [--train TRAIN] [--val VAL] [--label_names LABEL_NAMES]
                [--iteration ITERATION] [--lr LR] [--step_size STEP_SIZE]
                [--batchsize BATCHSIZE] [--gpu GPU] [--out OUT]
                [--val_iteration VAL_ITERATION] [--resume RESUME]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         The root directory of the training dataset
  --val VAL             The root directory of the validation dataset. If this
                        is not supplied, the data for train dataset is split
                        into two with ratio 8:2.
  --label_names LABEL_NAMES
                        The path to the yaml file with label names
  --iteration ITERATION
                        The number of iterations to run until finishing the
                        train loop
  --lr LR               Initial learning rate
  --step_size STEP_SIZE
                        The number of iterations to run before dropping the
                        learning rate by 0.1
  --batchsize BATCHSIZE
                        The size of batch
  --gpu GPU             GPU ID
  --out OUT             The directory in which logs are saved
  --val_iteration VAL_ITERATION
                        The number of iterations between every validation.
  --resume RESUME       The path to the trainer snapshot to resume from. If
                        unspecified, no snapshot will be resumed
```


### Dividing dataset into train/val
When calling `train.py` without supplying `--val`, the dataset is split into two with ratio 8:2.
The larger one is used as the training dataset and the smaller one is used as the validation dataset.

Since the validation dataset is randomly split, it can be better to fix the split.
You can use fixed split by supplying both `--train` and `--val` when calling `train.py`.

In that case, you can use a convenient script `randomly_split_directory.py`.
This script divides all data in `DATA_DIR` into `TRAIN_DIR` and `VAL_DIR`.

Example usage:
`python randomly_split_directory.py TRAIN_DIR VAL_DIR DATA_DIR`
