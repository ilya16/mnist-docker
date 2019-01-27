# PyTorch MNIST Docker image

A Docker image with MNIST model for training and inference 
written using PyTorch framework.

# Build

To build a Docker image from the sources use the following:

```sh
git clone https://github.com/ilya16/mnist-docker
cd mnist-docker
docker build -t pytorch-mnist .
``` 

Another option is pull the image from DockerHub using:
```sh
docker pull ib16/pytorch-mnist:latest
```

# Running

MNIST model supports three modes:
* `fit` – fitting a new model
* `predict` – making an inference on a valid input
* `eval` – evaluating a model on a valid input with labels

Note: prediction and evaluation can only run after the model is fit.

The general command for running the image is:
```sh
docker run -it --rm \
    -v volume-mnist:/app/state \
    -v "$(pwd)"/data:/app/data \
    pytorch-mnist --mode [running mode (default: fit)] \
    --model-path [(optinal) path to save the model] \
    --batch-size [(optinal) training batch_size] \
    --epochs [(optinal) number of training epochs]
```

Model parameters are saved at `/app/state` inside a container. 
To use the same model across multiple runs, 
option ``-v volume-mnist:/app/state`` is necessary.

To use own train or test data, create a directory `data` with files 
inside your current working directory and 
use ``-v "$(pwd)"/data:/app/data`` with `docker run`.


## `fit`

To train a model on original MNIST train data use:

```sh
docker run -it --rm -v volume-mnist:/app/state pytorch-mnist --mode fit
```

To train a model on your own data `X_train.npy` and `y_train.npy`, 
add files to directory `./data` on your host machine and use:

```sh
docker run -it --rm \
    -v volume-mnist:/app/state -v "$(pwd)"/data:/app/data pytorch-mnist \
    pytorch-mnist --mode fit
```

If input files `X_train.npy` and `y_train.npy` are not found or invalid, 
original MNIST train data will be used.


## `predict`

Assuming, you have trained a model, run a model inference on MNIST test data with:

```sh
docker run -it --rm -v volume-mnist:/app/state pytorch-mnist --mode predict
```

To make predictions on your own data `X_test.npy`,
add file to directory `./data` on your host machine and use::

```sh
docker run -it --rm \
    -v volume-mnist:/app/state -v "$(pwd)"/data:/app/data pytorch-mnist \
    pytorch-mnist --mode predict
```

Inference results will be saved at `data/y_pred.npy`.

If input file `X_test.npy` is not found or invalid, 
original MNIST test data will be used.


## `eval`

Assuming, you have trained a model, run a model evaluation on MNIST test data with:

```sh
docker run -it --rm -v volume-mnist:/app/state pytorch-mnist --mode eval
```

To make evaluation on your own data `X_test.npy` and  `y_test.npy`,
add these files to directory `./data` on your host machine and use::

```sh
docker run -it --rm \
    -v volume-mnist:/app/state -v "$(pwd)"/data:/app/data pytorch-mnist \
    pytorch-mnist --mode eval
```

If input files `X_test.npy` and `y_test.npy` are not found or invalid, 
original MNIST test data will be used.
