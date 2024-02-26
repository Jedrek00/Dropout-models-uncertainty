<div align="center">

# Dropout-models-uncertainty

<p align="center">
    Experiments for modeling uncertainty with dropout research.
    <br />
    <i>Project made for Deep Learning classes</i>
    <br/>
    <b>Date of completion: 📆 24.01.2024 📆</b>
  </p>

</div>

## About The Project

The goal of the project was to create relevant experiments to extend the scientific work [Gal, Y and Ghahramani, Z. Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf). We reproduced the research described in the aforementioned paper, and also checked whether we were able to obtain similar results using other dropout techniques.

We will soon describe the results of the experiments in a corresponding blog post!

## Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![MLFlow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=mlflow&logoColor=blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Installation

Poetry is used to manage dependencies for this repository. Check if poetry is installed on your machine using command:

```
poetry --version
```

If version of the poetry was displayed, you can install all required packages using command:

```
poetry install
```

To install poetry on your machine see [documentation](https://python-poetry.org/docs/cli/#install).

All code should be ran in the env created by the poetry. To ensure that you are running scripts in the env created by poetry run command:

```
poetry shell
```

## Installing new modules
Instead of using *pip install module_name*
Use:
```
poetry add module_name
```

## MLFlow
Run MLFlow tool inside virtual environment created by poetry:
```
mlflow ui
```

## Example results
The chart below illustrates the performance of our algorithm for arbitrarily chosen hyperparameters. This is a perfect example of correctly modeling uncertainty using the dropout technique. 

![Modelling uncertainty example plot](data/plots/fashion_mnist/0-morph-1022-densenet-drop_connect-0.1-50.png) 

Our project allows user to generate its own, custom charts for user-selected hyperparameters.