# Transfer learning for computer vision

The purpose of this project is to demonstrate various use cases for transfer learning in object classification, object localization and object recognition. Popular models are used on a bunch of datasets of varying sizes. The tooling is centered on PyTorch for model implementation and MLFlow for experiment tracking.

## Dataset overview

TBD

## Installation

The project is centered around a package of utility functions installable as `cvision`. Recommended approach is to use an editable install in a running virtual environment.

- Activate your venv.
- Install with `pip install --editable .` from project root. 

## MLFlow

MLFlow tracking server is used for centralized logging for all model training runs. The server is packaged as a Flask app that can be started directly on localhost or via Docker.

To use a bare metal instance:

- MLFlow package is installed along with other project dependencies.
- Run server with `mlflow server --host 127.0.0.1 --port 8888`.

To use Docker:

TBD