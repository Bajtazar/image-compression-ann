Image compression using artificial neural networks and discrete wavelet transform
==============================

Bachelor Thesis utilizing an artifical neural networks and discrete wavelet transform in order to implement a lossy image compression codec. Project based on paper "[Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436)"

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── external           <- Python dependencies that cannot be installed via package manager
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── models.csv     <- File that translates training session id to the model params
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make benchmarks
    │       │
    │       ├── networks   <- module with different neural network architectures
    │       ├── gym        <- module with neural network training suite
    │       ├── benchmarks <- module with neural network benchmarking suite
    │       ├── benchmark_model.py
    │       ├── train_model.py
    │       └── update_config.py
    |
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    │
    └── config.ini         <- configuration file with the current network params

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
