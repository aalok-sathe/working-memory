# `simWM` 🏊: simulation of working memory management with neural networks

## Introduction
Perceptual as well organism-internal information is plentiful---being able to use all of this information is essential to optimal planning and decision-making. 
Deciding what information is relevant, choosing to robustly maintain it, accessing it on demand, and overwriting existing information with new, potentially more 
relevant information, are all ecologically-relevant demands that make intelligent behavior possible. 
However, knowing what information is relevant in any given context is difficult and needs to be learned over developmental as well as evolutionary timescales.

This repository serves to enable computational simulations of learning and generalization in service of understanding working memory capacity better. A robustly-engineered 
framework allows researchers to define their own tasks and generate datasets with fine control over the parameters of those tasks. The framework allows for training 
neural models in an architecture-agnostic manner, so long as they support a PyTorch backend. This includes transformers, recurrent neural networks, and long short-term memory
networks that can flexibly act as _participants_ in the same tasks, allowing for clean, controlled comparisons across model instantiations.

Integration with the 
[weights & biases (`wandb`)](https://wandb.ai)
experiment-tracking platform promotes robust, open, and replicable science by constructing uniquely-tagged and documented config-driven experiments for each condition
a researcher may be interested in. These experimental conditions live in separate spaces on the disk, each with meticulously documented metadata that makes
discerning results and analyzing data pleasantly organized. Furthermore, the software supports exposure and transfer-learning experiments using these same condition-based
tags, allowing precise documentation of the entire training history of a computational model and subsequent experiments on pretrained models.

The main module (`python -m workingmem`), implemented as an entrypoint in `workingmem/__main__.py`, does the orchestrating of running experiments, i.e., 
loading/constructing datasets, training/evaluating models. However, much of the library's functionality exposed for programmatic use as well, and allows
researchers to construct datasets in a custom manner, manage their own training-eval routines, and handle data management.

**A typical experiment workflow looks like:**
1. identify manipulations of interest (see what variations the library already supports using `python -m workingmem -h`).
2. write/modify a config defining experimental conditions ([example](./configs/sample_conditional_config))
    - configs follow an "independent variables" and "conditional variables" format---independent variables are enumerated as lists
      that yield a cross-product over all possible combinations of their variation. "conditional variables" are typically hyperparameters
      that need to be looked up dependent on the particular condition.  
3. use `python -m workingmem --wandb.create_sweep` along with the flag `--wandb.from_config [path/to/config]` to define individual experimental conditions.
  at this point, the library evaluates a cross-product over all possible conditions in your experiment and creates individual
  W&B "sweeps" for each condition. this enables separate tracking of the progress of experiments in a web browser, as well as unique-ID-based
  retrieval after the experiment finishes for clean, reproducible science.

For programmatic use, components of the library can be imported in your program: `import workingmem`, or `from workingmem import LSTMModelWrapper, SIRDataset`.

To exhaustively see the CLI options, run `python -m workingmem -h`.

## Getting started / Install
1. Use with Weights and Biases (recommended)

   `simWM` is best used alongside Weights and Biases. In order to do so, you will have to create an account on the [W&B website](https://wandb.ai).
    There are many ways in which to do so, including using your GitHub login.
   
1. Install `uv`

    `simWM` uses [`uv`](https://atral.sh/uv) as its package- and environement-manager. `uv` makes painless the age-old task of managing dependencies
    in Python. In order to install the framework, you'll need to install `uv` on your system. This is fairly straightforward---visit the link from before.

1. Install `simWM` (this library)
    - `uv sync`: install the python virtual environment with all requisite packages (needed once)
    - `. ./.venv/bin/activate`: activate the virtual environment in the directory of the library (needed each time you log in to your compute node until you exit/log out)
