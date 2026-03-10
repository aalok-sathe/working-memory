# sWM 🏊: computational simulation for working memory research

Perceptual as well internally-generated information is plentiful---being able to use all of this information is essential to optimal planning and decision-making. 
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

The main module (`python -m workingmem`), implemented as an entrypoint in `workingmem/__main__.py` does the orchestrating, i.e., 
loading/constructing datasets, training/evaluating models. However, much of the library's functionality exposed for programmatic use as well, and allows
researchers to construct datasets in a custom manner, manage their own training-eval routines, and handle data management.
Components of the library may be imported as so: `import workingmem`, or `from workingmem import LSTMModelWrapper`.

To see the options, run `python -m workingmem -h`.

## Getting started / Install
**Using with Weights and Biases (recommended)**
This framework is best used alongside Weights and Biases. In order to do so, you will have to create an account on the [W&B website](https://wandb.ai).
There are many ways in which to do so, including using your GitHub login.

**Installing UV**
**sWM** uses [`uv`](https://atral.sh/uv) as its package- and environement-manager. `uv` makes painless the age-old task of managing dependencies
in Python. In order to install the framework, you'll need to install `uv` on your system. This is fairly straightforward---visit the link from before.

**Installing sWiMm**
