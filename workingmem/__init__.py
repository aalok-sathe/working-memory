"""
.. include:: ../README.md
"""

import typing
import copy
import dataclasses
import yaml
import logging
import random
from pathlib import Path
from collections import defaultdict
import os

# 3rd party packages
import wandb
from dacite import from_dict

# local
from workingmem.model import (
    ModelWrapper,
    ModelConfig,
    # TransformerConfig,
    # RNNConfig,
    TrainingConfig,
    TransformerModelWrapper,
    RNNModelWrapper,
    LSTMModelWrapper,
)
from workingmem.task import SIRDataset, SIRConfig, _T_dataset_or_collection_of_datasets
from workingmem.utils import print_gpu_mem, wandbapi
import workingmem.model
import workingmem.task.SIR

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(LOGLEVEL)


@dataclasses.dataclass
class WandbConfig:
    create_sweep: bool = False
    run_sweep: bool = False
    sweep_id: str = None  # required if do_sweep is True
    project_name: str = "wm-mechanisms-1"
    # method: str = "bayes"  # use this for a hparam sweep
    method: str = "grid"  # use this once hparams are fixed
    metric: dict = dataclasses.field(
        default_factory=lambda: {"goal": "maximize", "name": "eval_acc"}
    )
    program: str = "run_wm.py"  # the program to run with a wandb sweep agent
    from_config: typing.Union[str, None] = (
        None  # path to the config YAML file where an experimental setup is specified.
    )
    prefix: str = wandbapi.viewer.username  # account prefix where your wandb sweeps are created. login to wandb.ai in a browser to find out!
    download_runs: typing.Union[str, None] = (
        None  # path to a created config YAML file outputted by this
        # program as a result of the `create_sweep` and `from_config` flags.
        # triggers a workflow where the sweeps contained therein are fetched
        # from wandb servers and stored in csv files named as `sweep_id.csv`
        # in a directory called `downloaded_runs` in the same directory structure
        # nested under the parent config's experiments structure
    )
    """
    `from_config`: only applicable with `create_sweep=True`. reads in a config
    file (YAML) if supplied that enumerates variations over individual variables
    the product of each variable's possible values is used to create a product
    of that many new sweeps, also printed out as a table at the end of running
    this module with this option enabled (both `create_sweep` and `from_config`).
    expects a simple enumaration of values (e.g., `dataset.concurrent_reg: [2,4,8]`)
    rather than `wandb`-specific format (i.e., `dataset.concurrent_reg: {values: [2,4,8]}`)  
    """


@dataclasses.dataclass
class MainConfig:
    """
    Run a recipe of loading a dataset, training a model, and evaluating it.
    Coming soon: load a model from a checkpoint to cross-train or evaluate it (for this, we will need to implement training history recordkeeping).
    """

    model: ModelConfig
    dataset: SIRConfig
    trainer: TrainingConfig
    wandb: WandbConfig
    seed = None
    array_task_id: typing.Union[int, None] = None
    filter_by_accuracy: bool = None
    filter_by_accuracy_threshold: float = 0.7

    # names of partitions to utilize in submitting jobs to. we will uniformly alternate
    # between them for each condition we construct
    gpu_partition_names: tuple = (
        "3090-gcondo",
        "gpu --account=carney-frankmj-condo",
    )

    def __post_init__(self, *args, **kwargs):
        logger.info(f"running post-init hook to set seeds to {self.seed}")
        if self.seed is not None:
            # prefer to keep using the same dataset instance across model training seeds
            # unless the dataset seed is explicitly set, so we wont be setting
            # `self.dataset.seed` here.
            self.model.seed = self.seed
            self.trainer.seed = self.seed
            # additionally set the seed globally here?
            # NOTE do not set seed for dataset here---we don't want datasets to vary for each instance of
            # a model, because that would introduce too much variability in the model training outcomes

            import torch
            import numpy as np

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)


def main(config: MainConfig):
    """
    given a config, train a model on an SIR dataset, evaluate, and test, all described
    as per config. wandb is used for logging regardless of whether this is a 'sweep'.
    """
    supplied_batch_size = config.trainer.batch_size
    config.trainer.batch_size = 256
    logger.warning(
        f"OVERRIDE {supplied_batch_size=}: starting with {config.trainer.batch_size} to search over the memory limit"
    )
    logger.info(f"running main with config: {config}")
    if config.dataset.create_dataset_and_exit:
        pass  # no need to initiate a w&b run for this
    else:
        wandb.init(
            project=config.wandb.project_name,
            config=config,
            dir=str(Path("~/scratch/wandb").expanduser().resolve()),
        )

    # set up the dataset
    logger.info(f"loading datasets using {config.dataset}")

    if isinstance(config.dataset.concurrent_reg, int):
        # this condition indicates concurrent_reg is supplied a single integer value
        # (this is the typical case)
        # we proceed as normal, instantiating an SIRDataset object that generates and
        # caches the dataset on disk if necessary
        train_config = copy.deepcopy(config.dataset)
        eval_config = from_dict(SIRConfig, dataclasses.asdict(train_config))
        test_config = from_dict(SIRConfig, dataclasses.asdict(train_config))
        eval_config.split, test_config.split = "val", "test"

        train_dataset = SIRDataset(train_config)
        eval_dataset = SIRDataset(eval_config)
        test_dataset = SIRDataset(test_config)

        logger.info("train dataset size: %s", len(train_dataset))
        logger.info("eval dataset size: %s", len(eval_dataset))
        logger.info("test dataset size: %s", len(test_dataset))

        # we need to explicitly set `d_vocab` if it isn't supplied via CLI, only if we're not
        # loading a model from disk
        if not config.model.d_vocab:
            config.model.d_vocab = eval_dataset.vocab_size

    else:
        # this situation indicates we are supplied with a list/tuple of ints
        # indicating all the possible concurrent_reg values we want mixed into this
        # dataset, for meta-training
        # first, we will iterate through the list and generate parent datasets if
        # needed by initializing them in the "normal" way (construct `SIRDataset` instance,
        # which will trigger constructing dataset examples and caching it to disk).
        # second, we will draw a proportionate sample of train, eval, and test examples
        # from each of these parent datasets. the proportions will be n_examples / len(concurrent_reg_values).
        # we will assemble these samples into a new mixture dataset for
        # meta-training.

        concurrent_reg_values = tuple(config.dataset.concurrent_reg)
        # initialize a train, eval, and test dataset for each of the values and add it to a list
        train_dataset = []
        eval_dataset = []
        test_dataset = []
        for value in concurrent_reg_values:
            train_config = copy.deepcopy(config.dataset)
            train_config.concurrent_reg = value
            train_config.n_back = value
            eval_config = from_dict(SIRConfig, dataclasses.asdict(train_config))
            test_config = from_dict(SIRConfig, dataclasses.asdict(train_config))
            eval_config.split, test_config.split = "val", "test"
            logger.info(f"assembling dataset corresponding to {train_config}")

            _train_dataset = SIRDataset(train_config)
            _eval_dataset = SIRDataset(eval_config)
            _test_dataset = SIRDataset(test_config)

            train_dataset += [_train_dataset]
            eval_dataset += [_eval_dataset]
            test_dataset += [_test_dataset]

            logger.info("...train dataset size: %s", len(_train_dataset))
            logger.info("...eval dataset size: %s", len(_eval_dataset))
            logger.info("...test dataset size: %s", len(_test_dataset))

            # we need to explicitly set `d_vocab` if it isn't supplied via CLI, only if we're not
            # loading a model from disk
            if not config.model.d_vocab:
                config.model.d_vocab = _eval_dataset.vocab_size

    print_gpu_mem(train_dataset)
    print_gpu_mem(eval_dataset)
    print_gpu_mem(test_dataset)

    if config.dataset.create_dataset_and_exit:
        logger.info("STOP after creating dataset")
        exit()

    # set up the model
    logger.info("initializing model")

    # if we're loading a pretrained model, check if an explicit model is passed, or a directory containing many models is
    # provided, in which case, we'd use the `config.array_task_id` to load the Xth model (modulo total models in dir)
    if (
        config.model.from_pretrained
        and len(list(Path(config.model.from_pretrained).glob("*.pth"))) == 0
    ):
        # enumerate subdirectories within this dirctory
        # and load the Xth model modulo the number of models in the directory
        models_dir = Path(config.model.from_pretrained)
        models_dir = list(models_dir.glob("*"))
        assert all(len(list(m.glob("*.pth"))) == 1 for m in models_dir), (
            f"malformed model checkpoints dir passed: {models_dir}"
        )

        if config.filter_by_accuracy:
            threshold: float = config.filter_by_accuracy_threshold

            # filter models by the accuracy recorded in their history
            def filter_by_accuracy(m: Path, threshold=threshold) -> bool:
                with open(m / "history.yaml", "r") as f:
                    history = yaml.load(f, Loader=yaml.FullLoader)
                return history[-1]["eval_acc"] >= threshold

            prev_len = len(models_dir)
            models_dir = list(filter(filter_by_accuracy, models_dir))
            logger.info(
                f"filtering models by accuracy >= {threshold} in {models_dir}. {prev_len = }, {len(models_dir) = }"
            )

        # set `from_pretrained` path to one of the pretrained models after filtering for its end accuracy.
        # if a seed is provided, we actually use the seed as a modulo rotary operator to pick the Xth index.
        # if no seed is provided, we randomly pick from the list of models.
        if config.model.seed is not None:
            logger.info(
                f"{config.model.seed = }. picking {config.model.seed % len(models_dir)}th model from {len(models_dir)} models (post-filtering, if applicable)"
            )
            config.model.from_pretrained = str(
                models_dir[config.model.seed % len(models_dir)]
            )
        else:
            config.model.from_pretrained = str(random.choice(models_dir))
        # record the new pretrained model path corresponding to the model we're actually using
        wandb.config.update(
            {"model.from_pretrained": str(config.model.from_pretrained)},
            allow_val_change=True,
        )

    # once the `from_pretrained` path is set to a not-None value, we can just use the regular way to
    # load the model, since the `ModelWrapper` class will take care of loading the model from checkpoint
    # check model class to instantiate the correct model wrapper
    # model = ModelWrapper(config.model)
    if config.model.model_class == "transformer":
        model = TransformerModelWrapper(config.model)
    elif config.model.model_class == "rnn":
        model = RNNModelWrapper(config.model)
    elif config.model.model_class == "lstm":
        model = LSTMModelWrapper(config.model)
    else:
        raise ValueError(f"unknown model class: {config.model.model_class}")

    logger.info(f"{config.model.model_class} model initialized.")
    logger.info(
        f"model initialized with {config.model.n_layers} layers, {config.model.n_heads} heads, "
        f"{config.model.d_model} d_model, {config.model.d_vocab} d_vocab, "
        f"from pretrained: {config.model.from_pretrained}"
    )
    print_gpu_mem(model)

    new_epochs = int(config.trainer.epochs / (1 - config.trainer.sparsity))
    # adjust epochs for sparsity
    logger.info(
        f"adjusting epochs for sparsity: {config.trainer.epochs} -> {new_epochs}"
    )
    config.trainer.epochs = new_epochs
    wandb.config.update(
        {"trainer.epochs": config.trainer.epochs}, allow_val_change=True
    )

    logger.info(f"about to start training on: {repr(train_dataset)}")
    if config.dataset.split == "train":
        # train the model
        logger.info("Training the model")

        while config.trainer.batch_size >= 16:
            # if the batch size is too large, we won't be able to fit the model in memory
            # so we will reduce it until it fits
            try:
                model.train(
                    train_dataset,
                    config.trainer,
                    eval_dataset=eval_dataset,
                    test_dataset=test_dataset,
                )
                break  # if training succeeded, we can break out of the loop
            except RuntimeError as e:
                if "CUDA out of memory. Tried to allocate" in str(e):
                    logger.info(str(e))
                    logger.warning(
                        f"⚠ batch size {config.trainer.batch_size} is too large, reducing it by half to {config.trainer.batch_size // 2} and retrying"
                    )
                    config.trainer.batch_size //= 2
                    # remember to update the wandb config for logging
                    wandb.config.update(
                        {"trainer.batch_size": config.trainer.batch_size},
                        allow_val_change=True,
                    )
                else:
                    raise e
        else:
            logger.error(
                f"could not train the model with batch size {config.trainer.batch_size} even after reducing it to 16, exiting"
            )

        logger.info("Finished.")
