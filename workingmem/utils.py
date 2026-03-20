#!/bin/bash
from pandas.core.frame import DataFrame

from itertools import product
import typing
from functools import lru_cache
import logging
from pathlib import Path

import yaml
import pandas as pd
from tqdm.auto import tqdm
import wandb
import dacite

wandbapi = wandb.Api()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
logger.setLevel(logging.INFO)


def print_gpu_mem(obj: typing.Any = None):
    """
    Print the GPU memory usage.
    """
    import torch

    if torch.cuda.is_available():
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )
        if obj is not None:
            logger.info(
                f"GPU memory allocated for {obj.__class__.__name__}: "
                f"{torch.cuda.memory_allocated(obj) / 1024**3:.2f} GB"
            )
    else:
        logger.info("No GPU available; no memory report.")


@lru_cache(maxsize=None)
def _get_wandb_runs(
    project_name: str, sweep_id: str, prefix=wandbapi.viewer.username, samples=20_000
) -> DataFrame:
    """
    for a given workspace/user prefix and sweep ID retrieves all the runs corresponding
    to that sweep. in this experimental framework each sweep should represent a single
    experimental condition. so every sweep is associated with identical metadata save for
    random seed in cases where models are randomly initialized, and the random shuffling
    of datapoints during runs. this code, however, only includes so many hyperparameters
    as were included in the config sent to wandb to initialize the run. does not extensively
    catalog all the default parameters
    """
    from workingmem import MainConfig, ModelConfig, TrainingConfig, SIRConfig

    runs = wandbapi.sweep(f"{prefix}/{project_name}/{sweep_id}").runs
    dfs = []
    for run in tqdm([*runs]):
        metrics: pd.DataFrame = run.history(pandas=True, samples=samples)
        metrics["run_id"] = run.name
        metrics["sweep_id"] = sweep_id
        config: typing.Dict[str, typing.Any] = run.config
        all_configs = {
            **config["model"],
            **config["trainer"],
            **config["dataset"],
        }
        for key in all_configs:
            metrics[key] = all_configs[key]

        try:
            dfs += [metrics]
        except KeyError:
            # this run doesn't have enough data to have 'epoch' as a key; skip for now
            print(
                f"\tkey `epoch` not found. skipping run: https://wandb.ai/{prefix}/{project_name}/runs/{run.name}"
            )
            pass

    df = pd.concat(dfs).reset_index(drop=True)
    return df


@typing.overload
def get_wandb_runs(
    project_name: str, sweep_id: str, prefix: str, samples: int
) -> DataFrame: ...


@typing.overload
def get_wandb_runs(config_path: typing.Union[str, Path], samples: int) -> DataFrame: ...


def get_wandb_runs(
    project_name: str = None,
    sweep_id: str = None,
    prefix=wandbapi.viewer.username,
    config_path: typing.Union[str, Path] = None,
    samples=10_000,
) -> DataFrame:

    if sweep_id is None:
        config_path = (
            project_name  # accommodate alternate overridden function signature
        )

    if config_path is not None:
        assert "sweep_dict" in str(config_path), (
            f"are you sure you passed the correct config? input: {config_path}"
        )
        with Path(config_path).open("r") as f:
            created_config = yaml.load(f, yaml.FullLoader)
        for sweep in tqdm(created_config, desc="fetching sweeps from created config"):
            # for each sweep obtain the workspace prefix and experiment name and sweep_id
            # and fetch the runs co...rresponding to it
            project_name = sweep["project_id"]
            sweep_id = sweep["sweep_id"]
            prefix = sweep["username"]
            sweep_df = get_wandb_runs(project_name, sweep_id, prefix)
            sweep_df_grouped_by_epoch = (
                sweep_df.groupby(["epoch", "run_id"]).first().reset_index()
            )

            dest = Path(config_path).parent.parent / "downloaded_runs"
            dest.mkdir(exist_ok=True)
            sweep_df.to_csv(dest / (sweep_id + "_steps.csv"))
            sweep_df_grouped_by_epoch.to_csv(dest / (sweep_id + "_epochs.csv"))

    else:
        return _get_wandb_runs(project_name, sweep_id, prefix)


def _flatten_collection_of_tuples(
    keys_tuples_collection: typing.Collection[tuple],
    vals_tuples_collection: typing.Collection[tuple],
):
    keys_flat, vals_flat = [], []
    for keys_tuple, vals_tuple in zip(keys_tuples_collection, vals_tuples_collection):
        # deflate the tuples
        keys_flat += [*keys_tuple]
        vals_flat += [*vals_tuple]
    return keys_flat, vals_flat


def parse_config(config) -> typing.Generator[dict, None, None]:
    if "independent_variables" in config:
        independent_variables: typing.List[typing.Dict] = config[
            "independent_variables"
        ]

        conditional_variables: typing.List[dict] = config.get(
            "conditional_variables",
            [{"index": {}, "kwargs": {}}],  # default is no values to look up
        )

        def _lookup_kwargs(parameters):
            # we iterate through conditoinal variable entries in order
            # and check if
            kwargs = {}
            for cond_variable_set in conditional_variables or []:
                index = cond_variable_set["index"]
                if all(parameters[k] == v for k, v in index.items()):
                    this_kwargs = cond_variable_set["kwargs"]
                    kwargs.update(this_kwargs)
                    break
                continue
            return kwargs

        # we maintain tuples of keys and tuples of values
        # to enable grouping them together. after taking their product
        # we will uncouple them
        ind_keys_tuples, ind_vals_tuples = [], []

        for d in independent_variables:
            ind_keys_tuples += [d.keys()]
            ind_vals_tuples += [tuple(zip(*d.values()))]

        print(ind_keys_tuples, ind_vals_tuples)

        assert len(ind_keys_tuples) == len(ind_vals_tuples)

        values_product = [*product(*ind_vals_tuples)]

        # we want to create a sweep corresponding to each 'value set' at the end of the
        # cross product between all possible values of covarying independent variable sets
        for this_values_set in values_product:
            keys, vals = _flatten_collection_of_tuples(ind_keys_tuples, this_values_set)
            parameters = dict(zip(keys, vals))
            print(parameters)
            parameters.update(_lookup_kwargs(parameters))
            yield parameters

    else:  # this means the config file just contains (key: values) entries, old-style format
        logger.warning(
            "config supplied is old-style formatted; parsing assuming flat key-value structure."
        )
        keys, values = zip(*config.items())
        for this_values_set in product(*values):
            yield dict(zip(keys, this_values_set))
