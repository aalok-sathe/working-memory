#!/usr/bin/env python
import typing

from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import tyro
import pandas as pd
import yaml


@lru_cache(maxsize=8)
def gather_data(
    basedir: typing.Union[str, Path], kind: str, include: typing.Tuple[str] = None
) -> typing.Dict[str, dict]:
    """
    recursively searches through all children of basedir for YAMLs representing
    sweep dictionaries---dictionaries corresponding to attributes of sweeps
    created using configs defining the experiment. indexes these found sweep_dicts
    in order to be available to be referenced down the line.
    """

    all_dfs = []
    for downloaded_data_file in Path(basedir).rglob(f"downloaded_runs/*{kind}.csv"):
        if include is not None and any(
            sweep_id in str(downloaded_data_file) for sweep_id in include
        ):
            all_dfs += [pd.read_csv(downloaded_data_file)]

    return pd.concat(all_dfs)


def subset_data(
    basedir: typing.Union[str, Path],
    index: typing.Dict[str, typing.Callable] = {"sweep_id": lambda x: True},
    kind: str = "epochs",
):
    """
    subsets dataset by `index` of the form of a dictionary providing callable
    constraints. the callable constraints evaluate values of the keys in all_dfs.
    anything not appearing in `index` is treated as a variable and allowed to take
    on multiple values.
    """
    sweeps = gather_sweeps(basedir)
    properties = extract_sweep_properties(sweeps, list(index.keys()))
    # evaluate properties using index
    sweep_ids_to_keep = [
        sweep["sweep_id"]
        for sweep in sweeps.values()
        if all(index[prop](properties[sweep["sweep_id"]][prop]) for prop in index)
    ]
    return gather_data(basedir, kind=kind, include=tuple(sorted(sweep_ids_to_keep)))


def gather_sweeps(basedir: typing.Union[str, Path]) -> typing.Dict[str, dict]:
    """
    recursively searches through all children of basedir for YAMLs representing
    sweep dictionaries---dictionaries corresponding to attributes of sweeps
    created using configs defining the experiment. indexes these found sweep_dicts
    in order to be available to be referenced down the line.
    """

    all_sweeps = {}
    for sweep_dicts_file in Path(basedir).rglob("*sweep_dict.yaml"):
        with sweep_dicts_file.open("r") as f:
            sweep_dicts = yaml.load(f, yaml.FullLoader)
        for sweep_dict in sweep_dicts:
            sweep_id = sweep_dict["sweep_id"]
            all_sweeps[sweep_id] = sweep_dict

    return all_sweeps


def extract_sweep_properties(
    sweeps, properties: typing.List[str]
) -> typing.Dict[str, dict]:

    def _value_if_allowed(thing):
        if isinstance(thing, dict):
            return thing["value"]
        elif isinstance(thing, str):
            return thing
        raise ValueError(thing)

    extracted_properties = {
        key: {
            property: _value_if_allowed(sweeps[key].get(property, {"value": None}))
            for property in properties
        }
        for key in sweeps
    }
    return extracted_properties


@dataclass
class Config:
    basedir: Path  # directory where to look for sweep_dict configs
    property: typing.Union[typing.List[str], None] = (
        None  # variable to extract from each sweep_dict
    )
    kind: str = "epochs"


def main(config: Config):
    all_sweeps = gather_sweeps(config.basedir)
    if config.property is not None:
        extracted_properties = extract_sweep_properties(all_sweeps, config.property)
        print(extracted_properties)
    else:
        print(all_sweeps.keys())


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
