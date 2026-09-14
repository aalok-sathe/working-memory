from collections import defaultdict
from functools import lru_cache
from itertools import product
import json
import logging
from pathlib import Path
import typing

import pandas as pd
from pandas.core.frame import DataFrame
from tqdm.auto import tqdm
import wandb
import yaml

import workingmem.utils.plotting as plotting


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
_logger = logging.getLogger("workingmem")
_logger.setLevel(logging.INFO)

try:
    wandbapi = wandb.Api()
except wandb.errors.UsageError:
    _logger.warning(
        "wandb API initialization failed. if you are trying to use wandb features, make sure you have logged in to wandb using `wandb login` command and that you have access to the project and sweep you are trying to fetch runs from."
    )

    class _AttrMaker:
        """
        duck typed class allowing indefinitely many recursive __getattr__ calls
        to help pass github actions without needing to log into W&B
        """

        def __init__(self, name=""):
            self.name = name

        def __getattr__(self, name):
            return _AttrMaker(self.name + "." + name)

        def __repr__(self) -> str:
            return f"<placeholder for wandbapi.{self.name}>"

        def __str__(self) -> str:
            return repr(self)

    wandbapi = _AttrMaker("wandbapi")


def print_gpu_mem(obj: typing.Any = None):
    """
    Print the GPU memory usage.
    """
    import torch

    if torch.cuda.is_available():
        _logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )
        if obj is not None:
            _logger.info(
                f"GPU memory allocated for {obj.__class__.__name__}: "
                f"{torch.cuda.memory_allocated(obj) / 1024**3:.2f} GB"
            )
    else:
        _logger.info("No GPU available; no memory report.")


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
    from workingmem import MainConfig, ModelConfig, SIRConfig, TrainingConfig

    runs = wandbapi.sweep(f"{prefix}/{project_name}/{sweep_id}").runs
    dfs = []
    for run in tqdm([*runs]):
        metrics: pd.DataFrame = run.history(pandas=True, samples=samples)
        metrics["run_id"] = run.name
        metrics["sweep_id"] = sweep_id
        config: typing.Dict[str, typing.Any] = run.config
        # print(config)
        all_configs = {
            **config,
        }
        for additional_key in ["model", "dataset", "trainer"]:
            all_configs.update(config.get(additional_key, {}))

        new_columns = {}
        for key in all_configs:
            value = all_configs[key]
            # if value is not a singleton, we wrap it in a tuple
            if isinstance(value, list):
                value = " ".join(map(str, value))
            new_columns[key] = value

        try:
            metrics = pd.concat(
                [metrics, pd.DataFrame(new_columns, index=metrics.index)], axis=1
            )
        except ValueError as e:
            print(e, new_columns)
            exit()

        try:
            dfs += [metrics]
        except KeyError:
            # this run doesn't have enough data to have 'epoch' as a key; skip for now
            print(
                f"\tkey `epoch` not found. skipping run: https://wandb.ai/{prefix}/{project_name}/runs/{run.name}"
            )
            pass

    try:
        df = pd.concat(dfs).reset_index(drop=True)
    except ValueError as e:
        if "No objects to concatenate" in str(e):
            # skip due to no completed runs
            _logger.warning(f"skipping sweep {sweep_id} due to no completed runs: {e}")
            return DataFrame()
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
            f"are you sure you passed the correct config? "
            f"expected input is 1 YAML file created as a result of creating a sweep. what you provided as input: {config_path}"
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
            try:
                sweep_df_grouped_by_epoch = (
                    sweep_df.groupby(["epoch", "run_id"]).first().reset_index()
                )
            except KeyError as e:
                _logger.warning(f"SKIPPING {sweep=} due to {e}")

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
    """
    Yield parameter dictionaries from supplied YAML experiment config.
    (For how to structure a config to define an experiment, see the [tutorial here](http://localhost:8080/workingmem.html#config-structure).)

    If config contains "independent_variables", each dict entry's values are zipped (covary)
    and different entries are combined with a Cartesian product; optional "conditional_variables"
    provide index-based kwargs merged into the resulting parameters. Otherwise treats config
    as a flat mapping and yields the Cartesian product of its value lists.

    Yields one dict per parameter combination. Relies on itertools.product and a helper
    flatten_collection_of_tuples to expand covarying keys/values. Matching for conditional
    entries uses equality on specified keys; the first matching kwargs entry is applied.
    """
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
        _logger.warning(
            "config supplied is old-style formatted; parsing assuming flat key-value structure."
        )
        keys, values = zip(*config.items())
        for this_values_set in product(*values):
            yield dict(zip(keys, this_values_set))


def annotate_trial_seq(
    preds: typing.List[str], labels: typing.List[str], trial_seq: str
) -> pd.DataFrame:
    """
    processes trials of type (instr role_n item_m ans) where instr is one of St/Ig standing for
    store or ignore. first, we compile a unique list of roles appearing in this trial
    then we create a state (dict) mapping each role to the item it currently stores. at each trial,
    if the instruction is 'store', we update the state to store the item for that role.
    also in the state, we store the time elapsed since last accessing that role for any instruction,
    as well as last _updating_ that role (i.e., St instruction). we also annotate the current trial's
    answer (same/diff) as well as whether the model got the prediction right (preds vs labels).

    also includes `oracle_memory_state`, a json-serialized `{role: item}` snapshot of the ground-truth
    memory contents as of the *end* of that trial (i.e., after applying this trial's update, if any) --
    this is the oracle/ground-truth state a model would need to track internally to solve the task,
    as opposed to `role`/`item`/`instr`, which describe only the current trial's own instruction.
    """
    trial_seq: typing.List[str] = trial_seq.split()
    roles = set()
    for token in trial_seq:
        if token.startswith("reg_"):
            roles.add(token)
    state = defaultdict(
        lambda: dict(time_since_access=-1, time_since_update=-1, num_accesses=0)
    )
    oracle_memory: typing.Dict[str, str] = {}
    annotated_seq = []
    for i in range(0, len(trial_seq), 4):
        instr, role, item, ans = trial_seq[i : i + 4]

        for r in roles:
            if r == role:
                state[r]["num_accesses"] += 1
                continue
            if state[r]["time_since_access"] >= 0:
                state[r]["time_since_access"] += 1
            if state[r]["time_since_update"] >= 0:
                state[r]["time_since_update"] += 1

        if instr == "St":
            oracle_memory[role] = item

        annotated_seq.append({
            "trial_ix": i // 4,
            "instr": instr,
            "role": role,
            "item": item,
            "ans": ans,
            "correct": int(preds[i // 4] == labels[i // 4]),
            "label": labels[i // 4],
            **state[role].copy(),
            "oracle_memory_state": json.dumps(oracle_memory),
        })
        state[role]["time_since_access"] = 0
        if instr == "St":
            state[role]["time_since_update"] = 0

    return pd.DataFrame(annotated_seq)


def get_annotated_representations(
    model, example: dict, mask_answer_tokens: bool = True
) -> pd.DataFrame:
    """
    Runs `model.get_representations_over_sequence` on a single (unbatched)
    dataset example (e.g. `dataset[i]`) and returns a per-token DataFrame that
    merges the SIR trial annotations from `annotate_trial_seq` (instr, role,
    item, label, correct, time_since_access, time_since_update, num_accesses)
    with every representation the model reports for that example
    (`embeddings`, `hidden_states`/`cell_states`,
    `percell_hidden_states`/`percell_cell_states` for `LSTMMultiCellWrapper`,
    `logits`, etc. -- whatever keys that model class's
    `get_representations_over_sequence` returns).

    A trial spans 4 tokens (instr, role, item, ans) but `annotate_trial_seq`
    (like `ModelWrapper.evaluate`/`compute_masked_loss`) reasons about one row
    per trial, gathered at the answer position. To align with the
    per-timestep representation tensors (each of shape `(seq_len, ...)` where
    `seq_len` counts *tokens*, not trials), each trial's row from
    `annotate_trial_seq` is broadcast across its 4 constituent token rows
    here, plus a `token`/`token_type` ("instr"/"role"/"item"/"ans") column
    identifying which of the 4 each row corresponds to.

    Predictions are read off the logit *preceding* each answer token
    (`answer_locations.nonzero() - 1`), mirroring
    `compute_masked_loss`'s `gathered_logits` -- i.e., the model predicts the
    answer at the item token's position, one step before the answer token
    itself appears.
    """
    reprs = model.get_representations_over_sequence(
        example, mask_answer_tokens=mask_answer_tokens
    )

    # NOTE: `example` (unlike the returned `reprs`) is left batched in-place by
    # `get_representations_over_sequence` (shape (1, seq_len)) even though it was passed
    # in unbatched -- only `reprs` gets unbatched before being returned.
    answer_locations = example["answer_locations"].squeeze(0)
    answers = example["answers"].squeeze(0)
    answer_positions = answer_locations.nonzero(as_tuple=True)[0]
    pred_ids = (
        reprs["logits"][answer_positions - 1].argmax(dim=-1).detach().cpu().tolist()
    )
    label_ids = answers[answer_positions].detach().cpu().tolist()

    trial_df = annotate_trial_seq(pred_ids, label_ids, example["tokens"])

    tokens = example["tokens"].split()
    token_types = ["instr", "role", "item", "ans"] * len(trial_df)
    assert len(tokens) == len(token_types), (
        f"{len(tokens) = } tokens in example but {len(trial_df) = } trials "
        f"(expected {len(trial_df) * 4} tokens)"
    )

    annotations = trial_df.loc[trial_df.index.repeat(4)].reset_index(drop=True)
    annotations.insert(0, "token_ix", range(len(annotations)))
    annotations.insert(2, "token_type", token_types)
    annotations.insert(3, "token", tokens)
    # `label`/`correct` (from `annotate_trial_seq`) are only meaningful for the trial as a
    # whole (i.e. at its answer token); blank them out on the other 3 broadcast rows.
    is_ans = annotations["token_type"] == "ans"
    annotations.loc[~is_ans, ["label", "correct"]] = None

    for key, value in reprs.items():
        if not hasattr(value, "detach"):  # skip non-tensor entries, if any
            continue
        annotations[key] = list(value.detach().cpu().numpy())

    return annotations


def load_model_and_dataset(ckpt_path, split="test", epoch=None):
    """
    given a checkpoint path, load the model and the corresponding dataset it was most
    recently trained on. we can infer the model class from the config.yaml file
    in the checkpoint directory, and we can infer the dataset path from the
    history.yaml file in the checkpoint directory.
    """
    import torch

    from workingmem.model import (
        LSTMModelWrapper,
        LSTMMultiCellWrapper,
        RNNModelWrapper,
        TransformerModelWrapper,
    )
    from workingmem.task import SIRDataset

    model_config = yaml.load(
        (ckpt_path / "config.yaml").open("r").read(), Loader=yaml.SafeLoader
    )
    model_class = model_config["model_class"]
    if model_class == "lstm":
        model_class = LSTMModelWrapper
    elif model_class == "lstm_multicell":
        model_class = LSTMMultiCellWrapper
    elif model_class == "rnn":
        model_class = RNNModelWrapper
    elif model_class == "transformer":
        model_class = TransformerModelWrapper
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model = model_class.from_checkpoint_dir(ckpt_path, epoch=epoch)

    # also load the dataset corresponding to the model which is specified in the model's history.yaml file
    hist = yaml.load(
        (ckpt_path / "history.yaml").open("r").read(), Loader=yaml.SafeLoader
    )
    dataset_path = Path(hist[-1]["dataset_path"])
    dataset = SIRDataset.from_path(dataset_path, split=split, generate=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    return model, dataset
