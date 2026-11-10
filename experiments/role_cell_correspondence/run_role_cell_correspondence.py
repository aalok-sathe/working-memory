"""
Standalone (non-marimo) script version of the `compute_role_cell_correspondence`
analysis prototyped in `notebooks/decoder_playground_marimo.py`, for batch
submission to SLURM rather than interactive exploration.

Copied verbatim (plus `--necessity-only`, see below) from the main repo's
`experiments/role_cell_correspondence/run_role_cell_correspondence.py` onto
this branch (feature-multicell-competition-reg) -- all paths
(`REPO_ROOT`/`DOWNLOADED_RUNS_DIR`/`CHECKPOINTS_DIR`) are computed relative
to this file's own location, so running it from THIS worktree automatically
analyzes the `competition_reg_coef=1.0` staggered-sweep checkpoints here
instead of the baseline ones in the main repo -- no other changes needed.

For one (num_lstm_cells, concurrent_reg, rank) condition -- selected either via
`--job-index` or `$SLURM_ARRAY_TASK_ID` -- this: looks up the `rank`-th best
(by eval_acc) run for that (num_lstm_cells, concurrent_reg) pair from the
downloaded sweep CSVs (`query_runs_by_condition`), loads that run's checkpoint,
runs the per-cell ablation "which cells does each register's identity depend
on" analysis across several trial sequences (`compute_role_cell_correspondence`),
and writes the resulting dict to its own YAML file under `--output-dir`.

`--necessity-only` skips the sufficiency-side ablation fits entirely (not
just omits them from the output) -- roughly halves per-task compute for
callers that only want necessity.

Usage (manual, one job):
    python run_role_cell_correspondence.py --job-index 0 --necessity-only

Usage (SLURM array, one task per condition x rank):
    see `submit_role_cell_correspondence.sh` in this directory -- it sets
    `$SLURM_ARRAY_TASK_ID`, which this script reads automatically when
    `--job-index` is not given.
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from tqdm.auto import tqdm

from workingmem.utils import get_annotated_representations, load_model_and_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

# the grid of conditions to analyze: num_lstm_cells x concurrent_reg, picking
# the top `TOP_K` models (by eval_acc) for each of the 6 * 4 = 24 combinations.
NUM_LSTM_CELLS = [1, 2, 4, 6, 8, 10]
CONCURRENT_REGS = [2, 4, 6, 8]
TOP_K = 10  # ranks 0..9

DOWNLOADED_RUNS_DIR = (
    REPO_ROOT
    / "experiments/lstm_multicell_sweep/lstm_multicell_stress_test_staggered.yaml_experiments/downloaded_runs"
)
CHECKPOINTS_DIR = REPO_ROOT / "model_checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

RATIO_THRESHOLD = 0.95
TRIAL_INDICES = range(15)


# ---------------------------------------------------------------------------
# prerequisites copied from notebooks/decoder_playground_marimo.py -- these
# are notebook-local helpers (not part of the `workingmem` package), so they're
# duplicated here rather than imported. Keep in sync with the notebook if
# either changes.
# ---------------------------------------------------------------------------


def query_runs_by_condition(
    num_lstm_cells, concurrent_reg, downloaded_runs_dir=DOWNLOADED_RUNS_DIR
):
    """
    Reads every `<sweep_id>_epochs.csv` in `downloaded_runs_dir` (produced by
    `swim --wandb.download_runs ...`, see CLAUDE.md), filters to rows matching
    (num_lstm_cells, concurrent_reg) -- there can be more than one sweep_id per
    condition if it was resubmitted -- and returns one row per run with its
    FINAL (highest-epoch) eval/test accuracy, sorted best-eval_acc-first.
    """
    cols = [
        "epoch",
        "run_id",
        "eval_acc",
        "test_acc",
        "model.num_lstm_cells",
        "model.seed",
        "dataset.concurrent_reg",
        "sweep_id",
    ]

    rows = []
    for f in sorted(Path(downloaded_runs_dir).glob("*_epochs.csv")):
        df = pd.read_csv(f, usecols=lambda c: c in cols)
        if df.empty or "model.num_lstm_cells" not in df.columns:
            continue
        match = df[
            (df["model.num_lstm_cells"] == num_lstm_cells)
            & (df["dataset.concurrent_reg"] == concurrent_reg)
        ]
        if match.empty:
            continue
        for run_id, group in match.groupby("run_id"):
            last = group.sort_values("epoch").iloc[-1]
            rows.append({
                "sweep_id": last["sweep_id"],
                "run_id": run_id,
                "seed": last.get("model.seed"),
                "final_epoch": last["epoch"],
                "eval_acc": last["eval_acc"],
                "test_acc": last["test_acc"],
            })

    return (
        pd
        .DataFrame(rows)
        .sort_values("eval_acc", ascending=False)
        .reset_index(drop=True)
    )


def make_classifier(classifier_cls, classifier_kwargs=None):
    """
    Instantiates `classifier_cls(**classifier_kwargs)`, generically supporting
    both actual classifiers (e.g. `RidgeClassifier`, `LogisticRegression`) and
    plain regressors (e.g. `LinearRegression`, `Ridge`) as `classifier_cls`.

    A plain regressor has no notion of discrete classes -- `.fit` expects a
    numeric target, not string/arbitrary labels, and `.predict` returns a
    continuous value rather than a class label -- so it can't directly stand in
    for a classifier. We detect that case via `sklearn.base.is_classifier` and
    wrap it in `OneVsRestClassifier`, which fits one regressor per class against
    a one-hot target and predicts via argmax over the per-class continuous
    outputs. (This is, in fact, exactly how `RidgeClassifier` itself is
    implemented internally -- one-vs-rest ridge regression onto a one-hot
    target -- just generalized here to any regressor.) This lets
    `decode_memory_identity`/`decode_ordinal_identity` accept an arbitrary
    sklearn regressor or classifier interchangeably for `classifier_cls`.
    """
    classifier_kwargs = classifier_kwargs or {}
    estimator = classifier_cls(**classifier_kwargs)
    if is_classifier(estimator):
        return estimator
    return OneVsRestClassifier(estimator)


def random_partition(n_trials, train_frac=0.3, seed=None):
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_trials)
    partition_ix = int(n_trials * train_frac)
    return permuted[:partition_ix], permuted[partition_ix:]


def decode_memory_identity(
    annotated_df,
    register,
    representation="hidden_states",
    classifier_cls=LinearRegression,
    classifier_kwargs=None,
    partition_fn=random_partition,
    token_type="instr",
):
    """
    Model/dataset-agnostic: trains `classifier_cls` to decode, from
    `annotated_df[representation]` (any per-token representation column produced
    by `get_annotated_representations`, e.g. "hidden_states", "cell_states",
    "percell_hidden_states"), the identity of the item currently held by
    `register` -- i.e. `oracle_memory_state[register]` -- as of each trial.
    Only assumes `annotated_df` has the columns `get_annotated_representations`
    always produces (`trial_ix`, `token_type`, `oracle_memory_state`, and the
    requested `representation`), so it works for any model/checkpoint/dataset
    that function was run on.

    The target changes trial-by-trial (whatever `register` currently holds), so
    exactly one example is drawn per trial, at the row where
    `token_type == token_type` (default "ans": the representation after the
    model has fully processed that trial, i.e. seen instr+role+item). Trials
    before `register` has ever been stored to (nothing yet in
    `oracle_memory_state`) are excluded, since there is no target to decode.

    `partition_fn(n_trials) -> (train_idx, test_idx)` (positional indices into
    the per-trial examples, in trial order) controls the train/test split; see
    `default_partition`.

    `classifier_cls` may be an actual classifier (e.g. `RidgeClassifier`) or a
    plain regressor (e.g. `LinearRegression`); see `make_classifier`.

    Returns `(classifier, train_acc, test_acc)`.
    """
    trial_rows = (
        annotated_df[annotated_df["token_type"] == token_type]
        .sort_values("trial_ix")
        .reset_index(drop=True)
    )
    memory_state = trial_rows["oracle_memory_state_start"].map(json.loads)
    targets = memory_state.map(lambda state: state.get(register))
    has_target = targets.notna()

    trial_rows = trial_rows[has_target].reset_index(drop=True)
    targets = targets[has_target].reset_index(drop=True)

    X = np.stack(trial_rows[representation].to_numpy())
    y = targets.to_numpy()

    train_idx, test_idx = partition_fn(len(trial_rows))

    classifier = make_classifier(classifier_cls, classifier_kwargs)
    classifier.fit(X[train_idx], y[train_idx])

    train_acc = accuracy_score(y[train_idx], classifier.predict(X[train_idx]))
    test_acc = accuracy_score(y[test_idx], classifier.predict(X[test_idx]))

    return classifier, train_acc, test_acc


def ablate_cell_representation(annotated_df, cell, representation, invert=False):
    """
    Decoder-level ablation only -- does NOT touch the model or re-run it.
    Returns a copy of `annotated_df` with `representation`'s column zeroed out
    at the segment corresponding to `cell`, so it can be fed straight into
    `decode_memory_identity` (or any other decoder) exactly as-is.

    Assumes `representation` is the "concatenate"-merged flattening of the
    per-cell states in cell-major order (true for the default
    `"hidden_states"`/`"cell_states"`, matching `LSTMMultiCell`'s
    `_merge_outputs`): `num_cells`/`d_hidden` are inferred from
    `percell_hidden_states`'s shape, and `representation`'s own width must
    equal `num_cells * d_hidden`.

    `invert=False` (default) is a *necessity* test: zero out only `cell`'s
    segment, leaving every other cell intact -- does decoding break without
    this cell? `invert=True` flips this into a *sufficiency* test: zero out
    every cell EXCEPT `cell`, retaining only its segment -- is this cell
    alone enough to support decoding, with no help from the rest of the
    representation?
    """
    percell_shape = np.stack(annotated_df["percell_hidden_states"].to_numpy())[0].shape
    num_cells, d_hidden = percell_shape

    ablated_repr = np.stack(annotated_df[representation].to_numpy()).copy()
    assert ablated_repr.shape[1] == num_cells * d_hidden, (
        f"{representation!r} has width {ablated_repr.shape[1]}, expected "
        f"num_cells * d_hidden = {num_cells * d_hidden}"
    )

    if invert:
        retained = ablated_repr[:, cell * d_hidden : (cell + 1) * d_hidden].copy()
        ablated_repr[:, :] = 0
        ablated_repr[:, cell * d_hidden : (cell + 1) * d_hidden] = retained
    else:
        ablated_repr[:, cell * d_hidden : (cell + 1) * d_hidden] = 0

    ablated_df = annotated_df.copy()
    ablated_df[representation] = list(ablated_repr)
    return ablated_df


def compute_cell_ablation_accuracies(
    annotated_df,
    register,
    num_cells,
    representation,
    train_frac=0.25,
    seeds=range(42, 42 + 20),
    compute_sufficiency=True,
):
    """
    Repeats the pre-ablation baseline and each per-cell post-ablation fit
    across `seeds` (different random train/test partitions via
    `random_partition`), rather than a single train/test split's point
    estimate, so both can be reported as mean +/- SEM for error-barred
    plotting.

    By default computes BOTH ablation directions per cell in the same pass --
    necessity (`ablate_cell_representation(..., invert=False)`: zero out this
    cell only) and sufficiency (`invert=True`: retain only this cell) --
    reusing the same baseline fit and the same per-seed train/test partition
    for both, rather than calling this twice (once per `invert` value) and
    redundantly recomputing the (seed-matched) baseline each time.

    `compute_sufficiency=False` skips the sufficiency-side fits entirely
    (roughly halving this function's cost -- one decode fit per cell per
    seed instead of two) for callers that only want necessity; in that case
    `sufficiency_means`/`sufficiency_sems` are returned as `None`.

    Returns `(baseline_mean, baseline_sem, necessity_means, necessity_sems,
    sufficiency_means, sufficiency_sems)`, where the `*_means`/`*_sems` are
    length-`num_cells` arrays (one per cell).
    """
    baseline_accs = []
    necessity_accs_by_cell = [[] for _ in range(num_cells)]
    sufficiency_accs_by_cell = [[] for _ in range(num_cells)]

    for seed in seeds:
        partition_fn = partial(random_partition, train_frac=train_frac, seed=seed)

        _, _, baseline_acc = decode_memory_identity(
            annotated_df,
            register=register,
            classifier_cls=LinearRegression,
            classifier_kwargs={},
            representation=representation,
            partition_fn=partition_fn,
        )
        baseline_accs.append(baseline_acc)

        for cell in range(num_cells):
            necessity_df = ablate_cell_representation(
                annotated_df, cell, representation=representation, invert=False
            )
            _, _, necessity_acc = decode_memory_identity(
                necessity_df,
                register=register,
                classifier_cls=LinearRegression,
                classifier_kwargs={},
                representation=representation,
                partition_fn=partition_fn,
            )
            necessity_accs_by_cell[cell].append(necessity_acc)

            if compute_sufficiency:
                sufficiency_df = ablate_cell_representation(
                    annotated_df, cell, representation=representation, invert=True
                )
                _, _, sufficiency_acc = decode_memory_identity(
                    sufficiency_df,
                    register=register,
                    classifier_cls=LinearRegression,
                    classifier_kwargs={},
                    representation=representation,
                    partition_fn=partition_fn,
                )
                sufficiency_accs_by_cell[cell].append(sufficiency_acc)

    def _mean_sem(values):
        values = np.asarray(values)
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return values.mean(), sem

    baseline_mean, baseline_sem = _mean_sem(baseline_accs)
    necessity_means, necessity_sems = zip(
        *(_mean_sem(accs) for accs in necessity_accs_by_cell)
    )
    if compute_sufficiency:
        sufficiency_means, sufficiency_sems = zip(
            *(_mean_sem(accs) for accs in sufficiency_accs_by_cell)
        )
        sufficiency_means = np.array(sufficiency_means)
        sufficiency_sems = np.array(sufficiency_sems)
    else:
        sufficiency_means = None
        sufficiency_sems = None

    return (
        baseline_mean,
        baseline_sem,
        np.array(necessity_means),
        np.array(necessity_sems),
        sufficiency_means,
        sufficiency_sems,
    )


def compute_role_cell_correspondence(
    model,
    dataset,
    trial_indices=range(15),
    ratio_threshold=0.95,
    representation="hidden_states",
    compute_sufficiency=True,
):
    """
    For each trial sequence in `trial_indices`, runs the per-cell ablation
    analysis (`compute_cell_ablation_accuracies`) for every register
    present in that sequence, and records, per register:
    - which cells are NECESSARY: cells whose removal (`invert=False`) dropped
      decode accuracy below `ratio_threshold` of the un-ablated baseline.
    - which cells are SUFFICIENT (only when `compute_sufficiency=True`):
      cells that, kept ALONE (`invert=True`), still reach at least
      `ratio_threshold` of that same baseline.

    By default both directions come from the same
    `compute_cell_ablation_accuracies` call per register (one shared
    baseline/seed loop), so computing both costs the extra sufficiency-side
    decode fits only -- no redundant re-fitting of the baseline or
    re-running necessity itself. `compute_sufficiency=False` skips the
    sufficiency-side fits entirely (see that function), for callers that
    only want necessity and want to avoid paying for sufficiency's extra
    decode fits.

    `num_cells` is inferred per-trial from `percell_hidden_states`'s shape
    (first dim), same as `ablate_cell_representation`/
    `plot_decoder_cell_ablation` elsewhere in this notebook.

    Returns `(necessary_by_trial, sufficient_by_trial)`, each shaped
    `{trial_ix: {register: [cell_ix, ...], ..., "n_unique": n}}` -- the number
    of cells past threshold for a given role is just `len(entry[register])`.
    `sufficient_by_trial` is `{}` when `compute_sufficiency=False`.
    """
    necessary_by_trial = {}
    sufficient_by_trial = {}
    for trial_ix in tqdm([*trial_indices]):
        trial_seq = dataset[trial_ix]
        annotated_df = get_annotated_representations(model, trial_seq)
        num_cells = np.stack(annotated_df["percell_hidden_states"].to_numpy())[0].shape[
            0
        ]

        necessary_cells_by_role = {}
        sufficient_cells_by_role = {}
        for register in annotated_df["role"].unique():
            (
                baseline_mean,
                baseline_sem,
                necessity_means,
                necessity_sems,
                sufficiency_means,
                sufficiency_sems,
            ) = compute_cell_ablation_accuracies(
                annotated_df=annotated_df,
                register=register,
                num_cells=num_cells,
                representation=representation,
                compute_sufficiency=compute_sufficiency,
            )

            necessity_ratio = necessity_means / baseline_mean
            necessary_cells_by_role[register] = (
                np.argwhere(necessity_ratio < ratio_threshold).ravel().tolist()
            )

            if compute_sufficiency:
                sufficiency_ratio = sufficiency_means / baseline_mean
                sufficient_cells_by_role[register] = (
                    np.argwhere(sufficiency_ratio >= ratio_threshold).ravel().tolist()
                )

        necessary_cells_by_role["n_unique"] = len(
            {x for l in necessary_cells_by_role.values() for x in l}
        )
        necessary_by_trial[trial_ix] = necessary_cells_by_role

        if compute_sufficiency:
            sufficient_cells_by_role["n_unique"] = len(
                {x for l in sufficient_cells_by_role.values() for x in l}
            )
            sufficient_by_trial[trial_ix] = sufficient_cells_by_role

    return necessary_by_trial, sufficient_by_trial


# ---------------------------------------------------------------------------
# job selection + entrypoint
# ---------------------------------------------------------------------------


def build_job_list():
    """
    One job per (num_lstm_cells, concurrent_reg, rank) triple -- `TOP_K` models
    per (num_lstm_cells, concurrent_reg) condition, each analyzed in its own
    job for parallelism. Order is deterministic (itertools.product order), so
    a 1-indexed SLURM array task ID maps onto this list consistently across
    submissions.
    """
    return [
        {"num_lstm_cells": n_cells, "concurrent_reg": c_reg, "rank": rank}
        for n_cells, c_reg in itertools.product(NUM_LSTM_CELLS, CONCURRENT_REGS)
        for rank in range(TOP_K)
    ]


def run_job(job, output_dir, ratio_threshold, trial_indices, compute_sufficiency=True):
    n_cells = job["num_lstm_cells"]
    c_reg = job["concurrent_reg"]
    rank = job["rank"]

    runs = query_runs_by_condition(n_cells, c_reg)
    if rank >= len(runs):
        raise ValueError(
            f"requested rank={rank} but only {len(runs)} run(s) found for "
            f"num_lstm_cells={n_cells}, concurrent_reg={c_reg} -- re-run "
            "`swim --wandb.download_runs` if more runs have since finished."
        )
    run_row = runs.iloc[rank]
    sweep_id = run_row["sweep_id"]
    run_name = run_row["run_id"]  # see query_runs_by_condition: this is the
    # wandb run's *name* (e.g. "sweet-sweep-3"), which is also the checkpoint
    # directory name -- `run_id` is a historical/confusing column name
    # inherited from `get_wandb_runs` (`metrics["run_id"] = run.name`).

    ckpt_path = CHECKPOINTS_DIR / sweep_id / run_name
    print(
        f"[num_lstm_cells={n_cells}, concurrent_reg={c_reg}, rank={rank}] "
        f"loading {ckpt_path} (eval_acc={run_row['eval_acc']:.4f})"
    )

    model, dataset = load_model_and_dataset(ckpt_path, split="test", epoch=None)

    role_cell_correspondence_by_trial, sufficient_cell_correspondence_by_trial = (
        compute_role_cell_correspondence(
            model,
            dataset,
            trial_indices=trial_indices,
            ratio_threshold=ratio_threshold,
            representation="hidden_states",
            compute_sufficiency=compute_sufficiency,
        )
    )

    trial_indices = list(trial_indices)
    result = {
        "num_lstm_cells": int(n_cells),
        "concurrent_reg": int(c_reg),
        "rank": int(rank),
        "sweep_id": str(sweep_id),
        "run_id": str(run_name),
        "seed": None if pd.isna(run_row["seed"]) else int(run_row["seed"]),
        "eval_acc": float(run_row["eval_acc"]),
        "test_acc": float(run_row["test_acc"]),
        "ratio_threshold": float(ratio_threshold),
        "n_trials": len(trial_indices),
        "trial_indices": trial_indices,
        # necessity: removing this cell alone hurts the register's decodability.
        "role_cell_correspondence_by_trial": role_cell_correspondence_by_trial,
        # sufficiency: retaining ONLY this cell still supports decodability.
        # (empty dict when compute_sufficiency=False -- not computed at all.)
        "sufficient_cell_correspondence_by_trial": sufficient_cell_correspondence_by_trial,
    }

    # `n_trials` in the filename so a re-run at a different trial count (e.g.
    # the full 1000-trial eval set vs. an earlier 15-trial sanity check)
    # doesn't silently overwrite the other's results. `necessityonly` tag
    # similarly disambiguates from a (same n_trials) run that also computed
    # sufficiency.
    output_dir.mkdir(parents=True, exist_ok=True)
    necessity_tag = "_necessityonly" if not compute_sufficiency else ""
    out_path = (
        output_dir
        / f"cells{n_cells}_creg{c_reg}_rank{rank}_ntrials{len(trial_indices)}{necessity_tag}_{sweep_id}_{run_name}.json"
    )
    # json, not yaml -- much faster to load back for analysis across the
    # (now hundreds of) result files, and these are plain nested
    # dicts/lists/scalars with no need for yaml's extra features.
    with out_path.open("w") as f:
        json.dump(result, f)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help=(
            "0-indexed position into the job list (see `build_job_list`). "
            "Defaults to `$SLURM_ARRAY_TASK_ID - 1` (SLURM arrays are "
            "1-indexed by convention; see submit_role_cell_correspondence.sh)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ratio-threshold", type=float, default=RATIO_THRESHOLD)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=len(TRIAL_INDICES),
        help="number of trial sequences (dataset[0:n]) to run the analysis over",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="print the full job list (index, condition) and exit",
    )
    parser.add_argument(
        "--necessity-only",
        action="store_true",
        help=(
            "skip the sufficiency-side ablation fits entirely (roughly "
            "halves per-task compute) -- only role_cell_correspondence_by_trial "
            "is computed; sufficient_cell_correspondence_by_trial is left empty."
        ),
    )
    args = parser.parse_args()

    jobs = build_job_list()

    if args.list_jobs:
        for ix, job in enumerate(jobs):
            print(ix, job)
        return

    job_index = args.job_index
    if job_index is None:
        import os

        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task_id is None:
            raise SystemExit(
                "no --job-index given and $SLURM_ARRAY_TASK_ID is not set -- "
                "pass one explicitly, or run via "
                "submit_role_cell_correspondence.sh."
            )
        job_index = int(slurm_task_id) - 1  # SLURM arrays are 1-indexed

    if not (0 <= job_index < len(jobs)):
        raise SystemExit(
            f"job_index={job_index} out of range for {len(jobs)} total jobs "
            f"(0..{len(jobs) - 1}); see --list-jobs."
        )

    run_job(
        jobs[job_index],
        output_dir=args.output_dir,
        ratio_threshold=args.ratio_threshold,
        trial_indices=range(args.n_trials),
        compute_sufficiency=not args.necessity_only,
    )


if __name__ == "__main__":
    main()
