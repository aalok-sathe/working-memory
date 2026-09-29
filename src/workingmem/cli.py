import dataclasses
import yaml
from pathlib import Path
from datetime import datetime
import tyro


# 3rd party packages
import wandb

# local
from workingmem import _logger, main, MainConfig


def entrypoint():
    """
    CLI entrypoint for `swim`. This function is called when `swim` or `python -m workingmem` is invoked from the terminal. Collects options and invokes `main`.

    ### CLI-driven usage
    ```bash
    swim [OR: python -m workingmem]
    --------------------
      -h, --help:
        see a list of all options

      --[model|dataset|trainer|wandb].FOO BAR:
        pass the value 'BAR' to [model|dataset|trainer|wandb] parameter "FOO". e.g., "--model.model_class lstm" or "--dataset.n_back 5"

    ```

    ### Options: applicable to config and CLI

    - [Model options](https://aalok-sathe.github.io/working-memory/workingmem/model.html#ModelConfig) are passed in as `--model.XYZ` in the CLI invocation and as `model.XYZ: [...]` in the config-driven invocation of `swim`, for a parameter titled `XYZ`. This allows you to set model parameters and hyperparamas such as the model class (e.g., `lstm`), number of layers, dimensionality, etc.
    - [Dataset options](https://aalok-sathe.github.io/working-memory/workingmem/task/SIR/SIR.html#SIRConfig) are passed in as `--dataset.XYZ` for an option titled `XYZ` to set dataset parameters such as `concurrent_roles`, `n_back`, `n_train`, `n_trials`, etc.
    - [Trainer options](https://aalok-sathe.github.io/working-memory/workingmem/model.html#TrainingConfig) are passed in as `--trainer.XYZ` for an option titled `XYZ` to set trainer hyperparameters such as `learning_rate`.
    - [W&B integration options](https://aalok-sathe.github.io/working-memory/workingmem.html#WandbConfig) are passed in as `--wandb.XYZ` for an option titled `XYZ` to use W&B integration directives such as `create_sweep` or `run_sweep [SWEEP_ID]` .
    - [Additional options](https://aalok-sathe.github.io/working-memory/workingmem.html#MainConfig) can be supplied for a limited number of settings such as setting the compute cluster slurm account/partition names, specifying whether to filter model loading from pre-trained by accuracy threshold, etc.


    ### Example output of `swim -h` helptext as of June 2026 (run in your own terminal for the latest, or refer to the configs linked in the section above):

    ```bash
        ╭─ options ──────────────────────────────────────────────────────────────────────────╮
        │ -h, --help                                                                         │
        │       show this help message and exit                                              │
        │ --array-task-id {None}|INT                                                         │
        │       (default: None)                                                              │
        │ --filter-by-accuracy {None,True,False}                                             │
        │       (default: None)                                                              │
        │ --filter-by-accuracy-threshold FLOAT                                               │
        │       (default: 0.7)                                                               │
        │ --gpu-partition-names [STR [STR ...]]                                              │
        │       names of partitions to utilize in submitting jobs to. we will uniformly      │
        │       alternate between them for each condition we construct (default: 3090-gcondo │
        │       'gpu-he --account=carney-frankmj-condo2')                                    │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
        ╭─ model options ────────────────────────────────────────────────────────────────────╮
        │ --model.model-class STR                                                            │
        │       (default: lstm)                                                              │
        │ --model.from-pretrained {None}|STR                                                 │
        │       `from_pretrained` is a path to a directory containing the model checkpoints  │
        │       and config.yaml.                                                             │
        │           typically:                                                               │
        │           |                                                                        │
        │           +-- config.yaml                                                          │
        │           +-- history.yaml                                                         │
        │           +-- checkpoints/{epoch}.pth, ...                                         │
        │           +-- best_model.pth                                                       │
        │       if supplied, any options in the passed `ModelConfig` instance are ignored.   │
        │       model is initialized                                                         │
        │       using the config in the config.yaml file, and the state_dict is loaded from  │
        │       the *.pth file. (default: None)                                              │
        │ --model.n-layers INT                                                               │
        │       (default: 2)                                                                 │
        │ --model.d-model INT                                                                │
        │       dimensionality of the residual stream / embeddings, appropriately defined    │
        │       for each model (default: 256)                                                │
        │ --model.init-weights, --model.no-init-weights                                      │
        │       whether or not to initialize weights (default: True)                         │
        │ --model.seed {None}|INT|STR                                                        │
        │       (default: None)                                                              │
        │ --model.act-fn STR                                                                 │
        │       for transformer: "must be set unless using an attn-only model". for RNN-like │
        │       models: this gets passed as `nonlinearity=config.act_fn` (default: relu)     │
        │ --model.attn-only, --model.no-attn-only                                            │
        │       (default: True)                                                              │
        │ --model.n-heads INT                                                                │
        │       (default: 4)                                                                 │
        │ --model.n-ctx INT                                                                  │
        │       this should be set so that it is longer than the longest trial sequence      │
        │       length we expect to use with the model. i.e., 4 * seq_len + change. for 300, │
        │       we need at least 1201. (default: 1205)                                       │
        │ --model.d-head INT                                                                 │
        │       (default: 256)                                                               │
        │ --model.d-mlp INT                                                                  │
        │       (default: 0)                                                                 │
        │ --model.d-vocab {None}|INT                                                         │
        │       vocab dim is determined by the tokenizer (default: None)                     │
        │ --model.positional-embedding-type {None}|STR                                       │
        │       type of positional embedding to use: "rotary", "standard", None NOTE!        │
        │       passing None corresponds to NOPE (no positional embeddings) rather than some │
        │       default. use with caution! (default: rotary)                                 │
        │ --model.d-hidden INT                                                               │
        │       hidden state dimensionality (default: 256)                                   │
        │ --model.num-lstm-cells INT                                                         │
        │       number of parallel LSTM cells in LSTMMultiCell (default: 3)                  │
        │ --model.lstm-merge-strategy STR                                                    │
        │       "average", "concatenate", or "gated" (default: gated)                        │
        │ --model.num-mechanisms INT                                                         │
        │       number of independent mechanisms in RIM (default: 4)                         │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
        ╭─ dataset options ──────────────────────────────────────────────────────────────────╮
        │ --dataset.split {train,val,test}                                                   │
        │       the split of the dataset to use. if no data already exists on disk,          │
        │       data is generated for all splits (we need to make sure all examples          │
        │       are unique and non-repeating across splits). if data already exists          │
        │       (or once data has been generated), simply supplies examples from             │
        │       appropriate split. defaults to "train". (default: train)                     │
        │ --dataset.rootdir STR|PATH                                                         │
        │       where the dataset should be stored and/or read from (default: datasets)      │
        │ --dataset.seed {None}|INT                                                          │
        │       random seed for dataset generation as well as picking the random heldout     │
        │       combinations (default: None)                                                 │
        │ --dataset.generate, --dataset.no-generate                                          │
        │       whether to generate the dataset if it doesnt already exist on disk,          │
        │       or simply to initialize it to enable calling `generate_trial_sequence`       │
        │       (default: True)                                                              │
        │ --dataset.load, --dataset.no-load                                                  │
        │       should we load the created dataset? defaults to yes. (default: True)         │
        │ --dataset.create-dataset-and-exit, --dataset.no-create-dataset-and-exit            │
        │       if passed, this invocation will serve only for the purpose of creating a     │
        │       dataset and no model training (default: False)                               │
        │ --dataset.n-reg INT                                                                │
        │       total number of registers in vocab to draw from (default: 50)                │
        │ --dataset.n-items INT                                                              │
        │       total number of items in vocab to draw from (default: 50)                    │
        │ --dataset.seq-len INT                                                              │
        │       length of a trial sequence (default: 200)                                    │
        │ --dataset.concurrent-reg INT|{[INT [INT ...]]}                                     │
        │       number of registers to use concurrently within a trial. if this              │
        │       number is too high, we risk a simple heuristic solution such as:             │
        │       simply check if an item has appeared in the prior history, when              │
        │       number of total items n_items is high (default: 4)                           │
        │ --dataset.concurrent-items INT                                                     │
        │       number of items to use concurrently within a trial (default: 4)              │
        │ --dataset.heldout-reg INT                                                          │
        │       [DEPRECATED] number (absolute) of registers to hold out.                     │
        │       these registers will never make an appearance in the train set (default: 0)  │
        │ --dataset.heldout-items INT                                                        │
        │       [DEPRECATED] number (absolute) of items to hold out.                         │
        │       these items will never appear in the train (default: 0)                      │
        │ --dataset.heldout-items-per-reg INT                                                │
        │       number of items that will be held-out per register during training. these    │
        │       register-item                                                                │
        │       pairings will never appear in the training set (when mode='train') or        │
        │       validation set                                                               │
        │       but will appear with high probability in the challenge set (when             │
        │       mode='challenge', i.e., 'test' split) (default: 15)                          │
        │ --dataset.locality {None}|INT                                                      │
        │       the locality value, when supplied, is used to sample concurrent registers    │
        │       locally                                                                      │
        │       (numerically close to one another). i.e., register_i can only ever occur in  │
        │       the same                                                                     │
        │       trial sequence as register_{i pm locality}.  this allows us to break the     │
        │       locality                                                                     │
        │       constraint at test time to see out-of-locality-distribution generalization.  │
        │       TODO: option to manipulate locality of train/test split. alternatively, we   │
        │       could                                                                        │
        │       do this evaluation using a separate dataset with the locality parameter      │
        │       relaxed                                                                      │
        │       (which should make the test data OOD) (default: None)                        │
        │ --dataset.ignore-prob FLOAT                                                        │
        │       probability of an ignore instruction (default: 0.5)                          │
        │ --dataset.same-diff-prob FLOAT                                                     │
        │       probability of a 'same' outcome on a particular register.                    │
        │       varies independently of store/ignore instruction (default: 0.5)              │
        │ --dataset.td-prob FLOAT                                                            │
        │       temporal dependence probability: (X_N ~ Uniform[0,1]) the probability with   │
        │       which                                                                        │
        │       the corrent ANS at the current trial depends on the item that occurred at a  │
        │       previous trial N* trials ago                                                 │
        │       *another interpretation of N is f(N), where f(N) is ignore-trial-aware       │
        │       (default: 0.0)                                                               │
        │ --dataset.n-back {None}|INT|{[INT [INT ...]]}                                      │
        │       specify N for n-back-i-ness. must be >= 1 when provided.                     │
        │       must be provided when temporal dependence (`td_prob`) > 0.                   │
        │       does nothing when `td_prob` = 0.                                             │
        │       should be = `concurrent_reg` for `role_n_congruence` to be an                │
        │       effective signal                                                             │
        │       *f(N), where f(N) is ignore-trial-aware (default: None)                      │
        │ --dataset.role-n-congruence {None}|FLOAT                                           │
        │       role-N congruence probability: (Y ~ Uniform[0,1])                            │
        │       determines, at each trial generation step, whether the identity of           │
        │       the role sampled at that trial will be congruent with N*, should             │
        │       the trial be an N-back trial.                                                │
        │       *f(N), where f(N) is ignore-trial-aware (TODO; NotImplemented) (default:     │
        │       0.0)                                                                         │
        │ --dataset.global-split-set-control {None,True,False}                               │
        │       (stricter) control condition where each item is assigned to a single role    │
        │       (corollary: each role has a potentially small pool of items which are the    │
        │       only items that can co-occur with it).                                       │
        │       so a given item cannot occur with any other role.                            │
        │       also, a given role will never have any items outside of its small set of     │
        │       items                                                                        │
        │       ever occur with it                                                           │
        │       this is used in ORielly & Frank (2002) and Soni, Traylor, et al (in prep.)   │
        │       as a control for requiring role-addressable gating (i.e., there is never     │
        │       going                                                                        │
        │       to be a case when the same item is potentially stored across multiple roles  │
        │       and it needs to be differentiated). (default: None)                          │
        │ --dataset.local-split-set-control {None,True,False}                                │
        │       [DEPRECATED] (weak) control condition where, within each trial sequence,     │
        │       the role and item pairings are section off into split-sets                   │
        │       (mimics the global split set condition on a micro scale) (default: None)     │
        │ --dataset.dirichlet-priors {None,True,False}                                       │
        │       whether to set the priors on roles drawn within a trial sequence according   │
        │       to a dirichlet distribution with alphas=1 (concurrent_roles-dimensional).    │
        │       this approximates a uniform distribution over each role while maintaining a  │
        │       simplex constraint over probabilities of individual roles. (default: None)   │
        │ --dataset.n-train INT                                                              │
        │       (default: 100000)                                                            │
        │ --dataset.n-val INT                                                                │
        │       (default: 1000)                                                              │
        │ --dataset.n-test INT                                                               │
        │       (default: 1000)                                                              │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
        ╭─ trainer options ──────────────────────────────────────────────────────────────────╮
        │ --trainer.freeze-embeddings {None,True,False}                                      │
        │       (default: None)                                                              │
        │ --trainer.epochs INT                                                               │
        │       (default: 40)                                                                │
        │ --trainer.optimizer STR                                                            │
        │       (default: adamw)                                                             │
        │ --trainer.learning-rate FLOAT                                                      │
        │       (default: 0.0004)                                                            │
        │ --trainer.weight-decay FLOAT                                                       │
        │       (default: 0.0)                                                               │
        │ --trainer.sparsity FLOAT                                                           │
        │       (default: 0.0)                                                               │
        │ --trainer.checkpoint-dir {None}|STR                                                │
        │       this is where checkpoints are saved, if supplied. if available, a            │
        │       wandb.run.sweep_id AND a model random seed will be appended to the           │
        │       checkpoint directory name. e.g. `model_checkpoints/{sweep_id}/{run_name}/`   │
        │       (default: model_checkpoints/)                                                │
        │ --trainer.batch-size INT                                                           │
        │       this is where checkpoints are saved, if supplied. if available, a            │
        │       wandb.run.sweep_id AND a model random seed will be appended to the           │
        │       checkpoint directory name. e.g. `model_checkpoints/{sweep_id}/{run_name}/`   │
        │       (default: 128)                                                               │
        │ --trainer.seed {None}|INT                                                          │
        │       this is where checkpoints are saved, if supplied. if available, a            │
        │       wandb.run.sweep_id AND a model random seed will be appended to the           │
        │       checkpoint directory name. e.g. `model_checkpoints/{sweep_id}/{run_name}/`   │
        │       (default: None)                                                              │
        │ --trainer.logging-strategy STR                                                     │
        │       log every X epochs or X steps? (default: epoch)                              │
        │ --trainer.logging-steps INT                                                        │
        │       log every X epochs/steps (default: 1)                                        │
        │ --trainer.log-predictions {None,True,False}                                        │
        │       (default: None)                                                              │
        │ --trainer.logging-steps-per-epoch INT                                              │
        │       log X many times per epoch: the # of steps to log after is determined by the │
        │       dataset length and batch size (default: 5)                                   │
        │ --trainer.save-strategy {best,epoch}                                               │
        │       'best' saves a checkpoint each time we see a drop in validation loss, named  │
        │       'best_model.pth'                                                             │
        │       'epoch' saves a checkpoint at the end of 20 epochs named 'epoch_{epoch}.pth' │
        │       in a subdirectory called 'checkpoints/' (default: best)                      │
        │ --trainer.save-steps {None}|INT                                                    │
        │       if strategy is 'epoch', then we save every X epochs determined by            │
        │       `save_steps` (default: None)                                                 │
        │ --trainer.do-test {None,True,False}                                                │
        │       (default: True)                                                              │
        │ --trainer.mask-answer-tokens {None,True,False}                                     │
        │       (default: True)                                                              │
        │ --trainer.interleaved {None,True,False}                                            │
        │       (default: True)                                                              │
        │ --trainer.scaffolded {None,True,False}                                             │
        │       (default: False)                                                             │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
        ╭─ wandb options ────────────────────────────────────────────────────────────────────╮
        │ --wandb.create-sweep, --wandb.no-create-sweep                                      │
        │       (default: False)                                                             │
        │ --wandb.run-sweep, --wandb.no-run-sweep                                            │
        │       (default: False)                                                             │
        │ --wandb.sweep-id {None}|STR                                                        │
        │       required if do_sweep is True (default: None)                                 │
        │ --wandb.project-name STR                                                           │
        │       (default: wm-mechanisms-1)                                                   │
        │ --wandb.method STR                                                                 │
        │       use this once hparams are fixed (default: grid)                              │
        │ --wandb.program STR                                                                │
        │       the program to run with a wandb sweep agent (default: run_wm.py)             │
        │ --wandb.from-config {None}|STR                                                     │
        │       (default: None)                                                              │
        │ --wandb.prefix STR                                                                 │
        │       account prefix where your wandb sweeps are created. login to wandb.ai in a   │
        │       browser to find out! (default: aloxatel)                                     │
        │ --wandb.download-runs {None}|STR                                                   │
        │       `from_config`: only applicable with `create_sweep=True`. reads in a config   │
        │       file (YAML) if supplied that enumerates variations over individual variables │
        │       the product of each variable s possible values is used to create a product   │
        │       of that many new sweeps, also printed out as a table at the end of running   │
        │       this module with this option enabled (both `create_sweep` and                │
        │       `from_config`).                                                              │
        │       expects a simple enumaration of values (e.g., `dataset.concurrent_reg: [2,4, │
        │       8]`)                                                                         │
        │       rather than `wandb`-specific format (i.e., `dataset.concurrent_reg: {values: │
        │       [2,4,8]}`) (default: None)                                                   │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
        ╭─ wandb.metric options ─────────────────────────────────────────────────────────────╮
        │ --wandb.metric.goal STR                                                            │
        │       (default: maximize)                                                          │
        │ --wandb.metric.name STR                                                            │
        │       (default: eval_acc)                                                          │
        ╰────────────────────────────────────────────────────────────────────────────────────╯
    ```
    """
    from workingmem.utils import parse_config, get_wandb_runs

    config = tyro.cli(MainConfig, config=(tyro.conf.CascadeSubcommandArgs,))

    # case 1 is we create a new sweep
    if config.wandb.create_sweep:
        sweep_config = dataclasses.asdict(config.wandb)
        # Add default parameters for dataset, model, and trainer from CLI to sweep_config
        default_params = {
            **{
                f"dataset.{key}": {
                    "value": " ".join(map(str, value))
                    if isinstance(value, list)
                    else str(value)
                }
                for key, value in dataclasses.asdict(config.dataset).items()
                if not isinstance(value, bool)
            },
            **{
                f"model.{key}": {
                    "value": " ".join(map(str, value))
                    if isinstance(value, list)
                    else str(value)
                }
                for key, value in dataclasses.asdict(config.model).items()
                if not isinstance(value, bool)
            },
            **{
                f"trainer.{key}": {
                    "value": " ".join(map(str, value))
                    if isinstance(value, list)
                    else str(value)
                }
                for key, value in dataclasses.asdict(config.trainer).items()
                if not isinstance(value, bool)
            },
        }
        sweep_config.update({"parameters": default_params})

        ############
        # parameters to use when we want to optimize hyperparameters before fixing them for experimentation
        ############
        hparam_optimization_params = {
            # "model.n_heads": {"values": [2, 4, 6]},
            "model.n_layers": {"values": [2]},
            "model.d_model": {"values": [64, 128, 256, 512]},
            "model.d_hidden": {"values": [64, 128, 256, 512]},
            # we use a smaller range of seeds just to make sure out hparams aren't overly seed-specific.
            # TODO: this should actually be set to `None` at optimization-time so the sweep doesn't overfit
            # to a particular subset of seeds (there is unfortunately no way to fully exclude the random seed
            # from sweep parameters)
            "model.seed": {"values": [*map(str, range(162, 167))]},
            "trainer.learning_rate": {
                "min": 1e-6,
                "max": 1e-2,
                "distribution": "log_uniform_values",
            },
        }
        ############
        # parameters to use when we want to run a grid search over a fixed set of hyperparameters
        # NOTE: change these based on the outcomes of the hparam optimization sweep above!
        ############
        fixed_experimental_params = {
            "model.seed": {
                "values": [*map(str, range(42, 42 + 15))]
            },  # 15 random seeds; non-overlapping range with the seeds used for hparam sweep above
            # rnn x n_back
            # "trainer.learning_rate": {"value": 2e-4},
            # # rnn x ref_back
            # "trainer.learning_rate": {"value": 2e-4},
            # # lstm x n_back
            # "trainer.learning_rate": {"value": 3e-4},
            # # lstm x ref_back
            # "trainer.learning_rate": {"value": 1e-3},
        }
        ############

        which_params_to_use = (
            hparam_optimization_params
            if config.wandb.method == "bayes"
            else fixed_experimental_params
        )

        # additional default params to use for both a hparam sweep or regular experiments
        sweep_config["parameters"].update(
            which_params_to_use  # use either hparam optimization or fixed params
        )

        _logger.info(f"defaults: {sweep_config['parameters']}")

        if config.wandb.from_config is not None:
            # read the YAML file
            with open(config.wandb.from_config, "r") as f:
                from_config_params = yaml.load(f, Loader=yaml.FullLoader)
            with (Path(__file__).parent.parent / "scripts/template_run_sweep.sh").open(
                "r"
            ) as f:
                script_template_header = f.read()

            # for each of the variables (keys) in this config, we want to do
            # a product of all possible values each variable takes
            sweep_records = []
            sweep_commands = []

            for param_set in parse_config(from_config_params):
                this_sweep_config = sweep_config.copy()
                this_sweep_config["parameters"] = this_sweep_config["parameters"].copy()

                print("# ---- -------- new sweep ----")
                for key, val in param_set.items():
                    # overwrite the params with new values from supplied config yaml file
                    this_sweep_config["parameters"][key] = {"value": val}

                this_cumulative_param_set = this_sweep_config["parameters"]

                sweep_id = wandb.sweep(
                    this_sweep_config, project=config.wandb.project_name
                )
                python_command = f"python3 -m workingmem --wandb.run_sweep --wandb.sweep_id {config.wandb.prefix}/{config.wandb.project_name}/{sweep_id}"

                # what makes this sweep special?
                sweep_commands.append(
                    script_template_header
                    + "\n"
                    + "# "
                    + " ".join(
                        f"{k}={v}"
                        for k, v in param_set.items()
                        if k in this_sweep_config["parameters"]
                    )
                    + "\n# "
                    + (
                        sweep_url
                        := f"https://wandb.ai/{config.wandb.prefix}/{config.wandb.project_name}/sweeps/{sweep_id}"
                    )
                    + "\n"
                    + python_command
                    + "\n"
                )
                sweep_records += [
                    {
                        k: v
                        for k, v in this_cumulative_param_set.items()
                        if k in this_sweep_config["parameters"]
                    }
                    | {"username": config.wandb.prefix}
                    | {"sweep_id": sweep_id}
                    | {"project_id": config.wandb.project_name}
                    | {"sweep_url": sweep_url}
                ]

            timestamp = datetime.now().strftime("%y-%m-%d-%H-%M")
            P = Path(
                f"{config.wandb.from_config}_experiments/created_configs/{timestamp}_sweep_dict.yaml"
            )
            P.parent.mkdir(parents=True, exist_ok=True)
            with P.open("w") as f:
                yaml.dump(sweep_records, f)

            for ix, sweep_command in enumerate(sweep_commands):
                S = Path(
                    f"{config.wandb.from_config}_experiments/scripts/{timestamp}_{ix}.sh"
                )
                S.parent.mkdir(parents=True, exist_ok=True)
                with S.open("w") as f:
                    f.write(
                        sweep_command.format(
                            batch_output_prefix=str(S.parent) + "/",
                            slurm_partition_argument=config.gpu_partition_names[
                                ix % len(config.gpu_partition_names)
                            ],
                        )
                    )
                (S.parent / "batch-output").mkdir(exist_ok=True)

            S = Path(
                f"{config.wandb.from_config}_experiments/scripts/RUN_ALL_{timestamp}.sh"
            )
            with S.open("w") as f:
                f.write(
                    "\n".join([
                        "#!/bin/bash\n",
                        f"for script in {config.wandb.from_config}_experiments/scripts/{timestamp}_*.sh; do",
                        '\tif [ -f "$script" ]; then',
                        '\t\tsbatch "$script"',
                        "\telse",
                        f'\t\techo "No scripts found matching pattern: {config.wandb.from_config}_experiments/scripts/{timestamp}_*.sh"',
                        "\tfi",
                        "done",
                    ])
                )

        else:
            sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
            # dump all the parameters of this sweep to stdout
            _logger.info(f"parameters of {sweep_id}:\n{yaml.dump(sweep_config)}")
            _logger.info(f"created sweep with id: {sweep_id} !")

    # case 1.1 is we fetch the runs corresponding a YAML file provided.
    elif config.wandb.download_runs is not None:
        get_wandb_runs(
            config.wandb.download_runs, download_steps=config.wandb.download_steps
        )

    # case 2 is we run a sweep
    elif config.wandb.run_sweep:
        # if we're doing a sweep, we need to update the config with the sweep values
        _logger.info(
            f"running an agent part of sweep {config.wandb.sweep_id} with: {wandb.config}"
        )
        # this uses the wandb sweep_id to initialize a single wandb agent and runs
        # the designated script as specified in the `WandbConfig` argument that was
        # used when creating the sweep (see the first clause of this if-statement)
        wandb.agent(config.wandb.sweep_id, count=1)

    # case 3 is run using kwarg and default parameters, initiating a new wandb run
    # not tied to any particular sweep.
    else:  # run as normal in a single-run fashion using wandb only for logging
        main(config)
