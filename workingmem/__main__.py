#!/usr/bin/env python3
"""
run as: `python -m workingmem [-h]`
"""

import dataclasses
from datetime import datetime
from pathlib import Path
import logging
import os

import yaml
import tyro
import wandb

from workingmem import MainConfig, main
from workingmem.utils import parse_config, get_wandb_runs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(LOGLEVEL)


if __name__ == "__main__":
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

        logger.info(f"defaults: {sweep_config['parameters']}")

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
                (S.parent / "batch_output").mkdir(exist_ok=True)

            S = Path(
                f"{config.wandb.from_config}_experiments/scripts/RUN_ALL_{timestamp}.sh"
            )
            with S.open("w") as f:
                f.write(
                    "\n".join(
                        [
                            "#!/bin/bash\n",
                            f"for script in {config.wandb.from_config}_experiments/scripts/{timestamp}_*.sh; do",
                            '\tif [ -f "$script" ]; then',
                            '\t\tsbatch "$script"',
                            "\telse",
                            f'\t\techo "No scripts found matching pattern: {config.wandb.from_config}_experiments/scripts/{timestamp}_*.sh"',
                            "\tfi",
                            "done",
                        ]
                    )
                )

        else:
            sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
            # dump all the parameters of this sweep to stdout
            logger.info(f"parameters of {sweep_id}:\n{yaml.dump(sweep_config)}")
            logger.info(f"created sweep with id: {sweep_id} !")

    # case 1.1 is we fetch the runs corresponding a YAML file provided.
    elif config.wandb.download_runs is not None:
        get_wandb_runs(config.wandb.download_runs)

    # case 2 is we run a sweep
    elif config.wandb.run_sweep:
        # if we're doing a sweep, we need to update the config with the sweep values
        logger.info(
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
