# stdlib
import dataclasses
import typing
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from itertools import chain
import yaml

# installed packages
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

import wandb

# local
from workingmem.task.interface import (
    GeneratedCachedDataset,
    _T_dataset_or_collection_of_datasets,
)
from workingmem.model.interface import (
    AbstractPytorchModel,
    TrainingConfig,
    TrainingHistoryEntry,
    ModelConfig,
    # TransformerConfig,
    # RNNConfig,
    compute_masked_loss,
)


logger = logging.getLogger("workingmem")
logger.setLevel(logging.DEBUG)


class ModelWrapper(ABC):
    """
    this model wrapper treats the model as a first-class entity.
    the model(wrapper) is now responsible to train itself, and to evaluate itself on some supplied dataset.
    """

    model: AbstractPytorchModel
    # we want to document the unique identification of the dataset a model has been trained on
    history: typing.List[typing.Union[TrainingHistoryEntry, typing.Dict]] = None

    @abstractmethod
    def _init_model(self, config: ModelConfig):
        pass

    def load_state_dict(
        self, state_dict: typing.Dict[str, torch.Tensor], config: ModelConfig = None
    ):
        """
        simply make a call to the underlying model's `load_state_dict` method
        as provided by any standard pytorch model except in the case of a
        HookedTransformer, where we have to call the
        `load_and_process_state_dict` method
        """
        self.model.load_state_dict(state_dict)

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.from_pretrained is None:
            # if no pretrained path is supplied, we initialize the model from scratch
            logger.info(f"initializing model from scratch with config: {config}")
            # set the seed for initializing the model weights
            if config.seed is not None:
                logger.info(f"setting MODEL random seed to {config.seed}")
                torch.manual_seed(int(config.seed))
                np.random.seed(int(config.seed))

            # we call the abstract method _init_model which should be implemented in subclasses
            self._init_model(config)

        else:
            # if we're asked to load from a pretrained checkpoint, we load the model
            # using the stored config rather than the supplied config
            # note that any passed options about model parameters will be ignored!
            # we should make sure the user is aware of this.
            logger.warning(f"loading model from checkpoint: {config.from_pretrained}")
            logger.warning(
                f"any additional options passed to `ModelConfig` will be ignored!\n\t{config}"
            )
            self.load_checkpoint(config.from_pretrained)

        if self.history is None:
            self.history = []
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_dir: typing.Union[str, Path]):
        """
        `checkpoint_dir` points to a directory containing:
        - `config.yaml` which contains the `ModelConfig`
        - `*.pth`: a single .pth file that contains the model state_dict
        - `history.yaml` which details the training history (this is inherited and appended
            to the existing history, so a model that has been trained first on dataset X and then Y
            will say so in its history)
        """
        # 0. convert to Path
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # 1. load config
        with open(checkpoint_dir / "config.yaml", "r") as f:
            _config = ModelConfig(**yaml.load(f, Loader=yaml.FullLoader))
            self.config = _config  # NOTE: added 1/15/2026; seems that we were not updating self.config here before?
            # update the config with the checkpoint dir as the new `from_pretrained` path
            # NOTE: this is unnecessary if this method was called from __init__ since the config
            # would have been set to the checkpoint dir already---that is the preferred way.
            self.config.from_pretrained = checkpoint_dir
        logger.info(f"loaded config for pretrained model:\n\t{_config}")

        # 2. load history
        with open(checkpoint_dir / "history.yaml", "r") as f:
            self.history = yaml.load(f, Loader=yaml.FullLoader)

        # 3. load model
        # 3.1 load the state dict

        ################################################################
        # TODO: may be worth supporting state dicts other than `best_model.pth`,
        ################################################################
        # e.g. `epoch_{epoch}.pth` for taking a model trained for X epochs
        _state_dict_path = list(checkpoint_dir.glob("*.pth"))
        if len(_state_dict_path) != 1:
            raise ValueError(
                f"expected exactly one .pth file in {checkpoint_dir}, found {_state_dict_path}"
            )
        [_state_dict_path] = _state_dict_path
        # vocab_path = os.path.join(root_dir, d, "vocab.json")

        # 3.2 initialize a model instance just based on the config (this will have
        # random weights, but we are about to overwrite them)
        self._init_model(_config)

        # 3.3 load the state dict into the model: this should overwrite the weights
        _state_dict = torch.load(_state_dict_path, map_location=self.device)
        self.load_state_dict(_state_dict, _config)

        logger.info(f"finished loading model state dict from {_state_dict_path}")

    def save_checkpoint(
        self, checkpoint_dir: typing.Union[str, Path], epoch_num: int = None
    ):
        """
        saves model.state_dict(), config, and training history to checkpoint_dir.
        by default saves under 'best_model.pth' (overwriting if needed) unless an explicit
        epoch number is supplied, in which case, it is used as 'epoch_{epoch}.pth'.
        """
        # 0. convert to Path
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        # 0.1 if wandb.run.sweep_id is available, use it
        if self.history[-1].sweep_id is not None:
            checkpoint_dir /= self.history[-1].sweep_id

        # 0.2 if a run name is available, use it
        if self.history[-1].run_name is not None:
            checkpoint_dir /= self.history[-1].run_name
        else:
            # else, use a random prefix to avoid collisions
            import uuid

            # generate a random UUID
            random_string = str(uuid.uuid4())
            checkpoint_dir /= random_string[:6]

        self.history[-1].checkpoint_dir = str(checkpoint_dir)

        # 1. save model
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        checkpoint_path = (
            checkpoint_dir / "best_model.pth"
            if epoch_num is None
            else checkpoint_dir / "checkpoints" / f"epoch_{epoch_num}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)

        # 2. save config
        config_path = checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(self.config), f)

        def convert_dataclass_if_needed(obj):
            """
            convert dataclass to dict if needed
            """
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return obj

        # 3. save training history
        history_path = checkpoint_dir / "history.yaml"
        with open(history_path, "w") as f:
            yaml.dump([*map(convert_dataclass_if_needed, self.history)], f)

        logger.info(f"saved model checkpoint to {checkpoint_path}")

    def _deactivate_positional_embeddings(self) -> None:
        """placeholder hunk for use by the TransformerModel subclass"""
        raise NotImplementedError

    def set_embeddings(self, embeddings: typing.Union[np.ndarray, torch.Tensor]):
        """
        explicitly set the embeddings of the model to a supplied weight matrix W_E.
        the dimensionality of the matrix must be `vocab_size x d_model` (check `self.config`)
        """
        raise NotImplementedError

    def _evaluate_and_log(
        self,
        datasets: typing.List[GeneratedCachedDataset],
        log_prefix: str,
        state: typing.Any,
        training_config: TrainingConfig,
        predictions_table: wandb.Table = None,
        mask_answer_tokens: bool = True,
    ) -> typing.Tuple[typing.List[dict], float, float, float]:
        """
        Helper method to evaluate and log metrics for a list of datasets.

        Args:
        ---
        datasets: List[GeneratedCachedDataset]
            List of datasets to evaluate.
        log_prefix: str
            Prefix for logging metrics (e.g., "eval" or "test").
        state: TrainingState
            Current training state.
        training_config: TrainingConfig
            Configuration for training.
        predictions_table: wandb.Table (optional)
            Table for logging predictions.
        mask_answer_tokens: bool (default=True)
            Whether to mask answer tokens during evaluation.

        Returns:
        ---
        Tuple[float, float, float]
            Average loss, accuracy, and macro accuracy across datasets.
        """
        metrics = []
        for dataset in datasets:
            result = self.evaluate(
                dataset,
                train_epoch=state.epoch,
                predictions_table=predictions_table,
                mask_answer_tokens=mask_answer_tokens,
            )
            metrics.append(
                dict(
                    **result,
                    dataset=str(dataset),
                )
            )

        avg_loss = np.mean([entry["loss"] for entry in metrics])
        avg_acc = np.mean([entry["acc"] for entry in metrics])
        avg_macro_acc = np.mean([entry["macro_acc"] for entry in metrics])

        wandb.log(
            {
                **dataclasses.asdict(state),
                "step": state.step,
                f"{log_prefix}_loss": avg_loss,
                f"{log_prefix}_acc": avg_acc,
                f"{log_prefix}_macro_acc": avg_macro_acc,
                **{
                    f"{entry['dataset']}_{log_prefix}_acc": entry["acc"]
                    for entry in metrics
                },
                **{
                    f"{entry['dataset']}_{log_prefix}_loss": entry["loss"]
                    for entry in metrics
                },
            }
        )

        logger.info(
            f"{log_prefix.upper()}: {state.epoch = } {avg_loss = :.3f}, {avg_acc = :.3f}, {avg_macro_acc = :.3f}"
        )

        return metrics, avg_loss, avg_acc, avg_macro_acc

    def train(
        self,
        dataset: _T_dataset_or_collection_of_datasets,
        training_config: TrainingConfig,
        eval_dataset: _T_dataset_or_collection_of_datasets = None,
        test_dataset: _T_dataset_or_collection_of_datasets = None,
    ):
        """
        given an `eval_dataset` and `test_dataset`, periodically evaluates model and logs the results
        """

        # create an entry for history logging, which will be updated as we go
        self.history += [
            TrainingHistoryEntry(
                dataset_name=repr(
                    dataset
                ),  # repr should recursively call repr() on child datasets if a list is given
                dataset_path=(
                    str(dataset.config.basedir)
                    if isinstance(dataset, GeneratedCachedDataset)
                    else [str(d.config.basedir) for d in dataset]
                ),
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                sparsity=training_config.sparsity,
                weight_decay=training_config.weight_decay,
                freeze_embeddings=training_config.freeze_embeddings,
                sweep_id=(wandb.run.sweep_id if wandb.run else None),
                run_name=(wandb.run.name if wandb.run else None),
                run_url=(wandb.run.get_url() if wandb.run else None),
                checkpoint_dir=None,  # to be filled in later
                epoch=0,  # to be filled in later
                eval_acc=None,  # to be filled in later; will house the average acc across passed eval_dataset(s)
                eval_macro_acc=None,  # to be filled in later
                test_acc=None,  # to be filled in later; will house the average acc across passed eval_dataset(s)
                test_macro_acc=None,  # to be filled in later
                sub_metrics={},  # to be filled in later; will house either None or a list of child TrainingHistoryEntry objects per eval dataset
            )
        ]

        if isinstance(dataset, GeneratedCachedDataset):
            dataloaders = [
                DataLoader(
                    dataset,
                    batch_size=training_config.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                )
            ]
        else:
            # Create a DataLoader for each dataset with RandomSampler
            dataloaders = [
                DataLoader(
                    d,
                    sampler=RandomSampler(d, num_samples=len(d) // len(dataset)),
                    batch_size=training_config.batch_size,
                    num_workers=1,
                    pin_memory=True,
                )
                for d in dataset
            ]

        _len_train_dataset = (
            sum(len(d) for d in dataset) if isinstance(dataset, list) else len(dataset)
        )

        eval_datasets = (
            eval_dataset if isinstance(eval_dataset, list) else [eval_dataset]
        )
        test_datasets = (
            test_dataset if isinstance(test_dataset, list) else [test_dataset]
        )

        @dataclasses.dataclass
        class TrainingState:
            """
            this class is responsible for keeping track of the training state;
            it has a `step` property that is a function of the epoch, the epoch step,
            the dataset length, and batch size, and is computed on the fly and is
            therefore a function decorated with `@property`.
            when serializing this class, the `step` property will not be serialized
            automatically, so you should explicitly log it if you want to keep track
            """

            epoch: int = 0
            epoch_step: int = 0
            best_val_loss: float = np.inf
            best_val_acc: float = 0.0
            best_val_epoch: int = -1
            # cumulative AUC, to be updated during training. this is simply measured as an integration of eval_acc over epochs
            # so, the max possible value is 1.0 x num_epochs. for instance, a model that achieves 1.0 accuracy starting from
            # epoch 0 will have cumAUC = num_epochs
            cumAUC: float = 0.0

            @property
            def step(self):
                return self.epoch_step + np.ceil(
                    self.epoch * _len_train_dataset / training_config.batch_size
                )

        if training_config.log_predictions:
            predictions_table = wandb.Table(
                columns=[
                    "epoch",  # so that we can observe the evolution of the model's predictions over time
                    "example_ix",
                    "eval_example",
                    "eval_prediction",
                    "eval_labels",
                ]
            )
        else:
            predictions_table = None

        # set the model up for training
        # set up the optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        scaler = torch.amp.grad_scaler.GradScaler()

        state = TrainingState()
        for state.epoch in tqdm(range(total := training_config.epochs), total=total):
            ################################
            #### begin epoch            ####
            ################################
            # set the model to training mode at the beginning of each epoch, since there is
            # no guarantee that it will still be in training mode from the previous epoch
            # if we went into the eval subroutine
            # NOTE this might not matter, since we don't use standard language modeling
            # design decisions like dropout
            self.model.train()

            # freeze model embeddings (and unembeddings) if requested
            if training_config.freeze_embeddings:
                for param in self.model.embed.parameters():
                    param.requires_grad = False
                for param in self.model.unembed.parameters():
                    param.requires_grad = False

            # NOTE: as of yet NotImplemented: there is no such parameter.
            # if training_config.freeze_attention:
            #     for param in self.model.attn.parameters():
            #         param.requires_grad = False
            #     for param in self.model.attn_norm.parameters():
            #         param.requires_grad = False

            self.history[-1].epoch = state.epoch

            # combine the dataloaders into a single iterable right before use so
            # we can refresh the iterable each epoch
            train_dataloader = chain.from_iterable(dataloaders)
            for state.epoch_step, inputs in enumerate(train_dataloader):
                if state.best_val_acc >= 0.999:
                    logger.warning(
                        f"best validation accuracy {state.best_val_acc:.3f} reached, skipping training loop to directly evaluate the model"
                    )
                else:
                    torch.cuda.empty_cache()
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        loss = self._step(
                            inputs,
                            sparsity=training_config.sparsity,
                            mask_answer_tokens=training_config.mask_answer_tokens,
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    wandb.log(
                        wandb_logged := {
                            **dataclasses.asdict(state),
                            "step": state.step,
                            "train_loss": loss.item(),
                        }
                    )

                # evaluate the model when you reach the logging step within the epoch
                log_every_steps = (
                    _len_train_dataset
                    // training_config.batch_size
                    // training_config.logging_steps_per_epoch
                )
                if (
                    training_config.logging_steps_per_epoch
                    and state.epoch_step % log_every_steps == 0
                ):
                    ################################
                    # eval loop mid-epoch at however-many logging steps

                    eval_metrics, eval_loss, eval_acc, eval_macro_acc = (
                        self._evaluate_and_log(
                            eval_datasets,
                            log_prefix="eval",
                            state=state,
                            training_config=training_config,
                            mask_answer_tokens=training_config.mask_answer_tokens,
                        )
                    )
                    test_metrics, test_loss, test_acc, test_macro_acc = (
                        self._evaluate_and_log(
                            test_datasets,
                            log_prefix="test",
                            state=state,
                            training_config=training_config,
                            mask_answer_tokens=training_config.mask_answer_tokens,
                        )
                    )

                    # end eval loop mid-epoch at however-many logging steps
                    ################################
                    self.model.train()

            if (
                training_config.logging_steps
                and state.epoch % training_config.logging_steps == 0
            ):
                ################################
                # eval once at the end of every epoch
                eval_metrics, eval_loss, eval_acc, eval_macro_acc = (
                    self._evaluate_and_log(
                        eval_datasets,
                        log_prefix="eval",
                        state=state,
                        training_config=training_config,
                        predictions_table=predictions_table,
                        mask_answer_tokens=training_config.mask_answer_tokens,
                    )
                )
                test_metrics, test_loss, test_acc, test_macro_acc = (
                    self._evaluate_and_log(
                        test_datasets,
                        log_prefix="test",
                        state=state,
                        training_config=training_config,
                        mask_answer_tokens=training_config.mask_answer_tokens,
                    )
                )
                # update latest known eval_acc
                self.history[-1].eval_acc = float(eval_acc)
                self.history[-1].eval_macro_acc = float(eval_macro_acc)
                for entry in eval_metrics + test_metrics:
                    dataset_repr = entry["dataset"]
                    self.history[-1].sub_metrics[dataset_repr] = {**entry}

                state.cumAUC += eval_acc * 1

                logger.info(
                    f"EVAL: {state.epoch = } {eval_loss = }, {eval_acc = }, {test_loss = }, {test_acc = }"
                )

                wandb.log(
                    wandb_logged := {
                        **dataclasses.asdict(state),
                        "step": state.step,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "eval_macro_acc": eval_macro_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_macro_acc": test_macro_acc,
                        # "cumAUC": state.cumAUC,
                        # "cumAUC_normalized": state.cumAUC / state.epoch,
                    }
                )
                logger.debug(f"{wandb_logged = }")

                # check if we had an improvement in validation loss
                if eval_loss < state.best_val_loss:
                    logger.info(
                        f"found new best validation loss: {eval_loss} < {state.best_val_loss}"
                    )
                    state.best_val_loss = eval_loss
                    state.best_val_epoch = state.epoch
                    # update latest known eval_acc
                    self.history[-1].eval_acc = float(eval_acc)
                    self.history[-1].eval_macro_acc = float(eval_macro_acc)
                    self.save_checkpoint(training_config.checkpoint_dir)

                # end eval at the end of epoch
                ################################

            # if saving strategy is epoch, then make a call to save anyway
            if training_config.save_strategy == "epoch":
                if (
                    training_config.save_steps
                    and state.epoch % training_config.save_steps == 0
                ):
                    self.save_checkpoint(
                        training_config.checkpoint_dir,
                        epoch_num=state.epoch,
                    )

            ################################
            #### end epoch              ####
            ################################

        if predictions_table is not None:
            wandb.log({"predictions": predictions_table})

        if training_config.do_test and test_dataset is not None:
            test_table = wandb.Table(
                columns=[
                    "epoch",  # so that we can observe the evolution of the model's predictions over time
                    "test_step",  # this is the step within the training epoch
                    "test_example",
                    "test_prediction",
                    "test_labels",
                ]
            )
            test_result = self.test(
                test_dataset,
                test_table,
                mask_answer_tokens=training_config.mask_answer_tokens,
            )
            test_loss, test_acc, test_macro_acc = (
                test_result["loss"],
                test_result["acc"],
                test_result["macro_acc"],
            )
            logger.info(f"TEST: {test_loss = }, {test_acc = }, {test_macro_acc = }")
            wandb.log(
                {
                    "epoch": state.epoch,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_macro_acc": test_macro_acc,
                    "test_predictions": test_table,
                }
            )

    def test(
        self,
        dataset: GeneratedCachedDataset,
        test_predictions_table: wandb.Table = None,
        mask_answer_tokens: bool = True,
    ):
        """
        evaluates the model on the test set
        """
        return self.evaluate(
            dataset,
            predictions_table=test_predictions_table,
            mask_answer_tokens=mask_answer_tokens,
        )

    def evaluate(
        self,
        dataset: GeneratedCachedDataset,
        train_epoch: int = None,
        predictions_table: wandb.Table = None,
        batch_size: int = 128,
        return_predictions: bool = False,
        mask_answer_tokens=True,
    ) -> dict:
        """
        Returns the average loss and accuracy of the model on the dataset (assumed eval or test split)

        Args:
        ---
        dataset: `GeneratedCachedDataset`
            the dataset instance (and split) to evaluate the model on; typically one of val, test
        train_epoch: `int` (optional)
            the epoch number of the training run that made a call to the evaluation run
        """

        logger.info("evaluating model")
        self.model.eval()

        eval_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,  # TODO, should we parameterize this?
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        losses = []
        predictions = []
        actual_labels = []
        input_sequences = []

        with torch.no_grad():
            for eval_step, inputs in enumerate(eval_dataloader):
                torch.cuda.empty_cache()
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    loss, answer_logits, answers, labels = self._step(
                        inputs,
                        sparsity=0.0,
                        return_outputs=True,
                        mask_answer_tokens=mask_answer_tokens,
                    )
                # we have a single loss value per batch (this is a fine approximation)
                losses += [loss.item()]
                # answers and labels are of the shape (b, seq_len)
                predictions += [answers.detach().cpu().numpy()]
                actual_labels += [labels.detach().cpu().numpy()]

                # log the first batch of eval examples and predictions to `wandb`
                if train_epoch is not None and predictions_table is not None:
                    for example_ix in range(len(inputs["tokens"])):
                        predictions_table.add_data(
                            train_epoch,
                            example_ix,  # corresponds to batch
                            inputs["tokens"][example_ix],
                            dataset.tokenizer.decode(
                                answers[example_ix].detach().cpu().tolist()
                            ),
                            dataset.tokenizer.decode(
                                labels[example_ix].detach().cpu().tolist()
                            ),
                        )
                if return_predictions:
                    for example_ix in range(len(inputs["tokens"])):
                        input_sequences += [inputs["tokens"][example_ix]]

        # now `predictions` is of shape (N_batches, batch_size, seq_len)
        # we want it to be of shape (N_batches * batch_size, seq_len)
        predictions = np.concat(predictions)
        actual_labels = np.concat(actual_labels)
        # predictions.shape = (N_batches * batch_size, seq_len)
        # actual_labels.shape = (N_batches * batch_size, seq_len)

        # we want to aggregate over each example in val set rather than each individual answer location
        eval_num_correct = np.sum(
            all(predictions[i] == actual_labels[i])
            for i in range(actual_labels.shape[0])
        )
        acc = np.mean(predictions == actual_labels)

        logger.info(f"percent trials correct for dataset {dataset}: {acc:.5f}")
        logger.info(
            f"# sequences correct for dataset {dataset}: {eval_num_correct} / {len(actual_labels)}"
        )

        if return_predictions:
            return {
                "loss": np.mean(losses),
                "acc": acc,
                "macro_acc": eval_num_correct / len(actual_labels),
                "predictions": predictions,
                "actual_labels": actual_labels,
                "input_sequences": input_sequences,
            }

        return {
            "loss": float(np.mean(losses)),
            "acc": float(acc),
            "macro_acc": float(eval_num_correct / len(actual_labels)),
        }

    def _step(
        self,
        inputs: typing.Dict[str, torch.Tensor],
        sparsity: float = 0.0,
        return_outputs=False,
        mask_answer_tokens=True,
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        this method is responsible for computing the loss and optionally the labels
        batch of a batch of inputs
        """

        inputs["token_ids"] = inputs["token_ids"].to(self.device)
        inputs["answer_locations"] = inputs["answer_locations"].to(self.device)
        inputs["answer_locations"].requires_grad = False  # not backprop-able

        # a variation we can do here is to remove the actual answer tokens from the inputs
        # so this is less like a language modeling task and more like a classification task
        # (which it already is in principle due to not receiving loss on anything but the
        # answers). however, this way, it should take away the answers implicit in the input
        # text
        inputs["answers"] = inputs["token_ids"] * inputs["answer_locations"].to(
            self.device
        )  # only the relevant token_ids remain non-zeroed-out as `answers`

        if mask_answer_tokens:
            logger.debug(
                f"removing answer tokens from input: {inputs['token_ids'].gt(0).sum() = }"
            )
            inputs["token_ids"] = inputs["token_ids"] * (1 - inputs["answer_locations"])
            logger.debug(
                f"\tAFTER removing answer tokens from input: {inputs['token_ids'].gt(0).sum() = }"
            )

        # shape of logits: (b, seq_len, |V|)
        # TODO: ERROR: mismatch for model_class 'rnn' ---
        # TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
        logits = self.model(inputs["token_ids"])

        if return_outputs:
            outputs = compute_masked_loss(
                logits, inputs, sparsity=sparsity, return_outputs=return_outputs
            )
            loss, gathered_logits, gathered_answers, gathered_labels = (
                outputs["loss"],
                outputs["gathered_logits"],
                outputs["gathered_answers"],
                outputs["gathered_labels"],
            )

            logger.debug(f"{loss.shape = }, {inputs['token_ids'].shape = }")
            logger.debug(
                f"{gathered_answers.shape = }, {inputs['answer_locations'].shape = }"
            )
            logger.debug(
                f"{gathered_logits.shape = }, {gathered_answers.shape = }, {gathered_labels.shape = }"
            )

            return loss, gathered_logits, gathered_answers, gathered_labels
        else:
            loss = compute_masked_loss(
                logits, inputs, sparsity=sparsity, return_outputs=return_outputs
            )
            return loss


class RNNModelWrapper(ModelWrapper):
    """provides a wrapper for initializing an RNN"""

    class _forward_overridden_RNN(torch.nn.RNN):
        """
        overrides torch.nn.RNN to return only the output tensor, not the hidden state
        so we can plug into the existing ModelWrapper interface
        """

        def forward(self, input: torch.Tensor, hx: torch.Tensor = None) -> torch.Tensor:
            output, hidden = super().forward(input, hx)
            return output

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        """
        uses RNNConfig to initialize an RNN language model capable of using a word-level tokenizer's
        input_ids as input, converting them to learnable embeddings, and passing them through an n-layer
        RNN with a specified hidden size and model_dim (same as embed_dim), and finally projecting the outputs
        back to the vocabulary space for language modeling.
        uses boilerplate RNN code from pytorch wherever possible.
        """
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(config.d_vocab, config.d_model),
            self._forward_overridden_RNN(
                input_size=config.d_model,
                hidden_size=config.d_hidden,
                num_layers=config.n_layers,
                batch_first=True,
                nonlinearity=config.act_fn,
                bidirectional=False,
            ),
            torch.nn.Linear(config.d_hidden, config.d_vocab),
        )


class LSTMModelWrapper(RNNModelWrapper):
    class _forward_overridden_RNN(torch.nn.LSTM):
        """
        overrides torch.nn.LSTM to return only the output tensor, not the hidden state
        or cell states so it can plug into the existing ModelWrapper interface easily
        """

        def forward(self, input: torch.Tensor, hx: typing.Tuple = None) -> torch.Tensor:
            output, (hidden, cell) = super().forward(input, hx)
            # `hidden`, `cell` are ignored
            return output

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        """
        uses RNNConfig to initialize an LSTM language model capable of using a word-level tokenizer's
        input_ids as input, converting them to learnable embeddings, and passing them through an n-layer
        LSTM with a specified hidden size and model_dim (same as embed_dim), and finally projecting the outputs
        back to the vocabulary space for language modeling.
        uses boilerplate LSTM code from pytorch wherever possible.
        """
        self.model = torch.nn.Sequential(
            torch.nn.Embedding(config.d_vocab, config.d_model),
            self._forward_overridden_RNN(
                input_size=config.d_model,
                hidden_size=config.d_hidden,
                num_layers=config.n_layers,
                batch_first=True,
                bidirectional=False,
            ),
            torch.nn.Linear(config.d_hidden, config.d_vocab),
        )


class TransformerModelWrapper(ModelWrapper):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__(config)

    def _init_model(self, config: ModelConfig):
        # Only pass fields that HookedTransformerConfig actually accepts, and
        # continue to exclude fields that are not constructor arguments.
        config_dict = dataclasses.asdict(config)
        allowed_fields = HookedTransformerConfig.__dataclass_fields__.keys()
        hooked_config_kwargs = {
            k: v
            for k, v in config_dict.items()
            if k in allowed_fields
            and k not in ("from_pretrained", "positional_embedding_type")
        }
        hookedtfm_config = HookedTransformerConfig(
            # d_head=config.d_head, # NOTE: formerly, this was passed as a separate argument because it was a @property
            positional_embedding_type=(config.positional_embedding_type or "standard"),
            **hooked_config_kwargs,
        )
        self.model = HookedTransformer(hookedtfm_config)

        # only makes sense to deactivate positional embeddings at initialization
        # if applicable (only for Transformer models)
        if config.positional_embedding_type is None:
            self._deactivate_positional_embeddings()

    def load_state_dict(
        self,
        state_dict: typing.Dict[str, torch.Tensor],
        _config: ModelConfig = None,
    ):
        """
        we wrap the standard `load_and_process_state_dict` method of HookedTransformer to
        make the interface consistent with AbstractPytorchModel which expects a `load_state_dict`
        method implementation.
        """
        self.model.load_and_process_state_dict(
            state_dict,
            center_unembed=True,  # this shifts the unembedding matrix weights to be centered around 0
            center_writing_weights=True,  # this shifts the weights written to residual stream to be centered around 0
            fold_ln=False,
            # refactor_factored_attn_matrices=True,
        )

        # if checkpoint to be loaded has no positional embedding, set the positional embedding weight matrix to
        # zeroes and set grad off.
        if _config is not None and _config.positional_embedding_type is None:
            self._deactivate_positional_embeddings()

    def _deactivate_positional_embeddings(self) -> None:
        """
        Deactivates the positional embedding in the model by setting its weights to zero
        and freezing the gradient updates for the positional embedding parameters.
        This method modifies the `W_pos` attribute of the `pos_embed` module in the model:
        - sets all values in `W_pos` to 0.0.
        - disables gradient computation for `W_pos` by setting `requires_grad` to False.
        source: https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/No_Position_Experiment.ipynb#scrollTo=fVWrVHo9y0T2
        """
        self.model.pos_embed.W_pos.data[:] = 0.0
        self.model.pos_embed.W_pos.requires_grad = False
