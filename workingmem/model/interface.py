# stdlib
import dataclasses
import typing
from abc import ABC, abstractmethod

# from pathlib import Path
import logging
# import yaml

# installed packages
import numpy as np
import torch
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from transformer_lens import HookedTransformer, HookedTransformerConfig

# local
# from workingmem.task.interface import GeneratedCachedDataset

logger = logging.getLogger("workingmem")
logger.setLevel(logging.DEBUG)


@dataclasses.dataclass
class ModelConfig:
    """
    holds configuration parameters for the model and serves as a base class for configs of
    specific model architectures, such as transformer or RNN.
    """

    model_class: str = "transformer"  # "transformer" or "rnn" or "lstm"
    from_pretrained: typing.Union[str, None] = None
    """`from_pretrained` is a path to a directory containing the model checkpoints and config.yaml.
        typically:
        +-- config.yaml
        +-- history.yaml
        +-- checkpoints/{epoch}.pth, ...
        +-- best_model.pth 
    if supplied, any options in the existing `ModelConfig` are ignored.  model is initialized using the config in the config.yaml file, and the state_dict is loaded from the *.pth file.  
    """

    n_layers: int = 2
    d_model: int = 256  # dimensionality of the residual stream / embeddings, appropriately defined for each model
    init_weights: bool = True  # whether or not to initialize weights
    seed: typing.Union[int, str, None] = (
        None  # seeds passed as str must be convertible to int, e.g. "42" or "1234" via a simple cast: `int("42")`
    )

    # for transformer: "must be set unless using an attn-only model". for
    # RNN-like models: this gets passed as `nonlinearity=config.act_fn`
    act_fn: str = "relu"

    # @dataclasses.dataclass
    # class TransformerConfig(ModelConfig):
    ### transformer-specific parameters
    # model_class: str = "transformer"

    attn_only: bool = True
    n_heads: int = 4
    n_ctx: int = 1205  # this should be set so that it is longer than the longest trial sequence length we expect to use with the model. i.e., 4 * seq_len + change. for 300, we need at least 1201.
    d_head: int = 256
    d_mlp: int = 0
    d_vocab: typing.Union[int, None] = None  # vocab dim is determined by the tokenizer

    # type of positional embedding to use: "rotary", "standard", None
    # NOTE! passing None corresponds to NOPE (no positional embeddings) rather
    # than some default. use with caution!
    positional_embedding_type: typing.Union[str, None] = "rotary"

    # @dataclasses.dataclass
    # class RNNConfig(ModelConfig):
    d_hidden: int = 256  # hidden state dimensionality


@dataclasses.dataclass
class TrainingConfig:
    freeze_embeddings: typing.Union[bool, None] = None
    epochs: int = 40
    optimizer: str = "adamw"
    learning_rate: float = 4e-4
    weight_decay: float = 0.0
    sparsity: float = 0.0

    # this is where checkpoints are saved, if supplied.
    # if available, a wandb.run.sweep_id AND a model random seed will be appended
    # to the checkpoint directory name.
    # e.g. `model_checkpoints/{sweep_id}/{run_name}/`
    checkpoint_dir: typing.Union[str, None] = "model_checkpoints/"
    batch_size: int = 128
    seed: typing.Union[int, None] = None

    logging_strategy: str = "epoch"  # log every X epochs or X steps?
    logging_steps: int = 1  # log every X epochs/steps
    log_predictions: typing.Union[bool, None] = None

    # log X many times per epoch: the # of steps to log after is determined
    # by the dataset length and batch size
    logging_steps_per_epoch: int = 5
    # 'best' saves a checkpoint each time we see a drop in validation loss, named 'best_model.pth'
    # 'epoch' saves a checkpoint at the end of each epoch named 'epoch_{epoch}.pth' in a subdirectory called 'checkpoints/'
    save_strategy: typing.Literal["best", "epoch"] = "best"
    # if strategy is 'epoch', then we save every X epochs determined by `save_steps`
    save_steps: typing.Union[int, None] = None

    do_test: bool = True  # evaluate the model on the test set after training?

    mask_answer_tokens: bool = True  # whether we train the model using answer tokens in the input sequence or not


@dataclasses.dataclass
class TrainingHistoryEntry:
    """
    represents one entry in model training history. provides appropriate fields that
    should be populated when recording history
    """

    # basic info
    dataset_name: str
    dataset_path: str

    # training args
    epoch: int  # remember to update this (this is the epochs trained so far)
    batch_size: int
    sparsity: float
    learning_rate: float
    weight_decay: float
    sweep_id: str
    run_name: str
    run_url: str
    checkpoint_dir: str  # this is the directory where the model checkpoint is saved
    freeze_embeddings: bool

    # outcomes
    eval_acc: float
    eval_macro_acc: float
    test_acc: float
    test_macro_acc: float


class AbstractPytorchModel(ABC, torch.nn.Module):
    """"""

    @abstractmethod
    def to(self, device: torch.device):
        """move model to device"""
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: typing.Dict[str, torch.Tensor], **kwargs):
        """
        for generic pytorch models, we can simply call `model.load_state_dict(state_dict)`.
        for HookedTransformer, we make a call to `load_and_process_state_dict` instead.
        """
        ...


def compute_masked_loss(
    logits: torch.Tensor,
    inputs: typing.Dict[str, torch.Tensor],
    sparsity: float = 0.0,
    # rescale_loss: bool = True,
    rescale_loss: bool = False,
    return_outputs: bool = False,
) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]:
    """
    computes the loss for a batch of logits and inputs, limited to:
    - specific answer locations in the sequence
    - probabilistically chosen answer locations based on `sparsity`

    Parameters
    ----------
    - `logits` from a model (this can be softmaxed to get a distribution over the vocabulary elsewhere)
    - `inputs` is the dictionary of tensors supplied to the model, which includes the key `input_ids`
        and `answer_locations` which provides a 0-1 mask over which tokens in the sequence should be
        considered 'predictions' (only these are used for loss computation)
    - `sparsity` acts as an iid probability at each answer location to mask out the answer---this way, the model receives
       no loss at this location.
        Q: should we also mask the input answer immediately following the answer location?
    - `return_outputs` will cause the method to return the gathered logits, argmax-softmaxed logit answers, and true labels
        all as a dictionary with the keys `loss`, `gathered_logits`, `gathered_answers`, `gathered_labels`

    Example
    --------
    ```python
    outputs = compute_masked_loss(logits, inputs, return_outputs=True)
        loss, gathered_logits, gathered_answers, gathered_labels = (
            outputs["loss"],
            outputs["gathered_logits"],
            outputs["gathered_answers"],
            outputs["gathered_labels"],
        )
    ```
    """
    # logits have the shape (b, seq_len, |V|)
    b, seq_len, vocab_size = logits.shape

    # logger.debug(f"{logits.shape = }, {inputs['token_ids'].shape = }")
    gathered_logits = logits[
        :, inputs["answer_locations"][0].nonzero(as_tuple=True)[0] - 1, :
    ]
    gathered_labels = inputs["answers"][
        :, inputs["answer_locations"][0].nonzero(as_tuple=True)[0]
    ]

    # depending on the sparsity parameter, we want to additionally zero-out some of the answer locations
    # that are non-zero however, to achieve this, we can simply multiply the entire answer_locations
    # tensor by a randomly-generated mask, since each item in the mask is iid. for answer locations that
    # are zero already, nothing will change.
    if sparsity > 0.9:
        logger.warning(f"{sparsity = :.2f} is high")
        if sparsity >= 0.99:
            raise ValueError(
                f"{sparsity = } is too high. sparsity=1 corresponds to no feedback"
            )

    sparsity_mask = (
        torch.rand(b, gathered_labels.shape[1], device=logits.device).ge(sparsity).int()
    )
    sparsity_mask.requires_grad = False

    logger.debug(
        f"in COMPUTE_MASKED_LOSS: {gathered_logits.shape = }, {gathered_labels.shape = }, {sparsity_mask.shape = }"
    )
    # logger.debug(gathered_answers)

    # return shape: (b, seq_len)
    loss = torch.nn.functional.cross_entropy(
        # NOTE: a simple rearrange is not appropriate here! we need to permute the dimensions.
        # see here for why: https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
        # DONT: einops.rearrange(logits, "b seq_len vocab -> b vocab seq_len"),
        # DO:   torch.permute(gathered_logits, (0, 2, 1))
        torch.permute(gathered_logits, (0, 2, 1)),
        gathered_labels,
        reduction="none",
    )

    logger.debug(f"{loss.shape = } {loss.greater(0).sum() = }. applying mask")
    # apply sparsity mask to the loss to zero-out the loss at the locations dropped by sparsity computation
    # this is done by multiplying the loss by the sparsity mask, which is 1
    # at the locations we want to keep and 0 at the locations we want to drop
    old_loss = loss.mean()  # / gathered_labels.shape[0]  # average over the batch size
    loss = loss * sparsity_mask
    logger.debug(
        f"\tAFTER {loss.shape = } {loss.greater(0).sum() = }. AFTER applying mask"
    )
    loss = loss.mean()  # / gathered_labels.shape[0]  # average over the batch size
    logger.debug(
        f"old loss: {old_loss.item():.3f}, new loss after applying sparsity: {loss.item()}"
    )
    # at this point, our loss magnitude is smaller (rescaled by `sparsity` compared to the original loss)
    # if we want it to be comparable, we can rescale it back up by 1 / (1 - sparsity)
    if rescale_loss:
        loss /= 1 - sparsity
        logger.debug(f"new loss after rescaling: {loss.item()}")

    if return_outputs:
        return dict(
            loss=loss,
            gathered_logits=gathered_logits,
            gathered_answers=torch.nn.functional.softmax(
                gathered_logits, dim=-1
            ).argmax(-1),
            gathered_labels=gathered_labels,
        )
    return loss
