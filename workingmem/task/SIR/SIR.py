"""
This file houses classes and functions for the Store-Ignore-Recall (SIR) task
"""

# stdlib
import os
from dataclasses import dataclass
import typing
import logging
import random
import itertools

# packages
import tokenizers
import numpy as np
import torch
from tqdm.auto import tqdm

# local
from workingmem.task.interface import (
    GeneratedCachedDataset,
    GeneratedCachedDatasetConfig,
    SupportsGetitem,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("workingmem")


@dataclass
class SIRConfig(GeneratedCachedDatasetConfig):
    n_reg: int = 50
    """total number of registers in vocab to draw from"""
    n_items: int = 50
    """total number of items in vocab to draw from"""
    seq_len: int = 200
    """length of a trial sequence"""
    concurrent_reg: int | typing.List[int] = 4
    """number of registers to use concurrently within a trial. if this
        number is too high, we risk a simple heuristic solution such as: 
        simply check if an item has appeared in the prior history, when 
        number of total items n_items is high"""
    concurrent_items: int = 4
    """number of items to use concurrently within a trial"""
    heldout_reg: int = 0
    """[DEPRECATED] number (absolute) of registers to hold out. 
        these registers will never make an appearance in the train set"""
    heldout_items: int = 0
    """[DEPRECATED] number (absolute) of items to hold out. 
        these items will never appear in the train"""
    heldout_items_per_reg: int = 15
    """number of items that will be held-out per register during training. these register-item
       pairings will never appear in the training set (when mode='train') or validation set 
       but will appear with high probability in the challenge set (when
       mode='challenge', i.e., 'test' split)"""
    locality: typing.Union[int, None] = None
    """the locality value, when supplied, is used to sample concurrent registers locally
        (numerically close to one another). i.e., register_i can only ever occur in the same
        trial sequence as register_{i pm locality}.  this allows us to break the locality
        constraint at test time to see out-of-locality-distribution generalization.
        TODO: option to manipulate locality of train/test split. alternatively, we could
        do this evaluation using a separate dataset with the locality parameter relaxed
        (which should make the test data OOD)"""
    ignore_prob: float = 0.5
    """probability of an ignore instruction"""
    same_diff_prob: float = 0.5
    """probability of a 'same' outcome on a particular register. 
        varies independently of store/ignore instruction"""
    td_prob: float = 0.0
    """temporal dependence probability: (X_N ~ Uniform[0,1]) the probability with which 
        the corrent ANS at the current trial depends on the item that occurred at a 
        previous trial N* trials ago
        *another interpretation of N is f(N), where f(N) is ignore-trial-aware 
        """
    n_back: int | typing.List[int] | None = None
    """specify N for n-back-i-ness. must be >= 1 when provided. 
        must be provided when temporal dependence (`td_prob`) > 0. 
        does nothing when `td_prob` = 0.
        should be = `concurrent_reg` for `role_n_congruence` to be an
        effective signal
        *f(N), where f(N) is ignore-trial-aware 
        """
    role_n_congruence: typing.Union[float, None] = 0.0
    """role-N congruence probability: (Y ~ Uniform[0,1])
        determines, at each trial generation step, whether the identity of
        the role sampled at that trial will be congruent with N*, should 
        the trial be an N-back trial.
        *f(N), where f(N) is ignore-trial-aware (TODO; NotImplemented)
    """
    global_split_set_control: typing.Union[bool, None] = None
    """(stricter) control condition where each item is assigned to a single role 
        (corollary: each role has a potentially small pool of items which are the 
        only items that can co-occur with it). 
        so a given item cannot occur with any other role. 
        also, a given role will never have any items outside of its small set of items
        ever occur with it
        this is used in O'Rielly & Frank (2002) and Soni, Traylor, et al (in prep.) 
        as a control for requiring role-addressable gating (i.e., there's never going
        to be a case when the same item is potentially stored across multiple roles
        and it needs to be differentiated). """
    local_split_set_control: typing.Union[bool, None] = None
    """[DEPRECATED] (weak) control condition where, within each trial sequence, 
        the role and item pairings are section off into split-sets 
        (mimics the global split set condition on a micro scale)"""
    seed: typing.Union[int, None] = None
    """random seed for dataset generation as well as picking the random heldout combinations"""

    n_train: int = 100_000
    n_val: int = 1_000
    n_test: int = 1_000


# Create a custom tokenizer class by extending PreTrainedTokenizerFast
class SIRTokenizer:
    """
    taken from: https://discuss.huggingface.co/t/creating-a-custom-token-vocabulary-for-gpt-2/134522
    spaces act as a delimiter! spaces are just for human readability, and do not matter for the actual
    tokens.
    """

    template = "{instr} {reg} {item} {samediff} "
    query = "{instr} {reg} {item} "

    @dataclass
    class instructions:
        store = "St"
        ignore = "Ig"
        recall = "Re"

    @dataclass
    class labels:
        """
        this class could be used to create a loss mask, if the structure of each
        individual trial were variable. else, we could just use the positions to
        create the mask in the case of the SIR task.
        """

        same = "same"
        diff = "diff"

    @dataclass
    class symbols:
        register: typing.Callable = lambda i: f"reg_{int(i)}"
        item: typing.Callable = lambda i: f"item_{int(i)}"

    @classmethod
    def from_params(
        cls,
        n_reg: int,
        n_items: int,
        *args,
        **kwargs,
    ) -> tokenizers.Tokenizer:
        # token_to_id

        # dataclasses that have default instantiations represent text-form of symbols such
        # as instructions, labels, register identifiers and item identifiers

        # we define the vocabulary here with maximally identifiable token_ids:
        # 0-10 represents special symbols
        # 100-n_reg represents registers
        # 300(ish)-n_items represents items. typically, n_reg<=300 so n_items can start at 300
        #   (see highlighted line below)
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        vocab = {
            "UNK": 0,  # it is problematic if this is ever used
            "PAD": 1,
            cls.instructions.store: 3,
            cls.instructions.ignore: 4,
            cls.instructions.recall: 5,
            cls.labels.same: 7,
            cls.labels.diff: 8,
            **{cls.symbols.register(i): 100 + i for i in range(n_reg)},
            **{
                cls.symbols.item(i): max(300, 100 + n_reg) + i
                #                    ^^^^^^^^^^^^^^^^^^^^^
                for i in range(n_items)
            },
        }
        # id_to_token = {i: token for token, i in vocab.items()}

        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab, unk_token="UNK"), *args, **kwargs
        )
        # we want to tokenize on whitespace so that whitespace is ignored fully
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer.enable_padding(
            direction="left",
            length=None,
            pad_id=1,
            pad_token="PAD",
        )

        return tokenizer


class SIRDataset(GeneratedCachedDataset):
    """
    dataset instance for the SIR task that inherits methods for caching and loading from
    disk
    """

    data: SupportsGetitem
    trial_label_mask: tuple = (0, 0, 0, 1)
    # use_bos_token: bool = True # alternatively, we just suck it up and handle the offset on the model side
    config_class = SIRConfig

    def __init__(
        self,
        config: SIRConfig,
        tokenizer: tokenizers.Tokenizer = None,
    ):
        """
        Class representing an instance of a dataset for the SIR task.
        It encodes many relevant values that manipulate task demands, including,
        the vocabulary size, the number of things held out at the time of training,
        the length of a trial sequence, the number of registers to use concurrently,
        and whether concurrently-used registers have a local structure to them (locality).
        """
        # seed the random number generator
        # UPDATE: for held-out items per register, it would be nice to have a
        # random seed so that across different initializations of the dataset we
        # use the same held-out items per register (i.e., we want the test set to be internally consistent and
        # actually challending w.r.t. the train and eval set)
        # since the generated data is shuffled at training time anyway, I am not worried about a lack of
        # randomness, so I'm changing this code to always use a random seed (default to 42 if none is provided)
        # UPDATE 2025-11-17 this needed to happen BEFORE a call to `_heldout_setup`, but this code hunk
        # was *underneath* super()(config), meaning the seed was being set AFTER the heldout setup.
        # this is unfortunate; we won't be able to read into the heldout test set results for this round of expts.
        np.random.seed(config.seed or 42)
        random.seed(config.seed or 42)

        if config.n_back is None and config.concurrent_reg is not None:
            config.n_back = config.concurrent_reg

        super().__init__(config)
        self.tokenizer = tokenizer or SIRTokenizer.from_params(
            self.config.n_reg, self.config.n_items
        )

    def _heldout_setup(self):
        """
        sets up what's needed to construct a held-out challenge set by
        sampling a set of items to hold out per register
        """

        # instead of enumerating all possible combinations, simply sample as many
        # items as needed per register at the time of setting up that register
        np.random.seed(self.config.seed or 42)
        random.seed(self.config.seed or 42)
        self.reg_heldout_items = {
            i: tuple(
                [
                    *map(
                        int,
                        np.random.choice(
                            range(self.config.n_items),
                            self.config.heldout_items_per_reg,
                            replace=False,
                        ),
                    )
                ]
            )
            for i in range(self.config.n_reg)
        }  # len = n_reg

    def __getitem__(
        self, idx: int
    ) -> typing.Dict[str, typing.Union[typing.List, torch.Tensor, tokenizers.Encoding]]:
        # since our data supports __getitem__ (for now) we can index into it
        sequence = self.data[idx]["sequence"]
        # answer_locations = torch.LongTensor(self.data[idx]["answer_locations"])
        encoding = self.tokenizer.encode(sequence)
        return {
            "tokens": sequence,  # "raw" tokens
            "token_ids": torch.LongTensor(encoding.ids),
            "attention_mask": torch.LongTensor(encoding.attention_mask),
            "answer_locations": torch.LongTensor(self.answer_locations),
            # "encoding": encoding,
        }

    @property
    def vocab_size(self) -> int:
        """ """
        return (
            max(self.tokenizer.get_vocab().values()) + 1
        )  # because the token_ids are 0-indexed

        # WRONG: this returns the literal size of the vocab dict. what we instead we want the largest token_id in the vocab
        # return self.tokenizer.get_vocab_size()

    def generate_trial_sequence(
        self,
        mode=None,  # 'train' | 'challenge' | None
    ) -> dict:
        """
        Generates a sequence of `seq_len` trials for the SIR task
        ---
        mode: str = 'train' | 'challenge' | None (default)
            - 'train' mode uses the full set of registers and items minus held-out register-symbol pairs
            - 'challenge' mode uses the held-out registers and items
            - None disregards any consideration of register-item combinations and uses the full set of registers
              and items

        key things to remember:
        1. we have n_reg registers and n_items items in total, but not all of them are
           used in every trial. only concurrent_reg are used. and same_diff_prob determines
           how items are drawn: if a 'same' outcome is picked, the same item is used for that
           register in the next trial. if a 'diff' outcome is picked, a new item is uniformly
           drawn from the item pool without replacement.
        2. ignore_prob determines the likelihood of an ignore instruction. if an ignore instruction
           is given, the register is not updated.
        3. locality determines the locality of the registers. if locality is None, registers are
           drawn uniformly from the register pool. if locality is an integer, registers are drawn
           from a local pool of registers that are within locality distance of the current register.
           in practice, this means, picking a start_idx uniformly from the range [0, n_reg) with
           wraparound, and picking registers uniformly without replacement from this range

        algorithm
        +-------+
        +--------- (this happens once at the level of the trial seq) -----+
        | 1. pick a start_idx uniformly at random from the range [0, n_reg)
        |     (start_idx exists to serve a case where locality is not None: when registers
        |     within a trial sequence are likely to occur with nearby registers---and never
        |     occur with registers outside of locality bounds)
        |     in practice, locality has not been used so far---this means that by default
        |     the entire range of n_reg roles is utilized to sample concurrent_reg from
        |
        | 2. choose concurrent_reg registers from the range
        |       [start_idx, start_idx + (locality or n_reg)) % n_reg
        |    and likewise choose concurrent_items items from the range [0, n_items)
        |
        +-----------------------------------------------------------------+
        |
        |    +-------- (repeat for seq_len steps) ----------------------------+
        |    | 3. pick one register to operate on from the chosen registers
        |    |   [3'] with probability `td_prob`, this role will be the same
        |    |      as the one exactly `n_back` steps ago
        |    |      [this has no bearing on what symbol is picked---with 50%
        |    |       prob we still have different symbols]
        |    |     if not all roles have been picked yet, pick from among those
        |    |      that haven't yet made an appearance---this avoids the case
        |    |      where we have role_n_congruence and n_back>0.
        |    |
        |    | 4. pick an instruction using ignore_prob
        |    |
        |    | 5. pick an item using same_diff_prob, unless there was no
        |    |     previous item (note that 4 & 5 are independent)
        |    |     [oops, we forgot about the heldout items]
        |    |     [update: supporting held-out items]
        |    |
        |    | 6. update the register with the item if the instruction is
        |    |     'store'
        |    +----------------------------------------------------------------+
        |
        +

        """
        store = SIRTokenizer.instructions.store
        ignore = SIRTokenizer.instructions.ignore
        # recall = self.tokenizer.instructions.recall
        same = SIRTokenizer.labels.same
        diff = SIRTokenizer.labels.diff
        reg = SIRTokenizer.symbols.register
        item = SIRTokenizer.symbols.item

        # -----------------------------------------------------
        # step 1
        # -----------------------------------------------------
        # pick a start_idx uniformly from the range [0, n_reg)
        start_idx = np.random.randint(0, self.config.n_reg)

        # -----------------------------------------------------
        # step 2
        # -----------------------------------------------------
        # pick registers from the range [start_idx, start_idx + (locality or n_reg)) % n_reg
        # NB: we haven't been using locality at all since the start of this project.
        if self.config.locality is not None:
            assert self.config.locality >= self.config.concurrent_reg, (
                f"locality must be at least the number of concurrent registers to use. you supplied: {self.config.locality} < {self.config.concurrent_reg}"
            )
        reg_range = np.arange(
            start_idx,
            start_idx
            + (
                self.config.locality
                # WLG, heldout registers are numerically at the end of the reg pool
                or (self.config.n_reg - self.config.heldout_reg)
            ),
            dtype=int,
            # this way, we wraparound at the end of the available (non-held-out) register pool
        ) % (self.config.n_reg - self.config.heldout_reg)

        # sample w/o replacement
        # regs_chosen now contains the indexes of the registers to use.
        # when n_reg == concurrent_reg, this just makes
        # regs_chosen == reg_range
        regs_chosen: np.ndarray[int] = np.random.choice(
            reg_range, self.config.concurrent_reg, replace=False
        ).astype(int)

        register_item_pool = {}
        # typically, we'll be using split-set control when n_reg = 2 and
        # concurrent_reg = 2. then, we'll do a very simple serial mapping of
        # item ranges to each register.
        # NOTE: UPDATE: we want to now start using split-set control with different
        # numbers of concurrent registers, so we have expanded this hunk to be
        # more general
        if self.config.global_split_set_control:
            assert mode is None, (
                "held-out train/challenge modes not supported for split-set control"
                " (if you think about it, split-set control is already a held-out mode "
                "that exposes only certain register-item combinations during training)"
            )
            item_range = np.arange(self.config.n_items - self.config.heldout_items)
            # split the item_range up roughly equally into `concurrent_regs` parts
            # and assign each part to a register
            items_per_reg = (item_range[-1] + 1) // self.config.concurrent_reg
            # it isn't the most efficient to repeat this splitting up process
            # per trial, but as long as it's deterministic and not too costly
            # it should be OK---there is no randomness involved, so it produces the same
            # item subranges each time called
            # `how_many` is the number of items used with each register per trial sequence.
            # e.g., for 64 registers and 128 items, `how_many` will be 2, and 2 items
            # will be sampled from a larger pool of 4 items (out of 256) that always occur
            # with this register
            how_many = self.config.concurrent_items // self.config.concurrent_reg
            # within each trial sequence, we must have the same number of concurrent items
            assert (
                how_many * self.config.concurrent_reg == self.config.concurrent_items
            ), (
                f"something about the number of items per register is "
                f"wrong and it doesn't add up to the {self.config.concurrent_items=}"
            )

            # for every register, we assign `items_per_reg` items from the item pool
            # to the register, so that we can sample from it later
            for i in range(self.config.concurrent_reg):
                this_reg_item_range = item_range[
                    i * items_per_reg : (i + 1) * items_per_reg
                ]
                # this statement is where we sample `how_many` items from the pool of
                # `items_per_reg` items (`this_reg_item_range`) for this register
                register_item_pool[regs_chosen[i]] = np.random.choice(
                    this_reg_item_range, how_many, replace=False
                ).astype(int)

            # make sure each register has at least two items associated with it for this trial
            # sequence; otherwise the task doesn't make sense (and we wouldn't be able to generate
            # 'diff' labeled trials)
            assert all(len(v) >= 2 for v in register_item_pool.values()), (
                f"register_item_pool doesn't have at least two items "
                f"associated with each register: {register_item_pool}"
            )

        forbidden_items: typing.Set[int] = set()  # items forbidden across all regs
        if mode is not None:
            for r in regs_chosen:
                forbidden_items.update(self.reg_heldout_items[r])

        # before exclusions, this is the range we are working with
        # WLG, heldout items are numerically at the end of the item pool, and
        # so far we aren't doing anything like locality or wraparound with them
        # so this hunk is simpler
        allowable_item_range = np.arange(
            self.config.n_items - self.config.heldout_items, dtype=int
        )
        # of these, we exclude the forbidden items if we are in train mode
        if mode == "train":
            item_range = np.array(
                [i for i in allowable_item_range if i not in forbidden_items]
            )
        elif mode == "challenge":
            # in this special case, we want to sample items in such a way that there is a high
            # prevalence of using held-out register-item combinations (but not every register-item
            # combination can be held-out, because we will likely have a
            # non-overlapping held-out item subrange for each register)
            # for now, let's just sample from the set of forbidden items
            item_range = np.array(
                [i for i in allowable_item_range if i in forbidden_items]
            )
            if len(item_range) < self.config.concurrent_items:
                raise ValueError(
                    f"not enough items to sample from in challenge mode: {item_range=}; {self.config.concurrent_items=}"
                )
        else:  # no challenge set and no split set requested
            item_range = allowable_item_range

        # in the absence of global_split_set_control, we just sample `concurrent_items`
        # uniformly from the item pool
        items_chosen: np.ndarray[int] = np.random.choice(
            item_range,  # `ArrayLike`
            self.config.concurrent_items,
            replace=False,
        ).astype(int)

        # here is where we start maintaining the state of the registers (as in, the item they currently hold)
        reg_state: typing.Dict[int, int] = {i: -1 for i in regs_chosen}

        this_trial_seq: typing.List[str] = []
        this_reg_seq: typing.List[str] = []
        this_item_seq: typing.List[str] = []
        # ^ this stores just the sequence of roles (erstwhile "registers") [items]
        # so we don't have to go through the trial symbols and backwards-count.
        # this sequence should be automatically adjusted to `f(N)`, either including
        # or leaving out roles corresponding to ignore trials, depending on policy.
        # * current default is all roles are included, so only the absolute position
        # (i.e., absolute N) matters

        def _pick_maybe_congruent_reg(i: int) -> int:
            """reusable code hunk to probabilistically pick an N-congruent
            role or otherwise uniformly random role, given the
            current trial index `i`. returns the integer corresponding
            to the role; not the role token (e.g. `reg_23`)"""
            if (len(this_item_seq) < (self.config.n_back or 0)) or (
                np.random.rand() > self.config.role_n_congruence
            ):
                # uniformly at random pick a register to operate on from
                # among the chosen registers
                # EXCEPT: exclude all registers that have occurred thus far
                # so that we don't repeat registers before hitting all N,
                # to facilitate N-back-ness for Role-N congruence
                if len(this_item_seq) < (self.config.n_back or 0):
                    _regs_chosen = set(regs_chosen).difference(this_reg_seq)
                    return np.random.choice([*_regs_chosen], p=None).astype(int)
                return np.random.choice(regs_chosen, p=None).astype(int)
            else:
                # use the same role that occurred f(N)* trials ago
                # *f(N), if enabled, excludes ignore trials. whether or not
                # this is the case, the registers from f(N) ago will always be
                # tracked in the same data structure, `this_reg_seq`
                this_reg_idx = this_reg_seq[-self.config.n_back]
                return this_reg_idx
                # return int(this_reg_token.split("_")[1])

        # repeat `seq_len` times (each iteration of the loop generates a trial):
        for i in range(self.config.seq_len):
            # -----------------------------------------------------
            # step 3
            # -----------------------------------------------------

            # fork in the road!
            # sub-tree [3a]
            # ---------
            # with probability X_N* this trial is either an N*-back trial or NOT.
            # the only thing this affects is how the same/diff label is decided.
            # we will still sample item to respect `same_diff_prob`, but this sampling
            # will be relative to N* trials ago rather than first having sampled a register

            # OLD COMMENT:
            # | using the `td_prob` parameter, either pick the same role as `n_back` steps ago
            # | or, follow the familiar register-picking procedure (pick uniformly)
            # | constraint: in order for `n_back` to be meaningful when
            # | temporal dependence is 1, we cannot have `n_back` < `concurrent_reg`
            # | until we've accumulated at least `n_back` steps, we cannot do
            # | temporal dependence

            n_back = None
            # is this an N-back trial?
            if (len(this_item_seq) < (self.config.n_back or 0)) or (
                np.random.rand() >= self.config.td_prob
            ):  # NOT an N-back trial
                # 'correct' label (same/diff) is based on role identity
                n_back = False
                this_reg_idx = _pick_maybe_congruent_reg(i)

            # N-back trial: 'correct' label (same/diff) is based on N trials ago
            else:
                assert self.config.n_back is not None and self.config.n_back >= 1, (
                    "`config.n_back` must be specified and be >= 1 when `td_prob` != 0"
                )
                n_back = True
                this_reg_idx = _pick_maybe_congruent_reg(i)

            # -----------------------------------------------------
            # step 4
            # -----------------------------------------------------
            # pick an instruction using ignore_prob
            # NOTE: AMENDMENT: for short trial sequences, if 'ignore' is picked too often, we end up
            # with a situation where labels are highly imbalanced---'diff' appears way more often than
            # 'same' despite the `same_diff_prob` because of picking 'same' being conditioned on whether
            # anything is already stored in the register.
            if reg_state[this_reg_idx] == -1:
                this_instr = (
                    store  # force storing something in the register in the beginning
                )
            else:
                this_instr = (
                    ignore if np.random.rand() < self.config.ignore_prob else store
                )

            # -----------------------------------------------------
            # step 5
            # -----------------------------------------------------
            # pick an item using same_diff_prob, unless there was no previous item, in which case,
            # we must pick a new item and make the instruction be 'diff' by default

            if (
                # ref back; no item stored in this reg yet
                (not n_back and reg_state[this_reg_idx] == -1)
                or
                # n-back; fewer than n non-ignore trials so far; can't compare with n-back
                (n_back and (len(this_item_seq) < (self.config.n_back or 0)))
                # we picked 'diff'
                or (np.random.rand() > self.config.same_diff_prob)
            ):
                # this chunk: diff!

                # sample item uniformly at random and make it diff from N* trials back
                if n_back:
                    # split set for N-back:
                    # modulus of N, so that the item pool for the N-back trial
                    # is different from the item pool for the non-N-back trial.
                    # but for now, we are just sampling from the same pool of
                    # items regardless of whether it's an N-back trial or not.

                    # since the split-set-control condition is set up to
                    # time-lock items to registers, we have to frame N as though
                    # it were a register---so we can use the register-item
                    # mapping to determine the item pool for the N-back trial.
                    MODULO_INDEX = (
                        len(this_item_seq) % self.config.n_back
                    )  # 0, 1, ... N-1 (ignore-aware since we're measuring trial idx based on this_item_seq!)

                    this_trial_item_pool = (
                        register_item_pool[regs_chosen[MODULO_INDEX]]
                        if self.config.global_split_set_control
                        else items_chosen
                    )
                    this_item = np.random.choice(this_trial_item_pool, p=None).astype(
                        int
                    )

                    # the below while-loop samples items until drawing one that's different from N ago
                    while (len(this_item_seq) >= self.config.n_back) and (
                        this_item == this_item_seq[-self.config.n_back]
                    ):
                        this_item = np.random.choice(
                            this_trial_item_pool, p=None
                        ).astype(int)

                # sample item with respect to role and make it diff from what's
                # stored with this role
                else:
                    # depending on whether we're using global_split_set_control, we either
                    # sample a new item from the register_item_pool mapping for this register
                    # or more broadly from items_chosen
                    this_trial_item_pool = (
                        register_item_pool[this_reg_idx]
                        if self.config.global_split_set_control
                        else items_chosen
                    )
                    # NOTE this line by itself doesn't guarantee that the item is new
                    this_item = np.random.choice(this_trial_item_pool, p=None).astype(
                        int
                    )
                    # so we need this follow-up loop to keep drawing until it's new
                    while this_item == reg_state[this_reg_idx]:
                        this_item = np.random.choice(
                            this_trial_item_pool, p=None
                        ).astype(int)
                this_label = diff
            # enforce "same" condition
            # this is easier since we don't have to do anything special for
            # split set: the item only depends on the past
            else:
                if n_back:  # item same as that from N trials ago
                    this_item = this_item_seq[-self.config.n_back]
                else:  # item same as that stored in the same role
                    this_item = reg_state[this_reg_idx]

                this_label = same

            # right here is where we assemble the current trial
            this_trial = [
                this_instr,
                reg(this_reg_idx),
                item(this_item),
                this_label,
            ]
            this_trial_seq.extend(this_trial)

            # -----------------------------------------------------
            # step 6
            # -----------------------------------------------------
            # update the register with the new item if the instruction is not ignore
            if this_instr != ignore:
                # doesn't matter if it's the same or a new item; we update
                reg_state[this_reg_idx] = this_item

                # to be ignore-aware in our treatment of (i) N-back and (ii) Role-N
                # correlations, we will only store the item and also the role
                # identity when the instruction is 'store'. this way, role identity
                # corresponds to N adjusted for skipping ignore trials.
                # since the same/diff label is determined based on the contents of the
                # lists referenced below, doing the N-back task will mean looking N-back
                # in these lists, which the models will somehow have to learn to maintain
                this_reg_seq.append(this_reg_idx)
                this_item_seq.append(this_item)

        return {
            "sequence": " ".join(this_trial_seq),
            "regs_used": tuple(regs_chosen.tolist()),
            "items_used": tuple(items_chosen.tolist()),
            "locality": self.config.locality,
            # lots of gymnastics here to make sure the register_item_pool is serializable
            "split_set_control": tuple(
                (
                    {int(k): tuple(v.tolist()) for k, v in register_item_pool.items()}
                ).items()
            ),
            "heldout_items_used": tuple(
                (
                    int(r),
                    tuple(
                        [
                            *set(self.reg_heldout_items[r]).intersection(
                                set(items_chosen.tolist())
                            )
                        ]
                    ),
                )
                for r in regs_chosen
            ),
            "mode": mode or "none",
        }

    @property
    def answer_locations(self) -> typing.List[int]:
        """
        returns a mask for the locations of the "answers" in the sequence where loss should be computed (the only deterministic/structured)
        part of the SIR task.
        """
        return list(SIRDataset.trial_label_mask * self.config.seq_len)

    @classmethod
    def _serialize_trial(cls, trial: dict) -> frozenset:
        return frozenset(sorted(trial.items()))

    def _generate_split(
        self, n_examples, mode, conflicts=()
    ) -> typing.Collection[typing.Sequence[str]]:
        examples = set(map(SIRDataset._serialize_trial, conflicts))
        examples_list = []

        # we are constructing a fully in-distribution dataset
        # where train, val, and test are iid
        total = n_examples
        for _ in tqdm(
            range(total),
            desc=f"generating {total} SIR trials in {mode=}",
            total=total,
        ):
            # check for duplicate trials
            while True:
                try:
                    trial = self.generate_trial_sequence(mode=mode)
                except ValueError as e:
                    logger.warning(f"failed to generate single trial due to: {e}")
                    continue
                fstrial = SIRDataset._serialize_trial(trial)
                if fstrial in examples:
                    continue  # discard trial
                examples.add(fstrial)
                examples_list.append(trial)
                break  # break while-loop

        return examples_list

    def generate(self) -> typing.Dict[str, typing.Collection[typing.Sequence[str]]]:
        """
        makes repeated calls to `_generate_trial_sequence` to generate a total of
        n_train + n_val + n_test examples.
        uses `SIRConfig.heldout_items_per_reg` to determine whether any register-item
        pairings are held-out during training/validation and only appear during testing,
        and passes appropriate kwargs to `generate_trial_sequence` to enforce this.
        """
        logger.info("generating data for SIR task")

        # seed the random number generator
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        if self.config.heldout_items_per_reg == 0:
            # we are constructing a fully in-distribution dataset
            # where train, val, and test are iid
            total = self.config.n_train + self.config.n_val + self.config.n_test
            all_examples_list = self._generate_split(total, mode=None)

            n_train, n_val = (self.config.n_train, self.config.n_val)
            return {
                "train": all_examples_list[:n_train],
                "val": all_examples_list[n_train : n_train + n_val],
                "test": all_examples_list[n_train + n_val :],
            }

        else:  # self.config.heldout_items_per_reg > 0
            # first, we create a train set and an iid validation set using 'train' mode
            train_val_total = self.config.n_train + self.config.n_val
            train_val_examples_list = self._generate_split(
                train_val_total, mode="train"
            )

            # next, we create a challenge set using 'challenge' mode
            # and pass in the train_val_examples_list as conflicts to avoid overlap
            # with train/val data
            challenge_total = self.config.n_test
            challenge_examples_list = self._generate_split(
                challenge_total, mode="challenge", conflicts=train_val_examples_list
            )

            return {
                "train": train_val_examples_list[: self.config.n_train],
                "val": train_val_examples_list[self.config.n_train :],
                "test": challenge_examples_list,
            }
