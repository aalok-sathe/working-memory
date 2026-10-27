"""
Eligibility-trace-based gate credit assignment for `LSTMMultiCell`
(`TrainingConfig.credit_assignment_mode="eligibility"`).

Motivation (see conversation history / paper draft): PBWM (O'Reilly & Frank 2006;
Soni & Frank 2024) and the actor-critic gating model (Todd, Niv & Cohen 2008) both
credit a *discrete* gating decision using only a *scalar* reward or
reward-prediction-error, with no local derivative connecting the decision to the
outcome -- unlike backprop, which gives every gate an exact, signed, local
gradient every time regardless of feedback frequency (`sparsity`). This module
lets `LSTMMultiCell`'s `i`/`f`/`o` gates be trained the same (scalar-reward,
no-local-derivative) way instead, without changing anything else about how the
model learns (embeddings, unembedding, the `g` candidate-content gate, and the
cell-state recurrence all keep their true backprop gradients).

Mechanism: `EligibilityGatedLSTMCell` (see `model.py`) registers a backward hook on
each gate's pre-activation tensor at each timestep; the hook returns a precomputed
substitute value from this recorder instead of letting the true chain-rule
gradient through. Hooks are registered *during* the forward pass (when the
pre-activation tensors are created), but the substitute values they should return
are only computable *after* the full sequence's forward pass and loss/correctness
computation -- so there are two phases:
  1. Forward pass: hooks are registered (reading from `self._substitutes`, still
     empty) and each gate's post-activation value is recorded into
     `self._activations`.
  2. Once per-position correctness is known, `compute_and_populate_substitutes`
     fills `self._substitutes`; only then is `.backward()` called, at which point
     the hooks fire and return the now-populated values.
"""

import typing

import torch


class EligibilityRecorder:
    """
    One instance is created per training step (per full-sequence forward pass).
    Not thread-safe / not meant to be reused across steps.
    """

    def __init__(self, num_layers: int, num_cells: int, decay: float = 0.9):
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.decay = decay
        # (layer_idx, cell_idx, gate) -> {t: detached (batch, hidden) activation}
        self._activations: typing.Dict[tuple, typing.Dict[int, torch.Tensor]] = {}
        # (layer_idx, cell_idx, t, gate) -> (batch, hidden) substitute gradient;
        # empty until `compute_and_populate_substitutes` runs, after which the
        # hooks registered during the forward pass can resolve.
        self._substitutes: typing.Dict[tuple, torch.Tensor] = {}

    def record_activation(
        self, layer_idx: int, cell_idx: int, t: int, gate: str, value: torch.Tensor
    ):
        """Called from `EligibilityGatedLSTMCell.forward` with each gate's
        post-activation value (post-sigmoid), forward-pass only -- this never
        touches the backward graph, so it carries no information about
        downstream causal/loss impact, unlike e.g. `|gradient|` would."""
        self._activations.setdefault((layer_idx, cell_idx, gate), {})[t] = value.detach()

    def hook_factory(self, layer_idx: int, cell_idx: int, t: int, gate: str):
        """Returns a `tensor.register_hook`-compatible callable that substitutes
        the true incoming gradient with the precomputed value for this
        (layer, cell, timestep, gate), once populated."""
        key = (layer_idx, cell_idx, t, gate)

        def _hook(grad):
            substitute = self._substitutes.get(key)
            if substitute is None:
                raise RuntimeError(
                    f"no eligibility substitute populated for {key} before "
                    "backward() was called -- "
                    "compute_and_populate_substitutes() must run first"
                )
            return substitute

        return _hook

    def compute_and_populate_substitutes(self, rewards: torch.Tensor, scale: float = 1.0):
        """
        Args:
            rewards: (batch, seq_len) float tensor, zero everywhere except answer
                positions, where it's +1 (correct) / -1 (incorrect) (already
                `sparsity`-masked by the caller, i.e. dropped positions are 0).
            scale: multiplier on the resulting substitute gradients (see
                `TrainingConfig.credit_assignment_scale`).
        """
        batch, seq_len = rewards.shape

        # G[:, t] = discounted future reward from t onward: G_t = r_t + decay * G_{t+1}
        # (a backward scan over time -- this is what lets a gate's eligibility at an
        # earlier timestep still get credited by a reward several steps later, i.e.
        # the mechanism's temporal-credit-assignment half.)
        G = torch.zeros(batch, seq_len, device=rewards.device, dtype=rewards.dtype)
        running = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(seq_len)):
            running = rewards[:, t] + self.decay * running
            G[:, t] = running

        for (layer_idx, cell_idx, gate), per_t in self._activations.items():
            eligibility = None
            for t in range(seq_len):
                activation_t = per_t[t]
                # e_t = decay * e_{t-1} + activation_t (forward-pass-only trace,
                # analogous to Todd's e(s,a) <- gamma*lambda*e(s,a) + 1 for the
                # taken action -- here "taken" is graded by activation strength
                # rather than a hard 0/1, since these gates are continuous)
                eligibility = (
                    activation_t
                    if eligibility is None
                    else self.decay * eligibility + activation_t
                )
                # NOTE the negative sign: a reward of +1 should *reinforce*
                # (increase) whatever was eligible -- a gradient-ASCENT step on
                # reward. PyTorch optimizers do `theta -= lr * grad` (descent), so
                # an ascent step of `+lr * eligibility * G` requires substituting
                # `grad = -eligibility * G` here.
                substitute = -(eligibility * G[:, t].unsqueeze(-1)) * scale
                self._substitutes[(layer_idx, cell_idx, t, gate)] = substitute
