import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from workingmem.utils import simulate_matched_dirichlet

    return mo, np, plt, simulate_matched_dirichlet, sns


@app.cell
def _(sns):
    custom_params = {"axes.spines.right": True, "axes.spines.top": True}
    sns.set_theme(style="white", palette=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## dirichlet experiments
    """)
    return


@app.cell
def _(np):
    _dim = 4
    priors = np.random.dirichlet(np.ones(_dim), size=int(1e4))
    return


@app.cell
def _(simulate_matched_dirichlet):
    def simulate_lower_dirichlet(k, lower_k, alpha=1, size=None):
        """
        thin m=1 alias of `workingmem.utils.simulate_matched_dirichlet`,
        kept for backwards compatibility with earlier cells in this
        notebook -- the real (now generalized to arbitrary m) implementation
        used by `SIRDataset` lives in `workingmem.utils` so the notebook and
        the actual dataset-generation pipeline share one implementation.
        """
        return simulate_matched_dirichlet(k=k, lower_k=lower_k, m=1, alpha=alpha, size=size)

    return (simulate_lower_dirichlet,)


@app.cell
def _(simulate_lower_dirichlet):
    simulate_lower_dirichlet(3, 2, size=8)
    return


@app.cell
def _(np, plt, simulate_lower_dirichlet):
    _k = 4
    _rows = {
        f"dirichlet{_k - 2}": np.random.dirichlet(np.ones(_k - 2), size=int(1e4)),
        f"dirichlet{_k - 1}": np.random.dirichlet(np.ones(_k - 1), size=int(1e4)),
        f"dirichlet{_k}": np.random.dirichlet(np.ones(_k), size=int(1e4)),
        "simulate_lower_dirichlet": simulate_lower_dirichlet(_k, 3, size=int(1e4)),
        # "simulate_lower_dirichlet1": simulate_lower_dirichlet(_k, 3, size=int(1e4)),
        # "simulate_lower_dirichlet2": simulate_lower_dirichlet(_k, 4, size=int(1e4)),
    }

    _fig, _axes = plt.subplots(len(_rows), _k, figsize=(4 * _k, 3 * len(_rows)))
    _cmap = plt.get_cmap("tab20")
    for _row, (_name, _probs) in enumerate(_rows.items()):
        for _i in range(_k):
            _ax = _axes[_row, _i]
            if _i >= _probs.shape[1]:
                _ax.axis("off")
                continue
            _ax.hist(_probs[:, _i], bins="fd", color=_cmap(_i))
            _ax.set_title(f"{_name}: dim {_i}")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
