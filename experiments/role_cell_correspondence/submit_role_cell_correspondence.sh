#!/bin/bash
#SBATCH --job-name=role_cell_corr_compreg
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH -a 1-240%36

#SBATCH -o experiments/role_cell_correspondence/batch-output/run_%A_%a.out

# Mirrors the main repo's experiments/role_cell_correspondence/
# submit_role_cell_correspondence.sh (same TOP_K=10, n_trials=1000,
# 6 cell counts x 4 set sizes x 10 ranks = 240 tasks), run here on this
# branch (feature-multicell-competition-reg) instead -- run_role_cell_correspondence.py
# is a near-verbatim copy of the main repo's script (plus --necessity-only,
# added on this branch); its REPO_ROOT/DOWNLOADED_RUNS_DIR/CHECKPOINTS_DIR
# are all computed relative to the script's own location
# (`Path(__file__).resolve().parents[2]`), so running it from THIS worktree
# automatically picks up the competition_reg_coef=1.0 staggered sweep's
# models here instead of the baseline ones in the main repo.
#
# --necessity-only: per explicit request, only the necessity side (does
# removing this cell alone hurt decodability?) is wanted for this run, not
# sufficiency -- this skips the sufficiency-side ablation fits entirely
# (not just omits them from the output), roughly halving per-task compute
# vs. the main repo's necessity+sufficiency run.
#
# Run from the repo root of THIS worktree
# (`sbatch experiments/role_cell_correspondence/submit_role_cell_correspondence.sh`,
# cwd = this worktree, NOT the main repo).

set -x

. .venv/bin/activate

python experiments/role_cell_correspondence/run_role_cell_correspondence.py --n-trials 1000 --necessity-only
