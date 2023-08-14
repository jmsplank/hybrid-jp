"""Change point experiments.

The goal is to discover the best method/dataset or set of transformations
that will most accurately predict transitions from solar wind to shock
transition region to magnetosheath.
"""
# %% Importts
from pathlib import Path

import ruptures as rpt
from epoch_cheats import evaluate_deck, validate_deck

from hybrid_jp.sdf_files import load_sdf_verified

# %%

# %% Load data
DATA_PATH = Path().resolve().parent / "U6T40"
fname = DATA_PATH / "0128.sdf"

sdf = load_sdf_verified(fname)
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %%
