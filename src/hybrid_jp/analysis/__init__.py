from pathlib import Path

from epoch_cheats import evaluate_deck as evaluate_deck
from epoch_cheats import validate_deck as validate_deck

from .ffts import frame_power as frame_power
from .loading import load_sdfs_para as load_sdfs_para
from .shock_centering import CenteredShock as CenteredShock


def load_deck(data_dir: str | Path, deck_name: str = "input.deck"):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    deck_path = data_dir / deck_name
    if not deck_path.exists():
        raise FileNotFoundError(f"Deck file {deck_path} not found.")

    return validate_deck(evaluate_deck(deck_path))
