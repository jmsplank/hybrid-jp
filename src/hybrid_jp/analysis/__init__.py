from pathlib import Path

from epoch_cheats import evaluate_deck, validate_deck

from .loading import load_async as load_sdfs_para
from .shock_centering import CenteredShock


def load_deck(data_dir: str | Path, deck_name: str = "input.deck"):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    deck_path = data_dir / deck_name
    if not deck_path.exists():
        raise FileNotFoundError(f"Deck file {deck_path} not found.")

    return validate_deck(evaluate_deck(deck_path))
