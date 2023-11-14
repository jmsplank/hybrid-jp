from os import environ
from pathlib import Path

import numpy as np
from epoch_cheats.deck import Deck
from PIL import Image

import hybrid_jp as hj
from hybrid_jp import analysis as hja


def preload(toml_path: str) -> tuple[hj.config.ShockParams, Deck, list[Path]]:
    cfg = hj.config_from_toml(toml_path).shk
    deck = hja.load_deck(cfg.data_dir)
    paths = sorted(list(Path(cfg.data_dir).glob("*.sdf")))

    return cfg, deck, paths


def usr_input(n_paths: int, data_dir: Path) -> int:
    msg = f"Choose one SDF [0<=n<{n_paths}] n="

    sdf_choice = int(input(msg))

    return sdf_choice


def main(cfg: hj.config.ShockParams, deck: Deck, sdf_path: Path):
    SDF = hj.sdf_files.load_sdf_verified(sdf_path, dt=None)

    data = np.stack([np.array(i).T for i in SDF.mag], axis=2)
    data /= data.max()
    data *= 255.0
    data = data.astype(np.uint8)
    print(data)

    img = Image.fromarray(data, "RGB")
    img = img.resize((img.size[0], img.size[1] * 8), Image.Resampling.LANCZOS)
    img.save(Path(__file__).parent / (Path(__file__).stem + ".jpg"))


if __name__ == "__main__":
    cfg, deck, paths = preload(environ["TOML_PATH"])
    n_paths = len(paths)
    # sdf_idx = usr_input(n_paths, cfg.data_dir)
    sdf_idx = 80
    main(cfg, deck, paths[sdf_idx])
