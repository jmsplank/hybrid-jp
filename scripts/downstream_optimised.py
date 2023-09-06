import matplotlib.pyplot as plt
from epoch_cheats.deck import Deck
from phdhelper import mpl
from phdhelper.colours import sim as colours

from hybrid_jp.analysis import CenteredShock, load_deck, load_sdfs_para
from hybrid_jp.sdf_files import SDF

if __name__ == "__main__":
    data_dir = "U6T40"
    START, END = 20, 200
    deck = load_deck(data_dir=data_dir)
    SDFs, files = load_sdfs_para(
        data_dir,
        threads=7,
        dt=deck.output.dt_snapshot,
        start=START,
        stop=END,
    )

    cs = CenteredShock(SDFs, deck)
    plt.plot(cs.shock_x)
    plt.show()
