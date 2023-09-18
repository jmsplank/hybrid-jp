import matplotlib.pyplot as plt
import numpy as np
from dotenv import dotenv_values
from tqdm import tqdm

import hybrid_jp as hj
import hybrid_jp.analysis as hja

if __name__ == "__main__":
    # Environment variable DATA_DIR must be set to the directory containing
    # the deck and sdf files
    env = dotenv_values(".env")
    data_dir = str(env["DATA_DIR"])

    # Load deck and sdf files
    deck = hja.load_deck(data_dir=data_dir)
    SDFs, fpaths = hja.load_sdfs_para(
        sdf_dir=data_dir,
        dt=deck.output.dt_snapshot,
        threads=7,
        start=0,
        stop=100,
    )
    # Unit conversions
    for SDF in SDFs:
        SDF.mag *= 1e9  # Convert to nT
        SDF.mid_grid *= 1e-3  # Convert to km

    # Create a CenteredShock object
    cs = hja.CenteredShock(SDFs, deck)

    # Set the number of chunks
    N_CHUNKS = 10
    cs.n_chunks = N_CHUNKS
    # Get all frames in first downstream chunk
    chunk = cs.downstream_start_chunk
    all_frames = cs.valid_chunks[chunk, :]
    valid_frames: hj.arrint = np.nonzero(all_frames)[0]

    # Set fft properties
    subdivisions = 8  # split each cell into (2^n)**2 subdivisions
    num_radial_bins = 100  # These bins are logarithmically spaced

    # Define containers for the power and radial bins
    power = np.empty((valid_frames.size, num_radial_bins))
    k = np.empty(num_radial_bins)

    # Loop over all valid frames
    for i, t_idx in tqdm(
        enumerate(valid_frames),
        total=valid_frames.size,
        desc="Frame",
    ):
        power[i], k = hja.frame_power(cs, chunk, t_idx, subdivisions, num_radial_bins)

    PSD = power.mean(axis=0)
    fig, ax = plt.subplots()
    ax.loglog(k, PSD, color="k", lw=1)
    ax.set_xlabel(r"$k\,[km]$")
    ax.set_ylabel(r"$PSD(k)\,[km^2\,s^{-2}]$")
    plt.show()
