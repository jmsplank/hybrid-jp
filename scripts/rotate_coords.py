# %%
from functools import partial
from multiprocessing import set_start_method
from os import environ
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.linalg import inv
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy.ndimage import gaussian_filter  # type: ignore

import hybrid_jp as hj
from hybrid_jp import analysis as hja

# %%
set_start_method("fork")
mpl.format()
# Force U6T40 for now
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk

# %%
deck = hja.load_deck(cfg.data_dir)
SDFs, fpaths = hja.load_sdfs_para(
    sdf_dir=cfg.data_dir,
    dt=deck.output.dt_snapshot,
    threads=cfg.n_threads,
    start=cfg.start_sdf,
    stop=cfg.stop_sdf,
)

# %%
for SDF in SDFs:
    SDF.mag *= 1e9  # nT
    SDF.mid_grid *= 1e-3  # km
    SDF.numberdensity *= 1e-6  # cm^-3
cs = hja.CenteredShock(SDFs, deck)
cs.n_chunks = cfg.n_chunks
# %%

pos = np.zeros(2)

CHUNK = cs.downstream_start_chunk
T_IDX = int(np.argmax(cs.valid_chunks[CHUNK]))


def qty_b_xyz(SDF: hj.sdf_files.SDF, coord: Literal["bx", "by", "bz"]):
    return getattr(SDF.mag, coord)


nx, ny = cs.chunk_i, int(deck.control.ny)
ss = (0, nx)
frame = np.empty((3, nx, ny))
for i in range(3):
    coord = ["bx", "by", "bz"][i]
    qty = partial(qty_b_xyz, coord=coord)  # type: ignore
    frame[i], ss = cs.get_qty_in_frame(qty, CHUNK, T_IDX)


# axs: list[Axes]
# fig_w = 8
# fig, axs = plt.subplots(1, 3, figsize=(fig_w, fig_w * (3 * nx) / ny))

# for i, (ax, c) in enumerate(zip(axs, ["bx", "by", "bz"])):
#     im = ax.pcolormesh(frame[i].T)
#     ax.set_title(c)
#     ax.set_aspect("equal")
#     ax.grid(False)

# fig.tight_layout()
# plt.show()

# %%
# Get mean unit vector direction


def mean_b_unit_vec(frame_xyz: hj.arrfloat) -> hj.arrfloat:
    # frame: [3, nx, ny]
    components = frame.mean(axis=(1, 2))
    norm = np.linalg.norm(components)
    return components / norm


b_unit = mean_b_unit_vec(frame_xyz=frame)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection="3d")

# pos = np.zeros(1)
# ax.quiver(pos, pos, pos, *b_unit[:, np.newaxis], color=colours.blue(), length=0.2)
# lims = (-1, 1)
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_zlim(lims)  # type: ignore
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")  # type: ignore

# fig.tight_layout()
# plt.show()

# %%
# Look at directions for all frames

frame_norm = np.linalg.norm(frame, axis=0)

vecs = frame.copy()
xx, yy = np.meshgrid(cs.grid.x[ss[0] : ss[1] : 3], cs.grid.y[::3])
zz = np.zeros_like(xx)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# ax.quiver(
#     xx,
#     yy,
#     zz,
#     *np.moveaxis(vecs, (1, 2), (2, 1))[:, ::3, ::3],
#     color=colours.red(),
# )
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_zlim(vecs[2].min(), vecs[2].max())  # type: ignore
# plt.show()

# %%


def qty_b_xyz_mean(sdf: hj.sdf_files.SDF, coord: str) -> hj.arrfloat:
    mag: hj.arrfloat = getattr(sdf.mag, coord)
    return mag.mean()


def mean_vec_in_chunk(qty_xyz_func, cs: hja.CenteredShock, chunk: int):
    times = np.nonzero(cs.valid_chunks[chunk])[0]

    out = np.empty((times.size, 3, cs.chunk_i, int(cs.deck.control.ny)))
    for i, t_idx in enumerate(times):
        for j, c in enumerate(["bx", "by", "bz"]):
            out[i][j], _ = cs.get_qty_in_frame(
                partial(qty_xyz_func, coord=c), chunk, t_idx
            )
    return out.mean(axis=0)


vec = mean_vec_in_chunk(qty_b_xyz, cs, CHUNK)


def pcolormesh_3d(arr: hj.arrfloat):
    """arr.shape = (3, nx, ny)"""
    assert arr.shape[0] == 3

    nx, ny = arr.shape[1:]
    figw = 6
    axs: list[Axes]
    fig, axs = plt.subplots(1, 3, figsize=(figw, figw * (3 * nx) / ny))

    for i, ax in enumerate(axs):
        ax.pcolormesh(arr[i].T)
        ax.set_aspect("equal")
        ax.set_title("b" + "xyz"[i])

    return fig, axs


fig, axs = pcolormesh_3d(vec)
fig.tight_layout()
plt.show()

# %%
# All chunks


def func_in_chunk(func: Callable, chunk: int, cs: hja.CenteredShock) -> hj.arrfloat:
    times = np.nonzero(cs.valid_chunks[chunk])[0]
    out = np.empty((times.size, cs.chunk_i, int(cs.deck.control.ny)))
    for i, t_idx in enumerate(times):
        out[i], _ = cs.get_qty_in_frame(func, chunk, t_idx)
    return out


vec_evo = np.empty((cs.n_chunks, 3))
f_x = partial(qty_b_xyz, coord="bx")
f_y = partial(qty_b_xyz, coord="by")
f_z = partial(qty_b_xyz, coord="bz")
for chunk in range(cs.n_chunks):
    for i, func in enumerate([f_x, f_y, f_z]):
        vec_evo[chunk, i] = func_in_chunk(func, chunk, cs).mean()

axs: list[Axes]
fig, axs = plt.subplots(3, 1)
dists = (
    (np.arange(cs.n_chunks) - cs.downstream_start_chunk)
    * cs.dx
    * cs.chunk_i
    / (deck.constant.di / 1e3)
)

pairs = [(0, 1), (0, 2), (1, 2)]
for i, ax in enumerate(axs):
    pair = pairs[i]
    vec = vec_evo[:, pair].T
    ax.quiver(dists, np.zeros_like(dists), *vec, color=colours.blue())
    ax.set_xlim(dists.min() - vec[0, 0], dists.max() + vec[0, -1])
    ax.set_ylim(-np.abs(vec[1]).max(), np.abs(vec[1]).max())
    lb = ["b_x", "b_y", "b_z"]
    ax.set_ylabel(rf"$({lb[pair[0]]},\ {lb[pair[1]]})$")
fig.tight_layout()
plt.show()


# %%
def rot_matrix_x_to_unit_vec(vec: hj.arrfloat) -> hj.arrfloat:
    x = np.asarray([1, 0, 0], dtype=np.float64)
    if np.allclose(vec, x):
        print("Aligned")
        return np.identity(3)
    if np.allclose(vec, -x):
        print("antiparallel")
        return -np.identity(3)

    # Derived using https://math.stackexchange.com/a/476311
    rotation_matrix = np.asarray(
        [
            [
                1,
                (vec[1] ** 2 / (1 + vec[0])) - vec[1],
                (vec[2] ** 2 / (1 + vec[0])) + vec[2],
            ],
            [
                (vec[1] ** 2 / (1 + vec[0])) + vec[1],
                1,
                0,
            ],
            [
                (vec[2] ** 2 / (1 + vec[0])) - vec[2],
                0,
                1,
            ],
        ]
    )
    return rotation_matrix


vec_norm = np.repeat(np.linalg.norm(vec_evo, axis=1), 3).reshape((-1, 3))
vec_unit = vec_evo / vec_norm


def rotate_arr_by_matrix(arr: hj.arrfloat, rotation: hj.arrfloat) -> hj.arrfloat:
    # From: https://stackoverflow.com/a/61855753
    return np.einsum("ij,klj->kli", rotation, arr)  # type: ignore


mag_data = np.empty((cs.chunk_i, int(cs.deck.control.ny), 3))
for i, coord in enumerate(["bx", "by", "bz"]):
    mag_data[:, :, i] = func_in_chunk([f_x, f_y, f_z][i], CHUNK, cs).mean(axis=0)


rot_mat = rot_matrix_x_to_unit_vec(vec_unit[CHUNK])
rotated_mag = rotate_arr_by_matrix(mag_data, rot_mat)
print(rotated_mag)

fig, axs = plt.subplots(1, 3)

axs[0].pcolormesh(mag_data[:, :, 0].T)
axs[0].set_title("$B_x$")
b_para = rotated_mag[:, :, 0]
axs[1].pcolormesh(b_para.T)
axs[1].set_title(r"$B_\parallel$")
b_perp = np.linalg.norm(rotated_mag[:, :, 1:], axis=2)
axs[2].pcolormesh(b_perp.T)
axs[2].set_title(r"$B_\perp$")

for ax in axs:
    ax.set_aspect("equal")

fig.tight_layout()
plt.show()

# %%
# fig, ax = plt.subplots()

# diff_mat = b_para - b_perp

# # print(np.abs(diff_mat).mean(), diff_mat.std())
# # print(np.abs(b_para).mean(), b_para.std())
# print(np.abs(b_perp).mean(), b_perp.std())
# ax.pcolormesh(diff_mat.T)
# ax.set_aspect("equal")


# fig.tight_layout()
# plt.show()


# %%
# def b_par_per_from_chunk(chunk: int, cs: hja.CenteredShock):
#     # 1. Get magnetic field data for chunk
#     ny = int(cs.deck.control.ny)
#     mag_data = np.empty((cs.chunk_i, ny, 3))
#     for component in range(3):
#         # Get b_x, b_y, b_z for each in (chunk_i,ny)
#         # func_in_chunk returns timesteps as axis 0 so mean over all time
#         mag_data[:, :, component] = func_in_chunk(
#             [f_x, f_y, f_z][component], chunk, cs
#         ).mean(axis=0)

#     # 2. Get average B vector for chunk
#     mag_vectors = mag_data.mean(axis=(0, 1))  # shape (3,)
#     mag_norm = np.linalg.norm(mag_vectors)
#     mag_unit = mag_vectors / mag_norm  # mean direction of B for chunk

#     # 3. Obtain rotation matrix for rotating b_x into b_parallel
#     rotation = rot_matrix_x_to_unit_vec(mag_unit)

#     # 4. Rotate every (chunk_i, ny)_bx,by,bz into (chunk_i, ny)_bpara,bperp1,bperp2
#     mag_para_perps = rotate_arr_by_matrix(mag_data, rotation)
#     mag_para_perp = np.empty((cs.chunk_i, ny, 2))
#     mag_para_perp[:, :, 0] = mag_para_perps[:, :, 0]
#     mag_para_perp[:, :, 1] = np.linalg.norm(mag_para_perp[:, :, 1:], axis=2)

#     return mag_para_perp


# all_mag_para_perp = np.empty((cs.n_chunks, cs.chunk_i, int(deck.control.ny), 2))
# for chunk in range(cs.n_chunks):
#     all_mag_para_perp[chunk] = b_par_per_from_chunk(chunk, cs)

# mpp_mean = all_mag_para_perp.mean(axis=(1, 2))
# fig, ax = plt.subplots()
# ax.plot(dists, mpp_mean[:, 0], label=r"$B_\parallel$")
# ax.plot(dists, mpp_mean[:, 1], label=r"$B_\perp$")
# ax.legend()
# fig.tight_layout()
# plt.show()

# %%

ny = int(cs.deck.control.ny)
mag_data = np.empty((cs.chunk_i, ny, 3))
for component in range(3):
    # Get b_x, b_y, b_z for each in (chunk_i,ny)
    # func_in_chunk returns timesteps as axis 0 so mean over all time
    mag_data[:, :, component] = func_in_chunk(
        [f_x, f_y, f_z][component], CHUNK, cs
    ).mean(axis=0)

vec = mag_data.mean(axis=(0, 1))
vec_norm = np.linalg.norm(vec)

vec_unit = vec / vec_norm


def orthonorm_basis_of_vec(v: hj.arrfloat) -> hj.arrfloat:
    # Generate a random vector e2
    e2 = np.random.rand(3)

    # Project e2 onto the plane normal to v
    e2 -= np.dot(e2, v) * v
    e2 /= np.linalg.norm(e2)

    # Calculate e3 as the cross product of e1 and e2
    e3 = np.cross(v, e2)

    # e1 is the unit vector v
    e1 = v

    # Return the orthonormal basis
    return np.column_stack((e1, e2, e3))


def change_basis(Q: hj.arrfloat, arr: hj.arrfloat) -> hj.arrfloat:
    arr_flat = arr.reshape(-1, 3).T
    new_basis_flat = np.dot(Q, arr_flat).T
    out = new_basis_flat.reshape(arr.shape)
    return out


b_basis = orthonorm_basis_of_vec(vec_unit).T
mag_b_basis = change_basis(b_basis, mag_data)


def plot_components(arr: hj.arrfloat, axs: list[Axes]):
    for i, ax in enumerate(axs):
        ax.pcolormesh(arr[:, :, i].T, vmin=-abs(arr).max(), vmax=abs(arr).max())


fig, axs = plt.subplots(3, 1)
plot_components(mag_b_basis, axs)
fig.tight_layout()
plt.show()


# %%
def para_and_perp(arr: hj.arrfloat) -> hj.arrfloat:
    para = arr[:, :, 0]
    perp = np.linalg.norm(arr[:, :, 1:], axis=2)
    return np.stack((para, perp), axis=2)


fig, axs = plt.subplots(2, 1)
pnp = para_and_perp(mag_b_basis)
print(pnp.shape)
plot_components(pnp, axs)
fig.tight_layout()
plt.show()


# %%
def create_orthonormal_basis_from_vec(v: hj.arrfloat) -> hj.arrfloat:
    """create a basis of orthonormal vectors from an input vector `v`.

    The original components of the vector (i,j,k) in the original basis are transformed
    to a new orthonormal basis (e1,e2,e3) where the component (e1,0,0) in the new basis
    points in the direction of `v` in the original. Therefore, e2 and e3 lie in the
    plane normal to `v` and are mutually orthogonal.

    Args:
        v (hj.arrfloat): 1D vector length 3

    Returns:
        hj.arrfloat: square array shape (3,3). e_n=arr[n-1,:] where n=1,2,3
    """
    # v MUST be 1d
    assert len(v.shape) == 1
    # v MUST have shape 3
    assert v.size == 3

    # Make sure it's a unit vector
    e1 = v / np.linalg.norm(v)

    # 1. Generate a random vector e2
    #    e2 is NOT length 1 and is NOT orthogonal to `v`
    e2 = np.random.rand(3)

    # 2. Project e2 into a unit vector in the plane normal to `v`
    #    Done by subtracting the component parallel to `v` and normalising
    e2 -= np.dot(e2, e1) * e1
    e2 /= np.linalg.norm(e2)

    # 3. Get a third basis vector perpendicular to both v and e2
    e3 = np.cross(e1, e2)

    return np.stack((e1, e2, e3), axis=0)


orth = create_orthonormal_basis_from_vec(vec_unit)


# %%
def rotate_arr_to_new_basis(basis: hj.arrfloat, arr: hj.arrfloat) -> hj.arrfloat:
    """Rotate an array (nx, ny, 3) in i,j,k basis to a new `basis`.

    Args:
        basis (hj.arrfloat): (3,3) basis of (e1, e2, e3) where basis[0,:]=e1 etc.
        arr (hj.arrfloat): (nx, ny, 3) array of vectors on a grid (nx, ny)

    Returns:
        hj.arrfloat: (nx, ny, 3) array of vectors in e1,e2,e3 basis

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> scale = 100
        >>> xx, yy = np.mgrid[0:scale, 0:scale]
        >>> grid = np.empty((*xx.shape, 3))
        >>> grid[:, :, 0] = yy
        >>> grid[:, :, 1] = xx
        >>> # Add an oscillating part in k
        >>> grid[:, :, 2] = np.sin(2 * np.pi * xx / xx.max())
        >>> # e1=k in new basis
        >>> vec = np.array([0, 0, 1], dtype=np.float64)
        >>> print(f"{vec = }")
        >>> basis = create_orthonormal_basis_from_vec(vec)
        >>> print(f"{basis = }")
        >>> rot = rotate_arr_to_new_basis(basis, grid)
        >>> fig, axs = plt.subplots(2, 1)
        >>> # Plot i and j streamlines coloured by k
        >>> axs[0].streamplot(
        >>>     xx[:, 0],
        >>>     yy[0, :],
        >>>     grid[:, :, 0].T,
        >>>     grid[:, :, 1].T,
        >>>     density=2,
        >>>     arrowstyle="-",
        >>>     color=grid[:, :, 2].T,
        >>> )
        >>> # Visualise e1, should be a horizontal sine wave
        >>> axs[1].pcolormesh(xx[:, 0], yy[0, :], rot[:, :, 0].T)
        >>> axs[0].grid(False)
        >>> axs[1].grid(False)
        >>> fig.tight_layout()
        >>> plt.show()
    """
    # 1. Flatten the array into (3, nx*ny) where axis 0 is the ijk components
    arr_flat = arr.reshape(-1, 3).T
    # 2. Compute the dot product with the orthonormal basis
    new_basis_flat = np.dot(basis, arr_flat).T
    # 3. Reshape into oridinal dimension
    out = new_basis_flat.reshape(arr.shape)

    return out
