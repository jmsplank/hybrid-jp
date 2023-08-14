import numpy as np
import numpy.testing as npt
import pytest

from hybrid_jp.dtypes import Elec, Grid, Mag


@pytest.fixture
def grid():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 20)
    return Grid(x=x, y=y)


@pytest.fixture
def mag():
    bx = np.random.rand(10, 20)
    by = np.random.rand(10, 20)
    bz = np.random.rand(10, 20)
    return Mag(bx=bx, by=by, bz=bz)


@pytest.fixture
def elec():
    ex = np.random.rand(10, 20)
    ey = np.random.rand(10, 20)
    ez = np.random.rand(10, 20)
    return Elec(ex=ex, ey=ey, ez=ez)


def test_grid_shape(grid: Grid):
    assert grid.shape == (10, 20)


def test_grid_slice_x(grid: Grid):
    start, stop = 2, 7
    sliced = grid.slice_x(start, stop)
    npt.assert_array_equal(sliced.x, grid.x[start:stop])
    npt.assert_array_equal(sliced.y, grid.y)


def test_mag_slice_x(mag: Mag):
    start, stop = 2, 7
    sliced = mag.slice_x(start, stop)
    npt.assert_array_equal(sliced.bx, mag.bx[start:stop, :])
    npt.assert_array_equal(sliced.by, mag.by[start:stop, :])
    npt.assert_array_equal(sliced.bz, mag.bz[start:stop, :])


def test_elec_slice_x(elec: Elec):
    start, stop = 2, 7
    sliced = elec.slice_x(start, stop)
    npt.assert_array_equal(sliced.ex, elec.ex[start:stop, :])
    npt.assert_array_equal(sliced.ey, elec.ey[start:stop, :])
    npt.assert_array_equal(sliced.ez, elec.ez[start:stop, :])


def test_grid_mult_by_three(grid: Grid):
    multiplied = grid * 3
    assert multiplied.shape == (10, 20)
    npt.assert_array_equal(multiplied.x, grid.x * 3)
    npt.assert_array_equal(multiplied.y, grid.y * 3)


def test_grid_all(grid: Grid):
    all = grid.all
    assert all["x"] is grid.x
    assert all["y"] is grid.y


def test_mag_all(mag: Mag):
    all = mag.all
    assert all["bx"] is mag.bx
    assert all["by"] is mag.by
    assert all["bz"] is mag.bz


def test_elec_all(elec: Elec):
    all = elec.all
    assert all["ex"] is elec.ex
    assert all["ey"] is elec.ey
    assert all["ez"] is elec.ez


def test_mag_rmult(mag: Mag):
    multiplied = 3 * mag
    npt.assert_array_equal(multiplied.bx, mag.bx * 3)
    npt.assert_array_equal(multiplied.by, mag.by * 3)
    npt.assert_array_equal(multiplied.bz, mag.bz * 3)


def test_elec_imul(elec: Elec):
    elec2 = Elec(**elec.all)
    elec2 *= 3
    npt.assert_array_equal(elec2.ex, elec.ex * 3)
    npt.assert_array_equal(elec2.ey, elec.ey * 3)
    npt.assert_array_equal(elec2.ez, elec.ez * 3)


def test_grid_iter(grid: Grid):
    coords = ["x", "y"]
    for i, val in enumerate(grid):
        npt.assert_array_equal(val, getattr(grid, coords[i]))


def test_mean_over_1d():
    grid1d = Grid(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        grid1d.mean_over_axis(0)


def test_mag_mean_over_y(mag: Mag):
    mean = mag.mean_over_y()
    npt.assert_array_equal(mean.bx, np.mean(mag.bx, axis=1))
    npt.assert_array_equal(mean.by, np.mean(mag.by, axis=1))
    npt.assert_array_equal(mean.bz, np.mean(mag.bz, axis=1))
