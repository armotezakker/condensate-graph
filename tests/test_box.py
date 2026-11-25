import numpy as np
import pytest

from condensate_graph.box import BoxError, box_from_dimensions, resolve_box, validate_box, wrap_positions


def test_missing_box_raises():
    with pytest.raises(BoxError, match="no unit cell"):
        resolve_box(dimensions=None, box=None)


def test_zero_unit_cell_raises():
    # many readers report zero lengths when the file stores no cell
    with pytest.raises(BoxError, match="no valid unit cell"):
        resolve_box(dimensions=[0, 0, 0, 90, 90, 90])


def test_explicit_box_works():
    box = resolve_box(dimensions=None, box=(15.0, 15.0, 60.0))
    assert box.dtype == np.float64
    np.testing.assert_array_equal(box, [15.0, 15.0, 60.0])


def test_explicit_box_overrides_dimensions():
    box = resolve_box(dimensions=[100, 100, 100, 90, 90, 90], box=(15.0, 15.0, 60.0))
    np.testing.assert_array_equal(box, [15.0, 15.0, 60.0])


def test_dimensions_converted_from_angstrom_to_nm():
    np.testing.assert_allclose(box_from_dimensions([150, 150, 600, 90, 90, 90]), [15.0, 15.0, 60.0])


@pytest.mark.parametrize("bad", [(15.0, 15.0), (15.0, -1.0, 60.0), (15.0, np.nan, 60.0), (0.0, 15.0, 60.0)])
def test_invalid_explicit_box_raises(bad):
    with pytest.raises(BoxError):
        validate_box(bad)


def test_non_orthorhombic_raises():
    with pytest.raises(BoxError, match="orthorhombic"):
        box_from_dimensions([150, 150, 600, 90, 90, 120])


def test_wrap_positions_in_primary_cell():
    box = np.array([15.0, 15.0, 60.0])
    w = wrap_positions([[-0.5, 15.5, 61.0], [30.2, -14.9, -0.1]], box)
    np.testing.assert_allclose(w, [[14.5, 0.5, 1.0], [0.2, 0.1, 59.9]], atol=1e-12)
    assert np.all(w >= 0) and np.all(w < box)


def test_wrap_rejects_nonfinite_positions():
    with pytest.raises(ValueError, match="NaN"):
        wrap_positions([[np.nan, 0.0, 0.0]], (15.0, 15.0, 60.0))
