#!/usr/bin/env python

"""
Test units.
"""
import pytest
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+"/../unitframe/"))
from unitframe import *
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
import operator


def get_us():
    us = UnitSeries([4, 1.2], index=list('AB'), unit='m')
    return us


def get_uf():
    uf = UnitFrame([[4,5],[1.2,1]], index=list('AB'), units=['m','kg'], columns=list('ab'))
    return uf


def assert_unitframe(actual, expected, units_expected=None):
    assert isinstance(actual, UnitFrame)
    np.testing.assert_almost_equal(actual.values, expected.values)
    assert_frame_equal(actual, expected)
    assert hasattr(actual, 'units')
    assert hasattr(actual, '_units')
    assert_series_equal(actual.units, expected.units)
    if units_expected is not None:
        assert_series_equal(actual.units, units_expected)


def assert_unitseries(actual, expected, unit_expected=None):
    assert isinstance(actual, UnitSeries)
    np.testing.assert_almost_equal(actual.values, expected.values)
    assert_series_equal(actual, expected)
    assert hasattr(actual, 'unit')
    assert hasattr(actual, '_unit')
    assert actual.unit == expected.unit
    if unit_expected is not None:
        assert actual.unit == unit_expected


def test_unitframe_constructor():
    uf = get_uf()
    np.testing.assert_array_almost_equal(uf.values, np.array([[4,5],[1.2,1]]))
    assert hasattr(uf, 'units')
    assert hasattr(uf, '_units')
    assert_series_equal(uf.units, pd.Series([UREG('m'), UREG('kg')], index=list('ab')))


def test_unitseries_constructor():
    us = get_us()
    np.testing.assert_array_almost_equal(us.values, np.array([4, 1.2]))
    assert hasattr(us, 'unit')
    assert hasattr(us, '_unit')
    assert us.unit == UREG('m')


@pytest.mark.parametrize(
    'op,expected, units_expected',
    [
        (operator.add,
         UnitFrame([[8,10],[2.4,2]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.sub,
         UnitFrame([[0,0],[0,0]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.mul,
         UnitFrame([[16,25],[1.44,1]], index=list('AB'), units=['m**2','kg**2'], columns=list('ab')),
         pd.Series([UREG('m**2'), UREG('kg**2')], index=list('ab'))),
        (operator.truediv,
         UnitFrame([[1,1],[1,1]], index=list('AB'), units=[1, 1], columns=list('ab')),
         pd.Series([1, 1], index=list('ab'), dtype=object)),
    ],
    ids=['add', 'sub', 'mul', 'truediv'])
def test_unitframe_op(op, expected, units_expected):
    uf = get_uf()
    actual = op(uf, uf)
    assert_unitframe(actual, expected, units_expected)


@pytest.mark.parametrize(
    'op,expected, units_expected',
    [
        (operator.add,
         UnitFrame([[14,15],[11.2,11]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.sub,
         UnitFrame([[-6,-5],[-8.8,-9]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.mul,
         UnitFrame([[40, 50],[12,10]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.truediv,
         UnitFrame([[0.4,0.5],[0.12,0.1]], index=list('AB'), units=['m','kg'], columns=list('ab')),
         pd.Series([UREG('m'), UREG('kg')], index=list('ab'))),
        (operator.pow,
         UnitFrame([[16,25],[1.44,1]], index=list('AB'), units=['m**2','kg**2'], columns=list('ab')),
         pd.Series([UREG('m**2'), UREG('kg**2')], index=list('ab'))),
    ],
    ids=['add', 'sub', 'mul', 'truediv', 'pow'])
def test_unitframe_scalar_op(op, expected, units_expected):
    uf = get_uf()
    if op == operator.pow:
        actual = op(uf, 2)
    else:
        actual = op(uf, 10)
    assert_unitframe(actual, expected, units_expected)


@pytest.mark.parametrize(
    'op,expected',
    [
        (operator.add,
         UnitFrame([[14,15],[11.2,11]], index=list('AB'), units=['m','kg'], columns=list('ab'))),
        (operator.sub,
         UnitFrame([[-6,-5],[-8.8,-9]], index=list('AB'), units=['m','kg'], columns=list('ab'))),
        (operator.mul,
         UnitFrame([[40, 50],[12,10]], index=list('AB'), units=['m','kg'], columns=list('ab'))),
        (operator.truediv,
         UnitFrame([[0.4,0.5],[0.12,0.1]], index=list('AB'), units=['m','kg'], columns=list('ab'))),
    ],
    ids=['add', 'sub', 'mul', 'truediv'])
def test_unitframe_array_op(op, expected):
    uf = get_uf()
    # since rectangular,  might as well try a column vector
    actual = op(uf, np.array([[10, 10]]).T)
    assert_unitframe(actual, expected)
    # an a row vector
    actual = op(uf, np.array([[10, 10]]))
    assert_unitframe(actual, expected)


@pytest.mark.parametrize(
    'op,expected',
    [
        ('min', [Q_(1.2, 'm'), Q_(1, 'kg')]),
        ('max', [Q_(4, 'm'), Q_(5, 'kg')]),
        ('mean', [Q_(np.mean([1.2, 4]), 'm'), Q_(np.mean([1, 5]), 'kg')]),
        ('std', [Q_(np.std([1.2, 4], ddof=1), 'm'), Q_(np.std([1, 5], ddof=1), 'kg')])  # ddof set to n - 1
    ])
def test_unitframe_extreme_op(op, expected):
    uf = get_uf()
    actual = getattr(uf, op)()
    expected = pd.Series(expected, index=list('ab'))
    assert isinstance(actual, pd.Series)
    assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    'op,expected',
    [
        ('min', Q_(1.2, 'm')),
        ('max', Q_(4, 'm')),
        ('mean', Q_(np.mean([1.2, 4]), 'm')),
        ('std', Q_(np.std([1.2, 4], ddof=1), 'm'))  # ddof set to n - 1
    ])
def test_unitseries_extreme_op(op, expected):
    us = get_us()
    actual = getattr(us, op)()
    assert actual == expected


@pytest.mark.parametrize(
    'op,expected, unit_expected',
    [
        (operator.add,
         UnitSeries([8,2.4], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.sub,
         UnitSeries([0,0], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.mul,
         UnitSeries([16,1.44], index=list('AB'), unit=UREG('m**2')),
         UREG('m**2')),
        (operator.truediv,
         UnitSeries([1,1], index=list('AB'), unit=1),
         1),
    ],
    ids=['add', 'sub', 'mul', 'truediv'])
def test_unitseries_op(op, expected, unit_expected):
    us = get_us()
    actual = op(us, us)
    assert_unitseries(actual, expected, unit_expected)


@pytest.mark.parametrize(
    'op,expected, unit_expected',
    [
        (operator.add,
         UnitSeries([14,11.2], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.sub,
         UnitSeries([-6,-8.8], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.mul,
         UnitSeries([40,12], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.truediv,
         UnitSeries([0.4, 0.12], index=list('AB'), unit=UREG('m')),
         UREG('m')),
        (operator.pow,
         UnitSeries([16,1.44], index=list('AB'), unit=UREG('m**2')),
         UREG('m**2')),
    ],
    ids=['add', 'sub', 'mul', 'truediv', 'pow'])
def test_unitseries_scalar_op(op, expected, unit_expected):
    us = get_us()
    if op == operator.pow:
        actual = op(us, 2)
    else:
        actual = op(us, 10)
    assert_unitseries(actual, expected, unit_expected)


def test_unitframe_item():
    uf = get_uf()
    us = get_us()
    actual = uf['a']
    assert_unitseries(actual, us)


def test_unitframe_multiple_items():
    uf = get_uf()
    actual = uf[['a', 'b']]
    expected = uf
    assert_unitframe(actual, expected)


def test_unitframe_slice():
    uf = get_uf()
    actual = uf[:1]
    expected = UnitFrame([[4,5]], index=list('A'), units=['m','kg'], columns=list('ab'))
    assert_unitframe(actual, expected)


def test_unitframe_ix():
    uf = get_uf()
    actual = uf.ix[:1, 'a']
    expected = UnitSeries([4], index=list('A'), unit='m')
    assert_unitseries(actual, expected)


def test_unitframe_loc():
    uf = get_uf()
    actual = uf.loc[['A', 'B']]
    expected = uf
    assert_unitframe(actual, expected)


if __name__ == '__main__':
    sys.exit()
