#!/usr/bin/env python

"""
Units
"""
import pandas as pd
import numpy as np
from collections import Sequence, Mapping, OrderedDict, Iterable
from numbers import Number
from functools import partial
import operator
from copy import deepcopy
import pdb
from pint import UnitRegistry, UndefinedUnitError, DimensionalityError
UREG = UnitRegistry()
Q_ = UREG.Quantity


def to_unit(q):
    """Convert to unit

    """

    if q is None:
        return 1
    if isinstance(q, str):
        q = Q_(q)
    u = 1
    if isinstance(q, Q_):
        for k, v in q.units.items():
            u *= Q_(k)**v
    elif isinstance(q, Number):
        pass
    else:
        raise TypeError(('Incorrect unit initialization. '
                         '{} of type {} can not be converted to unit').format(
                             q, type(q)))
    return u


class UnitSeries(pd.Series):
    _metadata = ['_unit']

    def __init__(self,*args,**kwargs):
        is_unit = 'unit' in kwargs
        if is_unit:
            unit = kwargs['unit']
            del kwargs['unit']
        kwargs['dtype'] = np.double
        super().__init__(*args,**kwargs)
        if not self.dtype == np.double:
            raise TypeError('dtypes must all be doubles')
        if is_unit:
            self.unit = unit

    @property
    def _constructor(self):
        return UnitSeries

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit is None:
            unit = 1
        else:
            unit = to_unit(unit)
        self._unit = unit

    def do_op(self, other, op):
        series = self.copy()
        is_unit_change_possible = op not in (operator.add, operator.sub)
        if hasattr(other, '_unit'):
            v = op(self.values*self.unit, other.values*other.unit)
            series.values[:] = v
            if is_unit_change_possible and isinstance(v, Q_):
                series._unit = to_unit(v)
        elif isinstance(other, Q_):
            for i in range(self.shape[1]):
                v = op(self.values*self.unit, other)
                series.values[:] = v
                if is_unit_change_possible and isinstance(v, Q_):
                    series._unit = to_unit(v)
        else:
            if op == operator.pow:
                if not isinstance(other, Number):
                    raise ValueError('Only scalar numbers are supported for ** operator')
                series._unit = series._unit**other
            series.values[:] = op(series.values, other)
        return series

    def __add__(self, other):
        return self.do_op(other, operator.add)

    def __sub__(self, other):
        return self.do_op(other, operator.sub)

    def __mul__(self, other):
        return self.do_op(other, operator.mul)

    def __truediv__(self, other):
        return self.do_op(other, operator.truediv)

    def __pow__(self, other):
        return self.do_op(other, operator.pow)

    def to(self, unit):
        unit = to_unit(unit)
        try:
            self.values = (self.values * self._unit).to(unit)
        except TypeError:
            if isinstance(unit, Number) and unit == 1:
                pass
            else:
                raise
        self._unit = unit

    def min(self, *args, **kwargs):
        return super().min(*args, **kwargs)*self._unit

    def max(self, *args, **kwargs):
        return super().max(*args, **kwargs)*self._unit

    def mean(self, *args, **kwargs):
        return super().mean(*args, **kwargs)*self._unit

    def std(self, *args, **kwargs):
        return super().std(*args, **kwargs)*self._unit

    def __str__(self):
        old_name = self.name
        if self.name is None:
            self.name = ''
        try:
            self.name += ' [{}]'.format(self._unit.units)
        except AttributeError:
            if self._unit == 1:
                pass
            else:
                raise
        s = super().__str__()
        self.name = old_name
        return s


class UnitFrame(pd.DataFrame):
    _metadata = ['_units']

    def __init__(self,*args,**kwargs):
        is_units = 'units' in kwargs
        if is_units:
            units = kwargs['units']
            del kwargs['units']
        kwargs['dtype'] = np.double
        super().__init__(*args,**kwargs)
        if not all(self.dtypes == np.double):
            raise TypeError('dtypes must all be doubles')
        if is_units:
            self.units = units

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        if units is None:
            units = 1
        elif isinstance(units, Mapping):
            # for dict like
            units = [to_unit(units[k]) for k in self.columns]
        elif isinstance(units, (Sequence, np.ndarray, pd.Series)):
            # for lists,  tuples and array likes
            units = [to_unit(units[i]) for i,k in enumerate(self.columns)]
        else:
            units = to_unit(units)
        self._units = pd.Series(units,index=self.columns, dtype=object)

    @property
    def _constructor(self):
        def tmp_constr(*args, **kwargs):
            if 'units' not in kwargs:
                kwargs['units'] = self._units
            return UnitFrame(*args, **kwargs)
        return tmp_constr

    def __finalize__(self, other, method=None, **kwargs):
        """ propagate metadata from other to self """
        # NOTE: backported from pandas master (upcoming v0.13)
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        self._units = self._units.copy()
        return self

    def copy(self, deep=True):
        """
        Make a copy of this UnitFrame object
        Parameters
        ----------
        deep : boolean, default True
            Make a deep copy, i.e. also copy data
        Returns
        -------
        copy : UnitFrame
        """
        # FIXME: this will likely be unnecessary in pandas >= 0.13
        data = self._data
        if deep:
            data = data.copy()
        return UnitFrame(data).__finalize__(self)

    def do_op(self, other, op):
        df = self.copy()
        is_unit_change_possible = op not in (operator.add, operator.sub)
        if hasattr(other, '_units'):
            for i in range(self.shape[1]):
                v = op(self.values[:,i]*self.units.iat[i], (other.values[:,i])*other.units.iat[i])
                df.values[:,i] = v
                if is_unit_change_possible and isinstance(v, Q_):
                    df._units.iat[i] = to_unit(v)
        elif isinstance(other, Q_):
            for i in range(self.shape[1]):
                v = op(self.values[:,i]*self.units.iat[i], other)
                df.values[:,i] = v
                if is_unit_change_possible and isinstance(v, Q_):
                    df._units.iat[i] = to_unit(v)
        else:
            if op == operator.pow:
                if not isinstance(other, Number):
                    raise ValueError('Only scalar numbers are supported for ** operator')
                for i in range(self.shape[1]):
                    df._units.iat[i] = df._units.iat[i]**other
            try:
                df.values[:] = op(df.values, other)
            except Exception:
                if isinstance(other, pd.Series):
                    raise ValueError(
                        'UnitFrame cannot use operation {} on {}'.format(
                            op, type(other)))
                else:
                    raise
        return df

    def __add__(self, other):
        return self.do_op(other, operator.add)

    def __sub__(self, other):
        return self.do_op(other, operator.sub)

    def __mul__(self, other):
        return self.do_op(other, operator.mul)

    def __truediv__(self, other):
        return self.do_op(other, operator.truediv)

    def __pow__(self, other):
        return self.do_op(other, operator.pow)

    def to(self, unit_dict):
        for k,v in unit_dict.items():
            v = to_unit(v)
            if not isinstance(v, Number):
                self[k] = (self[k].values * self._units[k]).to(v)
            self._units[k] = v

    def __str__(self):
        if hasattr(self, '_units'):
            s = self.rename(columns={
                c:str(c) + ' [{}]'.format(u.units)
                for c, u in self._units.items() if isinstance(u, Q_)
            })
        else:
            s=self
        return super(UnitFrame,s).__str__()

    def __setitem__(self, key, value):
        self._units[key] = to_unit(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if item.ndim == 1:
            item = UnitSeries(item, unit=self._units[key])
        elif isinstance(key, slice):
            item._units = self.units
        return item

    def min(self, **kwargs):
        vals = super().min(axis=0, **kwargs)
        return pd.Series((v*u for v, u in zip(vals.tolist(), self._units)),
                         index=vals.index)

    def max(self, **kwargs):
        vals = super().max(axis=0, **kwargs)
        return pd.Series((v*u for v, u in zip(vals.tolist(), self._units)),
                         index=vals.index)

    def mean(self, **kwargs):
        vals = super().mean(axis=0, **kwargs)
        return pd.Series((v*u for v, u in zip(vals.tolist(), self._units)),
                         index=vals.index)

    def std(self, **kwargs):
        vals = super().std(axis=0, **kwargs)
        return pd.Series((v*u for v, u in zip(vals.tolist(), self._units)),
                         index=vals.index)

    def to_df(self):
        return pd.DataFrame(self)


def _get_demo_ddf():
    return UnitFrame([[4,5],[6,7],[13.2,1.2]], index=list('ABC'), units=['m','kg'], columns=list('ab'))


if __name__ == '__main__':
    rr = UnitFrame([[4,5],[6,7],[13.2,1.2]], index=list('ABC'), units=['m','kg'], columns=list('ab'))
    print(rr)
    b = rr + rr
    print(b)
    c = rr + 100
    print(c)

    print('mult')
    print(rr)
    b = rr * rr
    print(b)
    c = rr * 100
    print(c)

    print('div')
    print(rr)
    b = rr / rr
    print(b)
    c = rr / 100
    print(c)
