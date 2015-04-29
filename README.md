# unitframe

Use units and quantities with pandas.

Attaching physical units to your data seems important to most of us if we are
writing down calculations by hand or drawing graphs but as soon as we step into
the digital world of floats and doubles they don't seem to matter any more. The
[unitframe](http://www.github.com/daviskirk/unitframe) module is an attempt to
attach physical unit information to
[pandas](http://www.github.com/pydata/pandas) objects using
[pint](http://www.github.com/hgrecco/pint). In theory we would like to combine
the awesome numerical framework of
[pandas](http://www.github.com/pydata/pandas) with the nifty unit calculation
and parsing abilities of [pint](http://www.github.com/hgrecco/pint).


## Requirements

* [numpy](http://www.github.com/numpy/numpy)
* [pandas](http://www.github.com/pydata/pandas)
* [pint](http://www.github.com/hgrecco/pint)
* [pytest](http://pytest.org/) (only for testing)

## Installing

Because at the moment I am just using this for personal projects there is no
"nice" way of installing `unitframe` via setuptools or pip. Just copy the
`unitframe/unitframe.py` file to wherever you want it and import the objects
using:

```python
from unitframe import UnitFrame, UnitSeries
```

## Example

```python
from unitframe import UnitFrame, UnitSeries

uf = UnitFrame([[4.0,5.0],[1.2,1.0]],
               index=list('AB'),
               units=['m','kg'],
               columns=list('ab'))
```

Math with UnitFrames works along each column and works with scalars:
```python
uf_new = uf + 10
```

as well as with other UnitFrames if they are compatible:
```python
uf2 = UnitFrame([[100,200],[0,0]],
                index=list('AB'),
                units=['mm','kg'],
                columns=list('ab'))
uf_new = uf + uf2
```

the same goes for `UnitSeries` objects. These are `pandas.Series` objects with
a single attached unit:
```python
us = UnitSeries([5,6], index=list('AB'), unit='m')
uf_new = uf + uf2
```


## Tests

Testing requires [pytest](http://pytest.org/) to be installed. To run the tests

```sh
py.test tests
```


## TODO

* The unit registry `UREG` is hard coded into the module. If there is any way
  to use generic unit registries it would be nice. Some of the code relies on
  type checking for unit registry quantities so that using a different registry
  leads to type checking errors.
* Implement pickling... there seem to be many many problems with pickling
  metadata of subclassed pandas objects
* setuptools, pip and or conda stuff


## Contribute

As stated above I just threw this up here because I know other people have had
problems with getting physical units, conversions and quantities to work well
in a numerical framework like [numpy](http://www.github.com/numpy/numpy) or [pandas](http://www.github.com/pydata/pandas). If anyone has better Ideas
or ways of improving this (at best) modest solution I would be happy to help in
any way.
