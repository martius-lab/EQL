# Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Code for generating white and pink noise.'''

import numpy
import numpy.random as rng
import operator


def iterwhite():
    '''Generate a sequence of samples of white noise.

    Generates a never-ending sequence of floating-point values.
    '''
    while True:
        for n in rng.randn(100):
            yield n


def iterpink(depth=20):
    '''Generate a sequence of samples of pink noise.

    Based on the Voss-McCartney algorithm, discussion and code examples at
    http://www.firstpr.com.au/dsp/pink-noise/

    depth: Use this many samples of white noise to calculate the output. A
      higher number is slower to run, but renders low frequencies with more
      correct power spectra.

    Generates a never-ending sequence of floating-point values. Any continuous
    set of these samples will tend to have a 1/f power spectrum.
    '''
    values = rng.randn(depth)
    smooth = rng.randn(depth)
    source = rng.randn(depth)
    sum = values.sum()
    i = 0
    while True:
        yield sum + smooth[i]

        # advance the index by 1. if the index wraps, generate noise to use in
        # the calculations, but do not update any of the pink noise values.
        i += 1
        if i == depth:
            i = 0
            smooth = rng.randn(depth)
            source = rng.randn(depth)
            continue

        # count trailing zeros in i
        c = 0
        while not (i >> c) & 1:
            c += 1

        # replace value c with a new source element
        sum += source[i] - values[c]
        values[c] = source[i]


def _asarray(source, shape):
    noise = source()
    if shape is None:
        return noise.next()
    count = reduce(operator.mul, shape)
    return numpy.asarray([noise.next() for _ in range(count)]).reshape(shape)


def white(shape=None):
    '''Generate white noise.

    shape: If given, returns a numpy array of white noise with this shape. If
      not given, return just one sample of white noise.
    '''
    return _asarray(iterwhite, shape)


def pink(shape=None, depth=20):
    '''Generate an array of pink noise.

    shape: If given, returns a numpy array of noise with this shape. If not
      given, return just one sample of noise.
    depth: Use this many samples of white noise to calculate pink noise. A
      higher number is slower to run, but renders low frequencies with more
      correct power spectra.
    '''
    return _asarray(lambda: iterpink(depth), shape)

# added by Georg
def pink_zero_mean_std(shape=None, depth=20):
    '''Generate an array of pink noise.

    shape: If given, returns a numpy array of noise with this shape. If not
      given, return just one sample of noise.
    depth: Use this many samples of white noise to calculate pink noise. A
      higher number is slower to run, but renders low frequencies with more
      correct power spectra.
    '''
    dat = _asarray(lambda: iterpink(depth), shape)
    return (dat-numpy.mean(dat))/numpy.sqrt(numpy.log2(depth))


if __name__ == '__main__':
    from matplotlib import pylab

    k = numpy.ones(100.) / 10.
    def spectrum(s):
        a = abs(numpy.fft.rfft(list(s))) ** 2
        return numpy.convolve(a, k, 'valid')

    ax = pylab.gca()

    w = iterwhite()
    ax.loglog(spectrum(w.next() for _ in range(10000)), 'k')

    for p, a in enumerate(numpy.logspace(-0.5, 0, 7)):
        print 2 ** (p + 1)
        p = iterpink(2 ** (p + 1))
        ax.loglog(spectrum(p.next() for _ in range(10000)), 'r', alpha=a)


    ax.grid(linestyle=':')
    ax.set_xlim(10., None)
    ax.set_ylim(None, 1e8)

    pylab.show()