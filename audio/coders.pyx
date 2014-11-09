# coding: utf-8
# cython: profile = True
"""
Variable-Length Binary Codes
"""

# Python 2.7 Standard Library
pass

# Third-Party Libraries
import numpy as np
cimport numpy as np

# Digital Audio Coding
import bitstream

#
# Metadata
# ------------------------------------------------------------------------------
#
from .about_coders import *

#
# Unary Coding
# ------------------------------------------------------------------------------
# 
class unary(object):
    """Unary coding tag.

The unary scheme represents an integer `n` as `n` ones followed by one zero. 
The basic usage is:

    >>> import bitstream
    >>> stream = bitstream.BitStream()
    >>> stream.write([0, 1, 2, 3], unary)
    >>> stream
    0101101110
    >>> stream.read(unary, 4)
    [0, 1, 2, 3]
"""

def unary_encoder(stream, data):
    if np.isscalar(data):
        data = [data]
    for datum in data:
        stream.write(datum * [True] + [False])

# **Remark:** the following "optimized" code for `unary_encoder` is 3x slower 
# in realistic use cases (`audio.shrink`).
#
#    cdef object[:] data_ = np.array(data, ndmin=1, copy=True, dtype=object)
#    cdef unsigned long datum, _
#    for datum in data_:
#        for _ in range(datum):
#            stream.write(True)
#        stream.write(False)

def unary_decoder(stream, n=None):
    scalar = n is None
    if scalar:
        n = 1
    data = []
    try:
        snapshot = stream.save()
        for _ in range(n):
            datum = 0
            while stream.read(bool):
                datum += 1
            data.append(datum)
    except:
        stream.restore(snapshot)
        raise
    if scalar:
        data = data[0]
    return data

bitstream.register(unary, reader=unary_decoder, writer=unary_encoder)

#
# Rice Coding (a.k.a. Golomb-Rice or Golomb-Power-of-Two)
# ------------------------------------------------------------------------------
# 
class rice(object):
    """
    Golomb-Rice Coding Tag

    The unsigned Golomb-Rice scheme with fixed-width `n` represents a
    non-negative integer `i` as the code for `i % 2**n` in a fixed-width
    scheme of width `n`, followed by a unary coding of `i >> n`.
    The signed version of the scheme encodes the sign of the integer `i`
    first, then the integer `abs(i)` with the unsigned scheme.

    The basic usage is:

    >>> from bitstream import BitStream
    >>> ur2 = rice(2, signed=False)
    >>> stream = BitStream([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ur2)
    >>> stream
    00001010011000100110101011100011001110
    >>> stream.read(ur2, 10)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> sr2 = rice(2, signed=True)
    >>> stream = BitStream([0, -1, 1, -2, 2, -3, 3, -4, 4], sr2)
    >>> stream
    00001010001011000100111001101001000010
    >>> stream.read(sr2, 9)
    [0, -1, 1, -2, 2, -3, 3, -4, 4]
    """

    def __init__(self, n, *args, **kwargs):
        """
        Arguments
        ---------

          - `n`: the number of bits used for fixed-width encoding

          - `signed`: `True` if the integer sign shall be encoded, 
            `False` otherwise. 
        """
        self.n = n
        # workaround: in Cython, `signed` is a keyword.
        if args:
            self.signed = args[0]
        else:
            self.signed = kwargs["signed"]

    @staticmethod
    def from_frame(frame, *args, **kwargs):
        """
        Return a rice tag instance from a sample frame.
 
        The method performs (quasi-)optimal bit width selection.

        Arguments
        ---------

          - `frame`: a sequence of integers,

          - `signed`: `True` if the integer sign shall be encoded, 
            `False` otherwise. 

        **References:**
        ["Selecting the Golomb Parameter in Rice Coding"][Golomb] by A. Kiely.

        [Golomb]: http://ipnpr.jpl.nasa.gov/progress_report/42-159/159E.pdf
        """
        # The cast to object yields a correct computation of abs.
        # (otherwise, we have for example abs(int16(-2**15)) == -2**15.
        frame = np.array(frame, ndmin=1, copy=False, dtype=object)
        try:
            np_settings = np.seterr() #all="ignore") # need some control here.
            mean_ = np.mean(np.abs(frame))
            golden_ratio = 0.5 * (1.0 + np.sqrt(5))
            if mean_ < golden_ratio:
                n = 0
            else:
                theta = mean_ / (mean_ + 1.0)
                log_ratio = np.log(golden_ratio - 1.0) / np.log(theta)
                n = int(np.maximum(0, 1 + np.floor(np.log2(log_ratio))))
        finally:
            np.seterr(**np_settings)
        # Workaround: in Cython, `signed` is a keyword.
        if args:
            _signed = args[0]
        else:
            _signed = kwargs["signed"]
        return rice(n, signed=_signed)

    def __repr__(self):
        return "rice({0}, signed={1})".format(self.n, self.signed)

    __str__ = __repr__


def rice_encoder(r):
    def _rice_encoder(stream, data):
        cdef object[:] data_ = np.array(data, ndmin=1, copy=False, dtype=object)
        for datum in data_:
            if r.signed:
                stream.write(datum < 0)
            datum = abs(datum) # exact (datum is a Python int).
            remain, fixed = divmod(datum, 2 ** r.n)
            fixed_bits = []
            for _ in range(r.n):
                fixed_bits.insert(0, bool(fixed % 2)) 
                fixed = fixed >> 1
            stream.write(fixed_bits, bool)
            stream.write(remain, unary)  
    return _rice_encoder

def rice_decoder(r):
    def _rice_decoder(stream, n=None):
        scalar = n is None
        if n is None:
            n = 1
        data = []
        try:
            snapshot = stream.save()
            for _ in range(n):
                if r.signed and stream.read(bool):
                    sign = -1
                else:
                    sign = 1
                fixed_number = 0
                for _ in range(r.n):
                    fixed_number = (fixed_number << 1) + int(stream.read(bool))
                remain_number = 2 ** r.n * stream.read(unary)
                data.append(sign * (fixed_number + remain_number))
        except:
            stream.restore(snapshot)
            raise
        if scalar:
            data = data[0]
        return data

    return _rice_decoder

# **Remark:** the obvious Cython optimization (working with an array of
# objects `data` instead of a list) results only in <5% performance gain.
# Moreover, this is with a change of the API, the perf. is probably worse
# than the original if we have to return the result as a list, hence it's
# not worth it.

bitstream.register(rice, reader=rice_decoder, writer=rice_encoder)

#
# Unit Tests
# ------------------------------------------------------------------------------
#
def test_unary_coder():
    """
Unary coder Tests:

    >>> import bitstream
    >>> stream = bitstream.BitStream()
    >>> stream.write(0, unary)
    >>> stream.write(1, unary)
    >>> stream.write([2, 3, 4, 5, 6], unary)
    >>> stream
    0101101110111101111101111110

    >>> stream.read(unary)
    0
    >>> stream.read(unary)
    1
    >>> stream.read(unary, 2)
    [2, 3]
    >>> stream.read(unary, 3)
    [4, 5, 6]
    >>> 
    """

def test_unary_coder_exception():
   """
    >>> import bitstream
    >>> stream = bitstream.BitStream(8 * [True])
    >>> stream.read(unary) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ReadError: ...
    >>> stream
    11111111
"""

def test_rice_coder():
    """
Rice coder Tests:

    >>> import bitstream
    >>> r2 = rice(2, signed=False)
    >>> stream = bitstream.BitStream()
    >>> stream.write(0, r2)
    >>> stream
    000
    >>> stream.write([1,2,3], r2)
    >>> stream
    000010100110
    >>> stream.write([4,5,6], r2)
    >>> stream
    000010100110001001101010
    >>> stream.write([7,8,9], r2)
    >>> stream
    00001010011000100110101011100011001110

    >>> stream.read(r2)
    0
    >>> stream.read(r2)
    1
    >>> stream.read(r2, 3)
    [2, 3, 4]
    >>> stream.read(r2, 5)
    [5, 6, 7, 8, 9]
    """

def test_rice_coder_exception():
   """
    >>> import bitstream
    >>> stream = bitstream.BitStream(8 * [True])
    >>> stream.read(rice(4, signed=False)) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ReadError: ...
    >>> stream
    11111111
"""

def test_rice_coder_abs_overflow():
    """
    >>> import numpy as np
    >>> from bitstream import BitStream
    >>> BitStream(np.int16(-2**15), rice(16, signed=True))
    110000000000000000
"""





################################################################################
# ------------------------------------------------------------------------------
# "Sandbox" section, not considered a part of the public API
# ------------------------------------------------------------------------------
#
# Generic Stream Encoder/Decoder
# ------------------------------------------------------------------------------
#
def stream_encoder(symbol_encoder):
    def _stream_encoder(stream, data):
        # attempt to distinguish datum and data ... far from perfect:
        # the risk is to get a symbol that support length ... (think of a 
        # single character, tuple, etc.). 

        # the symbol_encoder probably knows better, but how can we tell ? 

        # TRY it with the data directly, catch
        # type (and value ?) errors and then test for len ? That would make
        # sense ... the value errors would be ambiguous, the type errors ok.
        # So except for the value error case (that requires more thinking)
        # yeah we should do that. For now, NOT catching value errors (and
        # encouraging symbol_encoder to raise TypeErrors in doubt when the
        # received data MAY be a list of symbols is probably the safest bet.

        # or: we add an `is_datum` optional predicate, or `is_symbol` ...
        try:
            len(data)
        except:
            data = [data]
        for datum in data:
            symbol_encoder(stream, datum)
    return _stream_encoder

def stream_decoder(symbol_decoder):
    """
    Make a stream decoder from a symbol decoder.

    The resulting decoder has the following arguments:
    
      - `stream`: a Bitstream instance.

      - `n`: an integer or `None` (the default), the number of symbols to read. 

        If `n` is `None`, a single symbol is read. Otherwise, the symbols are
        returned as a list of length `n`.
"""
    def _stream_decoder(stream, n=None):
        if n is None:
            return symbol_decoder(stream)
        else:
            data = []
            count = 0
            while count < n:
                data.append(symbol_decoder(stream))
                count += 1

# Attempting to support n=inf is pointless so far: once ReadError has been 
# issued, we can't test things such as the length of the remaining stream: 
# it may be borked. The only issue we have, as long as ReadError are not 
# coupled with a rollback from ALL readers is to disallow n = inf and more 
# generally any attempt to catch the bubbling of ReadError: ReadError IS 
# really an error, not a mere exception ...

# For variable-length codes it makes less sense to use n=inf anyway.

            return data
    return _stream_decoder
 
#   
# Generic Table Symbol Encoders/Decoders
# ------------------------------------------------------------------------------
#

# Q: merge into Huffman utils ?

def invert(table):
    inverse = {}
    for key, value in table.items():
        inverse[value] = key
    return inverse

def symbol_encoder(encode_table):
    def symbol_encoder(stream, symbol_):
        try:
            stream.write(encode_table[symbol_].copy())
        except KeyError:
            raise bitstream.WriteError("invalid symbol {0!r}".format(symbol_))
    return symbol_encoder

# NOOOOOOOOOOOOOOOOOOOOOOOOO, the decoder has to be smarter now, it is presented
# a stream, not a code and shall read what is necessary (but no more, copy may
# come in handy).

def symbol_decoder(encode_table):
    max_code_length = max(len(code) for code in encode_table.values())  
    decode_table = invert(encode_table)
    def symbol_decoder(stream):
        #print 40*"-"
        max_length_try = min(len(stream), max_code_length)
        for n in range(max_length_try + 1):
            code = stream.copy(n)
            #print "stream", stream
            #print "code", code, 
            try:
                datum = decode_table[code] # THIS is buggy.
                # Heisenbug in [] ... Sometimes we get an improper value
                # sometimes a valid key is considered invalid, etc ...
                # and getting the value with the right (identity) key
                # may turn 'invalid' keys into valid ones ... wow ...
                 
                _ = stream.read(n)
#                print "YES, datum", datum
#                print "   ***", code, datum, decode_table
#                print "stream", stream
                return datum
            except KeyError:
                 pass
#                print "NO"
#                print "   ***", code, None, decode_table
        else:
            error = bitstream.ReadError()
            error.code = code
            error.decode_table = decode_table
            raise error  
    return symbol_decoder

#
# Huffman Coding
# ------------------------------------------------------------------------------
#
# TODO: qtree support ?

# Rk: final (atomic) symbols need to be hashable.

class Node(object):
    """Function helpers to deal with nodes as `(symbol, weight)` pairs"""
    @staticmethod
    def symbol(node):
        return node[0]

    @staticmethod
    def weight(node):
        return node[1]

    @staticmethod
    def is_terminal(node):
        return not isinstance(Node.symbol(node), list)


# TODO: in make_binary_tree: secondary sorter/cmp between equal weights.
class huffman(object):
    def __init__(self, weighted_alphabet):
        self.weighted_alphabet = weighted_alphabet
        self.tree = huffman.make_binary_tree(weighted_alphabet)
        self.table = huffman.make_table(self.tree)
        self.encoder = stream_encoder(symbol_encoder(self.table))
        self.decoder = stream_decoder(symbol_decoder(self.table))

    @staticmethod
    def make_binary_tree(weighted_alphabet):
        nodes = weighted_alphabet.items()
        while len(nodes) > 1:
            nodes.sort(key=Node.weight)
            node1, node2 = nodes.pop(0), nodes.pop(0)
            node = ([node1, node2], Node.weight(node1) + Node.weight(node2))
            nodes.insert(0, node)
        return nodes[0]

    @staticmethod
    def make_table(root, table=None, prefix=None):
        if prefix is None:
            prefix = BitStream()
        if table is None:
            table = {}
        if not Node.is_terminal(root):
            for index in (0, 1):
                new_prefix = prefix.copy()
                new_prefix.write(bool(index))
                new_root = Node.symbol(root)[index]
                huffman.make_table(new_root, table, new_prefix)
        else:
            table[Node.symbol(root)] = prefix
        return table

bitstream.register(huffman, reader=lambda h: h.decoder, 
                            writer=lambda h: h.encoder)
   


#def sort_dict(dict_):
#    """
#    Represent a dict whose keys have been sorted.
#    """
#    keys = sorted(dict_.keys())
#    output = "{"
#    for key in keys:
#        output += repr(key) + ": " + repr(dict_[key]) + ", " 
#    output = output[:-2] + "}"
#    print output

#def _test_huffman():
#    """
#Huffman coders tests:

#    >>> alphabet = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.125}
#    >>> huffman_ = huffman(alphabet)
#    >>> huffman_.tree
#    ([([([(3, 0.125), (2, 0.25)], 0.375), (1, 0.5)], 0.875), (0, 1.0)], 1.875)
#    >>> sort_dict(huffman_.table)
#    {0: 1, 1: 01, 2: 001, 3: 000}
#    >>> stream = BitStream([0, 1, 2, 3], huffman_)
#    >>> stream
#    101001000
#    >>> stream.read(huffman_, 4)
#    [0, 1, 2, 3]
#    """



