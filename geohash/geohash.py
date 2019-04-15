import math
from collections import namedtuple
import logging
import numpy as np
from copy import deepcopy


logger = logging.getLogger(__name__)
Range = namedtuple('Range', ['min', 'max'])


PRECISION_TO_BITS_TIMES = 5
BASE32_VALUES = '0123456789bcdefghjkmnpqrstuvwxyz'
LATITUDE_RANGE = Range(min=-90.0, max=90.0)
LONGITUDE_RANGE = Range(min=-180.0, max=180.0)


def decode(geohash):
    """
    decode hash string to coordinate interval
    """
    def __decode(bits, data_range):
        for bit in bits:
            mid = np.average([data_range.min, data_range.max])
            data_range = __new_range(data_range, mid, bit == 1)
        return data_range

    chunks = np.array([[BASE32_VALUES.index(c)] for c in geohash]).astype(np.uint8)
    chunks = np.unpackbits(chunks, axis=1)[:, -1 * PRECISION_TO_BITS_TIMES:].flatten()

    lng_bits = np.take(chunks, np.arange(0, chunks.size, 2))
    lat_bits = np.take(chunks, np.arange(1, chunks.size, 2))

    longitude_range = __decode(lng_bits, deepcopy(LONGITUDE_RANGE))
    latitude_range = __decode(lat_bits, deepcopy(LATITUDE_RANGE))

    return latitude_range, longitude_range


def encode(latitude, longitude, precision=12):
    """
    precision: 5 bits for 1 precision, according to https://en.wikipedia.org/wiki/Geohash
    """
    def __encode(data, data_range):
        mid = np.average([data_range.min, data_range.max])
        flag_in_right = (data >= mid)
        data_range = __new_range(data_range, mid, flag_in_right)
        return int(flag_in_right), data_range

    latitude_range, longitude_range = LATITUDE_RANGE, LONGITUDE_RANGE
    bits = list()

    # 1. calculate the bits
    for _ in range(math.ceil(precision * PRECISION_TO_BITS_TIMES / 2)):
        bit_lng, longitude_range = __encode(longitude, longitude_range)
        bit_lat, latitude_range = __encode(latitude, latitude_range)
        bits.extend([bit_lng, bit_lat])

    # 2. encode bits to hash string
    chunks = np.split(np.array(bits[:precision * PRECISION_TO_BITS_TIMES]), precision)
    pad_shape = ((0, 0), (8 - PRECISION_TO_BITS_TIMES, 0))
    chunks = np.packbits(np.pad(chunks, pad_shape, 'constant').astype(int), axis=1).flatten()
    return ''.join(list(map(lambda i: BASE32_VALUES[i], chunks)))


def __new_range(old_range, mid, use_right):
    if use_right:
        return Range(min=mid, max=old_range.max)
    else:
        return Range(min=old_range.min, max=mid)

