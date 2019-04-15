from geohash import geohash


def test_hash():

    print(geohash.encode(36, -129, precision=9))
    print(geohash.decode('9nkkb9954'))
    print(geohash.decode('x1d'))
