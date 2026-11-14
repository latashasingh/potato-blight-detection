
import numpy as np


def _nd(a, b):
    a, b = a.astype("float32"), b.astype("float32")
    return (a - b) / (a + b + 1e-8)


def ndvi(nir, red): return _nd(nir, red)
def ndre(nir, red_edge): return _nd(nir, red_edge)
def gndvi(nir, green): return _nd(nir, green)


def zone_means(index_map, rows=4, cols=4):
    """Average an index map over a rows x cols grid of field zones."""
    h, w = index_map.shape
    return np.array([[index_map[r*h//rows:(r+1)*h//rows, c*w//cols:(c+1)*w//cols].mean()
                      for c in range(cols)] for r in range(rows)])


if __name__ == "__main__":
    rng = np.random.RandomState(0)           # demo with synthetic bands
    nir, red = rng.uniform(0.4, 0.6, (64, 64)), rng.uniform(0.05, 0.15, (64, 64))
    nir[:32, :32] *= 0.6                      # simulate a stressed corner
    print("NDVI zone means:\n", zone_means(ndvi(nir, red)).round(2))
