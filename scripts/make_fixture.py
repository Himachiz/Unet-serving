"""Generate a synthetic chest-shaped fixture for tests.

Not a real X-ray, and deliberately so: test fixtures must not carry data we
are not free to redistribute. Deterministic, so CI runs are reproducible.
"""
import numpy as np
from PIL import Image

SIZE = 512
rng = np.random.default_rng(0)          # fixed seed: same fixture every run

y, x = np.mgrid[0:SIZE, 0:SIZE]
img = np.full((SIZE, SIZE), 0.15)

# two ellipses roughly where lungs sit, so the model has something to find
for cx in (0.36, 0.64):
    a, b = 0.11 * SIZE, 0.20 * SIZE
    mask = ((x - cx * SIZE) / a) ** 2 + ((y - 0.45 * SIZE) / b) ** 2 < 1
    img[mask] = 0.55

img += rng.normal(0, 0.03, img.shape)   # mild texture
img = np.clip(img, 0, 1)

Image.fromarray((img * 255).astype(np.uint8)).save("samples/fixture.png")
print("wrote samples/fixture.png")
