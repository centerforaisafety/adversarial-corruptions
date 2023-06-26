import os
import config

imagenet_location = os.path.join(config.project_path, "datasets/imagenet")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

default_epsilons = {
    "none": [1, 1, 1],
    "jpeg": [1 / 255, 3 / 255, 6 / 255],
    "prison": [0.01, 0.03, 0.1],
    "pgd": [2 / 255, 4 / 255, 8 / 255],
    "wood": [0.03, 0.05, 0.1],
    "elastic": [0.1, 0.25, 0.5],
    "fbm": [0.02, 0.04, 0.08],
    "whirlpool": [10, 40, 100],
    "gabor": [0.02, 0.04, 0.06],
    "polkadot": [1, 2, 3],
    "klotski": [0.03, 0.05, 0.1],
    "blur": [0.1, 0.3, 0.6],
    "fog": [0.3, 0.4, 0.5],
    "snow": [10, 15, 25],
    "edge": [0.03, 0.1, 0.3],
    "hsv": [0.01, 0.02, 0.03],
    "mix": [5, 10, 40],
    "texture": [0.01, 0.1, 0.2],
    "glitch": [0.03, 0.05, 0.07],
    "pixel": [3, 5, 10],
    "kaleidoscope": [0.05, 0.1, 0.15],
}
