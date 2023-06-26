import os
import config

cifar10_location = os.path.join(config.project_path, "datasets/cifar10")

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
mean_wrn = (0.5, 0.5, 0.5)
std_wrn = (0.5, 0.5, 0.5)

default_epsilons = {
    "none": [0.3, 0.3, 0.3],
    "jpeg": [1 / 255, 3 / 255, 6 / 255],
    "prison": [0.01, 0.03, 0.1],
    "pgd": [2 / 255, 4 / 255, 8 / 255],
    "wood": [0.03, 0.05, 0.1],
    "elastic": [0.1, 0.25, 0.5],
    "fbm": [0.02, 0.04, 0.08],
    "whirlpool": [20, 100, 200],
    "gabor": [0.02, 0.03, 0.04],
    "polkadot": [1, 2, 3],
    "klotski": [0.03, 0.05, 0.1],
    "blur": [0.1, 0.3, 0.6],
    "fog": [0.3, 0.4, 0.5],
    "lighting": [1, 3, 5],
    "snow": [3, 4, 5],
    "edge": [0.03, 0.1, 0.3],
    "hsv": [0.01, 0.02, 0.03],
    "mix": [1, 5, 10],
    "texture": [0.01, 0.1, 0.2],
    "glitch": [0.03, 0.05, 0.1],
    "pixel": [1, 5, 10],
    "kaleidoscope": [0.05, 0.1, 0.15],
}
