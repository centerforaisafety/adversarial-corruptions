import os

logging_categories = "name,attack,avg_loss,accuracy"
min = 0
max = 1
device = "cuda"  # Set as global variable during initalisation.

project_path = os.path.dirname(os.path.realpath(__file__))
# Adding the trailing slash to make it compatible with other usages
project_path = os.path.join(project_path, "")

attack_list = [
    "blur",
    "edge",
    "elastic",
    "fbm",
    "fgsm",
    "fog",
    "gabor",
    "glitch",
    "hsv",
    "jpeg",
    "kaleidoscope",
    "klotski",
    "mix",
    "pgd",
    "pixel",
    "polkadot",
    "prison",
    "snow",
    "texture",
    "whirlpool",
    "wood",
]
attack_list = ["none"] + attack_list
