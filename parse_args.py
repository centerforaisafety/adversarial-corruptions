import argparse

def get_parser():
    '""Returns the parser object for the main function. NOTE: All arguments should have nargs=1 (so that they can be ran from an experiment file)"""'

    parser = argparse.ArgumentParser()

    # General experiment-running arguments
    parser.add_argument(
        "--save",
        type=str,
        default="none",
        choices=["fourier_analysis", "teaser", "attacked_samples", "humanstudy"],
    )

    parser.add_argument(
        "--attack_config",
        type=str,
        help="Path to the configuration file that contains the attack parameters",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="This specified the model architecture which is being tested",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="The path to the weights (stored as a state dictonary in a .pt file) which is to be loaded into the model.",
    )
    parser.add_argument(
        "--dataset", required=True, help="The dataset which is being evaluated."
    )
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=config.attack_list,
        help="Name of the attack to evaluate against.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="./log.txt",
        help="Name of the log file which is used to store the results of the experiment, in the jsonlines format.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="The name of this experiment, used when logging.",
    )
    parser.add_argument(
        "--image_dir",
        default="./",
        help='If the "--num_image_batches" argument has been set, thi is the directory where the images are saved.',
    )

    parser.add_argument(
        "--check_loss_reduction",
        action="store_true",
        help="If set to true then the proportion of images for which loss decreases after the attack is computed and logged. Mainly used for debugging attacks, costs one extra model evaluation per datapoint.",
    )
    parser.add_argument(
        "--cuda_autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set to true, then the CUDA autotuner is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed used for both numpy and pytorch. Defaults to random.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers which are used by the  Dataloader objects in the project. If not set, defaults to 4 times the number of GPUs available, or 1 if no GPUs are available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which experiments are to be run.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
        help="The batch sizes used when evaluating models against attacks.",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=-1,
        help="If set, evaluation will only run for the specified number of batches. This is useful for debugging. By default, is -1 which means that evaluation will run on the whole evaldataset.",
    )

    # General attack arguments
    parser.add_argument(
        "--epsilon",
        type=str,
        default="medium",
        help="The maximum perturbation allowed in the attack. Can be a floating point number, or one of 'low', 'medium', 'high'.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        help="The size of the steps used in the PGD optimization algorithm. By default, is epsilon/num_steps.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="The number of steps used in the PGD optimization algorithm.",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["linf", "l2"],
        help="The distance metric used for constraining the perturbation. Only affects some attacks (see the Github attack README for more details.",
    )

    # Argument for the fog attack
    parser.add_argument(
        "--fog_wibbledecay",
        type=float,
        help="Decay parameter for the fog attack, controls the amount of large-scale structure in the fog",
    )

    # Arguments for the glitch attack
    parser.add_argument(
        "--glitch_num_lines",
        type=int,
        help="The number of vertical lines which are added to the image in the glitch attack",
    )
    parser.add_argument(
        "--glitch_grey_strength",
        type=float,
        help="The strength of the greying which is applied to the image in the glitch attack",
    )
    # Arguments for the prison attack
    parser.add_argument(
        "--prison_num_bars",
        type=int,
        help="The number of vertical bars which are added to the image in the prison attack",
    )
    parser.add_argument(
        "--prison_bar_width",
        type=int,
        help="The width of the vertical bars which are added to the image in the prison attack",
    )
    # Arguments for the blur attack
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        help="The size of the kernel which is used for the gaussian blur attack (must be odd) ",
    )
    parser.add_argument(
        "--blur_kernel_sigma",
        type=float,
        help="The standard deviation of the blur applied in the gaussian blur attack",
    )
    parser.add_argument(
        "--blur_interp_kernel_size",
        type=int,
        help="The size of the kernel which is used on the interpolation matrix for the gaussian blur attack ",
    )
    parser.add_argument(
        "--blur_interp_kernel_sigma",
        type=float,
        help="The standard deviation of the blur applied on the interpolation matrix for the gaussian blur attack",
    )

    # Arguments for the hsv attack
    parser.add_argument(
        "--hsv_kernel_size",
        type=int,
        help="The size of the kernel which is used for the hsv attack (must be odd) ",
    )
    parser.add_argument(
        "--hsv_sigma",
        type=float,
        help="The standard deviation of the gaussian blur applied on the HSV channels of the image",
    )

    # Arguments for the gabor attack
    parser.add_argument(
        "--gabor_kernel_size",
        type=int,
        help="The kernel size of the kernels used in the Gabor attack",
    )
    parser.add_argument(
        "--gabor_sides",
        type=int,
        help="The number of times each kernel is rotated and overlaid in the Gabor attack",
    )
    parser.add_argument(
        "--gabor_weight_density",
        type=float,
        help="The density of non-zero matrix entries in the spare matrix used in the Gabor noise",
    )
    parser.add_argument(
        "--gabor_sigma", type=float, help="The sigma parameter of the Gabor Kernel"
    )
    # Arguments for the snow attack
    parser.add_argument(
        "--snow_flake_size",
        type=int,
        help="Size of snowflakes (size of the kernel we convolve with). Must be odd.",
    )
    parser.add_argument(
        "--snow_num_layers",
        type=int,
        help="Number of different layers of snow applied to each image",
    )
    parser.add_argument(
        "--snow_grid_size",
        type=int,
        help="Distance between adjacent snowflakes (non-zero matrix entries) in the snow grids.",
    )
    parser.add_argument(
        "--snow_init",
        type=float,
        help="Used to initalise the optimisaiton variables in the snow attack",
    )
    parser.add_argument(
        "--snow_image_discolour",
        type=float,
        help="The amount of discolouration applied to the image during the snow attack",
    )
    parser.add_argument(
        "--snow_normalizing_constant",
        type=int,
        help="The normalisation constant applied to the snow flake grid",
    )
    parser.add_argument(
        "--snow_kernel_size",
        type=int,
        help="The size of the kernel used in the snow attack",
    )
    parser.add_argument(
        "--snow_sigma_range_lower",
        type=int,
        help="The lower bound of the range of the sigma in the gaussian blur for the snow attack",
    )
    parser.add_argument(
        "--snow_sigma_range_upper",
        type=int,
        help="The upper bound of the range of the sigma in the gaussian blur for the snow attack",
    )
    # Arguments for the klotski attack
    parser.add_argument(
        "--klotski_num_blocks", type=int, help="Number of blocks in the klotski attack"
    )
    # Arguments for the whirlpool attack
    parser.add_argument(
        "--num_whirlpools", type=int, help="Number of whirlpools applied to each image"
    )
    parser.add_argument("--whirlpool_radius", type=float, help="Radius of whirlpool")
    parser.add_argument(
        "--whirlpool_min_strength", type=float, help="Minimum strength of whirlpool"
    )
    # Arguments for the wood attack
    parser.add_argument(
        "--wood_num_rings",
        type=int,
        help="Number of rings of wood applied to each image",
    )
    parser.add_argument(
        "--wood_noise_resolution",
        type=int,
        help="Resolution of the noise used in the wood attack",
    )
    parser.add_argument(
        "--wood_random_init",
        type=lambda x: x == "true" or x == "True",
        help="Whether to use random initialisation for the wood attack",
    )
    parser.add_argument(
        "--wood_normalising_constant",
        type=int,
        help="The normalising constant used in the wood attack",
    )
    # Arguments for the interp attack
    parser.add_argument(
        "--mix_interp_kernel_size",
        type=int,
        help="The size of the kernel used in the mix attack",
    )
    parser.add_argument(
        "--mix_interp_kernel_sigma",
        type=float,
        help="The sigma of the kernel used in the mix attack",
    )

    # Arguments for the elastic attack
    parser.add_argument(
        "--elastic_upsample_factor",
        type=int,
        help="The scale factor for upsampling the displacement kernel in the elastic attack.",
    )

    # Arguments for the pixel attack
    parser.add_argument(
        "--pixel_size", type=int, help="The size of the pixels in the pixel attack"
    )
    # Arguments for the polkadot attack
    parser.add_argument(
        "--polkadot_num_polkadots",
        type=int,
        help="The number of polkadots in the polkadot attack",
    )
    parser.add_argument(
        "--polkadot_distance_scaling",
        type=float,
        help="The distance scaling for the polkadot attack",
    )
    parser.add_argument(
        "--polkadot_image_threshold",
        type=float,
        help="The image threshold for the polkadot attack",
    )
    parser.add_argument(
        "--polkadot_distance_normaliser",
        type=float,
        help="The distance normaliser for the polkadot attack",
    )
    # Arguments for the kaleidoscope attack
    parser.add_argument(
        "--kaleidoscope_num_shapes",
        type=int,
        help="The number of shapes in the kaleidoscope attack",
    )

    parser.add_argument(
        "--kaleidoscope_shape_size",
        type=int,
        help="The size of the shapes in the kaleidoscope attack",
    )
    parser.add_argument(
        "--kaleidoscope_min_value_valence",
        type=float,
        help='The minimum value of the "value" parameter in the kaleidoscope attack',
    )
    parser.add_argument(
        "--kaleidoscope_min_value_saturation",
        type=float,
        help='The maximum value of the "value" parameter in the kaleidoscope attack',
    )
    parser.add_argument(
        "--kaleidoscope_transparency",
        type=float,
        help='The minimum value of the "value" parameter in the kaleidoscope attack',
    )
    parser.add_argument(
        "--kaleidoscope_edge_width",
        type=int,
        help="The width of the edges in the kaleidoscope attack",
    )

    # Arguments for the lighting attack
    parser.add_argument(
        "--lighting_num_filters",
        type=int,
        default=20,
        help="The number of filters in the lighting attack",
    )
    parser.add_argument(
        "--lighting_loss_function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "candw"],
        help="The loss function used in the lighting attack",
    )

    # Arguments for the texture attack
    parser.add_argument(
        "--texture_threshold",
        type=float,
        help="The threshold for edges used in the texture attack",
    )

    # Arguments for the edge attack
    parser.add_argument(
        "--edge_threshold",
        type=float,
        help="The threshold for edges used in the edge attack",
    )
    # Arguments for the smoke attack
    parser.add_argument(
        "--smoke_resolution",
        type=int,
        default=16,
        help="The resolution of the noise in the smoke attack",
    )
    parser.add_argument(
        "--smoke_freq",
        type=int,
        default=10,
        help="The spacing of the smoke tendrils in the smoke attack",
    )
    parser.add_argument(
        "--smoke_normaliser",
        type=float,
        default=4,
        help="The normaliser for the smoke attack",
    )
    parser.add_argument(
        "--smoke_squiggle_freq",
        type=float,
        default=10,
        help="The squiggle frequency for the smoke attack",
    )
    parser.add_argument(
        "--smoke_height",
        type=float,
        default=0.5,
        help="The height of the smoke in the smoke attack",
    )
    return parser