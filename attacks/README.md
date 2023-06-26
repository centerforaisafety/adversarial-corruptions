

# Attacks

## Threat model
As described in [the paper](), the attacks in this repostitory aim to generate images which cause incorrecrt classifications from the models, but are still correctly classisfied by humans.

More precisely, given an image classifier $f: R^k \to {1..K}$, and a correctly classified datapoint $(x,y)$. We aim to generate an example  $x_\text{adv}$ with the property that:

$$f(x_\text{adv}) \neq f(x) = y$$

Furthermore, we also require that a human "oracle" $O: R^k \to {1..K}$:
$$O(x_\text{adv}) \neq O(x) = y$$

This captures a more realistic threat model for the literature.

## Method for generating the adversarial inputs
All the attacks are generated using the same underlying gradient-based optimisation method. In particular, to generate an adversarial example $x_\text{adv}$ we use a differentiable function $A$:
$$x_\text{adv} = A(x,o_{\text{adv}})$$

Which takes in our original image $x$, and perturbs it. The exact form of the perturbation is controlled by the variable $o_{\text{adv}}$, and we choose this value to maximise the loss of our network:

$$o_{\text{adv}} =\underset{o : |o|_p \leq \epsilon}{\operatorname{argmax}}\{J(\theta,f(x,o),y)\}$$ 




 Since the function $a$ is differentiable, we can solve this constrained optimisation using gradient-based techniques. In particular, throughout the repository we use the 
 [Projected Gradient Descent (PGD)](https://arxiv.org/abs/1706.06083) algorithm to find suitable candidates for $o_\text{adv}$.

All of our attack code then follows the same structure: 
 - Some initalisation code for $o$.
 - A definoton a differentiable function $A$ to generate the corruptions.
 - An inner loop which iteratively optimises the value of $A(x,o)$.

# List of attacks and attack hyperparameters

## Setting Hyperparameters
Each attack can be ran by passing ``--attack attack_name`` into ``main.py``, with hyperparameters similar specified with 
 ``--hyperparameter_name hyperaparemeter_value `` and some hyperparameters being shared across all attacks:

- ``--epsilon``, a postiive floating point number. Controls the perurbation size allowed by each attack, with larger values leading to more extreme image distortions.
- ``--num_steps``, a positive integer. Controls the number of iterations in the PGD optimisation loop.
- ``--step_size``, a positive floating point number. Controls the size of the steps taken when aarrying out PGD.

Attack-specific hyperparameters are listed below.


## Attack 1: Blur
![Blur attack with epsilon = 0.4](/docs/attack_images/blur.png)
This attack functions by passing a gaussian filter over the original image and then doing a pixel-wise linear interpolation between the blurred version and the original. For each image pixel, we have a variable in (0,1) which controls the level of interpolation.

We also apply a gaussian filter to the grid of optimisation variables, to enforce some continuity in the strength of the blur between adjacent pixels.

This attack can be selected with  the `` --attack blur`` option, and has the following attack-specific hyperparameters:
`

- ``--blur_kernel_size``, an odd interger. The size of the gaussian kernel which is passed over the image. 
- ``--blur_kernel_sigma``, a positive floating point number. The sigma parameter for the image gaussian kernel.
- ``--blur_interp_kernel_size``, an odd integer. The size of the gaussian kernel which is passsed over the optimisation variables.
-  ``--blur_interp_kernel_size``, a positive floating point number. The sigma parameter for the optimisation variable gaussian kernel.

## Attack 2: Edge
![edge attack with epsilon = 0.4](/docs/attack_images/edge.png)

This attack functions by applying a [canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) over the 
image to locate pixels at the edge of objects, which then marked as "edge pixels". We then apply a PGD attack but *only* to these edge pixels.

This attack can be selected with the  `` --attack edge`` option, and has these attack-specific hyperparameters:

- ``--edge_threshold``, a floating point number in (0,1). After passing a canny edge filter over an image, we get an "edge score" for each pixel. This decides the threshold before pixels are labelled as "edge pixels".

## Attack 4: Elastic
![elastic attack with epsilon = 0.4](/docs/attack_images/elastic.png)

This attack, which originally appeared in [Testing Robustness Against Unforseen Adversaries](https://arxiv.org/abs/1908.08016), doing bi-linear interpolation to resample the colour of each pixel as the colour of a neighbouring pixel. The location to resample from is controlled by a per-pixel displacement vector, which is optimised in the attack.


This attack can be selected with the  `` --attack elastic`` option, and has the following attack-specific hyperparameters:

-  ``--elastic_kernel_size``, a postive odd integer. We pass a gaussian filter over the translation vectors, to nesure local smoothness in the translations.
- ``--elastic_kernel_sigma``, a positive floating point number. The sigma parameter for the gaussian kernel used in the gaussian filter.

## Attack 4: Fractional Brownian Motion
![fbm attack with epsilon = 1](/docs/attack_images/fbm.png)

The attack functions by overlaying several layers of Perlin noise at different frequencies (as described [here](https://thebookofshaders.com/13/)), creating a distinctive noise patern. 
The underlying gradient vectors which generate each instance of the Perlin noise are optimised by the attack. 

This attack can be selected with the  `` --attack fbm`` option.

## Attack 5: FGSM
![fbm attack with epsilon = 1](/docs/attack_images/fgsm.png)

This implements the classis [FGSM attack](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm).

This attack can be selected with the  `` --attack fgsm`` option, and has no attack-specific hyperparameters.

## Attack 6: Fog
![fog attack with epsilon = 1](/docs/attack_images/fog.png)

The attack, which originally appeared in (Testing Robustness Against Unforseen Adversaries)[https://arxiv.org/abs/1908.08016], functions by implementing the [diamond square algorithm](https://en.wikipedia.org/wiki/Diamond-square_algorithm), which is a classic method for generating fractal noise in the computer graphics literature. In the attack we replace the random perturbations added at each step with optimisable displacements.

This attack can be selected with the  ``--attack fog`` option, and has the following attack-specific hyperparameters:

- ``--fog_wibbledecay``, a floating point number in (0,1). This controls the amount of large-scale structure in the fog.

## Attack 7: Gabor noise
![L2 gabor noise attack with epsilon = 40](/docs/attack_images/gabor.png)
This attack, originally implemented in [Testing Robustness Against Unforseen Adversaries](https://arxiv.org/abs/1908.08016),
 functions by generating [gabor noise](https://lirias.kuleuven.be/retrieve/193766) and overlaying it on the image. Gabor noise is a popular
  type of noise used in the graphics literature, which is generated by convolving a gabor kernel over a sparse matrix.
   In the attack we optimise the non-zero values of this sparse matrix.

This attack can be selected with the ``--attack gabor`` option, and has the following attack-specific hyperparameters:

- ``--distance-metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--gabor_kernel_size``, an odd positive integer. Controls the size of the gabor kernels which are passed over the spare matrices.
-  ``--gabor_sides``, a positive integer. The gabor noise is actually created by many gabor filters overlaid with themselves, this controls how many gabor filters are overlaid.
- ``--gabor_weight_density``, a floating point number in (0,1). This is the density of the non-zero entries in the sparse matrix which is being optimised.
- ``--gabor_sigma``, a positive floating point number. This is the sigma parameter passed into the gabor kernels.

## Attack 8: Glitch
![glitch attack with epsilon = 0.1](/docs/attack_images/glitch.png)
This attack functions by greying out the image, splitting it into horizonal bars, and then differentiably shifting each colour channel of the bars.

This attack can be selected with the ``--attack glitch`` option, and has the following attack-specific hyperparameters:

-  ``--glitch_num_lines``, a positive integer. The number of horizontal lines to split the image into.
- ``--glitch_grey_strength``, a floating point number in (0,1). The amount of greying to apply to the image

## Attack 9: HSV
![L_inf HSV attack with epsilon = 0.1](/docs/attack_images/hsv.png)
This attack functions by transforming the image into the [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) colour space, performing a differentiable perturbation, and then translating back. We also smooth out these perturbations by passing a gaussian filter over the perturbation matrix.

This attack can be selected with the ``--attack hsv`` option, and has the following attack-specific hyperparameters:

- ``--distance-metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--hsv_kernel_size``, a positive odd integer. Controls the size of the gaussian kernel.
-  ``--hsv_sigma``,  a positive floating point number. Controls the sigma parameter of the gaussian kernel.

## Attack 10: JPEG
![JPEG attack with epsilon = 0.1](/docs/attack_images/jpeg.png)
This attack functions by transforming to the image into the frequency space
 used by the [JPEG](https://en.wikipedia.org/wiki/JPEG) compression algorithm, differentiably perturbing this space, and transforming back.

This attack can be selected with the ``--attack jpeg`` option.

## Attack 10: Kaleidoscope
![Kaleidoscope attack with epsilon = 0.3](/docs/attack_images/kaleidoscope.png)
This attack functions by placing random shapes onto the image, and then optimising their colours, and optimising the darkness of pixels on the edge of the shapes.

This attack can be selected by with the ``--attack kaleidoscope`` option, and has the following attack-specific hyperparameters:

- ``--kaleidoscope-num-shapes``, a positive integer. This controls the number of shapes to be placed on the image in the kaleidoscope attack.
- ``--kaleidoscope_shape_size``, a positive integer. This controls the size of the shapes in the kaleidoscope attack.
- ``--kaleidoscope_min_value_valence``, a floating point number in (0,1). To make sure the shapes have bright colours, their colour values are initalised in the HSV colour space with a high "valence" value. This controls the minimum valence value.
- ``--kaleidoscope_edge_size``, an odd positive integer. Controls the size of the edges.  


## Attack 11: Klotski

![Klotski attack with epsilon = 0.1](/docs/attack_images/klotski.png)

The klotski attack works by splitting the image into blocks, and then differntiably applying translation vectors to each block.

This attack can be selected with the ``--attack kaleidoscope`` option, and has the following attack-specific hyperparameters:

- ``--kaleidoscope-num-blocks``, a positive integer. This controls how many blocks make up one side of the grid.

## Attack 12: Lighting
![Lighting attack with epsilon = 40](/docs/attack_images/lighting.png)
This is a reimplementation of the attack in [ALA: Adversarial Lightness Attack via Naturalness-aware Regularizations](https://arxiv.org/abs/2201.06070). This attack works by changing the image
 into the [LAB colour space](https://en.wikipedia.org/wiki/CIELAB_color_space), and then performing a differtiably parameterised piece-wise linear transform on the "L" component.

This attack can be selected by passing with the ``--attack lighting`` option, and has the following attack-specific hyperparameters:

- ``--lighting_num_filters``, a positive integer. This controls the number of different piecewise linear sections within the function.
- ``--lighting_loss_function``, one of "candw" or "cross_entropy". The original paper uses the [Carlinin and Wagner](https://arxiv.org/abs/1608.04644) loss function to do the inner PGD optimisation,
 and this option allows choosing between that and the usual cross-entropy loss.

## Attack 13: Mix
![Lighting attack with epsilon = 40](/docs/attack_images/mix.png)
This attack functions by doing differntiable pixel-wise interpolation between the original image and an image of a different class. The level of interpolation at each pixel is optimised, and a gaussian filter is passsed over the pixel interpolation matrix to ensure that the interpolation is locally smooth.

This attack can be selected by passing with the ``--attack mix`` option, and has the following attack-specific hyperparameters:
- ``--distance-metric``, an be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--mix_interp_kernel_size``, an odd integer. The size of the gaussian kernel which is convolved with the optimisation matrix.
-  ``--mix_interp_kernel_size``, a positive floating point number. The sigma parameter for the gaussian kernel which is convolved with the optimisation matrix.

## Attack 14: Projected Gradient Descent
![L_inf pgd attack with epsilon = 0.4](/docs/attack_images/pgd.png)
This is the classic [PGD](https://arxiv.org/abs/1706.06083) attack.

This attack can be selected with the  ``--attack pgd`` option, and has the following attack-specific hyperparameters:

-``--distance-metric`` Can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.

## Attack 15: Pixel
![L_inf Pixel attack with epsilon = 0.4](/docs/attack_images/pixel.png)
This attack works by first taking the image, splitting it into squares, and then recording the average colour within each square.
Then, for each pixel in the square, we differentiably interpolate between the pixels original colour value and the average square value.

This attack can be selected with the  ``--attack pixel`` option, and has the following attack-specific hyperparameters:

- ``--distance-metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--pixel_size``, a positive integer. The size of the pixels in the attack.

## Attack 16: Prison Bars
![Prison attack with epsilon = 0.4](/docs/attack_images/prison.png)
This attack works by putting grey "prison bars" across the image, and then performing a PGD attac only on the pixels on these prison bars.

This attack can be selected with the  ``--attack prison`` option, and has the following attack-specific hyperparameters:

- ``--prison_num_bars``, a positive integer. Controls the number of prison bars in the attack
- ``--prison_bar_width``, a positive integer. Controls the width of the prison bars in the attack.

## Attack 17: Polkadot

![L_inf Pixel attack with epsilon = 0.4](/docs/attack_images/polkadot.png)

The attack works by randomly selecting points on the image to be the centers of the polkadots, assigning a random colour to each
of the centers and then calculating the distance to each center. The colour of each pixel is then updated to the colours of each center,
with centres that are closer to the pixel having a greater influence.

This attack can be selected with the  ``--attack polkadot`` option, and has the following attack-specific hyperparameters:

- ``--distance-metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.

- ``--polkadot_num_polkadots``, a positive integer. Controls the number of polkadots in the attack.

- ``--polkadot_distance_scaling``, a positive floating point number. Higher values of this parameter makes the transition between the colours of the polkadots sharper, and ensures that the polkadots are visible.

- ``--polkadot_image_threshold``, a positive floating point number. This controls the size of the polkadots.

- ``--polkadot_distance_normaliser``, a positive floating point number. This also controls the sizes of the polkadots.

## Attack 18: Snow
![L_2 Snow attack with epsilon = 80](/docs/attack_images/snow.png)

The snow attack, which is a modification of a similar attack implemented in [Testing Robustness Against Unforseen Adversaries](https://arxiv.org/pdf/1908.08016.pdf), works by convolving line-shaped kernels over a
 sparse matrix which has all of its non-zero entries laid out in a regular grid. 
 The convolution outputs a regular grid of snow-flakes, and several of thes grids are overlaid to form a 
 single snowy image. The attack optimises the non-zero entries in the sparse matrix.

This attack can be selected with the  ``--attack snow`` option, and has the following attack-specific hyperparameters:

- ``--distance-metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--snow_flake_size``, a positive odd integer. Controls the length of the snowflakes.

- ``--snow_num_layers``, positive integer. The number of  snow grids to overlay onto the image.

- ``--snow_grid_size``, a positive integer. The spacing between non-zero entries in the snow-flake grid. Corresponds to the spacing between snowflakes.

- ``--snow_init``, a positive float. Used to set the initalisation scale of the snowflakes.

- ``--snow_normalizing_constant``, a positive integer. Increasing this increases how sparse the snowflakes are.

- ``--snow_sigma_range_lower``, a positive float. The thickness of the snowflakes on each grid is chosen uniformly at random, this controls the lower bound of the thickness.

- ``--snow_sigma_range_upper``, a positive float. The thickness of the snowflakes on each grid is chosen uniformly at random, this controls the upper bound of the thickness.

## Attack 19: Texture

![L_2 texture attack with epsilon = 40](/docs/attack_images/texture.png)
The attack functions by passing a [canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) over the image to find all 
the pixels which are at the edges of objects, which are then filled in as black. The other non-edge (or "texture") pixels are  whitened, with a per-pixel variable controlling how white that pixel becomes. 


This attack can be selected with the  ``--attack texture`` option, and has the following attack-specific hyperparameters:

- ``--distance_metric``, can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--texture_threshold``, a floating point number in (0,1). After passing a canny edge filter over an image, we get an "edge score" for each pixel. This decides what the threshold is to consider a pixel an "edge pixel". 

## Attack 20: Whirlpool
![L_2 whirlpool attack with epsilon = 100](/docs/attack_images/whirlpool.png)

This attack functions by translating individual pixels in the image by a function which creates whirpool like displacements (see [here](https://scikit-image.org/docs/stable/auto_examples/transform/plot_swirl.html) for a more detailed description). The attack randomly generates several of these whirlpools, and then optimises their  individual strength

This attack can be selected by passing in ``--attack whirlpool``, and has the following attack-specific hyperparameters:
- ``--distance-metric`` Can be "l2" or "linf". Controls the distance metric used to constrain the optimised variables.
- ``--num_whirlpools`` A Positive integer. Controls the number of whirlpools present in the image.
- ``--whirlpool_radius`` A positive float. Controls the radius of the whirlpools.
- ``--whirlpool_min_strength``A positive float. Whirlpool strength is lower bounded by this value, which is added as a constant offset to the optimsation variables.
  
## Attack 21: Wood
![Wood attack with epsilon = 100](/docs/attack_images/wood.png)

The attack functions by first generating concentric circles on the image by overlaying the function $f(x,y) = sin(\sqrt{x^2 + y^2 } )$ on each pixel (where x,y are the coordinates of the pixel and (0,0) is the center of the image). Then, we add an optimisable perturbation to these (x,y) values.

This attack can be selected by passing in ``--attack wodd``, and has the following attack-specific hyperparameters:
- ``--wood_noise_resolution`` The number of rings overlaid on the image.
- ``--wood_noise_resolution`` To ensure locally smooth perturbations, only the perturbations on a sub-grid of pixels
  are directly controlled by the optimisation algorithm, with the other perturbation values being linearly interpolated
  from this grid. This parameter controls the spacing of this perturbation grid.