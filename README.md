# Robust Image Enhancement with DRL

## Description:
Example of image enhancement with Proximal Policy Optimization (PPO) algorithm, using MIT-Adobe FiveK dataset. The environment is wrapped into OpenAI Gym format.

## Dependencies:
* tensorflow >= 2.0.0
* scikit-image >= 0.16.2
* Pillow >= 6.2.1
* tqdm >= 4.36.0

## Usages:
1. Download [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek).
2. Process raw images and retouced images to JPEG format with quality 100 and color space sRGB by Adobe Lightroom.
3. Resize images so that  the maximal side consists of 512 pixels.
4. Split dataset into training and validation, and generate `train_pairs` and `valid_pairs` files where each line consists of `{raw_image_path}\t{retouched_image_path}`.
5. Run `python train.py --mode=train` with specific args.
