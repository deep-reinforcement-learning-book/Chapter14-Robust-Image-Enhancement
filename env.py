import random

import numpy as np
from PIL import Image, ImageEnhance
from skimage.color import rgb2lab

import tensorflow as tf

backbone = tf.keras.applications.ResNet50(include_top=False, pooling='avg')
preprocess = tf.keras.applications.resnet50.preprocess_input


class Env(object):
    """Training env wrapper of image processing RL problem"""
    def __init__(self, src, max_episode_length=20, reward_scale=0.1):
        """
        Args:
            src (list[str, str]): list of raw and retouched path, initial
                                  state will sample from it uniformly
            max_episode_length (int): max number of actions can be taken
        """
        self._src = src
        self._backbone = backbone
        self._preprocess = preprocess
        self._rgb_state = None
        self._lab_state = None
        self._target_lab = None
        self._current_diff = None
        self._count = 0
        self._max_episode_length = max_episode_length
        self._reward_scale = reward_scale
        self._info = dict()

    def _state_feature(self):
        s = self._preprocess(self._rgb_state)
        s = tf.expand_dims(s, axis=0)
        context = self._backbone(s).numpy().astype('float32')
        hist_rgb = get_rgb_hist(self._rgb_state).astype('float32')
        hist_lab = get_lab_hist(self._lab_state).astype('float32')
        return np.concatenate([context, hist_rgb, hist_lab], 1)

    def step(self, action):
        """One step"""
        self._count += 1
        self._rgb_state = self._transit(action)
        self._lab_state = rgb2lab(self._rgb_state)
        reward = self._reward()
        done = self._count >= self._max_episode_length or action == 0
        return self._state_feature(), reward, done, self._info

    def reset(self):
        """Reset"""
        self._count = 0
        raw, retouched = map(Image.open, random.choice(self._src))
        self._rgb_state = np.asarray(raw)
        self._lab_state = rgb2lab(self._rgb_state)
        self._target_lab = rgb2lab(np.asarray(retouched))
        self._current_diff = self._diff(self._lab_state)
        self._info['max_reward'] = self._current_diff
        return self._state_feature()

    def _diff(self, lab):
        reward = np.sqrt(((self._target_lab - lab) ** 2).sum(2)).mean()
        return reward * self._reward_scale

    def _reward(self):
        diff = self._current_diff
        self._current_diff = self._diff(self._lab_state)
        return min(max(diff - self._current_diff, -1), 1)  # reward clip

    def _transit(self, a):
        s = self._rgb_state.copy()
        if a == 0:
            return s
        elif a == 1:
            return np.asarray(adjust_contrast(Image.fromarray(s), 0.95))
        elif a == 2:
            return np.asarray(adjust_contrast(Image.fromarray(s), 1.05))
        elif a == 3:
            return np.asarray(adjust_saturation(Image.fromarray(s), 0.95))
        elif a == 4:
            return np.asarray(adjust_saturation(Image.fromarray(s), 1.05))
        elif a == 5:
            return np.asarray(adjust_brightness(Image.fromarray(s), 0.95))
        elif a == 6:
            return np.asarray(adjust_brightness(Image.fromarray(s), 1.05))
        elif a == 7:
            return adjust_channels(s, 0.95, [0, 1])
        elif a == 8:
            return adjust_channels(s, 1.05, [0, 1])
        elif a == 9:
            return adjust_channels(s, 0.95, [2, 1])
        elif a == 10:
            return adjust_channels(s, 1.05, [2, 1])
        elif a == 11:
            return adjust_channels(s, 0.95, [0, 2])
        elif a == 12:
            return adjust_channels(s, 1.05, [0, 2])
        else:
            raise NotImplementedError


def get_lab_hist(lab):
    """Get hist of lab image"""
    lab = lab.reshape(-1, 3)
    hist, _ = np.histogramdd(lab, bins=(10, 10, 10),
                             range=((0, 100), (-60, 60), (-60, 60)))
    return hist.reshape(1, 1000) / 1000.0


def get_rgb_hist(lab):
    """Get hist of lab image"""
    lab = lab.reshape(-1, 3)
    hist, _ = np.histogramdd(lab, bins=(10, 10, 10),
                             range=((0, 255), (0, 255), (0, 255)))
    return hist.reshape(1, 1000) / 1000.0


def adjust_contrast(image_rgb, contrast_factor):
    """Adjust contrast"""
    enhancer = ImageEnhance.Contrast(image_rgb)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(image_rgb, saturation_factor):
    """Adjust saturation"""
    enhancer = ImageEnhance.Color(image_rgb)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_brightness(image_rgb, brightness_factor):
    """Adjust brightness"""
    enhancer = ImageEnhance.Brightness(image_rgb)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_channels(array, factor, channels):
    """Adjust channel pixel value"""
    for c in channels:
        array[:, :, c] = (array[:, :, c] * factor).clip(0, 255).astype('uint8')
    return array

