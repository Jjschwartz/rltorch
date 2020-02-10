"""Image and reward processing for atari

References:
- Implementation of DQN papers
https://github.com/spragunr/deep_q_rl/tree/master/deep_q_rl
- Blog discussing details of DQN Papers
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
- Pillow Python library
https://pillow.readthedocs.io/en/latest/reference/Image.html
"""
import numpy as np
from PIL import Image


class ImageProcessor:

    def __init__(self, resized_height, resized_width):
        self.resized_height = resized_height
        self.resized_width = resized_width

    def process_frames(self, f1, f2):
        # 1. take maximum pixel values over two frames
        max_frame = np.maximum(f1, f2)
        # 2. resize image
        img = Image.fromarray(max_frame)
        img = img.resize((self.resized_width, self.resized_height))
        # 3. convert image to grey scale
        img = img.convert(mode="L")
        return np.asarray(img)

    def debug(self, f1, f2):
        raw1 = Image.fromarray(f1)
        raw2 = Image.fromarray(f2)
        processed = self.process_frames(f1, f2)
        processed = Image.fromarray(processed)
        raw1.show("raw1")
        raw2.show("raw2")
        processed.show("processed")


def clip_reward(r, r_neg=-1, r_pos=1):
    if r == 0:
        return r
    if r < 0:
        return max(r, r_neg)
    return min(r, r_pos)
