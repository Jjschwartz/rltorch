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
import matplotlib.pyplot as plt


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
        x = np.asarray(img)
        return x

    def debug(self, f1, f2):
        raw1 = Image.fromarray(f1)
        raw2 = Image.fromarray(f2)
        processed = self.process_frames(f1, f2)
        processed = Image.fromarray(processed)
        raw1.show("raw1")
        raw2.show("raw2")
        processed.show("processed")

    def show_image(self, x, wait_for_user=True):
        img = Image.fromarray(x)
        img.show()
        if wait_for_user:
            input("Press any key..")

    def show_stacked(self, x_stacked, wait_for_user=True):
        print(x_stacked.shape)
        for i in range(x_stacked.shape[1]):
            self.show_image(x_stacked[0][i], False)
        if wait_for_user:
            input("Press any key..")


class ImageHistory:

    def __init__(self, history_length, img_dims):
        self.length = history_length
        self.img_dims = img_dims
        self.history = np.empty((history_length, *img_dims), dtype=np.float32)
        self.size, self.ptr = 0, 0

    def push(self, x):
        self.history[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.length
        self.size = min(self.size+1, self.length)

    def get(self):
        assert self.size == self.length
        # must add 1 for N dim for DQN
        history_buffer = np.empty((1, self.length, *self.img_dims),
                                  dtype=np.float32)
        history_buffer[0][:self.length-self.ptr] = self.history[self.ptr:]
        history_buffer[0][self.length-self.ptr:] = self.history[:self.ptr]
        return history_buffer

    def clear(self):
        self.size, self.ptr = 0, 0
