"""Image and reward processing for atari

References:
- Implementation of DQN papers
https://github.com/spragunr/deep_q_rl/tree/master/deep_q_rl
- Blog discussing details of DQN Papers
https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
- Pillow Python library
https://pillow.readthedocs.io/en/latest/reference/Image.html
"""
import time
import numpy as np
from PIL import Image
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ImageProcessor:

    def __init__(self, resized_height,
                 resized_width,
                 normalize=False,
                 max_value=255.0):
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.normalize = normalize
        self.max_value = max_value

    def process_frames(self, f1, f2):
        # 1. take maximum pixel values over two frames
        max_frame = np.maximum(f1, f2)
        # 2. resize image
        img = Image.fromarray(max_frame)
        img = img.resize((self.resized_width, self.resized_height))
        # 3. convert image to grey scale
        img = img.convert(mode="L")
        x = np.asarray(img)
        if self.normalize:
            x = x / self.max_value
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
        for i in range(x_stacked.shape[0]):
            self.show_image(x_stacked[i], False)
            time.sleep(0.01)
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


def run_animation(args):
    """To be run on seperate process """
    img_queue, img_dims, img_min, img_max = args
    fig = plt.figure()
    tmp_img = Image.fromarray(np.ones((img_dims)))
    im = plt.imshow(tmp_img, cmap='gray', vmin=0, vmax=255)

    def _anim_init():
        im.set_data(tmp_img)
        return [im]

    def _anim_func(i):
        while img_queue.empty():
            time.sleep(0.1)
        x = Image.fromarray(img_queue.get())
        im.set_array(x)
        img_queue.task_done()
        return [im]

    anim = FuncAnimation(fig,
                         _anim_func,
                         init_func=_anim_init,
                         interval=1,
                         blit=True)
    plt.show()


class ImageAnimation:

    def __init__(self, img_dims=(84, 84), img_min=0, img_max=255):
        self.queue = mp.JoinableQueue()
        self.anim_proc = None
        self.img_dims = img_dims
        self.img_min = img_min
        self.img_max = img_max

    def start(self):
        args = (self.queue, self.img_dims, self.img_min, self.img_max)
        self.anim_proc = mp.Process(target=run_animation,
                                    args=(args,))
        self.anim_proc.start()

    def add_image(self, x):
        self.queue.put(x)

    def stop(self):
        self.queue.join()
        self.anim_proc.join()
