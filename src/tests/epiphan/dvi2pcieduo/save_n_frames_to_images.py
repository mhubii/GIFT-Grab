#!/usr/bin/env python3

from time import sleep

import scipy.misc
import numpy as np

from pygiftgrab import IObserver
from pygiftgrab import VideoSourceFactory
from pygiftgrab import ColourSpace, Device


class BgraFrameSaver(IObserver):

    def __init__(self, max_num_frames):
        super(BgraFrameSaver, self).__init__()
        self.max_num_to_save = max_num_frames
        self.num_saved = 0

    def update(self, frame):
        if self.num_saved >= self.max_num_to_save:
            return

        data_np = np.copy(frame.data(True))

        scipy.misc.imsave(f'snapshot-{self.num_saved}.png', data_np)
        self.num_saved += 1


if __name__ == '__main__':
    sfac = VideoSourceFactory.get_instance()
    epiphan = sfac.get_device(Device.DVI2PCIeDuo_DVI, ColourSpace.BGRA)
    saver = BgraFrameSaver(3)

    epiphan.attach(saver)

    sleep(10)  # operate pipeline for this duration

    epiphan.detach(saver)