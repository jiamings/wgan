import os
import numpy as np


class DataSampler(object):
    def __init__(self):
        self.shape = [64, 64, 3]
        self.name = "lsun"
        self.db_path = "/media/jiaming/data/data/lsun/bedroom"
        self.db_files = os.listdir(self.db_path)
        self.cur_batch_ptr = 0
        self.cur_batch = self.load_new_data()
        self.train_batch_ptr = 0
        self.train_size = len(self.db_files) * 10000
        self.test_size = self.train_size

    def load_new_data(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        if self.cur_batch_ptr == len(self.db_files):
            self.cur_batch_ptr = 0
        return np.load(filename) * 2.0 - 1.0

    def __call__(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch.shape[0]:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch = self.load_new_data()
        x = self.cur_batch[prev_batch_ptr:self.train_batch_ptr, :, :, :]
        return np.reshape(x, [batch_size, -1])

    def data2img(self, data):
        rescaled = np.divide(data + 1.0, 2.0)
        return np.reshape(np.clip(rescaled, 0.0, 1.0), [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])