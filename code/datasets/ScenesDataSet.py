from datasets import SceneData
import numpy as np


class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.n = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.n / self.batch_size))
        self.shuffle=shuffle
        self.permutation = self.init_permutation()
        self.current_batch = 0
        self.device = 'cpu'

    def init_permutation(self):
        return np.random.permutation(self.n) if self.shuffle else np.arange(self.n)

    def __iter__(self):
        self.current_batch = 0
        self.permutation = self.init_permutation()
        return self

    def __next__(self):
        if self.current_batch == self.num_batches:
            raise StopIteration
        start_ind = self.current_batch*self.batch_size
        end_ind = min((self.current_batch+1)*self.batch_size, self.n)
        current_indices = self.permutation[start_ind:end_ind]
        self.current_batch += 1
        return [self.dataset[i].to(self.device) for i in current_indices]

    def __len__(self):
        return self.n

    def to(self, device, **kwargs):
        self.device = device
        return self


class ScenesDataSet:
    def __init__(self, data_list, return_all, min_sample_size=10, max_sample_size=30):
        super().__init__()
        self.data_list = data_list
        self.return_all = return_all
        self.min_sample_size = min_sample_size
        self.max_sample_size = max_sample_size

    def __getitem__(self, item):
        current_data = self.data_list[item]
        if self.return_all:
            return current_data
        else:
            max_sample = min(self.max_sample_size, len(current_data.y))
            if self.min_sample_size >= max_sample:
                sample_fraction = max_sample
            else:
                sample_fraction = np.random.randint(self.min_sample_size, max_sample + 1)
            return SceneData.sample_data(current_data, sample_fraction)

    def __len__(self):
        return len(self.data_list)


