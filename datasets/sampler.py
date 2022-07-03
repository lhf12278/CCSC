from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import torch.utils.data as data
import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_sketsource,data_rgbsource, batch_size, num_instances):
        self.data_rgbsource = data_rgbsource
        self.data_sketsource = data_sketsource
        self.batch_size = batch_size
        self.num_instances = num_instances

        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.rgbindex_dic = defaultdict(list)  # dict with list value
        self.sketindex_dic = defaultdict(list)
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_rgbsource):
            self.rgbindex_dic[pid].append(index)
        self.rgbpids = list(self.rgbindex_dic.keys())

        for index, (_, pid, _, _) in enumerate(self.data_sketsource):
            self.sketindex_dic[pid].append(index)
        self.sketpids = list(self.sketindex_dic.keys())

        # estimate number of examples in an epoch
        self.rgblength = 0
        for pid in self.rgbpids:
            idxs = self.rgbindex_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.rgblength += num - num % self.num_instances

        self.sketlength = 0
        for pid in self.sketpids:
            idxs = self.sketindex_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.sketlength += num - num % self.num_instances

        self.length = np.maximum(len(self.data_rgbsource),len(self.data_sketsource))


        batch_rgbidxs_dict = defaultdict(list)
        batch_sketidxs_dict = defaultdict(list)

        for pid in self.rgbpids:
            idxs = copy.deepcopy(self.rgbindex_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_rgbidxs = []
            for idx in idxs:
                batch_rgbidxs.append(idx)
                if len(batch_rgbidxs) == self.num_instances:
                    batch_rgbidxs_dict[pid].append(batch_rgbidxs)
                    batch_rgbidxs = []

        for pid in self.sketpids:
            idxs = copy.deepcopy(self.sketindex_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_sketidxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.sketpids)
        rgbfinal_idxs = []
        sketfinal_idxs = []


        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_rgbidxs = batch_rgbidxs_dict[pid].pop(0)
                rgbfinal_idxs.extend(batch_rgbidxs)

                batch_sketidxs = batch_sketidxs_dict[pid].pop(0)
                sketfinal_idxs.extend(batch_sketidxs)

                if len(batch_rgbidxs_dict[pid]) == 0 or len(batch_sketidxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.rgbfinal_idxs = rgbfinal_idxs
        self.sketfinal_idxs = sketfinal_idxs

    def __iter__(self):

        return iter(self.sketfinal_idxs)

    def __len__(self):
        return self.length


class Seeds:

    def __init__(self, seeds):
        self.index = -1
        self.seeds = seeds

    def next_one(self):
        self.index += 1
        if self.index > len(self.seeds) - 1:
            self.index = 0
        return self.seeds[self.index]


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class UniformSampler(data.sampler.Sampler):

    def __init__(self, dataset, k, random_seeds):

        self.dataset = dataset
        self.k = k
        self.random_seeds = random_seeds

        self._process()

        self.sample_list = self._generate_list()


    def __iter__(self):
        self.sample_list = self._generate_list()
        return iter(self.sample_list)


    def __len__(self):
        return len(self.sample_list)


    def _process(self):
        pids, cids = [], []
        for sample in self.dataset:
            _, pid, cid, _ = sample
            pids.append(pid)
            cids.append(cid)

        self.pids = np.array(pids)
        self.cids = np.array(cids)


    def _generate_list(self):

        index_list = []
        pids = list(set(self.pids))
        pids.sort()

        seed = self.random_seeds.next_one()
        random.seed(seed)
        random.shuffle(pids)



        for pid in pids:
            # find all indexes of the person of pid
            index_of_pid = np.where(self.pids == pid)[0]
            # randomly sample k images from the pid
            if len(index_of_pid) >= self.k:
                index_list.extend(np.random.choice(index_of_pid, self.k, replace=False).tolist())
            else:
                index_list.extend(np.random.choice(index_of_pid, self.k, replace=True).tolist())

        return index_list