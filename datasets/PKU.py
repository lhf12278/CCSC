import glob
import re
import os
import copy
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def takeSecond(elem):
    return elem[1]

class PKU(BaseImageDataset):
    """

    """

    dataset_dir = 'PKUSketchRE-ID_V'

    def __init__(self, root='', verbose=True, pid_begin=0, relabel=True, **kwargs):
        super(PKU, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.sketrain_dir = osp.join(self.dataset_dir, 'sketch/')
        self.rgbtrain_dir = osp.join(self.dataset_dir, 'photo/')
        self.query_dir = osp.join(self.dataset_dir, 'query/')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery/')

        self._check_before_run()
        self.pid_begin = pid_begin
        skettrain = self._process_dir(self.sketrain_dir, relabel=True)
        skettrain.sort(key=takeSecond)
        rgbtrain = self._process_dir(self.rgbtrain_dir, relabel=True)
        rgbtrain.sort(key=takeSecond)
        train = copy.deepcopy(skettrain) + copy.deepcopy(rgbtrain)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PKU loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.sketrain = skettrain
        self.rgbtrain = rgbtrain
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.sketrain_dir):
            raise RuntimeError("'{}' is not available".format(self.sketrain_dir))
        if not osp.exists(self.rgbtrain_dir):
            raise RuntimeError("'{}' is not available".format(self.rgbtrain_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(\d+)')
        pid_container = set()
        for img_path in sorted(img_paths, key=lambda x: eval(x.split("/")[-1].split('.')[0])):
            pid = int(pattern.search(img_path).group())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths, key=lambda x: eval(x.split("/")[-1].split('.')[0])):
            pid = int(pattern.search(img_path).group())

            if pid == -1: continue  # junk images are just ignored

            if 'sketch/' in img_path or 'query/' in img_path:
                camid = 0
            else:
                camid = 1

            if relabel:
                pid = pid2label[pid]

            dataset.append([img_path, self.pid_begin + pid, camid, 1])
        return dataset
