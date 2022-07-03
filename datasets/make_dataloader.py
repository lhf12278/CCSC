import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import copy
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import Seeds,IterLoader,UniformSampler
from .PKU import PKU
from .ShoeV2 import Shoe
from .ChairV2 import Chair



__factory = {
    'PKU': PKU,
    'Shoe': Shoe,
    'Chair': Chair,
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)


    seeds = Seeds(np.random.randint(0, 1e8, 9999))


    rgb_train_set = ImageDataset(dataset.rgbtrain, train_transforms)
    sketch_train_set = ImageDataset(dataset.sketrain,  train_transforms)

    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:



        rgb_train_loader = DataLoader(
            rgb_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=UniformSampler(dataset.rgbtrain,  cfg.DATALOADER.NUM_INSTANCE,copy.copy(seeds)),
            num_workers=num_workers
        )

        sketch_train_loader = DataLoader(
            sketch_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=UniformSampler(dataset.sketrain,cfg.DATALOADER.NUM_INSTANCE,copy.copy(seeds)),
            num_workers=num_workers
        )



        rgb_train_iter = IterLoader(rgb_train_loader)
        sketch_train_iter = IterLoader(sketch_train_loader)

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    query_set = ImageDataset(dataset.query, val_transforms)

    gallery_loader = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )
    query_loader = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return rgb_train_iter, sketch_train_iter, train_loader_normal, gallery_loader, query_loader, len(dataset.query), num_classes
