from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from loss.cross_modal_loss import CrossEntropyLabelSmooth, TripletLoss
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config_file", default="configs/transformerPKU.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    rgb_train_iter, sketch_train_iter, train_loader_normal, gallery_loader, query_loader, num_query, num_classes = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes)

    cls = CrossEntropyLabelSmooth(num_classes)
    triploss = TripletLoss(0.3, 'euclidean')

    optimizer = make_optimizer(cfg, model)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        rgb_train_iter,sketch_train_iter,
        gallery_loader,
        query_loader,
        optimizer,
        scheduler,
        cls, triploss,
        num_query, args.local_rank
    )
