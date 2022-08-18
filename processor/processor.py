import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def do_train(cfg,
             model,
             rgb_train_iter,sketch_train_iter,

             gallery_loader,
             query_loader,
             optimizer,

             scheduler,

             cls, triploss,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)


    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc1_meter.reset()
        acc2_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter in range(cfg.SOLVER.TOTAL_STEP):
            rgbimg, rgbvid, _ = rgb_train_iter.next_one()
            sketimg, sketvid, _ = sketch_train_iter.next_one()
            assert torch.equal(rgbvid, sketvid)
            optimizer.zero_grad()

            img1 = rgbimg.to(device)
            target1 = rgbvid.to(device)

            img2 = sketimg.to(device)
            target2 = sketvid.to(device)

            with amp.autocast(enabled=True):
                if epoch >50:
                    score3, feat3, score4, feat4, align_loss = model(img1, img2, target1, target2, modal=3, epoch=epoch)
                else:
                    score3, feat3, score4, feat4 = model(img1, img2, target1, target2, modal=3, epoch=epoch)


                rgb_cls_loss2 = cls(score3, target1)
                ske_cls_loss2 = cls(score4, target2)
                cls_loss = (rgb_cls_loss2 + ske_cls_loss2) / 2.0

                triplet_loss_1 = triploss(feat3, feat4, feat4, target1, target2, target2)
                triplet_loss_2 = triploss(feat4, feat3, feat3, target2, target1, target1)
                triplet_loss = (triplet_loss_1 + triplet_loss_2) / 2.0

                if epoch >50:
                    loss1 = cls_loss + triplet_loss + align_loss
                else:
                    loss1 = cls_loss + triplet_loss

            scaler.scale(loss1).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch <= 50:
                with amp.autocast(enabled=True):
                    score1, feat1, score2, feat2 = model(img1, img2, target1, target2, modal=4)

                    rgb_cls_loss1 = cls(score1, target1)
                    ske_cls_loss1 = cls(score2, target2)
                    cls_loss = (rgb_cls_loss1 + ske_cls_loss1) / 2.0

                    triplet_loss_1 = triploss(feat1, feat2, feat2, target1, target2, target2)
                    triplet_loss_2 = triploss(feat2, feat1, feat1, target2, target1, target1)
                    triplet_loss = (triplet_loss_1 + triplet_loss_2) / 2.0

                    loss2 = cls_loss + triplet_loss

                scaler.scale(loss2).backward()
                scaler.step(optimizer)
                scaler.update()


            loss=loss1+loss2


            acc1 = (score3.max(1)[1] == target1).float().mean()
            acc2 = (score4.max(1)[1] == target2).float().mean()



            loss_meter.update(loss.item(), 2*img1.shape[0])
            acc1_meter.update(acc1, 1)
            acc2_meter.update(acc2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:

                logger.info("Epoch[{}]  Loss: {:.3f}, Acc: {:.3f}  {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, loss_meter.avg, acc1_meter.avg, acc2_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, 2*cfg.SOLVER. IMS_PER_BATCH / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 and epoch >= 50:
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                model.eval()
                for n_iter, (img, vid, camid) in enumerate(gallery_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img, img, modal=1)
                        evaluator.g_update((feat, vid, camid))

                model.eval()
                for n_iter, (img, vid, camid) in enumerate(query_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img, img, modal=2)
                        evaluator.q_update((feat, vid, camid))

                cmc, mAP, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()







def do_inference(cfg,
                 model,
                 gallery_loader,
                 query_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for n_iter, (img, vid, camid) in enumerate(gallery_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img, img, modal=1)
            evaluator.g_update((feat, vid, camid))

    model.eval()
    for n_iter, (img, vid, camid) in enumerate(query_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img, img, modal=2)
            evaluator.q_update((feat, vid, camid))

    cmc, mAP, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


