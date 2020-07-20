import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from bmn.models import BMN
from bmn.Alimedia_v2 import VideoDataSet
from bmn.loss_function import bmn_loss_func, get_mask
from bmn.utils.logging import logger


def train_BMN(data_loader, model, optimizer, epoch, bm_mask, opt):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            label_start = label_start.cuda()
            label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        if (n_iter + 1) % opt["log_steps"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info("BMN training loss(epoch %d): tem_loss: %.03f, "
                        "pem class_loss: %.03f, pem reg_loss: %.03f, "
                        "total_loss: %.03f, lr: %f" % (
                            epoch+1, epoch_tem_loss / (n_iter + 1),
                            epoch_pemclr_loss / (n_iter + 1),
                            epoch_pemreg_loss / (n_iter + 1),
                            epoch_loss / (n_iter + 1),
                            lr))


def valid_BMN(data_loader, model, epoch, bm_mask, opt):
    model.eval()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        if torch.cuda.is_available():
            input_data = input_data.cuda()
            label_start = label_start.cuda()
            label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

        if (n_iter + 1) % opt["log_steps"] == 0:
            logger.info("BMN valid loss(epoch %d): tem_loss: %.03f, "
                        "pem class_loss: %.03f, pem reg_loss: %.03f, "
                        "total_loss: %.03f" % (
                            epoch+1, epoch_tem_loss / (n_iter + 1),
                            epoch_pemclr_loss / (n_iter + 1),
                            epoch_pemreg_loss / (n_iter + 1),
                            epoch_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, os.path.join(opt["checkpoint_path"], "checkpoint_{}.pth.tar".format(epoch+1)))
    logger.info("save checkpoint: epoch {}.".format(epoch+1))
    if epoch_loss < opt["best_loss"]:
        opt["best_loss"] = epoch_loss
        torch.save(state, os.path.join(opt["checkpoint_path"], "best_checkpoint.pth.tar"))
        logger.info("save best checkpoint: epoch {}.".format(epoch+1))


def train(opt):
    model = BMN(opt)
    # use all gpus
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    if opt["optimizer"] == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt["training_lr"],
                               weight_decay=opt["weight_decay"])
    elif opt["optimizer"] == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt["training_lr"], momentum=0.9,
                              weight_decay=opt["weight_decay"])
    else:
        raise NotImplementedError("The optimizer is not support:{}".format(opt["optimizer"]))

    train_loader = DataLoader(VideoDataSet(opt, mode="train"),
                              batch_size=opt["batch_size"], shuffle=True,
                              num_workers=opt["num_works"], pin_memory=True, drop_last=True)

    valid_loader = DataLoader(VideoDataSet(opt, mode="valid"),
                              batch_size=opt["batch_size"], shuffle=False,
                              num_workers=opt["num_works"], pin_memory=True, drop_last=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"],
                                                gamma=opt["step_gamma"])
    opt["best_loss"] = 1e10
    bm_mask = get_mask(opt["temporal_scale"])
    if torch.cuda.is_available():
        bm_mask = bm_mask.cuda()
    for epoch in range(opt["train_epochs"]):
        train_BMN(train_loader, model, optimizer, epoch, bm_mask, opt)
        with torch.no_grad():
            valid_BMN(valid_loader, model, epoch, bm_mask, opt)
        scheduler.step()
