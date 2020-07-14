import sys
from bmn.Alimedia import VideoDataSet
from bmn.loss_function import bmn_loss_func, get_mask
import os
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from bmn import opts
from bmn.models import BMN
import pandas as pd
from bmn.post_processing import BMN_post_processing
from bmn.eval import evaluation_proposal

sys.dont_write_bytecode = True


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
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

        if (n_iter + 1) % 10 == 0:
            print(
                "BMN training loss(epoch %d): tem_loss: %.03f, "
                "pem class_loss: %.03f, pem reg_loss: %.03f, "
                "total_loss: %.03f" % (
                    epoch, epoch_tem_loss / (n_iter + 1),
                    epoch_pemclr_loss / (n_iter + 1),
                    epoch_pemreg_loss / (n_iter + 1),
                    epoch_loss / (n_iter + 1)))


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

        if (n_iter + 1) % 10 == 0:
            print(
                "BMN training loss(epoch %d): tem_loss: %.03f, "
                "pem class_loss: %.03f, pem reg_loss: %.03f, "
                "total_loss: %.03f" % (
                    epoch, epoch_tem_loss / (n_iter + 1),
                    epoch_pemclr_loss / (n_iter + 1),
                    epoch_pemreg_loss / (n_iter + 1),
                    epoch_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, os.path.join(opt["checkpoint_path"], "BMN_checkpoint_epoch_{}.pth.tar".format(epoch+1)))
    if epoch_loss < opt["best_loss"]:
        opt["best_loss"] = epoch_loss
        torch.save(state, os.path.join(opt["checkpoint_path"], "BMN_best.pth.tar"))


def BMN_Train(opt):
    model = BMN(opt)
    # use all gpus
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = DataLoader(VideoDataSet(opt, subset="train"),
                              batch_size=opt["batch_size"], shuffle=True,
                              num_workers=opt["num_works"], pin_memory=True, drop_last=True)

    valid_loader = DataLoader(VideoDataSet(opt, subset="valid"),
                              batch_size=opt["batch_size"], shuffle=False,
                              num_workers=opt["num_works"], pin_memory=True, drop_last=False)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"],
                                                gamma=opt["step_gamma"])
    opt["best_loss"] = 1e10
    bm_mask = get_mask(opt["temporal_scale"])
    if torch.cuda.is_available():
        bm_mask = bm_mask.cuda()
    for epoch in range(opt["train_epochs"]):
        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        with torch.no_grad():
            valid_BMN(valid_loader, model, epoch, bm_mask, opt)
        scheduler.step()


def BMN_inference(opt):
    model = BMN(opt)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = os.path.join(opt["checkpoint_path"], "BMN_best.pth.tar")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = DataLoader(VideoDataSet(opt, subset="test"),
                             batch_size=1, shuffle=False,
                             num_workers=opt["num_works"], pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_data[idx[0]]["video_name"]
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and end_index < tscale:
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_score", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv(opt["result_dir"] + video_name + ".csv", index=False)


def main():
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    if opt["mode"] == "train":
        BMN_Train(opt)
    elif opt["mode"] == "test":
        if not os.path.exists(opt["result_dir"]):
            os.makedirs(opt["result_dir"])
        BMN_inference(opt)
        BMN_post_processing(opt)
        evaluation_proposal(opt)


if __name__ == '__main__':
    main()
