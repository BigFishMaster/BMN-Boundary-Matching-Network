import sys
import time
from bmn.Alimedia import VideoDataSet
from bmn.loss_function import bmn_loss_func, get_mask
import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from bmn import opts
from bmn.models import BMN
import pandas as pd
from bmn.post_processing import post_processing
from bmn.eval import evaluation
from bmn.utils.logging import init_logger, logger, beautify_info
from torch import multiprocessing
sys.dont_write_bytecode = True


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
        train_BMN(train_loader, model, optimizer, epoch, bm_mask, opt)
        with torch.no_grad():
            valid_BMN(valid_loader, model, epoch, bm_mask, opt)
        scheduler.step()


def inference(opt):
    model = BMN(opt)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = opt["test_checkpoint"]
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = DataLoader(VideoDataSet(opt, subset="test"),
                             batch_size=1, shuffle=False,
                             num_workers=opt["num_works"], pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for v_idx, input_data in test_loader:
            if v_idx % 10 == 0:
                logger.info("processing video proposal: {}".format(v_idx[0]))
            video_name = test_loader.dataset.video_data[v_idx[0]]["video_name"]
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


def get_classify(model, opt, video_data, video_proposal, index_queue, result_queue):
    feature_path = opt["feature_path"]
    while True:
        video_name = index_queue.get()
        prop = video_proposal[video_name].copy()
        data = video_data[video_name]
        if feature_path is not None:
            feature_name = os.path.join(feature_path, data["video_feature"].split("/")[-1])
        else:
            feature_name = data["video_feature"]
        feature = np.load(feature_name)
        feature = torch.tensor(feature, dtype=torch.float)
        video_frame = data["sampled_frames"]
        video_second = data["video_seconds"]
        feature_frame = data["feature_frame"]
        corrected_second = float(feature_frame) / video_frame * video_second
        N = feature.shape[0]
        for dic0 in prop:
            score = dic0["score"]
            start = dic0["segment"][0]
            end = dic0["segment"][1]
            start_idx = int(start / corrected_second * N)
            end_idx = int(end / corrected_second * N)
            prop_feature = feature[start_idx:end_idx].contiguous()
            top1_score, top1_label = model.predict(prop_feature)
            dic0["label"] = top1_label
            dic0["score"] = score * top1_score
        result = {video_name: prop}
        result_queue.put(result)


class LinearModel:
    """a linear model with weight (dim x num_labels) and bias (num_labels)."""
    def __init__(self, opt):
        if opt["test_classifier"] is None:
            raise ValueError("test_classifier is None. Please specify a test_classifier")

        weight, bias = torch.load(opt["test_classifier"])
        self.weight = weight.t()
        self.bias = bias

    def predict(self, feat, mode="avg_feature"):
        """
        Args:
            feat: a float tensor, N x dim
            mode: a mode type for the result. it can be "avg_softmax" or "avg_feature".

        Returns:
            result: N x num_labels
        """
        if mode == "avg_feature":
            feat = feat.mean(0)
            result = feat.matmul(self.weight) + self.bias
            result = F.softmax(result, dim=0)
        elif mode == "avg_softmax":
            result = feat.matmul(self.weight) + self.bias
            result = F.softmax(result, dim=1)
            result = result.mean(0)
        else:
            pass
        top1_score, top1_label = torch.topk(result, k=1)
        top1_score = top1_score.item()
        top1_label = top1_label.item()
        return top1_score, top1_label


def classify(opt):
    # parse video info file.
    input = open(opt["test_info"], "r").readlines()
    video_data = {}
    for i, item in enumerate(input):
        json_str = item.strip()
        data = json.loads(json_str)
        video_name = data["video_name"].split(".mp4")[0]
        video_data[video_name] = data
    # parse video proposal file.
    video_proposal = json.load(open(opt["proposal_file"], "r"))
    assert len(video_data) == len(video_proposal), "video info and proposal must have same length."
    video_names = list(video_proposal.keys())

    model = LinearModel(opt)

    ctx = multiprocessing.get_context("spawn")

    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    video_workers = [ctx.Process(target=get_classify,
                                 args=(model, opt, video_data, video_proposal, index_queue, result_queue))
                     for i in range(opt["num_works"])]
    for w in video_workers:
        w.daemon = True
        w.start()

    for key in video_names:
        index_queue.put(key)

    start_time = time.time()
    result = {}
    for i in range(len(video_names)):
        out = result_queue.get()
        result.update(out)
        period = time.time() - start_time
        logger.info("video index: %d, period: %.2f sec, speed: %.2f sec/video."
                    % (i, period, period/(i+1)))
    fout = open(opt["detection_file"], "w")
    json_str = json.dumps(result)
    fout.write(json_str)
    fout.close()


def main():
    opt = opts.parse_opt()
    opt = vars(opt)

    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    init_logger(os.path.join(opt["checkpoint_path"], "bmn.log"))
    logger.info("create configure successfully: {}".format(beautify_info(opt)))

    if opt["mode"] == "train":
        train(opt)
    elif opt["mode"] == "test":
        folder = os.path.dirname(opt["test_checkpoint"])
        opt["result_dir"] = os.path.join(folder, "result/")
        if not os.path.exists(opt["result_dir"]):
            os.makedirs(opt["result_dir"])
        opt["proposal_file"] = os.path.join(folder, "result_proposal.json")
        opt["save_fig_path"] = os.path.join(folder, "result_evaluation.jpg")
        inference(opt)
        post_processing(opt)
        evaluation(opt)
    elif opt["mode"] == "classify":
        folder = os.path.dirname(opt["test_checkpoint"])
        opt["proposal_file"] = os.path.join(folder, "result_proposal.json")
        opt["detection_file"] = os.path.join(folder, "result_detection.json")

        classify(opt)
        evaluation(opt)


if __name__ == '__main__':
    main()
