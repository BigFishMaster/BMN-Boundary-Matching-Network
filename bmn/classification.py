import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import multiprocessing
from bmn.utils.logging import logger


class LinearModel:
    """a linear model with weight (dim x num_labels) and bias (num_labels)."""
    def __init__(self, opt):
        if opt["test_classifier"] is None:
            raise ValueError("test_classifier is None. Please specify a test_classifier")

        weight, bias = torch.load(opt["test_classifier"])
        self.weight = weight.t()
        self.bias = bias

        if opt["test_accum_feature"] is not None:
            self.accum_feature = opt["test_accum_feature"]
        else:
            self.accum_feature = None

        logger.info("In Linear model, weight: {}, bias: {}, accum: {}".format(
            self.weight.shape, self.bias.shape, self.accum_feature))

    def _accum(self, feat):
        N = feat.shape[0]
        for i in range(N):
            s = i
            e = min(s+self.accum_feature, N)
            feat[i] = feat[s:e].mean(0)
        return feat

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
            if self.accum_feature is not None:
                feat = self._accum(feat)
            result = feat.matmul(self.weight) + self.bias
            result = F.softmax(result, dim=1)
            result = result.mean(0)
        else:
            pass
        top1_score, top1_label = torch.topk(result, k=1)
        top1_score = top1_score.item()
        top1_label = top1_label.item()
        return top1_score, top1_label


def get_classify(model, opt, video_data, video_proposal, index_queue, result_queue):
    feature_path = opt["feature_path"]
    mode = opt["mode"]
    if mode == "submit":
        # convert label to name
        label_file = opt["submit_label_file"]
        label2name = {i: line.strip() for i, line in enumerate(open(label_file, "r", encoding="utf8").readlines())}
    else:
        label2name = None

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
            top1_score, top1_label = model.predict(prop_feature, opt["classifier_type"])
            if label2name is not None:
                top1_label = label2name[top1_label]
            dic0["label"] = top1_label
            dic0["score"] = score * top1_score
        result = {video_name: prop}
        result_queue.put(result)


def classify(opt):
    # parse video info file.
    mode = opt["mode"]
    input = open(os.path.join(opt["data_folder"], "info_{}.txt".format(opt["mode"]))).readlines()
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
