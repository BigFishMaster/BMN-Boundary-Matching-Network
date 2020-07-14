# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import torch.utils.data as data
import torch
from bmn.utils import ioa_with_anchors, iou_with_anchors
from bmn.logging import logger

class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.info_file = opt[subset+"_bmn"]
        self.data_file = opt[subset+"_info"]
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

        self._get_dataset_dict()

    def _get_dataset_dict(self):
        input = open(self.data_file, "r").readlines()
        self.video_data = []
        for i, item in enumerate(input):
            json_str = item.strip()
            data = json.loads(json_str)
            self.video_data.append(data)
        logger.info("In subset {}, length of video data: {}".format(self.subset, len(self.video_data)))
        if self.subset in ["train", "valid"]:
            input = open(self.info_file, "r").readlines()
            self.video_info = {}
            #self.video_info = self.m.dict()
            for i, item in enumerate(input):
                tts = item.strip().split()
                video_name = tts[0]
                self.video_info[video_name] = []
                #self.video_info[video_name] = self.m.list()
                video_seg = tts[1:]
                length = len(video_seg) // 3
                for l in range(length):
                    s = 3 * l
                    e = 3 * (l+1)
                    label, start, end = video_seg[s:e]
                    self.video_info[video_name].append((int(label), float(start), float(end)))
            logger.info("In subset {}, length of video info: {}".format(self.subset, len(self.video_info)))
            new_video_data = []
            #new_video_data = self.m.list()
            count = 0
            for i in range(len(self.video_data)):
                data = self.video_data[i]
                video_name = data["video_name"]
                if video_name not in self.video_info:
                    count += 1
                    continue
                new_video_data.append(data)
            logger.info("In subset {}, there are {} videos which lack video_info.".format(self.subset, count))
            self.video_data = new_video_data
        else:
            self.video_info = None

    def __getitem__(self, index):
        video_data = self._load_file(index)
        if self.mode == "train":
            score_start, score_end, confidence_score = \
                self._get_train_label(index, self.anchor_xmin, self.anchor_xmax)
            return video_data, confidence_score, score_start, score_end
        else:
            return index, video_data

    def _load_file(self, index):
        video_dict = self.video_data[index]
        if self.feature_path is not None:
            feature_name = os.path.join(self.feature_path, video_dict["video_feature"].split("/")[-1])
        else:
            feature_name = video_dict["video_feature"]

        video_feature = np.load(feature_name)
        video_feature = torch.Tensor(video_feature)
        # N x C -> C x N
        video_feature = torch.transpose(video_feature, 0, 1).unsqueeze(0)
        #B, C, N = video_feature.shape
        video_feature = torch.nn.functional.interpolate(video_feature, size=self.temporal_scale, mode="linear",
                                                        align_corners=False)
        video_feature = video_feature.squeeze(0)
        return video_feature

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        """ video_dict contains a dict with:
            {"video_frames": 3151, "video_fps": 15.0, "target_sampling_rate": 5, "sampled_frames": 631,
            "original_frames": 3151, "video_seconds": 210.06666666666666,
            "video_name": "02a8ca381d08097b7a25420327832b63.mp4", "start_seconds": -1.0, "end_seconds": -1.0,
            "video_label": -1, "feature_shape": [157, 2304], "feature_frame": 628,
            "video_feature": "./models/07/output/02a8ca381d08097b7a25420327832b63.mp4.feat.npy", "step_frames": 128}
         """
        video_dict = self.video_data[index]
        # video_frame: the sampled frames from the video by target_sampling rate.
        video_frame = video_dict["sampled_frames"]
        video_second = video_dict["video_seconds"]
        # feature_frame: the video frames used to extract video features.
        feature_frame = video_dict["feature_frame"]
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used

        video_name = video_dict["video_name"]
        video_labels = self.video_info[video_name]

        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info[1] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info[2] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            # e.g., calculate the maximum overlap ratio between [-0.05, 0.05] and all start points of GT boxes.
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            # e.g., calculate the maximum overlap ratio between [-0.05, 0.05] and all end points of GT boxes.
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_data)


if __name__ == '__main__':
    from bmn import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=False)
    for i, (video_data, confidence, score_start, score_end) in enumerate(train_loader):
        logger.info("iter: {}".format(i))
        logger.info("video_data: {}".format(video_data.shape))
        logger.info("confidence: {}".format(confidence.shape))
        logger.info("score_start: {}".format(score_start.shape))
        logger.info("score_end: {}".format(score_end.shape))