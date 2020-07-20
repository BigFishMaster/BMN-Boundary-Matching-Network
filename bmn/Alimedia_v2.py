# -*- coding: utf-8 -*-
import os
import json
import random
import torch.utils.data as data
import torch
from bmn.utils.misc import ioa_with_anchors, iou_with_anchors
from bmn.utils.logging import logger
import numpy as np


def check_label_is_available(labels, start_idx, end_idx, threshold=0.5):
    result = []
    for label in labels:
        start_label, end_label = label
        int_xmin = np.maximum(start_label, start_idx)
        int_xmax = np.minimum(end_label, end_idx)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        ratio = inter_len / (end_label - start_label)
        if ratio > threshold:
            result.append([int_xmin, int_xmax])
    return result


class VideoDataSet(data.Dataset):
    def __init__(self, opt, mode="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_factor = opt["temporal_factor"] # 1.5
        self.temporal_range = [self.temporal_scale, int(self.temporal_scale * self.temporal_factor)]
        self.temporal_gap = 1. / self.temporal_scale
        self.mode = mode
        self.feature_path = opt["feature_path"]
        # the bmn files are not used when mode = "test" or "submit".
        self.bmn_file = os.path.join(opt["data_folder"], "bmn_{}.txt".format(mode))
        self.info_file = os.path.join(opt["data_folder"], "info_{}.txt".format(mode))
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]
        self.sampled_counter = 0
        self._get_dataset_dict()

    def _get_dataset_dict(self):
        input = open(self.info_file, "r").readlines()
        self.video_data = []
        for i, item in enumerate(input):
            json_str = item.strip()
            data = json.loads(json_str)
            self.video_data.append(data)
        logger.info("In mode {}, length of video data: {}".format(self.mode, len(self.video_data)))
        if self.mode in ["train", "valid"]:
            input = open(self.bmn_file, "r").readlines()
            self.video_bmn = {}
            for i, item in enumerate(input):
                tts = item.strip().split()
                video_name = tts[0]
                self.video_bmn[video_name] = []
                video_seg = tts[1:]
                length = len(video_seg) // 3
                for l in range(length):
                    s = 3 * l
                    e = 3 * (l+1)
                    label, start, end = video_seg[s:e]
                    self.video_bmn[video_name].append((int(label), float(start), float(end)))
            logger.info("In mode {}, length of video info: {}".format(self.mode, len(self.video_bmn)))
            new_video_data = []
            count = 0
            for i in range(len(self.video_data)):
                data = self.video_data[i]
                video_name = data["video_name"]
                if video_name not in self.video_bmn:
                    count += 1
                    continue
                new_video_data.append(data)
            logger.info("In mode {}, there are {} videos which lack video_bmn.".format(self.mode, count))
            self.video_data = new_video_data
        else:
            self.video_bmn = None

    def __getitem__(self, index):
        if self.mode in ["train", "valid"]:
            start, end, confidence, v_start, v_end = self._get_train_label(index, self.anchor_xmin, self.anchor_xmax)
            video_data = self._load_file(index, v_start, v_end)
            return video_data, confidence, start, end
        else:
            video_data = self._load_file(index, -1, -1)
            return index, video_data

    def _load_file(self, index, v_start, v_end):
        video_dict = self.video_data[index]
        if self.feature_path is not None:
            feature_name = os.path.join(self.feature_path, video_dict["video_feature"].split("/")[-1])
        else:
            feature_name = video_dict["video_feature"]

        video_feature = np.load(feature_name)
        if 0 < v_start < v_end:
            self.sampled_counter += 1
            self.sampled_counter = self.sampled_counter % 1000
            if self.sampled_counter == 0:
                logger.info("Sampled video with length: {}, v_start: {}, v_end: {}".format(
                    len(video_feature), v_start, v_end))
            video_feature = video_feature[v_start:v_end]

        video_feature = torch.Tensor(video_feature)
        # N x C -> C x N
        video_feature = torch.transpose(video_feature, 0, 1).unsqueeze(0)
        # B, C, N = video_feature.shape
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
        # feature_frame = feature_shape[0] x 4
        feature_shape = video_dict["feature_shape"]
        video_name = video_dict["video_name"]
        video_labels = self.video_bmn[video_name]
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used

        gt_bbox = []

        flag_is_sampled = False
        flag_can_sample = feature_shape[0] > self.temporal_range[1]
        sampled_video_start = -1
        sampled_video_end = -1
        if flag_can_sample:
            # update labels
            labels = []
            for j in range(len(video_labels)):
                l, s, e = video_labels[j]
                s_idx = int(s / corrected_second * feature_shape[0])
                e_idx = int(e / corrected_second * feature_shape[0])
                labels.append([s_idx, e_idx])

            start_max = feature_shape[0] - self.temporal_range[1]
            num = 10
            result_labels = []
            start_idx = -1
            end_idx = -1
            for _ in range(num):
                start_idx = random.randint(0, start_max-1)
                end_idx = random.randint(start_idx + self.temporal_range[0], start_idx + self.temporal_range[1])
                result_labels = check_label_is_available(labels, start_idx, end_idx)
                if len(result_labels) > 0:
                    break
            if len(result_labels) > 0:
                flag_is_sampled = True
                # get the gt_bbox
                sampled_length = end_idx - start_idx + 1
                sampled_video_start = start_idx
                sampled_video_end = end_idx
                for j in range(len(result_labels)):
                    tmp_info = result_labels[j]
                    tmp_start = max(min(1, (tmp_info[0] - start_idx) / sampled_length), 0)
                    tmp_end = max(min(1, (tmp_info[1] - start_idx) / sampled_length), 0)
                    gt_bbox.append([tmp_start, tmp_end])

        if flag_can_sample is False or flag_is_sampled is False:
            # change the measurement from second to percentage
            gt_bbox = []
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

        return match_score_start, match_score_end, gt_iou_map, sampled_video_start, sampled_video_end

    def __len__(self):
        return len(self.video_data)


if __name__ == '__main__':
    from bmn import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, mode="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=False)
    for i, (video_data, confidence, score_start, score_end) in enumerate(train_loader):
        logger.info("iter: {}".format(i))
        logger.info("video_data: {}".format(video_data.shape))
        logger.info("confidence: {}".format(confidence.shape))
        logger.info("score_start: {}".format(score_start.shape))
        logger.info("score_end: {}".format(score_end.shape))