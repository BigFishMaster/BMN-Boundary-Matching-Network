import os
import torch
import numpy as np
import pandas as pd
from bmn.models import BMN
from bmn.utils.logging import logger
from bmn.Alimedia import VideoDataSet
from torch.utils.data import DataLoader


def inference(opt):
    model = BMN(opt)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    checkpoint_file = opt["test_checkpoint"]
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logger.warn("The checkpoint file {} does not exist.".format(checkpoint_file))
    model.eval()

    test_loader = DataLoader(VideoDataSet(opt, mode=opt["mode"]),
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
                    if start_index < end_index < tscale:
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
