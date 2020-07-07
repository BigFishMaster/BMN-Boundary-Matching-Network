import math
import numpy as np
import torch
import torch.nn as nn


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        # time scale: 100
        self.tscale = opt["temporal_scale"]
        # boundary ratio: 0.5
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]
        # num sample(N): 32
        self.num_sample = opt["num_sample"]
        # num in each bin: 1
        self.num_sample_perbin = opt["num_sample_perbin"]
        # feature dim: 400
        self.feat_dim=opt["feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # net: conv1d->relu->conv1d->relu
        #      feat dim -> hidden_dim_1d -> relu -> hidden_dim_1d -> relu
        base_feature = self.x_1d_b(x)
        # net: conv1d -> relu -> conv1d -> sigmoid
        #      hidden_dim_1d -> hidden_dim_1d -> relu -> 1 -> sigmoid
        # shape: batch size x time scale
        start = self.x_1d_s(base_feature).squeeze(1)
        # net: conv1d -> relu -> conv1d -> sigmoid
        #      hidden_dim_1d -> hidden_dim_1d -> relu -> 1 -> sigmoid
        # shape: batch size x time scale
        end = self.x_1d_e(base_feature).squeeze(1)
        # shape: batch size x hidden_dim_1d x time scale
        confidence_map = self.x_1d_p(base_feature)
        # shape: batch size x hidden_dim_1d x num samples x time scale x time scale
        confidence_map = self._boundary_matching_layer(confidence_map)
        # shape: batch size x hidden_dim3d x time scale x time scale
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        # shape: batch size x 2 x time scale x time scale
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        # shape x: batch size x dim x time scale
        #       sample_mask: time scale x (num samples x time scale x time scale)
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    # generate sample mask for a boundary-matching pair
    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # total lenght of the start-end duration
        plen = float(seg_xmax - seg_xmin)
        # the length of each bin
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        # the positions of each bin
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            # shape: time scale x num samples x time scale(s)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        # shape: time scale x num samples x time scale(s) x time scale(e)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


if __name__ == '__main__':
    from bmn import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    input=torch.randn(2,400,100)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)
