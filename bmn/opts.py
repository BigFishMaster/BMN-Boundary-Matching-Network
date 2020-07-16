import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument('--pipeline', type=str, default='train')
    parser.add_argument('--gt_json', type=str, default="gt.json")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--data_folder', type=str, default="./data_C07B100/")
    parser.add_argument('--training_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_works', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default="adam")
    # Overall Dataset settings
    parser.add_argument('--hidden_size', type=str, default='256,128,512')
    parser.add_argument('--temporal_scale', type=int, default=100)
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--num_sample_perbin', type=int, default=3)
    parser.add_argument('--prop_boundary_ratio', type=int, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=2304)
    # Post processing
    parser.add_argument('--submit_label_file', type=str, default='labels.txt')
    parser.add_argument('--test_checkpoint', type=str, default='best_checkpoint.pth.tar')
    parser.add_argument('--test_classifier', type=str, default='slowfast_checkpoint.pth.tar.linear')
    parser.add_argument('--num_proposals', type=int, default=100)
    parser.add_argument('--post_process_thread', type=int, default=8)
    parser.add_argument('--soft_nms_alpha', type=float, default=0.4)
    parser.add_argument('--soft_nms_low_thres', type=float, default=0.5)
    parser.add_argument('--soft_nms_high_thres', type=float, default=0.9)
    parser.add_argument('--eval_type', type=str, default="proposal+detection")

    args = parser.parse_args()

    return args

