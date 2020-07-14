import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gt_json', type=str, default="./data/gt.json")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--train_info', type=str, default="./data/info_train.txt")
    parser.add_argument('--train_bmn', type=str, default="./data/bmn_train.txt")
    parser.add_argument('--valid_info', type=str, default="./data/info_valid.txt")
    parser.add_argument('--valid_bmn', type=str, default="./data/bmn_valid.txt")
    parser.add_argument('--test_info', type=str, default="./data/info_test.txt")
    parser.add_argument('--test_bmn', type=str, default="./data/bmn_test.txt")
    parser.add_argument('--training_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_works', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_size', type=int, default=7)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--step_gamma', type=float, default=0.1)
    # Overall Dataset settings
    parser.add_argument('--temporal_scale', type=int, default=100)
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--num_sample', type=int, default=32)
    parser.add_argument('--num_sample_perbin', type=int, default=3)
    parser.add_argument('--prop_boundary_ratio', type=int, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=2304)
    # Post processing
    parser.add_argument('--test_checkpoint', type=str, default='./model_test/best_checkpoint.pth.tar')
    parser.add_argument('--test_checkpoint_linear', type=str, default=None)
    parser.add_argument('--num_proposals', type=int, default=100)
    parser.add_argument('--post_process_thread', type=int, default=8)
    parser.add_argument('--soft_nms_alpha', type=float, default=0.4)
    parser.add_argument('--soft_nms_low_thres', type=float, default=0.5)
    parser.add_argument('--soft_nms_high_thres', type=float, default=0.9)
    parser.add_argument('--eval_type', type=str, default="proposal")

    args = parser.parse_args()

    return args

