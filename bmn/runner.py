import os
from bmn import opts
from bmn.proposal import proposal
from bmn.eval import evaluation
from bmn.train import train
from bmn.inference import inference
from bmn.classification import classify
from bmn.utils.logging import init_logger, logger, beautify_info
from bmn.tools.save_classifier import save_classifier


def update(opt):
    folder = os.path.dirname(opt["data_folder"])
    if not os.path.exists(folder):
        raise FileNotFoundError("The data folder {} is not found.".format(folder))

    opt["checkpoint_path"] = os.path.join(folder, opt["checkpoint_path"])
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    init_logger(os.path.join(opt["checkpoint_path"], "bmn.log"))

    opt["result_dir"] = os.path.join(folder, "result/")
    if not os.path.exists(opt["result_dir"]):
        os.makedirs(opt["result_dir"])

    opt["proposal_file"] = os.path.join(folder, "result_proposal.json")
    opt["save_fig_path"] = os.path.join(folder, "result_evaluation.jpg")
    opt["detection_file"] = os.path.join(folder, "result_detection.json")
    opt["submit_label_file"] = os.path.join(folder, opt["submit_label_file"])
    opt["test_checkpoint"] = os.path.join(opt["checkpoint_path"], opt["test_checkpoint"])
    opt["test_classifier"] = os.path.join(folder, opt["test_classifier"])
    opt["gt_json"] = os.path.join(folder, opt["gt_json"])

    logger.info("create configure successfully: {}".format(beautify_info(opt)))


def main():
    # TODO: the info files for train, valid, test and submit should be prepared in advance.
    opt = opts.parse_opt()
    opt = vars(opt)

    update(opt)

    pipeline = opt["pipeline"].split("+")

    logger.info("The pipeline is {}".format(pipeline))

    if "train" in pipeline:
        train(opt)

    if "linear" in pipeline:
        save_classifier(opt)

    if "test" in pipeline:
        opt["mode"] = "test"
        folder = opt["data_folder"]
        opt["proposal_file"] = os.path.join(folder, "{}_result_proposal.json".format(opt["mode"]))
        opt["detection_file"] = os.path.join(folder, "{}_result_detection.json".format(opt["mode"]))
        if "inference" in pipeline:
            inference(opt)
        if "proposal" in pipeline:
            proposal(opt)
        if "classify" in pipeline:
            classify(opt)
        if "evaluation" in pipeline:
            evaluation(opt)

    if "submit" in pipeline:
        opt["mode"] = "submit"
        folder = opt["data_folder"]
        opt["proposal_file"] = os.path.join(folder, "{}_result_proposal.json".format(opt["mode"]))
        opt["detection_file"] = os.path.join(folder, "{}_result_detection.json".format(opt["mode"]))
        if "inference" in pipeline:
            inference(opt)
        if "proposal" in pipeline:
            proposal(opt)
        if "classify" in pipeline:
            classify(opt)


if __name__ == '__main__':
    main()
