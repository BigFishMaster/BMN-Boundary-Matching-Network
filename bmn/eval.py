from bmn.eval_proposal import AliMediaProposal
from bmn.eval_detection import AliMediaDetection
from bmn.logging import logger
import matplotlib.pyplot as plt
import numpy as np


def eval_proposal(ground_truth_filename, proposal_filename,
                  max_avg_nr_proposals=100,
                  tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    proposal = AliMediaProposal(ground_truth_filename, proposal_filename,
                                tiou_thresholds=tiou_thresholds,
                                max_avg_nr_proposals=max_avg_nr_proposals,
                                verbose=True)
    proposal.evaluate()
    
    recall = proposal.recall
    average_recall = proposal.avg_recall
    average_nr_proposals = proposal.proposals_per_video
    
    return average_nr_proposals, average_recall, recall


def eval_detection(ground_truth_filename, proposal_filename,
                   num_labels=53, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    detection = AliMediaDetection(ground_truth_filename, proposal_filename,
                                  num_labels=num_labels, tiou_thresholds=tiou_thresholds,
                                  verbose=True)
    detection.evaluate()


def plot_metric(opt, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(opt["save_fig_path"])


def evaluation(opt):

    # eval_type could be: proposal+detection, proposal or detection.
    eval_type = opt["eval_type"].split("+")
    if "proposal" in eval_type:
        uniform_average_nr_proposals, uniform_average_recall, uniform_recall = eval_proposal(
            opt["gt_json"],
            opt["result_file"],
            max_avg_nr_proposals=100,
            tiou_thresholds=np.linspace(0.5, 0.95, 10))
        plot_metric(opt, uniform_average_nr_proposals, uniform_average_recall, uniform_recall)
        logger.info("AR@1 is \t", np.mean(uniform_recall[:, 0]))
        logger.info("AR@5 is \t", np.mean(uniform_recall[:, 4]))
        logger.info("AR@10 is \t", np.mean(uniform_recall[:, 9]))
        logger.info("AR@100 is \t", np.mean(uniform_recall[:, -1]))

    if "detection" in eval_type:
        eval_detection(opt["gt_json"], opt["result_file"], opt["num_labels"],
                       tiou_thresholds=np.linspace(0.5, 0.95, 10))

