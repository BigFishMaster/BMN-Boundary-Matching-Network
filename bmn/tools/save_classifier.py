import os
import torch
from bmn.utils.logging import logger


def save_classifier(opt):
    outname = opt["test_classifier"]
    if os.path.exists(outname):
        logger.warn("When saving classifier, it already exists.")
        return
    filename = outname.replace(".linear", "")
    p = torch.load(filename)["model_state"]

    weight = p["head.projection.weight"].cpu()
    bias = p["head.projection.bias"].cpu()

    logger.info("save classifier from checkpoint file {} with weight: {} and bias: {}"
                .format(filename, weight.shape, bias.shape))

    torch.save([weight, bias], outname)
