
from torch import nn
from models import mobilenet, mobilenetv2, shufflenet, shufflenetv2
import torch

def get_classifier_layer( model, width_mult):

    assert model in ["mobilenet", "mobilenetv2", "shufflenet", "shufflenetv2"] , "model not supported"
    
    if model == "mobilenet":
        assert width_mult in [1.0, 2.0], "width_mult not supported"
        if width_mult == 1.0:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 84)
                                        )
        elif width_mult == 2.0:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(2048, 84)
                                        )
    elif model == "mobilenetv2":
        assert width_mult in [0.7, 1.0], "width_mult not supported"
        if width_mult == 1.0:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(1280, 84)
                                        )
        elif width_mult == 0.7:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(1280, 84)
                                        )
    elif model == "shufflenet":
        assert width_mult in [1.0, 2.0], "width_mult not supported"
        if width_mult == 1.0:
            classifier_layer = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(960, 84)
                                )
        elif width_mult == 2.0:
            classifier_layer = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(1920, 84)
                                )
    elif model == "shufflenetv2":
        assert width_mult in [1.0, 2.0], "width_mult not supported"
        if width_mult == 1.0:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 84)
                                        )
        elif width_mult == 2.0:
            classifier_layer = nn.Sequential(
                                        nn.Dropout(0.2),
                                        nn.Linear(2048, 84)
                                        )
            
    return classifier_layer


def get_model(model:str, width_mult:float):

    if model == "mobilenet":
        model = mobilenet.get_model(num_classes = 27, sample_size =16, width_mult = width_mult)
    elif model == "mobilenetv2":
        model = mobilenetv2.get_model(num_classes = 27, sample_size =16, width_mult = width_mult)
    elif model == "shufflenet":
        model = shufflenet.get_model(groups = 3, num_classes = 27, width_mult = width_mult)
    elif model == "shufflenetv2":
        model = shufflenetv2.get_model(num_classes = 27, width_mult = width_mult)

    return model

        