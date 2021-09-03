import torch.nn as nn
import torchvision.models as models
from networks.EfficientNet import *
from networks import efficientnet

def build_model(model_type, model_params, pretrained=False):
    print(model_type)
    if model_type == 'resnet50':
        if pretrained:
            print("=> using pre-trained model: {}".format(model_type))
            # self.model = models.resnet50(num_classes=self.num_classes)
            model = models.__dict__[model_type](pretrained=True)
            model.fc = nn.Linear(2048, model_params['num_classes'])
        else:
            print("=> creating model: {}".format(model_type))
            model = models.__dict__[model_type](num_classes=model_params['num_classes'])
    elif model_type == 'efficientnet-b7':
        print("=> using pre-trained model: {}".format(model_type))
        #model = efficientnet.__dict__['EfficientNet'](encoder='tf_efficientnet_b7_ns')
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    else:
        if pretrained:
            # self.model = models.resnet50(num_classes=self.num_classes)
            model = models.__dict__[model_type](pretrained=True)
            model.fc = nn.Linear(2048, model_params['num_classes'])
        else:
            model = models.__dict__[model_type](num_classes=model_params['num_classes'])
   
        # self.model = models.resnet50(preGtrained=False)
        # self.model.fc = nn.Linear(2048, self.num_classes)
    return model

if __name__ == '__main__':
    import argparse
    from argparse import Namespace
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        default='./configs/config.yaml')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    model = build_model(Namespace(**config))
