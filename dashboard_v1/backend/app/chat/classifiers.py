import torch
import torch.nn.functional as F
from torch import nn
import os


class LinearProbeClassification(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, logistic=False, Relu=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        if logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        elif Relu:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.ReLU(True)
            )
        else:
            
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
            )
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss


def load_probe_classifier(model_func, input_dim, num_classes, weight_path, **kwargs):
    """
    Instantiate a ProbeClassification model and load its pretrained weights.
    
    Args:
    - input_dim (int): Input dimension for the classifier.
    - num_classes (int): Number of classes for classification.
    - weight_path (str): Path to the pretrained weights.
    
    Returns:
    - model: The ProbeClassification model with loaded weights.
    """

    # Instantiate the model
    model = model_func("cuda", num_classes, input_dim, **kwargs)
    
    # Load the pretrained weights into the model
    model.load_state_dict(torch.load(weight_path))
    
    return model


num_classes = {"age": 4,
               "unknownage": 5,
               # "gender": 2,
               "gender": 3,
               "newgender": 2,
               "unknowngender": 3,
               "ethnics":6,
               "newethnics": 7,
               "unknownethnics": 8,
               "education": 6,
               "marital": 4,
               "newmarital": 4,
               "socioeco": 3, 
               "unknownsocioeco": 4,
               "hallucination": 2,
               "language": 8,
               "neweducation": 3,
               "unknowneducation": 4,
               "religion": 6,
               "political": 3,
               "unknownpolitical": 4,
               "knownpolitical": 3,
               "knownreligion": 6,
               "sycophancy": 2,
               "hallucination": 2}


def return_classifier_dict(directory, model_func, chosen_layer=None, mix_scaler=False, hidden_neurons=5120, **kwargs):
    checkpoint_paths = os.listdir(directory)
    # file_paths = [os.path.join(directory, file) for file in checkpoint_paths if file.endswith("pth")]
    classifier_dict = {}
    for i in range(len(checkpoint_paths)):
        category = checkpoint_paths[i][:checkpoint_paths[i].find("_")]
        num_class = num_classes[category]
        
        if category == "neweducation":
            category = "education"
        elif category == "newethnics":
            category = "ethnics"
        elif category == "newmarital":
            category = "marital"
        elif category == "marital" and (not "mistral" in directory):
            continue
            
        if "read_probes" in directory:
            if category == "political":
                # continue
                num_class = 3
            if category == "religion":
                continue
            elif category == "knownreligion":
                category = "religion"
            elif category == "knownpolitical":
                # category = "political"
                continue
        else:
            if category == "unknownage":
                continue
            elif category == "unknowneducation":
                continue
            elif category == "unknowngender":
                continue
            elif category == "unknownethnics":
                continue
            elif category == "unknownsocioeco":
                continue
            elif category == "unknownpolitical":
                continue
                
        weight_path = os.path.join(directory, checkpoint_paths[i])
        if category not in classifier_dict.keys():
            classifier_dict[category] = {}
        if mix_scaler:
            classifier_dict[category]["all"] = load_probe_classifier(model_func, hidden_neurons, 
                                                                     num_classes=num_class,
                                                                     weight_path=weight_path, **kwargs)
        else:
            layer_num = int(checkpoint_paths[i][checkpoint_paths[i].rfind("_") + 1: checkpoint_paths[i].rfind(".pth")])

            if chosen_layer is None or layer_num == chosen_layer:
                try:
                    classifier_dict[category][layer_num] = load_probe_classifier(model_func, hidden_neurons, 
                                                                                 num_classes=num_class,
                                                                                 weight_path=weight_path, **kwargs)
                except Exception as e:
                    if category == "gender":
                        classifier_dict[category][layer_num] = load_probe_classifier(model_func, hidden_neurons, 
                                                                                     num_classes=2,
                                                                                     weight_path=weight_path, **kwargs)
                    if category == "ethnics":
                        classifier_dict[category][layer_num] = load_probe_classifier(model_func, hidden_neurons, 
                                                                                     num_classes=7,
                                                                                     weight_path=weight_path, **kwargs)
                    if category == "education":
                        classifier_dict[category][layer_num] = load_probe_classifier(model_func, hidden_neurons, 
                                                                                     num_classes=3,
                                                                                     weight_path=weight_path, **kwargs)
                    if category == "marital":
                        classifier_dict[category][layer_num] = load_probe_classifier(model_func, hidden_neurons, 
                                                                                     num_classes=4,
                                                                                     weight_path=weight_path, **kwargs)
    return classifier_dict
