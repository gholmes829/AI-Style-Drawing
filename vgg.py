"""

"""

import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
	# Defining a class that for the model
    def __init__(self):
        nn.Module.__init__(self)
        self.req_features = ["0", "5", "10", "19", "28"]
        # Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model = models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers
    
    # x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        # initialize an array that wil hold the activations from the chosen layers
        features = []
        # Iterate over all the layers of the mode
        for layer_num, layer in enumerate(self.model):
            # activation of the layer will stored in x
            x = layer(x)
            # appending the activation of the selected layers and return the feature array
            if (str(layer_num) in self.req_features):
                features.append(x)
                
        return features

