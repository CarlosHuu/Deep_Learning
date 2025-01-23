import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
### Write your model architecture

### End
class CIFAR100ShuffleNet(nn.Module):
    def __init__(self, num_classes=100,dropout_prob=0.2):
        super(CIFAR100ShuffleNet, self).__init__()
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
        self.base_model = shufflenet_v2_x0_5(weights=weights)  
        self.base_model.fc = nn.Linear(1024, num_classes) 
        

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x
    
def load_model(MODEL_PATH): 
    #call model
    model = CIFAR100ShuffleNet()
    model = torch.load(MODEL_PATH)
    return model