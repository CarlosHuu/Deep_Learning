import sys
import torch
from torch.utils.data.dataloader import DataLoader
from thop import profile
import network
from network import *

# Path
if len(sys.argv) >= 3: 
    DATA_PATH = sys.argv[1]
else: 
    DATA_PATH = "cifar100/valid.pt"
if len(sys.argv) >= 4:
    MODEL_PATH = sys.argv[2]
else: 
    MODEL_PATH = "model.pth"

# inference parameters
BATCH_SIZE = 1

# load model
model = network.load_model(MODEL_PATH)
model.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

# load data
test_set = torch.load(DATA_PATH)
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)


# ==============================================
# Test FLOPs
# ==============================================
# calculate FLOPs
input_test = test_set[0][0].unsqueeze(0).to(DEVICE)
flops, params = profile(model, inputs=(input_test,))

# output
print("flops = {:.0f}".format(flops))


# ==============================================
# Test parameter size
# ==============================================
# calculate parameter size
param_size = 0
for param in model.parameters(): 
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
total_size_KB = (param_size + buffer_size) / 1024.0

# output
print("Model parameter size = {:.3f} kB".format(total_size_KB))


# ==============================================
# Test accuracy
# ==============================================
# calculate accuracy
total_acc = 0
for inputs, labels in test_loader: 
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(inputs)
    _, preds = torch.max(outputs, dim=1); 

    total_acc += torch.sum(preds == labels).item()
total_acc = total_acc / len(test_set) * 100.0

# output
print("Accuracy = {:.2f} %".format(total_acc))

