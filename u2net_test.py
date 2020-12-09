import os

import torch

from model import U2NET

model_name = 'u2net'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

print("...load U2NET---173.6 MB")
net = U2NET(3, 1)
map_location = 'cpu'
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()

net.load_state_dict(torch.load(model_dir, map_location=map_location))
if torch.cuda.is_available():
    net.cuda()
net.eval()
