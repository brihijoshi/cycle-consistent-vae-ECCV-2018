
# coding: utf-8

# In[ ]:


import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import norm
from itertools import cycle
from random import shuffle
from sklearn.model_selection import train_test_split



import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from livelossplot import PlotLosses
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import OrderedDict

from cycle_consistent_vae_in import Encoder, Decoder
from latent_classifier import Classifier


np.random.bit_generator = np.random._bit_generator


# In[ ]:


cuda = 1
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./models/cycle_vae_512_64_in_13052020-175056_499.pth"
checkpoint = torch.load(MODEL_PATH)

Z_DIM = 512 #Style Dimension (Unspecified)
S_DIM = 64 # Class Dimension (Specified)

encoder = Encoder(style_dim=Z_DIM, class_dim=S_DIM)
encoder.load_state_dict(checkpoint['encoder'])

encoder.to(device)
# encoder.eval()


# # Dataset processing

# In[ ]:


class Latent(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][:,:,:3]
        
        return (img, self.labels[index])


# In[ ]:


# PATH = './processed/'
# files = os.listdir(PATH)
# data = []
# labels = []

# counter=0
# for folder in files:
#     print(counter)
#     sprites = os.listdir(PATH+folder)
#     for sprite in sprites:
#         data.append(plt.imread(PATH+folder+"/"+sprite))
#         labels.append(int(folder))
#     counter+=1


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(data, labels, \
#                                                     test_size=0.33, random_state=42, \
#                                                     shuffle=True, stratify=labels )


# In[ ]:


# train_data = Latent(X_train, y_train)
# test_data = Latent(X_test, y_test)


# In[ ]:


# with open('train_latent.pkl','wb') as f:
#     pickle.dump(train_data, f)
# with open('test_latent.pkl','wb') as f:
#     pickle.dump(test_data, f)


# In[ ]:


t1 = pickle.load(open('train_latent.pkl','rb'))
t2 = pickle.load(open('test_latent.pkl','rb'))


# In[ ]:


BATCH_SIZE = 16
train_loader = DataLoader(t1,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
test_loader = DataLoader(t2,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)


# In[ ]:


NUM_CLASSES=672


# # Latent Classifier - Unspecified Features

# In[ ]:


TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")

NUM_EPOCHS = 200

LEARNING_RATE = 1e-3

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CUDA = True


# In[ ]:


model = Classifier(z_dim=S_DIM, num_classes=NUM_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# In[ ]:


is_better = True
prev_acc = 0
name = "specified_latent_classifier_512_64_in_499"

liveloss = PlotLosses(fig_path='./figures/'+name+".png")


# In[ ]:


dataloaders = {'train':train_loader, 'validation':test_loader}


# In[ ]:


for epoch in range(NUM_EPOCHS):
    logs = {}
    t_start = time.time()
    
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
            
        else:
            model.eval()
        model.to(device)
        
        print("Started Phase")

        running_loss = 0.0
                
        predicted_phase = torch.zeros(len(dataloaders[phase].dataset), NUM_CLASSES)
        target_phase = torch.zeros(len(dataloaders[phase].dataset))
        
        if phase == 'validation':
            
            with torch.no_grad():
                
                for (i,batch) in enumerate(dataloaders[phase]):
                    input_tensor = batch[0]
                    input_tensor = input_tensor.to(device)
                    input_tensor = torch.transpose(input_tensor, 2,3)
                    input_tensor = torch.transpose(input_tensor, 1,2)

                    _, _, latent_vector = encoder(input_tensor)

                    bs = input_tensor.shape[0]
                    target_tensor = batch[1].to(device).reshape(bs)

                    softmaxed_tensor = model(latent_vector)

                    loss = criterion(softmaxed_tensor, target_tensor.long())

                    predicted_phase[i*bs:(i+1)*bs] = softmaxed_tensor.cpu()
                    target_phase[i*bs:(i+1)*bs] = target_tensor.cpu()

                    input_tensor = input_tensor.cpu()
                    target_tensor = target_tensor.cpu()

                    running_loss += loss.detach() * bs
                
     
        else:
            
            for (i,batch) in enumerate(dataloaders[phase]):
                with torch.no_grad():
                    input_tensor = batch[0]
                    input_tensor = input_tensor.to(device)
                    input_tensor = torch.transpose(input_tensor, 2,3)
                    input_tensor = torch.transpose(input_tensor, 1,2)

                    _, _, latent_vector = encoder(input_tensor)

                bs = input_tensor.shape[0]
                target_tensor = batch[1].to(device).reshape(bs)

                softmaxed_tensor = model(latent_vector)

                loss = criterion(softmaxed_tensor, target_tensor.long())

                predicted_phase[i*bs:(i+1)*bs] = softmaxed_tensor.cpu()
                target_phase[i*bs:(i+1)*bs] = target_tensor.cpu()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                input_tensor = input_tensor.cpu()
                target_tensor = target_tensor.cpu()

                running_loss += loss.detach() * bs
    

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_predicted = torch.argmax(predicted_phase, dim=1)
        epoch_f1 = f1_score(target_phase, epoch_predicted, average = 'macro')
        epoch_accuracy = accuracy_score(target_phase, epoch_predicted)

        
        model.to('cpu')

        prefix = ''
        if phase == 'validation':
            prefix = 'val_'

        logs[prefix + 'log loss'] = epoch_loss.item()
        logs[prefix + 'f1'] = epoch_f1.item()
        logs[prefix + 'accuracy'] = epoch_accuracy.item()
        
        print('Phase time - ',time.time() - t_start)

    delta = time.time() - t_start
    is_better = logs['val_accuracy'] > prev_acc
    if is_better:
        prev_acc = logs['val_accuracy']
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': logs['log loss']}, "./models/"+name+"_"+TIME_STAMP+"_"+str(logs['val_accuracy'])+".pth")


    liveloss.update(logs)
    liveloss.draw()

