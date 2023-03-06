# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:19:42 2022

@author: Shaun McKnight
"""

import torch
import numpy as np


from utils import *
from classifier import *
from train import *

def VisualiseData(dataloader):
    test_data = next(iter(dataloader))
    image = test_data[0][0].squeeze().cpu().detach().numpy()
    label = test_data[1][0].squeeze().cpu().detach().numpy()
    # print('Type: ', type(test_data[:,150:450,:]), test_data[:,150:450,:].shape)

    plt.figure()
    plt.imshow(np.amax(image[:,150:450,:], axis=1))
    plt.title(('Label ', (label)))
    plt.show()
    
    fig = plt.figure().add_subplot(projection="3d")
    fig.set_xlabel("sample")
    fig.set_ylabel("Amplitude")
    fig.set_zlabel("Element no.")
    frame = 32
    for i in range(image[:,:,0:64].shape[2]):       
        xlist = [x for x in range(1, (len(image[frame,:, i])+1))]
        plt.plot(
            xs=xlist, ys=image[frame,:, i], zs=i, label='Raw ' + str(i))
        plt.title(("All A-scans of frame: " + str(frame)))
    plt.show()
    
def experimental(HP, transforms, exp_path, iteration):

    root_test = r"C:\Users\CUE-ML\shaun\Datasets\3D\experimental\val test split"
    root_train = r"C:\Users\CUE-ML\shaun\Datasets\3D\synthetic\complete"

    train_dataloader = DataLoader(
        SyntheticDataset(root_train, mode='train', transforms_ = transforms_),
        batch_size=HP['batch_size'],
        shuffle=True,
        # num_workers=1,
    )
    
    # valid_dataloader = DataLoader(
    #     SyntheticDataset(root_train, mode='valid', transforms_ = transforms_),
    #     batch_size=HP['batch_size'],
    #     shuffle=True,
    #     # num_workers=1,
    # )
    
    valid_dataloader = DataLoader(
        ExperimentalDataset(root_test, mode='valid', transforms_ = transforms_),
        batch_size=HP['batch_size'],
        shuffle=True,
        # num_workers=1,
    )
       
    experimental_dataloader = DataLoader(
        ExperimentalDataset(root_test, mode='test', transforms_=transforms_),
        batch_size=HP['batch_size'],
        shuffle=False
    )
    
    # split = 0.5
    # split_train, split_valid = round(len(synthetic_dataloader.dataset)*split), round(
    #     len(synthetic_dataloader.dataset))-round(len(synthetic_dataloader.dataset)*split)
    
    # train, valid = torch.utils.data.random_split(
    #     synthetic_dataloader.dataset, (split_train, split_valid))
    
    # train_dataloader = DataLoader(
    #     train,
    #     batch_size=hp.batch_size,
    #     shuffle=True,
    # )
    
    # valid_dataloader = DataLoader(
    #     valid,
    #     batch_size=hp.batch_size,
    #     shuffle=False,
    # )
    
    VisualiseData(train_dataloader)
    VisualiseData(valid_dataloader)
    VisualiseData(experimental_dataloader)

    accuracy, precision, recall, f_score, cm = main(HP, 
         train_dataloader=train_dataloader,
         validation_dataloader=valid_dataloader, 
         test_dataloader=experimental_dataloader, 
         iteration = iteration)
    
    return accuracy, precision, recall, f_score, cm
    


#optimisation of experimental data
HP = {'n_epochs': 10,
      'batch_size': 32, #4
      'lr': 0.0005,#0.01#0.03#0.013870869810956584, #0.03,
      'momentum': 0.175764011181887,
      'early_stop': 0.0,
      'reduction_layers': 4,
      'conv_layers': 5,# 4 to 6
      'out_channel_ratio': 2,
      'kernel_size_reduction': 7,
      'kernel_size': 3, #3,5,7
      'pool_layers': 0, #1 to 4
      'pool_mode': 'max',
      'FC_layers': 1, #1 to 2
      'Norm': 'batch'
      }


"""
HPO parameters

architecture
        num CNN layers 1-5
    kernal size 3,5,7 for internal (3 for others)
        num FC layers 1-3
        nodes in FC layers 
    pool - 2,4,6
    normalisation

HYPERS
    batch size
    LR
    early stop
    momentum
    num epochs"""


transforms_ = [
    # transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5)),
]

exp_path = "C:/Users/Shaun McKnight/OneDrive - University of Strathclyde/PhD/Data/classifier/simple/experimental"

accuracies = []
precisions = []
recalls = []
f_scores = []
confusion_matrixes = []
trps = []
fprs = []

for i in range(15):
    
    torch.cuda.empty_cache()

    print('Model iteration ~ ', i)
    
    """Experimental"""
    
    accuracy, precision, recall, f_score, cm = experimental(HP, transforms=transforms_, exp_path=exp_path, iteration = i)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)
    confusion_matrixes.append(cm)
    

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Mean results ~")
print('~~~~~~~~~~~~~~~~~')
print("")
print('Accuracy ~ mu {}. std {}. '.format(np.mean(accuracies),np.std(accuracies)))
print('Precision ~ mu {}. std {}. '.format(np.mean(precisions),np.std(precisions)))
print('Recall ~ mu {}. std {}. '.format(np.mean(recalls),np.std(recalls)))
print('F score ~ mu {}. std {}. '.format(np.mean(f_scores),np.std(f_scores)))

print('Confusion matrix')
cm = np.array(confusion_matrixes)
cm = np.mean(cm, axis = 0)
print(cm)

print('')
print('~~~~~~~~~~~~~~~~~')
print("~ Max results ~")
print('~~~~~~~~~~~~~~~~~')
print('')
print('Accuracy ~ Max {}. '.format(np.amax(accuracies)))
print('Precision ~ Max {}. '.format(np.amax(precisions)))
print('Recall ~ Max {}. '.format(np.amax(recalls)))
print('F score ~ Max {}. '.format(np.amax(f_scores)))



"""

TO DO:
    add in training lists for 100 iterations to get std and averages
    re-train GAN
    
"""
