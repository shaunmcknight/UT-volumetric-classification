import numpy as np
import itertools
import time
import datetime

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F

from torchinfo import summary

import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.pyplot import figure
from IPython.display import clear_output
from sklearn import metrics 

# from PIL import Image
import PIL
import matplotlib.image as mpimg

from utils import *
from classifier import *

##############################################
# Defining all hyperparameters
##############################################

plt.style.use(['science', 'ieee','no-latex', 'bright'])
    
HP = {'n_epochs': 10,
      'batch_size': 32, #4
      'lr': 0.0005,#0.01#0.03#0.013870869810956584, #0.03,
      'momentum': 0.175764011181887,
      'early_stop': 0.0,
      'reduction_layers': 4,
      'conv_layers': 3,# 4 to 6
      'out_channel_ratio': 2,
      'kernel_size_reduction': 7,
      'kernel_size': 3, #3,5,7
      'pool_layers': 0, #1 to 4
      'pool_mode': 'max',
      'FC_layers': 1, #1 to 2
      'Norm': 'batch'
      }




# if model_iter  != None:
#     model_path = r"C:\Users\CUE-ML\shaun\Python\3d_classification/saved_model_" + model_iter
    

########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze())# , cmap = 'inferno')
    plt.colorbar()
    plt.figure(figsize=(size, size))
    plt.show()

def show_grid(im1, im2, im3, im4, im5, im6):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, [im1, im2, im3, im4, im5, im6]):
        # Iterating over the grid returns the Axes.
        ax.imshow(np.transpose(im, (1, 2, 0)).squeeze())
    
    plt.show()
    
def plotAllImages(synthetic_dataset, civa_dataset):
    for i in range(0, np.shape(synthetic_dataset)[0]):
        
        plt.figure(figsize = (5,2))
        plt.subplot(1,2,1)
        plt.imshow(civa_dataset[i])
        # plt.colorbar()
        # plt.axis('off')

        # plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(synthetic_dataset[i])
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        plt.show()
    
def plotAllImagesDb(synthetic_dataset, civa_dataset):
    for i in range(0, np.shape(synthetic_dataset)[0]):
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(civa_dataset[i])
        # plt.colorbar()
        # plt.axis('off')

        # plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(synthetic_dataset[i])
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        
        plt.subplot(2,2,3)
        plt.imshow(20*np.log10(civa_dataset[i]/np.max(civa_dataset[i])))
        plt.clim(0, -6)
        # plt.colorbar()
        # plt.axis('off')

        db_synthetic = synthetic_dataset[i]-np.min(synthetic_dataset[i])
        db_synthetic = db_synthetic/np.nanmax(db_synthetic)
        # plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(20*np.log(abs(db_synthetic)))
        plt.clim(0, -6)
        # plt.colorbar()
        # plt.axis('off')
        
        plt.show()
        

def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()

def plot_roc(true_list, pred_list_raw):
    fpr, tpr, thresholds = metrics.roc_curve(true_list,  pred_list_raw)
    auc = metrics.roc_auc_score(true_list, pred_list_raw)
    
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()
    

    print('true list len ', len(true_list))
    print('pred list len', len(pred_list_raw))
    print('TPR ', tpr)
    print('FPR', fpr)
    print('Thresholds', thresholds)
    print('Optimum threshold ', thresholds[np.argmax(tpr - fpr)])
    
    return fpr, tpr, auc, thresholds
    
def plot_threshold(fpr, tpr, thresholds):
    plt.figure()
    plt.plot(thresholds,fpr,label="FPR")
    plt.plot(thresholds,tpr,label="TPR")
    plt.ylabel('True Positive Rate')
    plt.xlabel('Threshold')
    plt.xlim(0,1)
    plt.legend()
    plt.show()
    
def showFailures(true_list, pred_list, dataloader):
    list_incorrect_defects = []
    list_incorrect_clean = []
    for i, item in enumerate(pred_list):
        if item != true_list[i]:
            if item == 1:
                list_incorrect_defects.append(
                    dataloader.dataset[i][0].squeeze())
            elif item == 0:
                list_incorrect_clean.append(
                    dataloader.dataset[i][0].squeeze())
            else:
                print('ERROR')

    if len(list_incorrect_defects) > 0:
        for img in list_incorrect_defects:
            plt.figure()
            plt.title('Incorrect Defect Prediction')
            plt.imshow(img[32,:,:], aspect='auto')
            plt.show()
            
    if len(list_incorrect_clean) > 0:
        for img in list_incorrect_clean:
            plt.figure()
            plt.title('Incorrect Clean Prediction')
            plt.imshow(img[32,:,:], aspect='auto')
            plt.show()

def test(dataloader, description, disp_CM, Net, Tensor, device):
    true_list = []
    pred_list = []
    pred_list_raw = []
    Net.eval()
    Net.cuda()
    images = []
    labels = []
    val_losses=0
    loss=nn.BCEWithLogitsLoss()
    print(description)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images.type(Tensor)
            labels.type(Tensor)

            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(labels.cpu().detach().numpy())):
                true_list.append(labels.cpu().detach().numpy()[i])

            output_raw = Net(images)
            val_loss = loss(output_raw.squeeze(), labels)


            output = torch.sigmoid(output_raw)
            pred_tag = torch.round(output)
            
            [pred_list.append(pred_tag[i]) for i in range(
                pred_tag.squeeze().cpu().numpy().size)]
            
            [pred_list_raw.append(output[i]) for i in range(
                output.squeeze().cpu().numpy().size)]
            
            val_losses+=val_loss.item()
               
            
    print("\r[Validation loss %f]" 
          %(val_losses))
   
                
                
            
    pred_list = [a.squeeze().tolist() for a in pred_list]
    pred_list_raw = [a.squeeze().tolist() for a in pred_list_raw]

    true_list = np.array(true_list)
    pred_list = np.array(pred_list)
    pred_list_raw = np.array(pred_list_raw)

    correct = np.sum(true_list == pred_list)
    total = np.shape(true_list)
    
    accuracy = correct/total

    print('')
    print('~~~~~~~~~~~~~~~~~')
    print(description)
    print('Prediciton Accuracy: ', (accuracy)*100)

    print('Confusion matrix || {}'.format(description))
    cm = metrics.confusion_matrix(true_list, pred_list)
    print(cm)

    if disp_CM == True:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['No defect', 'Defect'])
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(description)
        plt.show()  
        metrics.ConfusionMatrixDisplay.from_predictions(true_list, pred_list, normalize='true', cmap=plt.cm.Blues)

        
    print('Pred ', pred_list)
    print('True ', true_list)

    precision, recall, f_score, support = metrics.precision_recall_fscore_support(
        true_list, pred_list)

    print('Precision ', precision[1])
    print('Recall ', recall[1])
    print('F score ', f_score[1])

    return true_list, pred_list, pred_list_raw, accuracy, precision[1], recall[1], f_score[1], cm

##############################################
# Initialize generator and discriminator
##############################################

accuracies = []
precisions = []
recalls = []
f_scores = []
confusion_matrixes = []
trps = []
fprs = []

for i in range(15):
    print('testing model ', i)
    batch = str(i)#iter37/"
    
    input_shape = (1, 64, 1024, 64)
     
     # Net = Network(input_shape)
     # CNN3D
     # CNN3D_Designed
    Net = CNN3D_Designed(input_shape, 
                num_reduction=HP['reduction_layers'],
                num_conv=HP['conv_layers'],
                channel_ratio=HP['out_channel_ratio'],
                kernel_reduction=HP['kernel_size_reduction'],
                kernel=HP['kernel_size'], 
                pool_layers=HP['pool_layers'],
                pooling=HP['pool_mode'],
                num_fc=HP['FC_layers'],
                normalisation=HP['Norm'])
    
    dummy_input = torch.randn((1,1,64,1024,64))
    Net(dummy_input)
            
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net.to(device)
    
    print("Using CUDA" if torch.cuda.is_available() else "Not using CUDA")
    if torch.cuda.device_count() > 1:
        print("USing ", torch.cuda.device_count(), "GPUs!")
        Net = nn.DataParallel(Net)
    
    Net.to(device)
    
    print('Network')
    # print(Net)
    print(summary(Net.float(), input_size=(HP['batch_size'], 
                                           input_shape[0], input_shape[1], input_shape[2], input_shape[3])))
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    Net = Net.double()
    
    ##############################################
    # Load weights
    ##############################################
    
    # load_path = os.path.join('adam', batch, "saved_model_"+batch+'.pt')
    load_path = os.path.join('best_models_reduction', "saved_model_"+batch+'.pt')
    Net.load_state_dict(torch.load(load_path)) #saved_Gen_AB_37600
    
    ##############################################
    # Final Validation Function
    ##############################################
    
    root_test = r"C:\Users\CUE-ML\shaun\Datasets\3D\experimental\val test split"
    
    experimental_dataloader = DataLoader(
        ExperimentalDataset(root_test, mode='test', transforms_=[]),
        batch_size=HP['batch_size'],
        shuffle=False
    )
    
    test_dataloader=experimental_dataloader
    true_list, pred_list, pred_list_raw, accuracy, precision, recall, f_score, cm = test(test_dataloader,
                                'Experimental Test', disp_CM=True, Net = Net, Tensor=Tensor, device=device)
        
    fpr, tpr, auc, thresholds = plot_roc(true_list, pred_list_raw)
    plot_threshold(fpr, tpr, thresholds)
    
    plt.figure()
    plt.scatter(true_list, pred_list_raw, marker ='x')
    plt.show()
    
    showFailures(true_list, pred_list, test_dataloader)
    
    
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