# utils
import os
import time
import sys
import matplotlib.pyplot as plt
import pathlib

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import lookahead

# fastmri
import fastmri
from common import subsample
from data import transforms, mri_data

# SETTINGS and HYPERPARAMETERS
TRAIN_DIR = '...'                       # path to training set directory
VAL_DIR = '...'                         # path to validation set directory
WEIGHTS_DIR = "./weights/"              # path to directory to save weights
BATCH_SIZE = 1                          

# DEBUGGING
PLOT_DEBUG = False                      # plots target and prediction for each slice
SAVE_MODEL = False                      # saves model to WEIGHTS_DIR

# SETTINGS FOR TRAINING
learning_rate = 1e-4
epochs = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using available device:{}'.format(device))



"""
AUX FUNCTIONS
Some aux functions for debugging and for plotting intermediate results
"""
def print_slice(complex_slice, name = None):
    """
    This function should be used only to print the output of a CRNN-i unit, 
    in that it considers a complex input image in 4 dimensions, 
    with shape (1, 2, 320, 320)
    """
    if name: print("***********" + name + "**********")
    slice = complex_slice.clone()
    slice = slice.detach()
    slice = slice.cpu()
    slice = slice.permute(0, 2, 3, 1)
    slice = slice.squeeze()
    slice_image_abs = fastmri.complex_abs(slice)
    plt.imshow(slice_image_abs, cmap = 'gray')
    plt.show()


def print_real_image(r_img, name = None, batch_size = BATCH_SIZE):
    """
    Plots a real image
    """
    if name: print("***********" + name + "**********")
    to_print = []
    c_img = r_img.clone()
    for i in range(batch_size):
        img = c_img[i,:,:]
        img = torch.squeeze(img)
        img = img.detach()
        img = img.cpu()
        plt.imshow(img, cmap = 'gray')
        plt.show()


"""
PYRAMID CONVOLUTIONAL NEURAL NETWORK
Layers and blocks for PCRNN
"""

class DataConsistencyLayer(nn.Module): 
    """
    Data consistency layer.
    Filters out output pixels that were not masked.
    """
    def __init__(self):
        super(DataConsistencyLayer, self).__init__()

    def proper_padding(self, prediction, k_space_slice): 
        """
        Pad prediction to be same size as k_space_slice
        """
        h = prediction.shape[-3]
        w = prediction.shape[-2]
        w_pad = (k_space_slice.shape[-2] - w) // 2
        h_pad = (k_space_slice.shape[-3]-h) // 2
        return torch.nn.functional.pad(prediction, (0,0,w_pad,w_pad,h_pad,h_pad), "constant", 0)

    def data_consistency_kspace(self, prediction, k_space_slice, mask):
        """
        Args:
            - prediction: net (or block) predicted real image in complex domain
            - k_space_slice: initially sampled elements in k-space
            - mask: corresponding nonzero location in kspace
        Res:
            image in k space where:
                - masked entries of initial slice are replaced with entries predicted by output
                - non masked entries of initial slice stay the same
        """

        prediction = prediction[:,:,0:320, 0:320]
        prediction = prediction.permute(0,2,3,1) # prediction from 1 x 2 x h x w to 1 x h x w x 2
        prediction = self.proper_padding(prediction, k_space_slice) # pad prediction to be 640 x 372 x 2
        k_space_prediction = fastmri.fft2c(prediction) # transform prediction to kspace domain

        k_space_out = (1 - mask) * k_space_prediction + mask * k_space_slice # apply mask
        prediction = fastmri.ifft2c(k_space_out) # back to cplx image
        prediction = transforms.complex_center_crop(prediction, (320, 320)) # crop image to 320 x 320
        prediction = prediction.permute(0,3,1,2) # back to 1 x 2 x h x w
        return prediction

    def forward(self, prediction, original_kspace_slice, mask):
        out = self.data_consistency_kspace(prediction, original_kspace_slice, mask)
        return out


class ResBlock(nn.Module):
    """
    Residual convolutions layer.
    Performs two residual convolutions on input data. 
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding)
        self.act2 = nn.ReLU()
        self.norm = nn.BatchNorm2d(in_channels, affine=False)


    def forward(self, input):
        #ResConv1
        identity1 = input               #skip connection
        out = self.conv1(input)
        out += identity1                #apply skip connection
        out = self.act1(out)
        
        #ResConv2
        identity2 = out                 #skip connection
        out = self.conv2(out)
        out += identity2                #apply skip connection
        out = self.act2(out)

        out = self.norm(out)			# batch normalization
        return out



class ConvRNN1(nn.Module):
    """
    ConvRNN-1 layer.
    Recurrent convolutional block.
    """
    def __init__(self):
        super(ConvRNN1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.resblock = ResBlock(in_channels= 384, out_channels=384, kernel_size=3, stride=1,
                                 padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=384, out_channels=2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dc = DataConsistencyLayer()


    def forward(self, input, hidden_input, kspace, mask):
                
        enc_out = self.encoder(input)                       # encoder
        resblock_out = self.resblock(hidden_input)          # residual conv
        current_hidden_state = enc_out.add(resblock_out)    # add to form hidden state
        dec_out = self.decoder(current_hidden_state)        # decoder
        dc_out = self.dc(dec_out, kspace, mask)             # data consistency
        
        return dc_out, current_hidden_state



class ConvRNN2(nn.Module): 
    """
    ConvRNN-2 layer.
    Recurrent convolutional block.
    """
    def __init__(self):
        super(ConvRNN2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.resblock = ResBlock(in_channels= 192, out_channels=192, kernel_size=3, stride=1,
                                 padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=192, out_channels=2, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )
        self.dc = DataConsistencyLayer()


    def forward(self, input, hidden_input, kspace, mask):

        enc_out = self.encoder(input)                       # encoder
        resblock_out = self.resblock(hidden_input)          # residual conv
        current_hidden_state = enc_out.add(resblock_out)    # add to form hidden state
        dec_out = self.decoder(current_hidden_state)        # decoder
        dc_out = self.dc(dec_out, kspace, mask)             # data consistency

        return dc_out, current_hidden_state


class ConvRNN3(nn.Module): 
    """
    ConvRNN-3 layer.
    Recurrent convolutional block. 
    """
    def __init__(self):
        super(ConvRNN3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.resblock = ResBlock(in_channels= 96, out_channels=96, kernel_size=3, stride=1,
                                 padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=96, out_channels=2, kernel_size=4, stride=1, padding=1),
            nn.ReLU()
        )
        self.dc = DataConsistencyLayer()


    def forward(self, input, hidden_input, kspace, mask):

        enc_out = self.encoder(input)                       # encoder
        resblock_out = self.resblock(hidden_input)          # residual conv
        current_hidden_state = enc_out.add(resblock_out)    # add to form hidden state
        dec_out = self.decoder(current_hidden_state)        # decoder
        dc_out = self.dc(dec_out, kspace, mask)             # data consistency

        return dc_out, current_hidden_state




class FinalBlock(nn.Module): 
    """
    Final Block.
    Four final convolutional layers to process concatenated output of ConvRNN-i. 
    This layer is fed the output of the concatenation layer,
    hence its input size is 6 x 320 x 320.
    """
    def __init__(self):
        super(FinalBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dc = DataConsistencyLayer()

    def forward(self, input, kspace, mask):
        out = self.cnn(input)
        out = self.dc(out, kspace, mask)
        out = out.permute(0, 2, 3, 1)
        return out



class PyramidConvRNN(nn.Module):
    """
    Complete Net.
    """
    def __init__(self):
        super(PyramidConvRNN, self).__init__()
        self.convrnn1 = ConvRNN1()
        self.convrnn2 = ConvRNN2()
        self.convrnn3 = ConvRNN3()
        self.final = FinalBlock()


    def forward(self, slice, kspace, mask):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # hidden state tensors for first iteration, initialized to 0
        h1 = torch.zeros(1, 384, 80, 80)  
        h2 = torch.zeros(1, 192, 160, 160)  
        h3 = torch.zeros(1, 96, 320, 320)  

        h1, h2, h3 = h1.to(device), h2.to(device), h3.to(device)

        out1 = slice
        for _ in range(5):
            out1, h1 = self.convrnn1(input=out1, hidden_input=h1, kspace=kspace, mask=mask)
        
        out2 = out1
        for _ in range(5):
            out2, h2 = self.convrnn2(input=out2, hidden_input=h2, kspace=kspace, mask=mask)
        
        out3 = out2
        for _ in range(5):
            out3, h3 = self.convrnn3(input=out3, hidden_input=h3, kspace=kspace, mask=mask)
        
        
        out = torch.cat((out1, out2, out3), 1)  # concatenation of outputs of ConvRNN layers
        out = self.final(out, kspace, mask)     # final block

        out = transforms.complex_abs(out)       # transform complex image into real image; out is 320 x 320
        out = torch.unsqueeze(out, 0)           # adding dimension to have 1 x 1 x 320 x 320

        return out



net = PyramidConvRNN()
net.to(device)



"""
LOSS FUNCTIONS
NMSE and SSIM modules, to be combined for total loss function.
"""

class NMSE_loss(nn.Module):
    """
    Module to compute Normalized mean squared error.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Args:
            - prediction: output of nn
            - target: target image
        Returns:
            - Normalized mse NOT as defined in the paper, because it is only divided by img
            (not volume) TODO edit so as to return volume norm
        """
        norm = torch.norm(prediction)
        mse = nn.MSELoss()
        return mse(input=prediction, target=target) / norm


class SSIMLoss(nn.Module):
    """
    Module to compute SSIM loss.
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Args:
            win_size (int, default=7): Window size for SSIM calculation.
            k1 (float, default=0.1): k1 parameter for SSIM calculation.
            k2 (float, default=0.03): k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


nmse_loss = NMSE_loss()
nmse_loss = nmse_loss.to(device)

ssim_loss = SSIMLoss()                                    
ssim_loss = ssim_loss.to(device)


def combined_loss(prediction, target, max_value, ssim_weight=0.5):
    """
    Function that combines NMSE and SSIM loss, to get the total loss
    """
    ssim_l = ssim_loss(prediction, target, max_value)
    nmse_l = nmse_loss(prediction, target)
    total_loss = nmse_l + (ssim_l * ssim_weight)
    return total_loss


"""
DATA LOADING 
Here we load the data from the dataset and preprocess each slice.
"""

# masking function to be applied to kspace
mask_func = subsample.RandomMaskFunc(center_fractions=[0.04], accelerations=[2])


def data_transform(kspace, mask_function, target, data_attributes, filename, slice_num):
    """
    Perform preprocessing of the kspace image, in order to get a proper input for the net. Should be invoked from
    the SliceData class.
    Args:
        - kspace: complete sampled kspace image
        - mask_func: masking function to apply mask to kspace (TODO not working: we are passing from outside)
        - target: the target image to be reconstructed from the kspace
        - data_attributes: attributes of the whole HDF5 file

    Returns:
        - normalized_masked_image: original kspace with mask applied and cropped to 320 x 320
        - mask: mask generated by masking function
        - normalized_target: normalized target
        - max_value: highest entry in target tensor (for SSIM loss)
    """
    
    kspace_t = transforms.to_tensor(kspace)
    kspace_t = transforms.normalize_instance(kspace_t)[0]
    
    
    masked_kspace, mask = transforms.apply_mask(data=kspace_t, mask_func=mask_func)           # apply mask: returns masked space and generated mask
    masked_image = fastmri.ifft2c(masked_kspace)                                              # Apply Inverse Fourier Transform to get the complex image
    masked_image = transforms.complex_center_crop(masked_image, (320, 320))                   # center crop masked image
    masked_image = masked_image.permute(2, 0, 1)                                              # permuting the masked image fot pytorch n x c x h x w format
    masked_image = transforms.normalize_instance(masked_image)[0]                             # normalize
   
    target = transforms.to_tensor(target)
    target = transforms.normalize_instance(target)[0]                                         # normalize
    target = torch.unsqueeze(target, 0)                                                       # add dimension

    return kspace_t, masked_image, target, mask, data_attributes['max'], slice_num


# Note: we are not replicating the choices of the paper here: they started with 1e-5, then
# switched to 1e-4 for 2nd epoch, and then reduced learning rate by a factor of 2
# every 10 epochs. We are just starting with lr = 1e-4 and then decaying every 15 with gamma = 0.5

_train_dataset = mri_data.SliceData(
    root=pathlib.Path(TRAIN_DIR),        
    transform=data_transform,
    challenge='singlecoil',
    #reduce_slices = (19,20)
)
train_dataset = torch.utils.data.DataLoader(_train_dataset, batch_size = BATCH_SIZE)

_val_dataset = mri_data.SliceData(
    root=pathlib.Path(VAL_DIR),
    transform=data_transform,
    challenge='singlecoil',
    #reduce_slices = (19,20)
)
val_dataset = torch.utils.data.DataLoader(_val_dataset, batch_size = BATCH_SIZE)


"""
TRAINING 
"""

import lookahead

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)     # We use default parameters for Adam here
lookahead = lookahead.Lookahead(optimizer, k=5, alpha=0.5)
scheduler = StepLR(lookahead, step_size=10, gamma=0.5)              # lr decaying. With verbose = True we have a msg for each step



for epoch in range(epochs):
    is_last_epoch = (epoch == epochs-1)
    for kspace, slice, target, mask, max_value, num in train_dataset:
        kspace, slice, target, mask, max_value = kspace.to(device), slice.to(device), target.to(device), mask.to(device), max_value.to(device)
        
        #print("----------SLICE NUM---------", num.item())
        if PLOT_DEBUG: 
          print_slice(slice, "slice")
          print_real_image(target, "target")
        
        # forward step
        out = net(slice, kspace, mask)
        if PLOT_DEBUG: print_real_image(out)
        # compute loss
        total_loss = combined_loss(out, target, max_value)
        # zero gradient
        lookahead.zero_grad()
        # backward
        total_loss.backward()
        # optimizer step
        lookahead.step()
        # scheduler step
        scheduler.step()

    with torch.no_grad():
        val_losses = []
        for kspace, slice, target, mask, max_value, num in val_dataset: 
            # print("----------SLICE NUM---------", num.item())
            kspace, slice, target, mask, max_value = kspace.to(device), slice.to(device), target.to(device), mask.to(device), max_value.to(device)
            out = net(slice, kspace, mask)
            val_loss = combined_loss(out, target, max_value)
            val_losses.append(val_loss)

        current_val_loss = torch.mean(torch.tensor(val_losses))
        print('validation loss at epoch {epoch_index} is {current_loss}:'.format(epoch_index = epoch, current_loss = current_val_loss.item()))
        destination = WEIGHTS_DIR+'weights_epoch{}'.format(epoch)
        if SAVE_MODEL: torch.save(net.state_dict(), destination)

