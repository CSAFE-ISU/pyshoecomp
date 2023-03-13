# -*- coding: utf-8 -*-
"""
    shoeshiny.neuralnet
    ~~~~~~~~~~~~~~~~~~~

    Contains classes and functions to handle
    the "pattern score", i.e. obtaining MCNCC/MCPOC
    using the truncated ResNet
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as tmod

nn = torch.nn

np.random.seed(0)
torch.manual_seed(0)


def tensorify(x: np.ndarray):
    """
    Converts a numpy array into a torch.FloatTensor.
    It accounts for whether the input is color or grayscale

    Parameters
    ----------
    x : ndarray
        the input image

    Returns
    -------
    ans :
        a 4-D torch.FloatTensor that can be passed to a neural network for processing

    """
    if len(x.shape) == 2:
        x2 = np.zeros((3, x.shape[0], x.shape[1]))
        x2[0, :, :] = x
        x2[1, :, :] = x
        x2[2, :, :] = x
    elif len(x.shape) == 3:
        x2 = np.transpose(x, (2, 0, 1))
    else:
        raise AssertionError("Image dimensions are not appropriate")

    ans = torch.from_numpy(np.float32(x2)).unsqueeze(0)
    return ans


def freeze_params(mod: nn.Module):
    """
    Freezes the parameters of the pretrained model
    so that it can act as a fixed feature extractor.

    Parameters
    ----------
    mod :
        a neural network model

    Returns
    -------
    None
    """
    for x in mod.parameters():
        x.requires_grad_(False)


def pretrained_model_partial():
    """
    Function to truncate the pretrained ResNet model and return all layers upto res2bx

    Returns
    -------
    torch.nn.Module:
        A neural network consisting of the trained weights of the pretrained ResNet model,
        upto the res2bx layer.
    """

    # raise AssertionError("Testing version shouldn't need to load from torchvision")
    z = tmod.resnet50(weights=tmod.resnet.ResNet50_Weights.IMAGENET1K_V1)
    # SO RES_2BX corresponds to layer1, bottleneck2
    t = list(z.named_children())
    my_layers = []
    for x in t:
        if x[0] == "layer1":
            # stopping at res2bx
            a = list(x[1].children())[:2]
            z2 = nn.Sequential(*a)
            my_layers.append(z2)
            break
        else:
            my_layers.append(x[1])

    return nn.Sequential(*my_layers)


class res2bx_sub(nn.Module):
    """
    Extracted layers from the pretrained ResNet-50 model,
    Stopping at 'res2bx`, `res2b_relu`, or right after the second bottleneck set,
    depending on what reference you use

    The output of the truncated CNN has 256 channels.
    """

    def __init__(self, load=False):
        super(res2bx_sub, self).__init__()

        self.conv1 = pretrained_model_partial()

        # for x in list(self.conv1.children()):
        # print(x)
        self.n_channels = 256
        self.freeze()

    def freeze(self):
        freeze_params(self.conv1)

    def save(self):
        torch.save(self.conv1, "./data/resnet_2bx.pth")

    def forward(self, inp1, inp2=None):
        x1 = self.conv1(inp1)
        if inp2 is not None:
            x2 = self.conv1(inp2)
        else:
            x2 = 0
        return [x1, x2]


def get_channel_ncc(data1, data2):
    """
    Compute per-channel NCC of two tensors.

    Parameters
    ----------
    data1 : torch.FloatTensor
    data2 : torch.FloatTensor

    Returns
    -------
    per-channel NCC for the above inputs

    """
    d_m1 = data1.view(data1.shape[:2] + (-1,)).mean(-1).view(data1.shape[:2] + (1, 1))
    d_m2 = data2.view(data2.shape[:2] + (-1,)).mean(-1).view(data1.shape[:2] + (1, 1))

    d_s1 = data1.view(data1.shape[:2] + (-1,)).std(-1).view(data1.shape[:2] + (1, 1))
    d_s2 = data2.view(data2.shape[:2] + (-1,)).std(-1).view(data1.shape[:2] + (1, 1))

    d_s1[d_s1 == 0] = 1e-10
    d_s2[d_s2 == 0] = 1e-10

    d1 = (data1 - d_m1) / d_s1
    d2 = (data2 - d_m2) / d_s2

    ncc = (d1 * d2).view(data1.shape[:2] + (-1,)).mean(-1)
    return ncc


def get_channel_poc(data1, data2):
    """
    Compute per-channel POC of two tensors.

    Don't override this method unless necessary.
    Parameters
    ----------
    data1 : torch.FloatTensor
    data2 : torch.FloatTensor

    Returns
    -------
    per-channel POC for the above inputs
    """
    g1 = torch.fft.fft(data1)
    g2 = torch.fft.fft(data2).conj()
    prod = g1 * g2
    abs_prod = torch.maximum(prod.abs(), torch.tensor(1e-9))
    norm_prod = prod / abs_prod

    poc = torch.fft.ifft(norm_prod).real
    poc_max, indices = poc.view(poc.shape[:2] + (-1,)).max(-1)

    return poc_max[0]


def avg_poc(im1, im2):
    mod = res2bx_sub(load=False)
    # mod.save()
    img1 = tensorify(im1)
    img2 = tensorify(im2)

    channels = mod(img1, img2)
    poc = get_channel_poc(*channels)
    score = np.mean(poc.cpu().detach().numpy())
    return score


def avg_ncc(im1, im2):
    mod = res2bx_sub(load=False)
    img1 = tensorify(im1)
    img2 = tensorify(im2)

    channels = mod(img1, img2)
    ncc = get_channel_ncc(*channels)
    score = np.mean(ncc.cpu().detach().numpy())
    return score


def main():
    mod = res2bx_sub(load=False)
    print(mod._modules)
    mod.eval()


if __name__ == "__main__":
    main()
