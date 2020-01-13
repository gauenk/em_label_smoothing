import _init_paths
import sys
import numpy as np
import mnist_loader
import argparse
import torchvision,torch


class ParametricFunction():
    def __init__(self,t_model):
        self._model = t_model
        self._history = []

    def __call__(self,*args):
        raise NotImplementedError("Please define a calling function")

    def update(self,*args):
        raise NotImplementedError("Please define a calling function")

    @property
    def history(self):
        return self._history

    
class f_parametric(ParametricFunction):
    def __call__(self,S):
        self.model.forward()
        return S.labels
    
    def update(self,*args):
        pass

class g_parametric(ParametricFunction):
    def __call__(self,y_tilde,S): 
       return y_tilde

    def update(self,*args):
        pass

def label_smoothing(f,g,S,num_iter):
    S_k = S
    for i in range(num_iters):
        y_tilde = f(S)
        S_k_1 = g(y_tilde,S_k)
        f.update(S_k,S_k_1)
        g.update(S_k,S_k_1,f.update)
        S_k = S_k_1
    S = S_k
    return f,g,S

def cuda_handler(args):
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=None, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    args = cuda_handler(args)
    return args


if __name__ == "__main__":

    # read user input settings
    args = parse_args()

    # init the dataset
    S = mnist_loader.mnist
    mnist = mnist_loader.mnist()
    preproc = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    

    # init the models
    resnet18_f = torchvision.models.resnet18(pretrained=True)
    resnet18_g = torchvision.models.resnet18(pretrained=True)
    # f = f_parametric(resnet18_f)
    # g = g_parametric(resnet18_g)

    # # run EM
    # num_iters = 100
    # f,g,S_hat = label_smoothing(f,g,S,num_iters)
    
    # # show example label-switches
    # show_flipped_labels(S,S_hat)

    
