from enum import Enum
from base_classes.base import ActivationProcessor
import numpy as np
import torch
class NoProcessing(ActivationProcessor):
    def __init__(self, config):
        self.config = config
    def process(self, **kwargs):
        activation=  kwargs.get('activation')
        #if the activation is a tensor, convert it to numpy
        if isinstance(activation,torch.Tensor):
            activation = activation.detach().cpu().numpy()
        return activation
class StatisticalFeatureExtraction(ActivationProcessor):
    def __init__(self, config):
        self.config = config
    def process(self, **kwargs):
        processed_activation_list= []
        activation = kwargs.get('activation')
        functions = self.config['functions']
        #if the activation is a tensor, convert it to numpy
        if isinstance(activation,torch.Tensor):
            activation = activation.detach().cpu().numpy()
        #if the activation is a list, convert it to numpy, may be not needed
        if isinstance(activation,list):
            activation = np.array(activation)
        #if the activation is a numpy array, process it
        #Assumption is that channel is 0th index, which makes the operation global pooling
        for function in functions:
            processed_activation = eval(f"np.{function}(activation,axis=(2,3),keepdims=True)")
            processed_activation_list.append(processed_activation.squeeze((2,3)))
            # print(processed_activation.shape)
        return np.concatenate(processed_activation_list,axis=1)
    
class ExtremelySimpleActivationShaping(ActivationProcessor):
    def __init__(self, config):
        self.config = config
    def process(self, **kwargs):
        activation = kwargs.get('activation')
        percentile = self.config['percentile']
        method = self.config['method']
        if isinstance(activation,np.ndarray):
            activation = torch.from_numpy(activation)
        result = eval(f"self.ash_{method}(activation,percentile)")
        return result.detach().cpu().numpy()
    
    def ash_b(self,x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x

    def ash_p(self,x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100

        b, c, h, w = x.shape

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        return x

    def ash_s(self,x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])

        return x


class VisualDNA(ActivationProcessor):
    def __init__(self, config):
        self.config = config
    def process(self, **kwargs):
        processed_activation_list= []
        activation = kwargs.get('activation')
        bins = self.config['bins']
        #if the activation is a tensor, convert it to numpy
        if isinstance(activation,torch.Tensor):
            activation = activation.detach().cpu().numpy()
        #if the activation is a list, convert it to numpy, may be not needed
        if isinstance(activation,list):
            activation = np.array(activation)
        #if the activation is a numpy array, process it
        #Assumption is that channel is 0th index, which makes the operation global pooling
        base_histogram = np.zeros((activation.shape[0],bins))
        for i in range(activation.shape[0]):
            for j in range(activation.shape[1]):
                act_map = activation[i,j,:,:]
                histogram = np.histogram(act_map,bins=bins)[0]
                base_histogram[i,:] += histogram
        # print(base_histogram.shape)
        return base_histogram
class ProcessorEnum(Enum):
    SF = StatisticalFeatureExtraction
    ASH = ExtremelySimpleActivationShaping
    RAW = NoProcessing
    VDNA = VisualDNA