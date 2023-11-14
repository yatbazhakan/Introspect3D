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
        #Normalize the histogram as min max
        base_histogram = base_histogram - np.min(base_histogram,axis=1,keepdims=True)
        base_histogram = base_histogram / np.max(base_histogram,axis=1,keepdims=True)
        base_histogram = base_histogram.astype(np.float32)

        return base_histogram
class SpatioChannelReshaping(ActivationProcessor):
    def __init__(self, config):
        self.config = config
    def process(self, **kwargs):
        processed_activation_list= []
        activation = kwargs.get('activation')
        b,c,h,w = activation.shape
        assert c == 256
        H,W = h,w
       
        if isinstance(activation,torch.Tensor):
            activation = activation.detach().cpu().numpy()
        #if the activation is a list, convert it to numpy, may be not needed
        if isinstance(activation,list):
            activation = np.array(activation)
        grid_size = self.config['grid_size']  # since 256 = 16x16
        Hn = grid_size * H  # new height
        Wn = grid_size * W  # new width

        # Create an empty array for the new grid
        output_array = np.zeros((b,1,Hn, Wn))

        # Iterate over each channel and place it in the correct position on the grid
        for i in range(grid_size):
            for j in range(grid_size):
                channel_index = i * grid_size + j
                # Place the channel's activation map in the correct grid position
                output_array[:,0,i*H:(i+1)*H, j*W:(j+1)*W] = activation[:,channel_index, :, :]
        return output_array
import numpy as np

class GraphActivationProcessor(ActivationProcessor):
    def __init__(self, config):
        self.config = config

    def process(self, **kwargs):
        def index_to_1d(index, c, h):
            return (index[0] * h + index[1]) * c + index[2]

        # Extract 'activation' from kwargs and assume it's a 4D numpy array.
        activation = kwargs.get('activation')  # activation shape is (batch_size, channel, height, width)
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()
        # Process each activation in the batch separately.
        batch_edge_indices = []
        batch_node_features = []

        for batch in range(activation.shape[0]):  # Loop over the batch dimension
            single_activation = activation[batch]  # Get the single activation map for the current batch
            c, h, w = single_activation.shape[-3:]
            c_h_product = c * h
            non_zero_indices = np.transpose(np.nonzero(single_activation))
            # Compute linear indices for non-zero elements.
            linear_indices = c_h_product * non_zero_indices[:, 0] + c * non_zero_indices[:, 1] + non_zero_indices[:, 2]
            node_features = [[*idx, single_activation[tuple(idx)]] for idx in non_zero_indices]
            node_index_dict = {tuple(idx): i for i, idx in enumerate(non_zero_indices)}
            # Pad the array with zeros to avoid boundary checks.
            padded_arr = np.pad(single_activation, ((1, 1), (1, 1), (1, 1)), mode='constant')

            # Find connections by shifting the padded array and checking non-zero overlaps.
            shift_i = padded_arr[2:, 1:-1, 1:-1] != 0
            shift_j = padded_arr[1:-1, 2:, 1:-1] != 0
            shift_k = padded_arr[1:-1, 1:-1, 2:] != 0

            # Calculate edges based on shifted indices.
            edges_i = np.argwhere(shift_i & (single_activation != 0))
            edges_j = np.argwhere(shift_j & (single_activation != 0))
            edges_k = np.argwhere(shift_k & (single_activation != 0))
            # print(edges_i.shape, edges_j.shape, edges_k.shape)
            # Calculate edge connections.
            edge_index = [[node_index_dict[tuple(idx)], node_index_dict[tuple(idx+[1,0,0])]] for idx in edges_i] + \
                         [[node_index_dict[tuple(idx)], node_index_dict[tuple(idx+[0,1,0])]] for idx in edges_j] + \
                         [[node_index_dict[tuple(idx)], node_index_dict[tuple(idx+[0,0,1])]] for idx in edges_k]

            edge_index = np.unique(edge_index, axis=0).T  # Remove duplicate edges., transpose to get 2, as the first dimension.

            # Collect node features.
            
            # print("--",*non_zero_indices[0])
            # print(node_features[0])
            #create an n dimensional numpy array out of the list of lists
            node_features = np.array(node_features)
            # print("Number of nodes (size of x):", node_features.shape)
            # print("Number of edges:", edge_index.max())

            # Append results to the batch lists.
            batch_edge_indices.append(edge_index)
            batch_node_features.append(node_features)

        # Depending on the use case, you might want to return a list of batch results,
        # or aggregate them into a single structure.
        return batch_edge_indices, batch_node_features

# The output_array is now a single grayscale image of shape (Hn, Wn)
class ProcessorEnum(Enum):
    SF = StatisticalFeatureExtraction
    ASH = ExtremelySimpleActivationShaping
    RAW = NoProcessing
    VDNA = VisualDNA
    SCR = SpatioChannelReshaping
    GAP = GraphActivationProcessor