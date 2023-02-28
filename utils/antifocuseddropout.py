"""
Simple implementation of Anti Focused Dropout
From paper : <AntiFocusedDropout for Convolutional Neural Network>
Created by Kent Utama
Date : 2023/02/27
"""
import torch as t

class AntiFocusedDropout(t.nn.Module):
    """Define Anti Focused Dropout module"""

    def __init__(self, low = 0.6, high = 0.9, par_rate = 0.1):
        """
        Args :
            --low: left value in random range, default is 0.6
            --high: right value in random range, default is 0.9
        """
        super(AntiFocusedDropout, self).__init__()

        self.low = low
        self.high = high
        self.avg_pool = t.nn.AdaptiveAvgPool2d(1)
        self.par_rate = par_rate

    def forward(self, x):
        if self.training:
            # 1. First, we need to do global average pooling
            x_ = self.avg_pool(x) # [m, C, 1, 1]
            x_ = x_.squeeze() # [m, C] Each channel activation per image

            # 2. Find the maximum one
            x_max_val, _ = t.max(x_, dim = -1, keepdim = True) # [m, 1] Returns max channel value and channel index per image
            hidden_mask = (x_ == x_max_val) # [m, C]
            hidden_mask = t.unsqueeze(t.unsqueeze(hidden_mask, dim = -1), dim = -1) # Gets max channel index per image
            x_max = x * hidden_mask # Masks every other channel per image
            x_max = t.max(t.max(t.sum(x_max, dim = 1), dim = -1)[0], dim = -1)[0] # [m, 1] Gets highest channel's total activation per image
            x_max = t.unsqueeze(x_max, dim = -1) 
            

            #x_max, _ = t.max(x[x_max_indice, ...], dim = -1, keepdim = True) # [m, 1]
            x_max = t.unsqueeze(t.unsqueeze(x_max, dim = -1), dim = -1) # [m, 1, 1, 1]
            x_max = x_max.repeat(1, x.size(1), x.size(2), x.size(3))
            mask = t.zeros_like(x) # [m, C, H, W]

            # 3. sample
            rand = t.rand_like(x) # Normal distribution mean 0 var 1
            rand = rand * (self.high - self.low) + self.low # [0, 1] -> [low, high]
            x_max *= rand # [m, C, H, W]
            indices = x < x_max # Flipped focuseddropout
            mask += indices
            
            examples = x.size(0)
            num_par = int(self.par_rate * examples)
            mask[num_par:] = 1. # Only par_rate% is masked
            
            # 4. Focused Dropout on some content
            return x * mask
        else:
            return x