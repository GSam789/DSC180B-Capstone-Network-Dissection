import os
from six.moves.urllib.request import urlretrieve

from adversarial_robustness_v2.dataset import *
import tensorflow_datasets

class MNIST(Dataset):
    def __init__(self, data_dir=default_data_dir, **kwargs):
        self.X, self.y, self.Xt, self.yt = load_mnist(data_dir)
        self.feature_names = [str(i) for i in range(728)]
        self.label_names = [str(i) for i in range(10)]
        self.image_shape = (28, 28)
    
def load_mnist(datadir=default_data_dir):
    mnist = tensorflow_datasets.load('mnist')
    print(mnist)
    X, y = mnist['train']['image'], mnist['train']['labels']
    Xt, yt = mnist['test']['image'], mnist['test']['labels']
    return X, y, Xt, yt

if __name__ == '__main__':
    import pdb
    dataset = MNIST()
    pdb.set_trace()
    pass
