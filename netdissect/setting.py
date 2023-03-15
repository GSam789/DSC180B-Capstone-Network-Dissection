import torch, torchvision, os, collections
import oldalexnet
import oldresnet152
import oldvgg16
import renormalize
import parallelfolder
import segmenter
import torchvision.datasets as datasets
import matplotlib.image as mpimg
import numpy as np
import torchvision.transforms as transforms

def load_proggan(domain):
    # Automatically download and cache progressive GAN model
    # (From Karras, converted from Tensorflow to Pytorch.)
    from . import proggan
    weights_filename = dict(
        bedroom='proggan_bedroom-d8a89ff1.pth',
        church='proggan_churchoutdoor-7e701dd5.pth',
        conferenceroom='proggan_conferenceroom-21e85882.pth',
        diningroom='proggan_diningroom-3aa0ab80.pth',
        kitchen='proggan_kitchen-67f1e16c.pth',
        livingroom='proggan_livingroom-5ef336dd.pth',
        restaurant='proggan_restaurant-b8578299.pth',
        celebhq='proggan_celebhq-620d161c.pth')[domain]
    # Posted here.
    url = 'https://dissect.csail.mit.edu/models/' + weights_filename
    try:
        sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1+
    except:
        sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
    model = proggan.from_state_dict(sd)
    return model

def load_classifier(architecture, data = None):
    if data == 'cifar':
        model_factory = dict(
                alexnet=oldalexnet.AlexNet,
                vgg16=oldvgg16.vgg16,
                resnet152=oldresnet152.OldResNet152)[architecture]
        weights_filename = '/content/drive/MyDrive/vgg16_cifar100.pth'
        sd = torch.load(weights_filename)
        model = model_factory(num_classes=100)

        model.load_state_dict(sd)
        model.eval()
    else:
        model_factory = dict(
                alexnet=oldalexnet.AlexNet,
                vgg16=oldvgg16.vgg16,
                resnet152=oldresnet152.OldResNet152)[architecture]
        weights_filename = dict(
                alexnet='alexnet_places365-92864cf6.pth',
                vgg16='vgg16_places365-0bafbc55.pth',
                resnet152='resnet152_places365-f928166e5c.pth')[architecture]
        model = model_factory(num_classes=365)
        baseurl = 'https://dissect.csail.mit.edu/models/'
        url = baseurl + weights_filename
        try:
            sd = torch.hub.load_state_dict_from_url(url) # pytorch 1.1
        except:
            sd = torch.hub.model_zoo.load_url(url) # pytorch 1.0
        model.load_state_dict(sd)
        model.eval()
    return model

def load_dataset(domain, split=None, full=False, crop_size=None, download=True):
    if domain in ['places', 'imagenet', 'cifar']:
        if split is None:
            split = 'val'
        dirname = 'datasets/%s/%s' % (domain, split)
        if download and not os.path.exists(dirname) and domain == 'places':
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'https://dissect.csail.mit.edu/datasets/' +
                'places_%s.zip' % split,
                'datasets',
                md5=dict(val='593bbc21590cf7c396faac2e600cd30c',
                         train='d1db6ad3fc1d69b94da325ac08886a01')[split])
            places_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(crop_size or 224),
                torchvision.transforms.ToTensor(),
                renormalize.NORMALIZER['imagenet']])
            return parallelfolder.ParallelImageFolders([dirname],
                    classification=True,
                    shuffle=True,
                    transform=places_transform)
        elif download and not os.path.exists(dirname) and domain == 'cifar':
            os.makedirs('datasets', exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'https://www.cs.toronto.edu/~kriz/' +
                'cifar-100-python.tar.gz',
                'datasets',
                md5='eb9058c3a382ffc7106e4002c42a8d85')

            cifar_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(crop_size or 224),
                torchvision.transforms.ToTensor(),
                renormalize.NORMALIZER['cifar']])

            labels = unpickle('datasets/cifar-100-python/meta')[b'fine_label_names']
            for i in labels:
                os.makedirs('datasets/cifar/val/{}'.format(i.decode('utf-8')), exist_ok=True)

            test_image_name = unpickle('datasets/cifar-100-python/test')[b'filenames']
            test_image_label = unpickle('datasets/cifar-100-python/test')[b'fine_labels']
            test_image_data = unpickle('datasets/cifar-100-python/test')[b'data']

            for data in tuple(zip(test_image_name, test_image_data, test_image_label)):
                with open(os.path.join('datasets/cifar/val/{}'.format(labels[data[2]].decode('utf-8')), data[0].decode('utf-8')), 'wb') as f:
                    img = data[1].reshape((3, 32, 32)).transpose((1, 2, 0))
                    mpimg.imsave(f, img, format='jpg')
                with open('datasets/cifar/val/val.txt', 'a') as f:
                    f.write('val/{}/{}\n'.format(labels[data[2]].decode('utf-8'), data[0].decode('utf-8')))
                                                                

            return parallelfolder.ParallelImageFolders([dirname],
                    classification=True,
                    shuffle=True,
                    transform=cifar_transform)
        
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
        
def load_segmenter(segmenter_name='netpqc'):
    '''Loads the segementer.'''
    all_parts = ('p' in segmenter_name)
    quad_seg = ('q' in segmenter_name)
    textures = ('x' in segmenter_name)
    colors = ('c' in segmenter_name)

    segmodels = []
    segmodels.append(segmenter.UnifiedParsingSegmenter(segsizes=[256],
            all_parts=all_parts,
            segdiv=('quad' if quad_seg else None)))
    if textures:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'texture')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="texture", segarch=("resnet18dilated", "ppm_deepsup")))
    if colors:
        segmenter.ensure_segmenter_downloaded('datasets/segmodel', 'color')
        segmodels.append(segmenter.SemanticSegmenter(
            segvocab="color", segarch=("resnet18dilated", "ppm_deepsup")))
    if len(segmodels) == 1:
        segmodel = segmodels[0]
    else:
        segmodel = segmenter.MergedSegmenter(segmodels)
    seglabels = [l for l, c in segmodel.get_label_and_category_names()[0]]
    segcatlabels = segmodel.get_label_and_category_names()[0]
    return segmodel, seglabels, segcatlabels

if __name__ == '__main__':
    main()