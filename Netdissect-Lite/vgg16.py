def create_vgg_rescaled(subsample, feature):
    tmp = models.vgg16()
    tmp.features = tmp.features[0:17]
    vgg16_rescaled = nn.Sequential()
    modules = []
    
    if feature == 'raw':
        first_in_channels = 1
        first_in_features = 6144
    else:
        first_in_channels = 3
        first_in_features = 576
        
    for layer in tmp.features.children():
        if isinstance(layer, nn.Conv2d):
            if layer.in_channels == 3:
                in_channels = first_in_channels
            else:
                in_channels = int(layer.in_channels/subsample)
            out_channels = int(layer.out_channels/subsample)
            modules.append(nn.Conv2d(in_channels, out_channels, layer.kernel_size, layer.stride, layer.padding))
        else:
            modules.append(layer)
    vgg16_rescaled.add_module('features',nn.Sequential(*modules))
    vgg16_rescaled.add_module('flatten', nn.Flatten())

    modules = []
    for layer in tmp.classifier.children():
        if isinstance(layer, nn.Linear):
            if layer.in_features == 25088:
                in_features = first_in_features
            else:
                in_features = int(layer.in_features/subsample) 
            if layer.out_features == 1000:
                out_features = 2
            else:
                out_features = int(layer.out_features/subsample) 
            modules.append(nn.Linear(in_features, out_features))
        else:
            modules.append(layer)
    vgg16_rescaled.add_module('classifier', nn.Sequential(*modules))
    return vgg16_rescaled
