import torchvision
import torch.nn.functional as F 
from torch import nn
#from config_reid import DefaultConfigs
#config = DefaultConfigs()
def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))

def get_net(config, model_name='resnet50', is_pretrained=True):
    #return MyModel(torchvision.models.resnet101(pretrained = is_pretrained))
    #import ipdb;ipdb.set_trace()
    if model_name=='alexnet':
        model = torchvision.models.alexnet(pretrained = is_pretrained)  
        model.classifier[-1] = nn.Linear(4096,config.num_classes)
        
    if model_name=='resnet18':
        model = torchvision.models.resnet18(pretrained = is_pretrained)   
        
    if model_name=='resnet34':
        model = torchvision.models.resnet34(pretrained = is_pretrained)   
        
    if model_name=='resnet50':
        model = torchvision.models.resnet50(pretrained = is_pretrained)   
        
    if model_name=='resnet101':
        model = torchvision.models.resnet101(pretrained = is_pretrained)   
        
    if model_name=='resnet152':
        model = torchvision.models.resnet152(pretrained = is_pretrained)   
        
    if model_name=='vgg11':
        model = torchvision.models.vgg11(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg13':
        model = torchvision.models.vgg13(pretrained = is_pretrained)
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg16':
        model = torchvision.models.vgg16(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg19':
        model = torchvision.models.vgg19(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        
    if model_name=='vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)

    if model_name=='squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained = is_pretrained) 
        model.classifier[1]=nn.Conv2d(512, config.num_classes, kernel_size=(1, 1), stride=(1, 1))

    if model_name=='densenet161':
        model = torchvision.models.densenet161(pretrained = is_pretrained)  
        input_features = model.classifier.in_features
        model.classifier=nn.Linear(input_features,config.num_classes)

    if model_name=='inception_v3':
        model = torchvision.models.inception_v3(pretrained = is_pretrained)  
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)

    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained = is_pretrained)  
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)

    if model_name=='mnasnet1_0':
        model = torchvision.models.mnasnet1_0(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)

    if model_name=='shufflenet_v2_x1_0':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained = is_pretrained)   
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)

    if model_name=='mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)

    if model_name=='mobilenet_v3_large':
        model = torchvision.models.mobilenet_v3_large(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)

    if model_name=='mobilenet_v3_small':
        model = torchvision.models.mobilenet_v3_small(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)

    if model_name=='resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained = is_pretrained) 
        
    if model_name=='wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(pretrained = is_pretrained)  
        

    #for param in model.parameters():
    #    param.requires_grad = False
    if 'resnet' in model_name:
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)
        #print(1111111111111111111111)

    #model.avgpool = nn.AdaptiveAvgPool2d(1)
    #model.fc = nn.Linear(2048,config.num_classes)

    return model

def get_net_target(config, model_name='resnet50', is_pretrained=True):
    #return MyModel(torchvision.models.resnet101(pretrained = is_pretrained))
    #import ipdb;ipdb.set_trace()
    if model_name=='alexnet':
        model = torchvision.models.alexnet(pretrained = is_pretrained)  
        model.classifier[-1] = nn.Linear(4096,config.num_classes)
        targets = model.features
        
    if model_name=='resnet18':
        model = torchvision.models.resnet18(pretrained = is_pretrained) 

        
    if model_name=='resnet34':
        model = torchvision.models.resnet34(pretrained = is_pretrained)   
        
    if model_name=='resnet50':
        model = torchvision.models.resnet50(pretrained = is_pretrained)   
        
    if model_name=='resnet101':
        model = torchvision.models.resnet101(pretrained = is_pretrained)   
        
    if model_name=='resnet152':
        model = torchvision.models.resnet152(pretrained = is_pretrained)   
        
    if model_name=='vgg11':
        model = torchvision.models.vgg11(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features

    if model_name=='vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg13':
        model = torchvision.models.vgg13(pretrained = is_pretrained)
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg16':
        model = torchvision.models.vgg16(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg19':
        model = torchvision.models.vgg19(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features
        
    if model_name=='vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features

    if model_name=='squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained = is_pretrained) 
        model.classifier[1]=nn.Conv2d(512, config.num_classes, kernel_size=(1, 1), stride=(1, 1))
        targets = model.features

    if model_name=='densenet161':
        model = torchvision.models.densenet161(pretrained = is_pretrained)  
        input_features = model.classifier.in_features
        model.classifier=nn.Linear(input_features,config.num_classes)
        targets = model.features

    if model_name=='inception_v3':
        model = torchvision.models.inception_v3(pretrained = is_pretrained)  
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)
        targets = model.Mixed_7c

    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained = is_pretrained)  
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)
        targets = model.inception5b

    if model_name=='mnasnet1_0':
        model = torchvision.models.mnasnet1_0(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.layers

    if model_name=='shufflenet_v2_x1_0':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained = is_pretrained)   
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)
        targets = model.conv5

    if model_name=='mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained = is_pretrained)  
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features[-1]

    if model_name=='mobilenet_v3_large':
        model = torchvision.models.mobilenet_v3_large(pretrained = is_pretrained)    
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features[-1]

    if model_name=='mobilenet_v3_small':
        model = torchvision.models.mobilenet_v3_small(pretrained = is_pretrained)   
        input_features = model.classifier[-1].in_features
        model.classifier[-1]=nn.Linear(input_features,config.num_classes)
        targets = model.features[-1]

    if model_name=='resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained = is_pretrained) 
        
    if model_name=='wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(pretrained = is_pretrained)  
        

    #for param in model.parameters():
    #    param.requires_grad = False
    if 'resnet' in model_name:
        #import ipdb;ipdb;set_trace()
        input_features = model.fc.in_features
        model.fc=nn.Linear(input_features,config.num_classes)
        targets = model.layer4  
    #model.avgpool = nn.AdaptiveAvgPool2d(1)
    #model.fc = nn.Linear(2048,config.num_classes)
    return model,targets

    return model
