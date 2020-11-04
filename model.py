import torch.nn as nn
import torch.nn.functional as F
import torch
from functions import ReverseLayerF

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class CNNModel1(nn.Module):

    def __init__(self):
        super(CNNModel1, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
    def transform(self,X,layer="fc",batch_size=128):
        assert layer in ("feature","fc")
        device=next(self.parameters()).device
        X_tensor=torch.from_numpy(X)
        
        dataset=torch.utils.data.TensorDataset(X_tensor)
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        self.eval()
        X_transformed_tensor_list=list()
        def fc_transformation(fc,val):
            module_list=list(fc._modules.values())[:6]
            for m in module_list:
                val=m(val)
            return val
        with torch.no_grad():
            for i,(X_batch,) in enumerate(loader):
                X_batch=X_batch.to(device)
                X_batch_transformed_tensor=self.feature(X_batch)
                X_batch_transformed_tensor=X_batch_transformed_tensor.view(X_batch.shape[0],-1)
                if layer=="fc":
                    X_batch_transformed_tensor=fc_transformation(self.class_classifier,X_batch_transformed_tensor)
                X_transformed_tensor_list.append(X_batch_transformed_tensor)
            X_transformed_tensor=torch.cat(X_transformed_tensor_list,dim=0)
        X_transformed_arr=X_transformed_tensor.cpu().numpy()
        return X_transformed_arr

class CNNModel2(nn.Module):

    def __init__(self):
        super(CNNModel2, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_flatten', Flatten())
        self.feature.add_module('f2_fc1', nn.Linear(50 * 4 * 4, 100))
        self.feature.add_module('f2_bn1', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu1', nn.ReLU(True))
        self.feature.add_module('f2_drop1', nn.Dropout())
        self.feature.add_module('f2_fc2', nn.Linear(100, 100))
        self.feature.add_module('f2_bn2', nn.BatchNorm1d(100))
        self.feature.add_module('f2_relu2', nn.ReLU(True))
        self.class_classifier = nn.Sequential()
        
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output