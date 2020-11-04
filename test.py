import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
import sklearn.model_selection
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import contextlib
from MulticoreTSNE import MulticoreTSNE as TSNE

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class DomainAdaptationTester(object):
    def __init__(self,X1,Y1,X2,Y2):
        assert len(X1)==len(Y1) and len(X2)==len(Y2)
        assert X1.shape[1:]==X2.shape[1:]
        self.X1,self.Y1,self.X2,self.Y2=X1,Y1,X2,Y2
    def test_on_source(self,model,test_size=0.2):
        X1_train,X1_test,Y1_train,Y1_test=sklearn.model_selection.train_test_split(self.X1,self.Y1,test_size=test_size,random_state=123)
        if isinstance(model,nn.Module):
            model.fit(X1_train,Y1_train,test_data=(X1_test,Y1_test))
        else:
            model.fit(X1_train,Y1_train)
        scores=model.score(X1_test,Y1_test)
        return scores
    def test_on_target(self,model):
        if isinstance(model,nn.Module):
            model.fit(self.X1,self.Y1,test_data=(self.X2,self.Y2))
        else:
            model.fit(self.X1,self.Y1)
        scores=model.score(self.X2,self.Y2)
        return scores
    def __show_tsne_plot(self,X,labels):
        unique_labels=np.unique(labels)
        unique_labels=unique_labels[~np.isnan(unique_labels)]
        print(unique_labels)
        n_labels=len(unique_labels)
        
        tsne=TSNE(n_jobs=50,random_state=789)
        X_2d=tsne.fit_transform(X)
        
        plt.figure()
        X_2d_X=X_2d[:,0]
        X_2d_Y=X_2d[:,1]
        for i,c in enumerate(unique_labels):
            idx=(labels==c)
            plt.plot(X_2d_X[idx],X_2d_Y[idx],".",markersize=1,label=str(c))
        plt.legend(bbox_to_anchor=(1.20, 1))

    def show_tsne_plot(self,set="all",sample=2000):
        with temp_seed(124):
            idx_sample1=np.random.choice(len(self.X1),size=sample,replace=False)
            idx_sample2=np.random.choice(len(self.X2),size=sample,replace=False)

        if set=="source":
            X_sample=np.take(self.X1,idx_sample1,0)
            Y_sample=np.take(self.Y1,idx_sample1,0)
        elif set=="target":
            X_sample=np.take(self.X2,idx_sample2,0)
            Y_sample=np.take(self.Y2,idx_sample2,0)
        elif set=="all":
            X_sample1=np.take(self.X1,idx_sample1,0)
            Y_sample1=np.take(self.Y1,idx_sample1,0)
            X_sample2=np.take(self.X2,idx_sample2,0)
            Y_sample2=np.take(self.Y2,idx_sample2,0)

            X_sample=np.concatenate([X_sample1,X_sample2],axis=0)
            Y_sample=np.concatenate([Y_sample1,Y_sample2],axis=0)
        else:
            assert False
        self.__show_tsne_plot(X_sample,Y_sample)
        if set=="all":
            domain_label=np.array([1]*sample+[2]*sample)
            self.__show_tsne_plot(X_sample,domain_label)




def score(model,X,Y,batch_size=128):
    model.eval()
    X_tensor=torch.from_numpy(X)
    Y_tensor=torch.from_numpy(Y)
    
    test_dataset=torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)
    
    pred_list=list()
    device=next(iter(model.parameters())).device
    with torch.no_grad():
        for i,(X_batch,Y_batch) in enumerate(test_loader):
            X_batch=X_batch.to(device)
            Y_batch=Y_batch.to(device)
            pred,_=model(X_batch,1.0)
            pred=pred.argmax(dim=1)
            pred_list.append(pred)
    predictions_tensor=torch.cat(pred_list)
    predictions_arr=predictions_tensor.cpu().numpy()
    score=dict()
    score["accuracy"]=np.mean(predictions_arr==Y)
    return score

def test(dataset_name):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = 'models'
    image_root = os.path.join('dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='dataset',
            train=False,
            transform=img_transform_source,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
