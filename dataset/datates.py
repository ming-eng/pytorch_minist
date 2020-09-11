from torch.utils.data import Dataset
from dataset.loaddata import make_dataset
from utils.tools import  img_loader
import torch
class CaptchaData(Dataset):
    def __init__(self,datapath,transform,sample_conf):
        super(CaptchaData, self).__init__()
        self.datapath=datapath
        self.transform=transform
        self.samples=make_dataset(datapath,sample_conf)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        img_path,target=self.samples[index]
        img=img_loader(img_path)
        if self.transform:
            img=self.transform(img)
        return img,torch.Tensor(target)



