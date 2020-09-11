from dataset.datates import  CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.tools import vec_text
import time
from loguru import logger
import torch.nn as nn
from Model.net import CNN
import json
import os
import torch
class Ocr():
    def __init__(self):
      self.transforms = Compose([ToTensor()])
      # 初始化tensorboard
      self.writer =SummaryWriter()
      self._init_pro()
      self.cnn = CNN(self.sample_json)
      self.train()


    def _init_pro(self):
        logger.info("正在加载配置")
        with open('./conf/sample_config.json',encoding='utf-8') as f:
            self.sample_json=json.loads(f.read())

        self.train_image_dir=self.sample_json['train_image_dir']
        self.image_width=self.sample_json['image_width']
        self.image_height=self.sample_json['image_height']
        self.max_captcha=self.sample_json['max_captcha']
        self.char_set=self.sample_json['char_set']
        self.train_batch_size=self.sample_json['train_batch_size']
        self.test_batch_size=self.sample_json['test_batch_size']
        self.model_save_dir=self.sample_json['model_save_dir']
        self.model_save_path=self.sample_json['model_save_path']
        self.base_lr=self.sample_json['base_lr']
        self.max_epoch=self.sample_json['epoch']
        train_dataset = CaptchaData(self.train_image_dir,
                                    transform=self.transforms,
                                    sample_conf=self.sample_json)
        self.train_data_loader = DataLoader(train_dataset,
                                       batch_size=self.train_batch_size,
                                       num_workers=0,
                                       shuffle=True, drop_last=True)


    def calculat_acc(self,output, target):
        """
        首先   output.shape[128 144]
              target shape[128 144]
        然后view转化（-1 36）
              output.shape[512, 36]
              target shape[512, 36]
        求出每一行的最大值
                torch.argmax(output, dim=1)
                output.shape[512]
        再view转化（-1 4）
               output.shape[128,4]
        即是[batch char_set_index]

        :param output:
        :param target:
        :return:
        """
        output, target = output.view(-1, len(self.char_set)), target.view(-1,  len(self.char_set))
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        correct_list = []
        for i, j in zip(target, output):
            if torch.equal(i, j):
                correct_list.append(1)
            else:
                correct_list.append(0)
        acc = sum(correct_list) / len(correct_list)
        return acc


    def train(self):
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

        if os.path.exists(self.model_save_path):
            logger.info("开始读取模型文件")
            self.cnn.load_state_dict(torch.load(self.model_save_path))

        if torch.cuda.is_available():
            self.cnn=self.cnn.cuda()
            logger.info("cuda 可用")
        else:
            logger.info("cuda 无法使用")

        logger.info("开始训练数据")

        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.base_lr)
        criterion = nn.MultiLabelSoftMarginLoss()

        train_loss_history = []
        train_acc_history = []
        self.cnn.train()
        for epoch in range(self.max_epoch):
            start_ = time.time()
            for img, target in self.train_data_loader:
                img = Variable(img)
                target = Variable(target)

                if torch.cuda.is_available():
                    img=img.cuda()
                    target=target.cuda()
                output = self.cnn(img)

                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = self.calculat_acc(output, target)
                train_acc_history.append(float(acc))
                train_loss_history.append(float(loss))

            logger.info('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor( train_loss_history)),
                torch.mean(torch.Tensor( train_acc_history)),
            ))
            if epoch % 10 == 0:
                torch.save(self.cnn.state_dict(), self.model_save_path)
                logger.info("第{}次保存模型成功".format(int(epoch / 10) + 1))


            # self.writer.add_scalar('data/scalar1', s1[0],epoch)
            #
            # self.writer.add_scalar('data/scalar2', s2[0], epoch)

            self.writer.add_scalar('data/train_loss', torch.mean(torch.Tensor(train_loss_history)), epoch)
            self.writer.add_scalar('data/train_acc', torch.mean(torch.Tensor(train_acc_history)), epoch)





if __name__ == '__main__':
    train_path=''
    test_path=''
    ocr=Ocr()
