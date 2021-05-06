import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from module import MimgnetEncoder
from data import MimgNetDataset
from utils import NTXentLoss

class Trainer(object):
    def __init__(self, args):
        self.args = args
        dataset = MimgNetDataset(os.path.join(self.args.data_dir), mode='train', simclr=True)
        self.trloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True
        )

        self.encoder = MimgnetEncoder(
            hidden_size=args.hidden_size
        ).to(args.device)
        last_hidden_size = 2*2*args.hidden_size
        self.l1 = nn.Linear(last_hidden_size, last_hidden_size).to(args.device)
        self.l2 = nn.Linear(last_hidden_size, int(0.5*last_hidden_size)).to(args.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())+list(self.l1.parameters())+list(self.l2.parameters()),
            lr=args.lr
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(self.trloader), 
            eta_min=0,
            last_epoch=-1
        )

        self.criterion = NTXentLoss(
            device=args.device, 
            batch_size=args.batch_size, 
            temperature=0.5, 
            use_cosine_similarity=True
        )
        
    def train(self):
        global_step = 0
        best_loss = 1000000.0

        for global_epoch in range(self.args.train_epochs):
            avg_loss = []
            with tqdm(total=len(self.trloader)) as pbar:
                for xis, xjs in self.trloader:
                    self.encoder.train()
                    self.l1.train()
                    self.l2.train()

                    self.encoder.zero_grad()
                    self.l1.zero_grad()
                    self.l2.zero_grad()

                    xis = xis.to(self.args.device)
                    xjs = xjs.to(self.args.device)

                    zis = self.l2(F.relu(self.l1(self.encoder(xis))))
                    zjs = self.l2(F.relu(self.l1(self.encoder(xjs))))

                    zis = F.normalize(zis, dim=1)
                    zjs = F.normalize(zjs, dim=1)

                    loss = self.criterion(zis, zjs)           
                    loss.backward()          
                    self.optimizer.step()

                    postfix = OrderedDict(
                        {'loss': '{0:.4f}'.format(loss)}
                    )
                    pbar.set_postfix(**postfix)

                    pbar.update(1)
                    global_step += 1
                    avg_loss.append(loss.item())

                    if self.args.debug:
                        break

            if global_epoch >= 10:
                self.scheduler.step()

            avg_loss = np.mean(avg_loss)
            state = {
                'encoder_state_dict': self.encoder.state_dict(),
                'l1_state_dict': self.l1.state_dict(),
                'l2_state_dict': self.l2.state_dict()
            }
            torch.save(state, os.path.join(self.args.save_dir, 'best.pth'))

            print("{0}-th EPOCH Loss: {1:.4f}".format(global_epoch, avg_loss))

            if self.args.debug:
                break
        
        self.save_feature()

    def save_feature(self):
        self.encoder.eval()
        os.makedirs(self.args.feature_save_dir, exist_ok=True)
        
        for mode in ['train', 'val', 'test']:
            dataset = MimgNetDataset(os.path.join(self.args.data_dir), mode=mode)
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=False
            )            
            features = []
            print("START ENCODING {} SET".format(mode))
            with tqdm(total=len(loader)) as pbar:
                for x in loader:
                    x = x.to(self.args.device)
                    f = self.encoder(x)
                    features.append(f.detach().cpu().numpy())
                    pbar.update(1)
            features = np.concatenate(features, axis=0)
            print("SAVE ({0}, {1}) shape array".format(features.shape[0], features.shape[1]))
            np.save(os.path.join(self.args.feature_save_dir, "{}_features.npy".format(mode)), features)
