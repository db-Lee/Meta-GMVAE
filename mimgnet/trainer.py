import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from model import GMVAE
from data import Data

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data = Data(args.data_dir)

        dataset = TensorDataset(torch.from_numpy(self.data.x_mtr).float())
        sampler = RandomSampler(dataset, replacement=True)
        self.trloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size*args.sample_size,
            sampler=sampler,
            drop_last=True
        )

        self.input_shape = self.data.x_mtr.shape[-1]
        
        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=args.unsupervised_em_iters,
            semisupervised_em_iters=args.semisupervised_em_iters,
            fix_pi=args.fix_pi,   
            component_size=args.way,        
            latent_size=args.latent_size, 
            train_mc_sample_size=args.train_mc_sample_size,
            test_mc_sample_size=args.test_mc_sample_size
        ).to(self.args.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.writer = SummaryWriter(
            log_dir=os.path.join(args.save_dir, "tb_log")
        )

    def train(self):        
        global_epoch = 0
        global_step = 0
        best = 0.0
        iterator = iter(self.trloader)

        while (global_epoch * self.args.freq_iters < self.args.train_iters):
            with tqdm(total=self.args.freq_iters) as pbar:
                for _ in range(self.args.freq_iters):
                        
                    self.model.train()
                    self.model.zero_grad()

                    try:
                        H = next(iterator)[0]
                    except StopIteration:
                        iterator = iter(self.trloader)
                        H = next(iterator)[0]
                                        
                    H = H.to(self.args.device).float()
                    H = H.view(self.args.batch_size, self.args.sample_size, self.input_shape)

                    rec_loss, kl_loss = self.model(H)
                    loss = rec_loss + kl_loss
                    
                    loss.backward()          
                    self.optimizer.step()

                    postfix = OrderedDict(
                        {'rec': '{0:.4f}'.format(rec_loss), 
                        'kld': '{0:.4f}'.format(kl_loss)
                        }
                    )
                    pbar.set_postfix(**postfix)                    
                    self.writer.add_scalars(
                        'train', 
                        {'rec': rec_loss, 'kld': kl_loss}, 
                        global_step
                    )

                    pbar.update(1)
                    global_step += 1

                    if self.args.debug:
                        break

            with torch.no_grad():
                mean, conf = self.eval(shot=self.args.shot, test=False)
                
            self.writer.add_scalars(
                'test', 
                {'acc-mean': mean, 'acc-conf': conf}, 
                global_epoch
            )

            if best < mean:
                best = mean
                state = {
                    'state_dict': self.model.state_dict(),
                    'accuracy': mean,
                    'epoch': global_epoch,
                }
                torch.save(state, os.path.join(self.args.save_dir, 'best.pth'))

            print("{0}shot {1}-th EPOCH Val Accuracy: {2:.4f}, BEST Val Accuracy: {3:.4f}".format(self.args.shot, global_epoch, mean, best))

            global_epoch += 1
        
        del self.model

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,   
            component_size=self.args.way,        
            latent_size=self.args.latent_size, 
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)
        
        state_dict = torch.load(os.path.join(self.args.save_dir, 'best.pth'))['state_dict']
        self.model.load_state_dict(state_dict)

        with torch.no_grad():
            mean, conf = self.eval(shot=1, test=True)
        print("1 shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean, conf))

        with torch.no_grad():
            mean, conf = self.eval(shot=5, test=True)
        print("5 shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean, conf))

        with torch.no_grad():
            mean, conf = self.eval(shot=20, test=True)
        print("20 shot Final Test Accuracy: {0:.4f} Confidnece Interval: {1:.4f}".format(mean, conf))
        
        with torch.no_grad():
            mean, conf = self.eval(shot=50, test=True)
        print("50 shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean, conf))

    def eval(self, shot, test=True):
        
        self.model.eval()
        all_accuracies = np.array([])
        while(True):
            H_tr, y_tr, H_te, y_te = self.data.generate_test_episode(
                way=self.args.way,
                shot=shot,
                query=self.args.query,
                n_episodes=self.args.batch_size,
                test=test
            )
            H_tr = torch.from_numpy(H_tr).to(self.args.device).float()
            y_tr = torch.from_numpy(y_tr).to(self.args.device)
            H_te = torch.from_numpy(H_te).to(self.args.device).float()
            y_te = torch.from_numpy(y_te).to(self.args.device)

            if len(all_accuracies) >= self.args.eval_episodes or self.args.debug:
                break
            else:
                y_te_pred = self.model.prediction(H_tr, y_tr, H_te)
                accuracies = torch.mean(torch.eq(y_te_pred, y_te).float(), dim=-1).cpu().numpy()
                all_accuracies = np.concatenate([all_accuracies, accuracies], axis=0)
        
        all_accuracies = all_accuracies[:self.args.eval_episodes]
        return np.mean(all_accuracies), 1.96*np.std(all_accuracies)/float(np.sqrt(self.args.eval_episodes))
    
