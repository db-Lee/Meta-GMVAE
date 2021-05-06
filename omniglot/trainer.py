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

        self.input_shape = [1, 28, 28]

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=args.unsupervised_em_iters,
            semisupervised_em_iters=args.semisupervised_em_iters,
            fix_pi=args.fix_pi,
            hidden_size=args.hidden_size,
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
        best_1shot = 0.0
        best_5shot = 0.0
        iterator = iter(self.trloader)

        while (global_epoch * self.args.freq_iters < self.args.train_iters):
            with tqdm(total=self.args.freq_iters) as pbar:
                for _ in range(self.args.freq_iters):
                    self.model.train()
                    self.model.zero_grad()

                    try:
                        X = next(iterator)[0]
                    except StopIteration:
                        iterator = iter(self.trloader)
                        X = next(iterator)[0]
                                        
                    X = X.to(self.args.device).float()
                    X = X.view(self.args.batch_size, self.args.sample_size, *self.input_shape)

                    rec_loss, kl_loss = self.model(X)
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
                mean_1shot, conf_1shot = self.eval(shot=1)
                mean_5shot, conf_5shot = self.eval(shot=5)
            
            self.writer.add_scalars(
                'test', 
                {'1shot-acc-mean': mean_1shot, '1shot-acc-conf': conf_1shot,
                 '5shot-acc-mean': mean_5shot, '5shot-acc-conf': conf_5shot}, 
                global_epoch
            )

            if best_1shot < mean_1shot:
                best_1shot = mean_1shot
                state = {
                    'state_dict': self.model.state_dict(),
                    'accuracy': mean_1shot,
                    'epoch': global_epoch,
                }
                torch.save(state, os.path.join(self.args.save_dir, '1shot_best.pth'))
                
            print("1shot {0}-th EPOCH Val Accuracy: {1:.4f}, BEST Accuracy: {2:.4f}".format(global_epoch, mean_1shot, best_1shot))

            if best_5shot < mean_5shot:
                best_5shot = mean_5shot
                state = {
                    'state_dict': self.model.state_dict(),
                    'accuracy': mean_5shot,
                    'epoch': global_epoch,
                }
                torch.save(state, os.path.join(self.args.save_dir, '5shot_best.pth'))

            print("5shot {0}-th EPOCH Val Accuracy: {1:.4f}, BEST Accuracy: {2:.4f}".format(global_epoch, mean_5shot, best_5shot))

            global_epoch += 1

            if self.args.debug:
                break

        del self.model

        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,
            hidden_size=self.args.hidden_size,
            component_size=self.args.way,        
            latent_size=self.args.latent_size, 
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)
        
        state_dict = torch.load(os.path.join(self.args.save_dir, '1shot_best.pth'))['state_dict']
        self.model.load_state_dict(state_dict)
        with torch.no_grad():
            mean_1shot, conf_1shot = self.eval(shot=1, test=True)
        print("1shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean_1shot, conf_1shot))

        del self.model
        
        self.model = GMVAE(
            input_shape=self.input_shape,
            unsupervised_em_iters=self.args.unsupervised_em_iters,
            semisupervised_em_iters=self.args.semisupervised_em_iters,
            fix_pi=self.args.fix_pi,
            hidden_size=self.args.hidden_size,
            component_size=self.args.way,        
            latent_size=self.args.latent_size, 
            train_mc_sample_size=self.args.train_mc_sample_size,
            test_mc_sample_size=self.args.test_mc_sample_size
        ).to(self.args.device)
        
        state_dict = torch.load(os.path.join(self.args.save_dir, '5shot_best.pth'))['state_dict']
        self.model.load_state_dict(state_dict)
        with torch.no_grad():
            mean_5shot, conf_5shot = self.eval(shot=5, test=True)
        print("5shot Final Test Accuracy: {0:.4f} Confidence Interval: {1:.4f}".format(mean_5shot, conf_5shot))
        
    def eval(self, shot, test=False):
        
        self.model.eval()
        all_accuracies = np.array([])
        while(True):
            X_tr, y_tr, X_te, y_te = self.data.generate_test_episode(
                way=self.args.way,
                shot=shot,
                query=self.args.query,
                n_episodes=self.args.batch_size,
                test=test
            )
            X_tr = torch.from_numpy(X_tr).to(self.args.device).float()
            y_tr = torch.from_numpy(y_tr).to(self.args.device)
            X_te = torch.from_numpy(X_te).to(self.args.device).float()
            y_te = torch.from_numpy(y_te).to(self.args.device)

            if len(all_accuracies) >= self.args.eval_episodes:
                break
            else:
                y_te_pred = self.model.prediction(X_tr, y_tr, X_te)
                accuracies = torch.mean(torch.eq(y_te_pred, y_te).float(), dim=-1).cpu().numpy()
                all_accuracies = np.concatenate([all_accuracies, accuracies], axis=0)
        
        all_accuracies = all_accuracies[:self.args.eval_episodes]
        return np.mean(all_accuracies), 1.96*np.std(all_accuracies)/float(np.sqrt(self.args.eval_episodes))
