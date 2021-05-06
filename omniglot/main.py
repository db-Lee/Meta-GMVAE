import os
import argparse
import torch
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meta-GMVAE for Omniglot')

    # Directory Argument
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    
    # Model Argument
    parser.add_argument('--unsupervised-em-iters', type=int, default=10)
    parser.add_argument('--semisupervised-em-iters', type=int, default=10)
    parser.add_argument('--fix-pi', action='store_true')
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--latent-size', type=int, default=64)
    parser.add_argument('--train-mc-sample-size', type=int, default=32)   
    parser.add_argument('--test-mc-sample-size', type=int, default=32)
    
    # Training Argument
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--sample-size', type=int, default=300)
    parser.add_argument('--train-iters', type=int, default=60000)
    parser.add_argument('--freq-iters', type=int, default=1000)
    parser.add_argument('--eval-episodes', type=int, default=1000)
    parser.add_argument('--way', type=int, default=20)
    parser.add_argument('--query', type=int, default=15)    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-6)


    # System Argument
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    args.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()
