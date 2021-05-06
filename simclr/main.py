import os
import argparse
import torch
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimCLR pretrainig for Mini-ImageNet')

    # Directory Argument
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--feature-save-dir', type=str, required=True)
       
    # Model Argument
    parser.add_argument('--hidden-size', type=int, default=64)
    
    # Training Argument
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--train-epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    
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
