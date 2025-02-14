import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader_torch import TSFDataLoader
from models.tsmixer_torch import build_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TSMixer-PyTorch for Time Series Forecasting')
    
    # Basic config
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', type=str, default='tsmixer',
                      help='model name, options: [tsmixer, tsmixer_rev_in]')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='device to use for training')
    
    # Data loader
    parser.add_argument('--data', type=str, default='weather',
                      choices=['electricity', 'exchange_rate', 'national_illness',
                              'traffic', 'weather', 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'],
                      help='dataset name')
    parser.add_argument('--feature_type', type=str, default='M',
                      choices=['S', 'M', 'MS'],
                      help='forecasting task type')
    parser.add_argument('--target', type=str, default='OT',
                      help='target feature in S or MS task')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                      help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=336,
                      help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                      help='prediction sequence length')
    
    # Model hyperparameters
    parser.add_argument('--n_block', type=int, default=2,
                      help='number of TSMixer blocks')
    parser.add_argument('--ff_dim', type=int, default=2048,
                      help='fully-connected feature dimension')
    parser.add_argument('--dropout', type=float, default=0.05,
                      help='dropout rate')
    parser.add_argument('--norm_type', type=str, default='B',
                      choices=['L', 'B'], help='LayerNorm or BatchNorm')
    parser.add_argument('--activation', type=str, default='relu',
                      choices=['relu', 'gelu'], help='activation function')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                      help='number of workers for data loading')
    parser.add_argument('--train_epochs', type=int, default=100,
                      help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='learning rate')
    parser.add_argument('--patience', type=int, default=5,
                      help='patience for early stopping')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    return args

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment ID
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}'
    exp_id += f'_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}'
    exp_id += f'_dp{args.dropout}_fd{args.ff_dim}'
    
    # Initialize data loaders
    data_loader = TSFDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
        args.num_workers
    )
    train_loader = data_loader.get_train()
    val_loader = data_loader.get_val()
    test_loader = data_loader.get_test()
    
    # Initialize model
    model = build_model(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        n_block=args.n_block,
        dropout=args.dropout,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = checkpoint_dir / f'{exp_id}_best.pth'
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.train_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch + 1}/{args.train_epochs}:')
        logger.info(f'  Train Loss: {train_loss:.6f}')
        logger.info(f'  Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break
    
    training_time = time.time() - start_time
    logger.info(f'Training finished in {training_time:.2f} seconds')
    logger.info(f'Best epoch: {best_epoch + 1} with validation loss: {best_val_loss:.6f}')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss = validate(model, test_loader, criterion, device)
    logger.info(f'Test Loss: {test_loss:.6f}')

if __name__ == '__main__':
    main() 