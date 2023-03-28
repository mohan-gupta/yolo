import torch

from tqdm import tqdm

import config

def train_one_batch(model, data, loss_fn, optimizer):
    optimizer.zero_grad()
    
    for k, v in data.items():
        data[k] = v.to(config.DEVICE)
    
    labels = data['label']

    preds = model(data['image'])

    loss = loss_fn(preds, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss


def train_loop(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    loop = tqdm(data_loader)
    
    for data in loop:
        batch_loss = train_one_batch(model, data, loss_fn, optimizer)
        
        with torch.no_grad():
            total_loss += batch_loss.item()
        
        loop.set_postfix(dict(
                loss = batch_loss.item(),
            ))
        
    avg_loss = round(total_loss/num_batches, 3)

    return avg_loss

def validate_one_batch(model, data, loss_fn):
    for k, v in data.items():
        data[k] = v.to(config.DEVICE)
    
    labels = data['label']

    preds = model(data['image'])

    loss = loss_fn(preds, labels)
    
    return loss

def validate_loop(model, data_loader, loss_fn):
    model.eval()

    num_batches = len(data_loader)
    loop = tqdm(data_loader)

    total_loss = 0
    
    with torch.no_grad():
        for data in loop:
                loss = validate_one_batch(model, data, loss_fn)

                total_loss += loss.item()
                
                loop.set_postfix(dict(loss = loss.item()))
    
    val_loss = round(total_loss/num_batches, 3)
    return val_loss