import einops
import torch

from tqdm import tqdm

import config

def train_one_batch(model, data, loss_fn, optimizer):
    optimizer.zero_grad()
    
    data['image'] = data['image'].to(config.DEVICE)
    
    data['targets'][0] = data['targets'][0].to(config.DEVICE)
    data['targets'][1] = data['targets'][1].to(config.DEVICE)
    data['targets'][2] = data['targets'][2].to(config.DEVICE)

    preds = model(data['image'])
    
    anchors = torch.tensor(config.ANCHORS)
    scales = torch.tensor(config.S)
    scales = einops.repeat(scales, "n -> n x y", x=anchors.shape[1], y=anchors.shape[2])
    #scaling anchors from 0-1 to [0-13, 0-26 and 0-52]
    scaled_anchors = (anchors * scales).to(config.DEVICE)

    loss = (
        loss_fn(preds[0], data['targets'][0], scaled_anchors[0]) +
        loss_fn(preds[1], data['targets'][1], scaled_anchors[1]) +
        loss_fn(preds[2], data['targets'][2], scaled_anchors[2])
        )
    
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
    data['image'] = data['image'].to(config.DEVICE)
    
    data['targets'][0] = data['targets'][0].to(config.DEVICE)
    data['targets'][1] = data['targets'][1].to(config.DEVICE)
    data['targets'][2] = data['targets'][2].to(config.DEVICE)

    preds = model(data['image'])

    anchors = torch.tensor(config.ANCHORS)
    scales = torch.tensor(config.S)
    scales = einops.repeat(scales, "n -> n x y", x=anchors.shape[1], y=anchors.shape[2])
    
    scaled_anchors = (anchors * scales).to(config.DEVICE)

    loss = (
        loss_fn(preds[0], data['targets'][0], scaled_anchors[0]) +
        loss_fn(preds[1], data['targets'][1], scaled_anchors[1]) +
        loss_fn(preds[2], data['targets'][2], scaled_anchors[2])
        )
    
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