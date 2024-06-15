import torch
from tqdm.auto import tqdm
import time
import numpy as np
from losses import calc_prob_uncertinty
tic, toc = (time.time, time.time)


def train(probe, device, train_loader, optimizer, epoch, loss_func, 
          class_names=None, report=False, verbose_interval=5, layer_num=40,
          head=None, verbose=True, return_raw_outputs=False, one_hot=False, uncertainty=False, **kwargs,):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param train_loader: torch.utils.data.DataLoader of train dataset
    :param optimizer: optimizer for the model
    :param epoch: current epoch of training
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param report: whether to print a classification report of training 
    :param train_verbose: print a train progress report after how many batches of training in each epoch
    :return: average loss, train accuracy, true labels, predictions
    """
    assert (verbose_interval is None) or verbose_interval > 0, "invalid verbose_interval, verbose_interval(int) > 0"
    starttime = tic()
    # Set the model to the train mode: Essential for proper gradient descent
    probe.train()
    loss_sum = 0
    correct = 0
    tot = 0
    
    preds = []
    truths = []
    
    # Iterate through the train dataset
    for batch_idx, batch in enumerate(train_loader):
        batch_size = 1
        target = batch["age"].long().cuda()
        if one_hot:
            target = torch.nn.functional.one_hot(target, **kwargs).float()
        optimizer.zero_grad()

        if layer_num or layer_num == 0:
            act = batch["hidden_states"][:, layer_num,].to("cuda")
        else:
            act = batch["hidden_states"].to("cuda")
        output = probe(act)
        if not one_hot:
            loss = loss_func(output[0], target, **kwargs)
        else:
            loss = loss_func(output[0], target)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.sum().item()  
        if uncertainty:
            pred, uncertainty = calc_prob_uncertinty(output[0].detach().cpu().numpy())
        pred = torch.argmax(output[0], axis=1)

        # In the Scikit-Learn's implementation of OvR Multi-class Logistic Regression. They linearly normalized the predicted probability and then call argmax
        # Below is an equivalent implementation of the scikit-learn's decision function. The only difference is we didn't do the linearly normalization
        # To save some computation time
        if len(target.shape) > 1:
            target = torch.argmax(target, axis=1)
        correct += np.sum(np.array(pred.detach().cpu().numpy()) == np.array(target.detach().cpu().numpy()))
        if return_raw_outputs:
            preds.append(pred.detach().cpu().numpy())
            truths.append(target.detach().cpu().numpy())
        tot += pred.shape[0] 
    
    train_acc = correct / tot
    loss_avg = loss_sum / len(train_loader)
    
    endtime = toc()
    if verbose:
        print('\nTrain set: Average loss: {:.4f} ({:.3f} sec) Accuracy: {:.3f}\n'.\
              format(loss_avg, 
                     endtime-starttime,
                     train_acc))
        
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
        
    if return_raw_outputs:
        return loss_avg, train_acc, preds, truths
    else:
        return loss_avg, train_acc
    

def test(probe, device, test_loader, loss_func, return_raw_outputs=False, verbose=True,
         layer_num=40, scheduler=None, one_hot=False, uncertainty=False, **kwargs):
    """
    :param model: pytorch model (class:torch.nn.Module)
    :param device: device used to train the model (e.g. torch.device("cuda") for training on GPU)
    :param test_loader: torch.utils.data.DataLoader of test dataset
    :param loss_func: loss function for the training
    :param class_names: str Name for the classification classses. used in train report
    :param test_report: whether to print a classification report of testing after each epoch
    :param return_raw_outputs: whether return the raw outputs of model (before argmax). used for auc computation
    :return: average test loss, test accuracy, true labels, predictions, (and raw outputs \ 
    from model if return_raw_outputs)
    """
    # Set the model to evaluation mode: Essential for testing model
    probe.eval()
    test_loss = 0
    tot = 0
    correct = 0
    preds = []
    truths = []
        
    # Do not call gradient descent on the test set
    # We don't adjust the weights of model on the test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_size = 1
            target = batch["age"].long().cuda()
            if one_hot:
                target = torch.nn.functional.one_hot(target, **kwargs).float()
            if layer_num or layer_num == 0:
                act = batch["hidden_states"][:, layer_num,].to("cuda")
            else:
                act = batch["hidden_states"].to("cuda")
            output = probe(act)
            if uncertainty:
                pred, uncertainty = calc_prob_uncertinty(output[0].detach().cpu().numpy())
            pred = torch.argmax(output[0], axis=1)
            
            if not one_hot:
                loss = loss_func(output[0], target, **kwargs)
            else:
                loss = loss_func(output[0], target)
            test_loss += loss.sum().item()  # sum up batch loss

            # In the Scikit-Learn's implementation of OvR Multi-class Logistic Regression. They linearly normalized the predicted probability and then call argmax
            # Below is an equivalent implementation of the scikit-learn's decision function. The only difference is we didn't do the linearly normalization
            # To save some computation time
            if len(target.shape) > 1:
                target = torch.argmax(target, axis=1)
            
            
            pred = np.array(pred.detach().cpu().numpy())
            target = np.array(target.detach().cpu().numpy())
            correct += np.sum(pred == target)
            tot += pred.shape[0] 
            if return_raw_outputs:
                preds.append(pred)
                truths.append(target)
                
    test_loss /= len(test_loader)
    if scheduler:
        scheduler.step(test_loss)
    
    test_acc = correct / tot

    if verbose:
        print('Test set: Average loss: {:.4f},  Accuracy: {:.3f}\n'.format(
              test_loss,
              test_acc))
        
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
        
    # If return the raw outputs (before argmax) from the model
    if return_raw_outputs:
        return test_loss, test_acc, preds, truths
    else:
        return test_loss, test_acc

import torch
from tqdm.auto import tqdm
import time
import numpy as np
from losses import calc_prob_uncertinty
tic, toc = (time.time, time.time)
