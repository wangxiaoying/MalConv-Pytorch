# coding: utf-8
import os
import time
import sys
import yaml
import numpy as np
import pandas as pd
from src.util import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

tt_start = time.time();

# Load config file for experiment
try:
    config_path = sys.argv[1]
    seed = int(sys.argv[2])
    conf = yaml.load(open(config_path,'r'))
except:
    print('Usage: python3 run_exp.py <config file path> <seed>')
    sys.exit()

print('Time lapse after loading config: ', time.time()-tt_start)

exp_name = conf['exp_name']+'_sd_'+str(seed)
print('Experiment:')
print('\t',exp_name)

np.random.seed(seed)
torch.manual_seed(seed)

train_data_path = conf['train_data_path']
train_label_path = conf['train_label_path']

valid_data_path = conf['valid_data_path']
valid_label_path = conf['valid_label_path']

log_dir = conf['log_dir']
pred_dir = conf['pred_dir']
checkpoint_dir = conf['checkpoint_dir']


log_file_path = log_dir+exp_name+'.log'
chkpt_acc_path = checkpoint_dir+exp_name+'.model'
pred_path = pred_dir+exp_name+'.pred'

# Parameters
use_gpu = conf['use_gpu']
use_cpu = conf['use_cpu']
learning_rate = conf['learning_rate']
max_step = conf['max_step']
test_step = conf['test_step']
batch_size = conf['batch_size']
first_n_byte = conf['first_n_byte']
window_size = conf['window_size']
display_step = conf['display_step']

sample_cnt = conf['sample_cnt']

print('Time lapse after setup all parameters: ', time.time()-tt_start)

# Load Ground Truth.
tr_label_table = pd.read_csv(train_label_path,header=None,index_col=0)
tr_label_table = tr_label_table.rename(columns={1:'ground_truth'})
val_label_table = pd.read_csv(valid_label_path,header=None,index_col=0)
val_label_table = val_label_table.rename(columns={1:'ground_truth'})

print('Time lapse after load training and test label: ', time.time()-tt_start)

# Merge Tables and remove duplicate
tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
val_table = val_label_table.groupby(level=0).last()
del val_label_table
tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

print('Time lapse after merge and remove duplicate labels: ', time.time()-tt_start)

print('Training Set:')
print('\tTotal',len(tr_table),'files')
print('\tMalware Count :',tr_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:',tr_table['ground_truth'].value_counts()[0])


print('Validation Set:')
print('\tTotal',len(val_table),'files')
print('\tMalware Count :',val_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:',val_table['ground_truth'].value_counts()[0])

if sample_cnt != 1:
    tr_table = tr_table.sample(n=sample_cnt,random_state=seed)


dataloader = DataLoader(ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth),first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=use_cpu)
validloader = DataLoader(ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth),first_n_byte),
                        batch_size=batch_size, shuffle=False, num_workers=use_cpu)

valid_idx = list(val_table.index)
del tr_table
del val_table

print('Time lapse after load all data: ', time.time()-tt_start)

malconv = MalConv(input_length=first_n_byte,window_size=window_size)
cross_entropy_loss = nn.CrossEntropyLoss()
adam_optim = optim.Adam([{'params':malconv.parameters()}],lr=learning_rate)

if use_gpu:
    malconv = malconv.cuda()
    cross_entropy_loss = cross_entropy_loss.cuda()

print('Time lapse after setup model: ', time.time()-tt_start)

step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
valid_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}'
log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
history = {}
history['tr_loss'] = []
history['tr_acc'] = []

log = open(log_file_path,'w')
log.write('step,tr_loss, tr_acc, val_loss, val_acc, time\n')

valid_best_acc = 0.0
total_step = 0
step_cost_time = 0

print('Time lapse before start running model: ', time.time()-tt_start)

while total_step < max_step:

    # Training
    for step,batch_data in enumerate(dataloader):
        start = time.time()

        adam_optim.zero_grad()

        cur_batch_size = batch_data[0].size(0)

        exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
        exe_input = Variable(exe_input.long(),requires_grad=False)

        label = batch_data[1].cuda() if use_gpu else batch_data[1]
        label = Variable(label.long(),requires_grad=False).view(-1)

        pred = malconv(exe_input)
        loss = cross_entropy_loss(pred, label)
        loss.backward()
        adam_optim.step()
        _, predval = torch.max(pred, 1)

        history['tr_loss'].append(loss.cpu().data.numpy())
        history['tr_acc'].extend(list(torch.eq(label.cpu().data, predval.cpu().data).cpu().data.numpy()))

        step_cost_time = time.time()-start

        if (step+1)%display_step == 0:
            print(step_msg.format(total_step,np.mean(history['tr_loss']),
                                  np.mean(history['tr_acc']),step_cost_time),end='\n',flush=True)
        total_step += 1

        # Interupt for validation
        if total_step%test_step ==0:
            break


    # Testing
    test_start = time.time()
    history['val_loss'] = []
    history['val_acc'] = []
    history['val_pred'] = []

    for _,val_batch_data in enumerate(validloader):
        cur_batch_size = val_batch_data[0].size(0)

        exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
        exe_input = Variable(exe_input.long(),requires_grad=False)

        label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
        label = Variable(label.long(),requires_grad=False).view(-1)

        pred = malconv(exe_input)
        loss = cross_entropy_loss(pred, label)

        history['val_loss'].append(loss.cpu().data.numpy())
        _, predval = torch.max(pred.cpu().data, 1)
        history['val_acc'].extend(list(torch.eq(label.cpu().data, predval).cpu().data.numpy()))
        history['val_pred'].append(list(predval.cpu().data.numpy()))

    print(log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                    np.mean(history['val_loss']), np.mean(history['val_acc']),step_cost_time),
          file=log,flush=True)

    print(valid_msg.format(total_step,np.mean(history['tr_loss']),np.mean(history['tr_acc']),
                           np.mean(history['val_loss']),np.mean(history['val_acc'])))
    if valid_best_acc < np.mean(history['val_acc']):
        valid_best_acc = np.mean(history['val_acc'])
        torch.save(malconv,chkpt_acc_path)
        print('Checkpoint saved at',chkpt_acc_path)
        write_pred(history['val_pred'],valid_idx,pred_path)
        print('Prediction saved at', pred_path)

    history['tr_loss'] = []
    history['tr_acc'] = []
    print('Time cost for one test round: ', time.time()-test_start)

print('Time lapse for all: ', time.time()-tt_start)
