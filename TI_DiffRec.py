"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import io
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils.utils import *

import models.gaussian_diffusion as gd
from models.DNN import DNN
from copy import deepcopy
import math
import random
import ipdb

def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')
parser.add_argument('--seed', type=int, default=2024, help='the random seed')
parser.add_argument('--num_samples', default=100, type=int)

parser.add_argument('--version', default=None, type=str)
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--cross_dataset', default=None, type=str)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--index', default=0, type=int)

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--reweight_version', type=str, default='AllOne', help='in AllOne, AllLinear, MinMax')

parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
args = parser.parse_args()

# ipdb.set_trace()
random_seed = args.seed
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn

print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

result_path = './results_file/' + str(args.dataset) + '/TI_DiffRec/'
print("Save in path:", result_path)
if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()


### DATA LOAD ###
dataset = data_partition(args.version, args.dataset, args.cross_dataset, args.maxlen)

[user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_valid_ti_target, user_test_ti_target] = dataset

num_batch = math.ceil(len(user_train_source) / args.batch_size) # 908
cc_source = 0.0
cc_target = 0.0
for u in user_train_source:
    cc_source = cc_source + len(user_train_source[u])
    cc_target = cc_target + len(user_train_target[u])
    
# Toy_Game----Toy: 8.22 / 12.32 / 20.54
# Toy_Game----Game: 11.73 / 8.36 / 20.10
print('average sequence length in source domain: %.2f' % (cc_source / len(user_train_source)))
print('average sequence length in target domain: %.2f' % (cc_target / len(user_train_source)))
print('average sequence length in both domain: %.2f' % ((cc_source + cc_target) / len(user_train_source)))


random_min = 0
random_max = 0
random_source_min = 0
random_source_max = 0
if args.dataset == 'amazon_toy':
    random_min = 1
    random_max = interval + 1
    random_source_min = interval + 1
    random_source_max = itemnum + 1
    print("The min is {} and the max is {} in amazon_toy".format(random_min, random_max))
    print("The min is {} and the max is {} in source domain".format(random_source_min, random_source_max))
elif args.dataset == 'douban_book':
    random_min = 1
    random_max = interval + 1
    random_source_min = interval + 1
    random_source_max = itemnum + 1
    print("The min is {} and the max is {} in douban_book".format(random_min, random_max))
    print("The min is {} and the max is {} in source domain".format(random_source_min, random_source_max))

sampler = WarpSampler_T_DiffCDR_TI(random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, usernum, itemnum, batch_size=args.batch_size, w_min = args.w_min, w_max = args.w_max, reweight_version = args.reweight_version, n_workers=3)
    

### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

# ipdb.set_trace()
diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, 'cuda').cuda()

### Build MLP ###
out_dims = eval(args.dims) + [itemnum+1] # [1000, 94949]
in_dims = out_dims[::-1] # [94949, 1000]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).cuda()

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")
if args.lrscheduler == 'Steplr': # 
    learningrate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay, gamma=args.lr_decay_rate, verbose=True)
elif args.lrscheduler == 'ExponentialLR': # 
    learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate, last_epoch=-1, verbose=True)
elif args.lrscheduler == 'CosineAnnealingLR':
    learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1, verbose=True)
        
param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

best_recall, best_epoch = -100, 0
best_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 10:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
        
#     ipdb.set_trace()
    t_start = time.time()
    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        user, seq_mix, seq_source, seq_target, seq_mix_temp, seq_source_temp, seq_target_temp = sampler.next_batch()   
        user, seq_mix, seq_source, seq_target, seq_mix_temp, seq_source_temp, seq_target_temp = np.array(user), np.array(seq_mix), np.array(seq_source), np.array(seq_target), np.array(seq_mix_temp), np.array(seq_source_temp), np.array(seq_target_temp)
        user, seq_mix, seq_source, seq_target, seq_mix_temp, seq_source_temp, seq_target_temp = torch.tensor(user,device='cuda'), torch.tensor(seq_mix,device='cuda'), torch.tensor(seq_source,device='cuda'), torch.tensor(seq_target,device='cuda'), torch.tensor(seq_mix_temp,device='cuda'), torch.tensor(seq_source_temp,device='cuda'), torch.tensor(seq_target_temp,device='cuda')
        batch_count += 1
        optimizer.zero_grad()
#         ipdb.set_trace()
        losses = diffusion.training_losses(model, seq_target_temp, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("    In epoch {} iteration {}: loss={:.4f}".format(epoch, step, loss.item())) 
#     ipdb.set_trace()
    t_end = time.time()
    print("Time interval of one epoch:{:.4f}".format(t_end-t_start))
#     ipdb.set_trace()
    learningrate_scheduler.step()
    print("The end batch_count is:",batch_count)
    print("In epoch {}: loss_mean={:.4f}, lr={}".format(epoch, total_loss/step, learningrate_scheduler.get_last_lr())) 

    if epoch % 1 == 0:
        model.eval()
#         ipdb.set_trace()
        t_test = evaluate_T_DiffRec_TI(model, diffusion, dataset, args, random_min, random_max, random_source_min, random_source_max)
        print('epoch:%d, epoch_time: %.4f(s): NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f\n' % (epoch, time.time()-start_time, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10]))
        with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
            file.write('epoch:%d, epoch_time: %.4f(s), NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f\n' % (epoch, time.time()-start_time, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10]))
        
        if t_test[2] > best_recall: # NDCG@10 as selection
            best_recall, best_epoch = t_test[2], epoch
            best_results = t_test

            torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth' \
                .format(result_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
print('Best results: epoch:{:d}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}\n'.format(epoch, best_results[0], best_results[1], best_results[2], best_results[3], best_results[4], best_results[5], best_results[6], best_results[7], best_results[8], best_results[9], best_results[10]))
with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
    file.write('======================================================\n')
    file.write('Best results: epoch:{:d}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}\n'.format(epoch, best_results[0], best_results[1], best_results[2], best_results[3], best_results[4], best_results[5], best_results[6], best_results[7], best_results[8], best_results[9], best_results[10]))        
    
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





