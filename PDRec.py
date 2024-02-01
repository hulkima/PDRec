import os
import time
import torch
import argparse
import os
import io
import math
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import ipdb
from models.model import GRU4Rec_withNeg_Dist, SASRec_V1_withNeg_Dist
from models.model import EarlyStopping_onetower
from models.DNN import DNN
import models.gaussian_diffusion as gd
from utils.utils import *


# -*- coding: UTF-8 -*-
plt.switch_backend('agg')
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=2000)
from matplotlib.font_manager import FontManager
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print(mat_fonts)


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def map_into_BCELoss(scores):
    return 1/2 * (scores + 1)

def min_max_normalize_batch(tensor):
    min_val = torch.min(tensor, dim=1)[0]
    max_val = torch.max(tensor, dim=1)[0]
    normalized_tensor = torch.div(tensor - min_val.unsqueeze(1), max_val.unsqueeze(1) - min_val.unsqueeze(1))
    return normalized_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--cross_dataset', default='111', type=str)
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_samples', default=100, type=int)
parser.add_argument('--decay', default=4, type=int)
parser.add_argument('--lr_decay_rate', default=0.99, type=float)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--version', default=None, type=str)
parser.add_argument('--lr_linear', default=0.01, type=float)
parser.add_argument('--start_decay_linear', default=8, type=int)
parser.add_argument('--temperature', default=5, type=float)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)
parser.add_argument('--patience', default=10, type=int)

parser.add_argument('--lr_diff', type=float, default=0.00005, help='learning rate')
parser.add_argument('--weight_decay_diff', type=float, default=0.0)
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for the DNN model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.01, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0005, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.005, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--reweight_version', type=str, default='AllLinear', help='in AllOne, AllLinear, MinMax')
parser.add_argument('--result_path', type=str, default=True, help='the path of result')
parser.add_argument('--filter_prob', type=float, default=0.1, help='the path of result')
parser.add_argument('--scale_weight', type=float, default=1.0, help='the path of result')
parser.add_argument('--scale_max', type=float, default=0.0, help='the path of result')
parser.add_argument('--rank_weight', type=float, default=0.0, help='the path of result')

parser.add_argument('--cal_version', type=int, default=1, help='the path of result')
parser.add_argument('--candidate_min_percentage_user', default=0, type=int)
parser.add_argument('--candidate_max_percentage_user', default=99, type=int)
parser.add_argument('--top_candidate_coarse_num', default=10, type=int)
parser.add_argument('--top_candidate_fine_num', default=10, type=int)
parser.add_argument('--top_candidate_weight', default=0.1, type=float)
parser.add_argument('--base_model', default='GRU4Rec', type=str)

args = parser.parse_args()


SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

result_path = './results_filedile/' + str(args.dataset) + '/PDRec_'+str(args.base_model)+'/'
print("Save in path:", result_path)
if not os.path.isdir(result_path):
    os.makedirs(result_path)
with open(os.path.join(result_path, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()
args.result_path = result_path
        
if __name__ == '__main__':
    dataset = data_partition(args.version, args.dataset, args.cross_dataset, args.maxlen)
    [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_valid_ti_target, user_test_ti_target] = dataset

    print("the user number is:", usernum)
    print("the item number is:", itemnum)

    cc_source = 0.0
    for u in user_train_target:
        if len(user_train_target[u]) > 0:
            cc_source = cc_source + len(user_train_target[u])
    print('average sequence length in source domain: %.2f' % (cc_source / len(user_train_target)))

    
    random_min = 0
    random_max = 0
    random_source_min = 0
    random_source_max = 0
    if args.dataset == 'amazon_toy':
        item_number = interval
        random_min = 1
        random_max = interval + 1
        random_source_min = interval + 1
        random_source_max = itemnum + 1
        print("The min is {} and the max is {} in amazon_toy".format(random_min, random_max))
        print("The min is {} and the max is {} in source domain".format(random_source_min, random_source_max))
    elif args.dataset == 'douban_book':
        item_number = interval
        random_min = 1
        random_max = interval + 1
        random_source_min = interval + 1
        random_source_max = itemnum + 1
        print("The min is {} and the max is {} in amazon_book".format(random_min, random_max))
        print("The min is {} and the max is {} in source domain".format(random_source_min, random_source_max))
    candidate_min_user = math.floor(item_number * args.candidate_min_percentage_user / 100)
    candidate_max_user = math.ceil(item_number * args.candidate_max_percentage_user / 100)
    item_list = torch.arange(start=random_min, end=random_max, step=1, device='cuda', requires_grad=False)
    print("The item_number is:",item_number)
    print("The candidate_min_user is:",candidate_min_user)
    print("The candidate_max_user is:",candidate_max_user)

    #     ipdb.set_trace()
    user_list = []
    for u_i in range(1, usernum):
        if len(user_train_source[u_i]) >= 1 and len(user_train_target[u_i]) >= 2: 
            user_list.append(u_i)    
    num_batch = math.ceil(len(user_list) / args.batch_size) # 908
    if args.base_model == 'GRU4Rec':
        model = GRU4Rec_withNeg_Dist(usernum, itemnum, args).cuda() # no ReLU activation in original SASRec implementation?
    elif args.base_model == 'SASRec':
        model = SASRec_V1_withNeg_Dist(usernum, itemnum, args).cuda() # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    epoch_start_idx = 1
        
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none') # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # set the early stop
    early_stopping = EarlyStopping_onetower(args, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    # set the learning rate scheduler
    if args.lrscheduler == 'Steplr': # 
        learningrate_scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer, step_size=args.decay, gamma=args.lr_decay_rate, verbose=True)
    elif args.lrscheduler == 'ExponentialLR': # 
        learningrate_scheduler = torch.optim.lr_scheduler.ExponentialLR(adam_optimizer, gamma=args.lr_decay_rate, last_epoch=-1, verbose=True)
    elif args.lrscheduler == 'CosineAnnealingLR':
        learningrate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1, verbose=True)

        
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
    model_diff = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).cuda()

    optimizer_diff = torch.optim.AdamW(model_diff.parameters(), lr=args.lr_diff, weight_decay=args.weight_decay_diff)
    print("model_diff ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model_diff.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)

    # Same as the pre-trained TI-DiffRec hyper-parameters
    if args.dataset == 'amazon_toy':
        model_diff_path = './Checkpoint/amazon_toy.pth' 
        args.lr_diff=5e-5
        args.weight_decay_diff=0.5
        args.dims='[1000]'
        args.emb_size=10
        args.mean_type='x0'
        args.steps=10
        args.noise_scale=0.01
        args.noise_min=0.0005
        args.noise_max=0.005
        args.sampling_steps=0
        args.reweight=1
        args.w_min=0.5
        args.w_max=1.0  
        args.reweight_version='AllOne'
        args.log_name='log'
        args.round=1
    elif args.dataset == 'douban_book':
        model_diff_path = './Checkpoint/douban_book.pth'
        args.lr_diff=5e-5
        args.weight_decay_diff=0.5
        args.dims='[256]'
        args.emb_size=8
        args.mean_type='x0'
        args.steps=10
        args.noise_scale=0.01
        args.noise_min=0.0005
        args.noise_max=0.01
        args.sampling_steps=0
        args.reweight=1
        args.w_min=0.3
        args.w_max=1.0  
        args.reweight_version='AllOne'
        args.log_name='log'
        args.round=1
#     ipdb.set_trace()
    model_diff = torch.load(model_diff_path).to('cuda')
    model_diff.eval()

    sampler = WarpSampler_V13_final_please_Diff_TI(random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, None, None, w_min = args.w_min, w_max = args.w_max, reweight_version=args.reweight_version, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    user_nearest_item_raw = torch.zeros([usernum+1, candidate_max_user-candidate_min_user], dtype=torch.int32, device='cuda')
    user_top_item_raw = torch.zeros([usernum+1, args.top_candidate_coarse_num], dtype=torch.int32, device='cuda')
    diff_matrix = torch.zeros([usernum+1, itemnum+1], device='cuda')
    weight_matrix = torch.zeros([usernum+1, itemnum+1], device='cuda')
    with torch.no_grad():
        for u in range(1, usernum + 1):
            if len(user_train_mix[u]) < 1 or len(user_train_source[u]) < 1 or len(user_train_target[u]) < 1: 
                continue
            # init the tensor  
            corpus_target_temp = np.zeros([itemnum+1], dtype=np.float32)
            # set the position-aware weight
            weight_target = scale_withminmax(user_train_ti_target[u], args.w_min, args.w_max, args.reweight_version)
            corpus_target_temp[user_train_target[u]] = weight_target
            corpus_target_temp =  torch.tensor(corpus_target_temp, device='cuda')
            diff_prob = diffusion.p_sample(model_diff, corpus_target_temp.unsqueeze(0), args.sampling_steps, args.sampling_noise).detach().squeeze()
#             ipdb.set_trace()
            sort_indices = torch.sort(input=diff_prob[random_min: random_max], dim=0, descending=True, stable=True)[1][candidate_min_user:candidate_max_user+len(user_train_target[u])+2] # torch.Size([4966])
            user_indices = copy.deepcopy(item_list[sort_indices]) # torch.Size([1001])
        
            sort_top_indices = torch.sort(input=diff_prob[random_min: random_max], dim=0, descending=True, stable=True)[1][0:args.top_candidate_coarse_num] # torch.Size([4966])
            user_top_indices = copy.deepcopy(item_list[sort_top_indices]) # torch.Size([1001])
            for it in (user_train_target[u]+user_valid_target[u]+user_test_target[u]):
                if it in user_indices:
                    user_equal_it_index = torch.nonzero(user_indices == it).squeeze(1)
                    user_indices = del_tensor_ele(user_indices, user_equal_it_index)

            user_nearest_item_raw[u] = user_indices[:candidate_max_user-candidate_min_user]
            user_top_item_raw[u] = user_top_indices[:args.top_candidate_coarse_num]
            diff_matrix[u] = diff_prob
            weight_matrix[u] = corpus_target_temp
            if u % 1000 == 0:
                print("Diffusion user:", u)

    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        t1 = time.time()
        loss_epoch = 0
        loss_weight_epoch = 0
        model.train()
        nearest_index = torch.tensor(np.random.randint(low=0, high=candidate_max_user-candidate_min_user, size=[usernum+1, args.maxlen]), device='cuda')
        user_nearest_item = torch.gather(user_nearest_item_raw, dim=-1,index=nearest_index)

        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices, seq_mix_inter, seq_source_inter, seq_target_inter, seq_mix_inter_temp, seq_source_inter_temp, seq_target_inter_temp = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg, seq_target_inter, seq_target_inter_temp = np.array(u), np.array(seq_target), np.array(pos_target), np.array(neg_target), np.array(seq_target_inter), np.array(seq_target_inter_temp)
            u, seq, pos, neg, seq_target_inter, seq_target_inter_temp = torch.tensor(u,device='cuda'), torch.tensor(seq,device='cuda'), torch.tensor(pos,device='cuda'), torch.tensor(neg,device='cuda'), torch.tensor(seq_target_inter,device='cuda'), torch.tensor(seq_target_inter_temp,device='cuda')
            # NNS module
            neg_diff = torch.index_select(user_nearest_item, dim=0, index=u)
            neg_diff = torch.where(pos>0, neg_diff, torch.zeros(pos.shape, dtype=torch.int32, device='cuda'))
            soft_diff = torch.index_select(user_top_item_raw, dim=0, index=u)
            seq_new = torch.cat([seq[:,1:], pos[:,-1].unsqueeze(1)],dim=1)
            neg_list = []
            neg_list.append(neg.squeeze())
            neg_list.append(neg_diff)
            
            # DPA module
            pos_logits, neg_logits, soft_logits = model(u, seq, pos, neg_list, seq_new, soft_diff)
            sort_soft_index = torch.sort(input=soft_logits, dim=1, descending=True, stable=True)[1][:,0:args.top_candidate_fine_num]
            soft_logits = torch.gather(soft_logits, dim=1, index=sort_soft_index)
            pos_labels, neg_labels, soft_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(pos_logits.shape).cuda(), torch.ones(soft_logits.shape).cuda()

            adam_optimizer.zero_grad()
            indices = torch.where(pos != 0)

            # HBR module
            pos_position = torch.where(pos>0, torch.ones(pos.shape,device='cuda'), torch.zeros(pos.shape,device='cuda'))
            diff_prob = torch.index_select(diff_matrix, dim=0,index=u)
            diff_prob_batch = torch.gather(diff_prob, dim=-1,index=pos.long())
            diff_weight_times = torch.where(torch.min(diff_prob_batch,dim=1)[0]>0, torch.ones(torch.min(diff_prob_batch,dim=1)[0].shape,device='cuda')*0.5, torch.ones(torch.min(diff_prob_batch,dim=1)[0].shape,device='cuda')*1.5)
            diff_prob_batch = torch.where(pos>0, diff_prob_batch, (torch.min(diff_prob_batch,dim=1)[0]*diff_weight_times).unsqueeze(1).repeat([1, diff_prob_batch.shape[1]]))
            diff_prob_batch = min_max_normalize_batch(diff_prob_batch)

            _, sorted_indices_diff = torch.sort(diff_prob_batch, dim=1, descending=False)  # 对每行进行升序排序并获取排序后的索引
            sorted_rank_diff = sorted_indices_diff.argsort(dim=1)
            sorted_rank_diff = sorted_rank_diff - (args.maxlen -1 - pos_position.sum(-1).unsqueeze(1))
            
            rescale = pos_position.sum(-1) / torch.where(torch.isnan(diff_prob_batch.sum(-1)), torch.ones(diff_prob_batch.sum(-1).shape,device='cuda'), diff_prob_batch.sum(-1))
            diff_prob_batch = torch.mul(diff_prob_batch, rescale.unsqueeze(1))
            diff_intermedia_batch = sorted_rank_diff / torch.where(pos_position.sum(-1)>0, pos_position.sum(-1), torch.ones(diff_prob_batch.sum(-1).shape,device='cuda')*1e3).unsqueeze(1)
            diff_rank_batch = torch.where(pos>0, diff_intermedia_batch, torch.zeros(pos.shape,device='cuda'))
            diff_prob_batch = diff_prob_batch * (1-args.rank_weight) + diff_rank_batch * args.rank_weight
            diff_prob_batch = torch.clamp(diff_prob_batch, 0.0, args.scale_max)
        
            diff_weight = diff_prob_batch[indices]
            loss_reweight = diff_weight*args.scale_weight

            loss = (bce_criterion(pos_logits[indices], pos_labels[indices])*loss_reweight).mean()
            loss += bce_criterion(soft_logits, soft_labels).mean() * args.top_candidate_weight 
            for k in range(0, len(neg_logits)):
                loss += bce_criterion(neg_logits[k][indices], neg_labels[indices]).mean() / len(neg_logits)
                
            loss_epoch += loss.item()
            loss_weight_epoch += loss_reweight.mean().item()
            
            loss.backward()
            adam_optimizer.step()
            print("In epoch {} iteration {}: loss is {}, loss_weight_mean is {}".format(epoch, step, loss.item(), loss_reweight.mean().item())) 
            with io.open(result_path + 'loss_log.txt', 'a', encoding='utf-8') as file:
                file.write("In epoch {} iteration {}: loss is {}, loss_weight is {}\n".format(epoch, step, loss.item(), loss_reweight.mean().item()))
        learningrate_scheduler.step()

        t2 = time.time()
        print("In epoch {}: loss is {}, loss_weight is {}, time is {}\n".format(epoch, loss_epoch / num_batch, loss_weight_epoch / num_batch, t2 - t1)) 
        with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
            file.write("In epoch {}: loss is {}, loss_weight is {}, time is {}\n".format(epoch, loss_epoch / num_batch, loss_weight_epoch / num_batch, t2 - t1))

        model.eval()
        # Speed-up evaluation
        if epoch > 50:
            print('Evaluating', end='')
            t_test = evaluate_PDRec(model, dataset, args, user_list)
            t3 = time.time()
            print('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
            print('        test: NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11]))
            
            with io.open(result_path + 'test_performance.txt', 'a', encoding='utf-8') as file:
                file.write('epoch:%d, epoch_time: %.4f(s), total_time: %.4f(s), test:\n' % (epoch, t3-t1, t3-t0))
                file.write('        NDCG@1: %.4f, NDCG@5: %.4f, NDCG@10: %.4f, NDCG@20: %.4f, NDCG@50: %.4f, HR@1: %.4f, HR@5: %.4f, HR@10: %.4f, HR@20: %.4f, HR@50: %.4f, AUC: %.4f, loss: %.4f\n' % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5], t_test[6], t_test[7], t_test[8], t_test[9], t_test[10], t_test[11]))
                
            early_stopping(epoch, model, result_path, t_test)
            if early_stopping.early_stop:
                print("Save in path:", result_path)
                print("Early stopping in the epoch {}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}".format(early_stopping.save_epoch, early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10]))
                with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
                    file.write("Early stopping in the epoch {}, NDCG@1: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@20: {:.4f}, NDCG@50: {:.4f}, HR@1: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@20: {:.4f}, HR@50: {:.4f}, AUC: {:.4f}\n".format(early_stopping.save_epoch, early_stopping.best_performance[0], early_stopping.best_performance[1], early_stopping.best_performance[2], early_stopping.best_performance[3], early_stopping.best_performance[4], early_stopping.best_performance[5], early_stopping.best_performance[6], early_stopping.best_performance[7], early_stopping.best_performance[8], early_stopping.best_performance[9], early_stopping.best_performance[10]))
                break
    
    sampler.close()