import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import ipdb
import dill as pkl
import time
from sklearn.metrics import roc_auc_score


def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def scale_withminmax(given_list, min_value, max_value, reweight_version):
    min_list = np.min(given_list)
    max_list = np.max(given_list)
    if min_list == max_list:
#         ipdb.set_trace()
        if reweight_version == "AllOne":
            scaled_array = np.ones(shape=len(given_list),dtype=np.float32)*max_value
        elif reweight_version == "AllLinear":
            scaled_array = np.linspace(min_value, max_value, len(given_list), dtype=np.float32)
        elif reweight_version == "MinMax":
            max_list = max_list + 1e3
            scale_factor = (max_value - min_value) / (max_list - min_list)
            scaled_array = min_value + (np.array(given_list, dtype=np.float32) - min_list) * scale_factor
    else:
        scale_factor = (max_value - min_value) / (max_list - min_list)
        scaled_array = min_value + (np.array(given_list, dtype=np.float32) - min_list) * scale_factor

    return scaled_array
            
def get_exclusive(t1, t2):
    t1_exclusive = t1[(t1.view(1, -1) != t2.view(-1, 1)).all(dim=0)]
    return t1_exclusive

    
# source:book----range[1,interval+1);target:movie[interval+1, itemnum + 1)
def sample_function_T_DiffCDR_TI(random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, usernum, itemnum, batch_size, w_min, w_max, reweight_version, result_queue):        
        
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train_mix[user]) <= 1 or len(user_train_source[user]) <= 1 or len(user_train_target[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)

        # init the tensor
        seq_mix = np.zeros([itemnum+1], dtype=np.float32)
        seq_source = np.zeros([itemnum+1], dtype=np.float32)
        seq_target = np.zeros([itemnum+1], dtype=np.float32)
        seq_mix_temp = np.zeros([itemnum+1], dtype=np.float32)
        seq_source_temp = np.zeros([itemnum+1], dtype=np.float32)
        seq_target_temp = np.zeros([itemnum+1], dtype=np.float32)
        
        # set the position-aware weight
        weight_mix = scale_withminmax(user_train_ti_mix[user], w_min, w_max, reweight_version)
        weight_source = scale_withminmax(user_train_ti_source[user], w_min, w_max, reweight_version)
        weight_target = scale_withminmax(user_train_ti_target[user], w_min, w_max, reweight_version)
        
        mask = np.logical_and(random_source_min <= np.array(user_train_mix[user]), np.array(user_train_mix[user]) < random_source_max)
        weight_mix = np.where(mask, weight_mix / 2, weight_mix)

        # generate the 
        seq_mix[user_train_mix[user]] = 1.0
        seq_source[user_train_source[user]] = 1.0
        seq_target[user_train_target[user]] = 1.0
        
        seq_mix_temp[user_train_mix[user]] = weight_mix
        seq_source_temp[user_train_source[user]] = weight_source
        seq_target_temp[user_train_target[user]] = weight_target
    
        return (user, seq_mix, seq_source, seq_target, seq_mix_temp, seq_source_temp, seq_target_temp)


    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))
                
            
            
class WarpSampler_T_DiffCDR_TI(object):
    def __init__(self, random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, usernum, itemnum, batch_size=64, w_min=0.1, w_max=1.0, reweight_version='AllLinear', n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_T_DiffCDR_TI, args=(random_min,
                                                          random_max, 
                                                          random_source_min, 
                                                          random_source_max,
                                                          user_train_mix,
                                                          user_train_source,
                                                          user_train_target,
                                                          user_train_ti_mix, 
                                                          user_train_ti_source, 
                                                          user_train_ti_target,
                                                          usernum,
                                                          itemnum,
                                                          batch_size,
                                                          w_min,
                                                          w_max,
                                                          reweight_version,
                                                          self.result_queue
                )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()   
            
            
            
# source:book----range[1,interval+1);target:movie[interval+1, itemnum + 1)
def sample_function_V13_final_please_Diff_TI(random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, w_min, w_max, reweight_version, batch_size, maxlen, sample_ratio, result_queue):        
    
    def sample():      
        user = np.random.randint(1, usernum + 1)
        while len(user_train_mix[user]) < 1 or len(user_train_source[user]) < 1 or len(user_train_target[user]) < 1: 
            user = np.random.randint(1, usernum + 1)
        
        user_train_mix_u = np.array(user_train_mix[user])
        user_train_source_u = np.array(user_train_source[user])
        user_train_target_u = np.array(user_train_target[user])

        seq_mix = np.zeros([maxlen], dtype=np.int32)
        seq_source = np.zeros([maxlen], dtype=np.int32)
        seq_target = np.zeros([maxlen], dtype=np.int32)
        pos_target = np.zeros([maxlen], dtype=np.int32)
        neg_target = np.zeros([sample_ratio, maxlen], dtype=np.int32)
        user_train_mix_sequence_for_target_indices = np.zeros([maxlen], dtype=np.int32)
        user_train_source_sequence_for_target_indices = np.zeros([maxlen], dtype=np.int32)

        nxt_target = user_train_target_u[-1] # # 最后一个交互的物品

        idx_mix = maxlen - 1 #49
        idx_source = maxlen - 1 #49
        idx_target = maxlen - 1 #49

        ts_target = set(user_train_target_u) # a set
        for i in reversed(range(0, len(user_train_mix_u))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_mix[idx_mix] = user_train_mix_u[i]
            idx_mix -= 1
            if idx_mix == -1: break
                
        for i in reversed(range(0, len(user_train_source_u))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_source[idx_source] = user_train_source_u[i]
            idx_source -= 1
            if idx_source == -1: break
                
        for i in reversed(range(0, len(user_train_target_u[:-1]))): # reversed是逆序搜索，这里的i指的是交互的物品
            seq_target[idx_target] = user_train_target_u[i]
            pos_target[idx_target] = nxt_target
            if user_train_mix_sequence_for_target[user][i] < -maxlen:
                user_train_mix_sequence_for_target_indices[idx_target] = 0
            else:
                user_train_mix_sequence_for_target_indices[idx_target] = user_train_mix_sequence_for_target[user][i] + maxlen
                
            if user_train_source_sequence_for_target[user][i] < -maxlen or user_train_source_sequence_for_target[user][i] == -len(user_train_source_u)-1:
                user_train_source_sequence_for_target_indices[idx_target] = 0
            else:
                user_train_source_sequence_for_target_indices[idx_target] = user_train_source_sequence_for_target[user][i] + maxlen
            if nxt_target != 0: 
                for j in range(0,sample_ratio):
                    neg_target[j, idx_target] = random_neq(random_min, random_max, ts_target)
            nxt_target = user_train_target_u[i]
            idx_target -= 1
            if idx_target == -1: break
                
                        # init the tensor
        seq_mix_inter = np.zeros([itemnum+1], dtype=np.float32)
        seq_source_inter = np.zeros([itemnum+1], dtype=np.float32)
        seq_target_inter = np.zeros([itemnum+1], dtype=np.float32)
        seq_mix_inter_temp = np.zeros([itemnum+1], dtype=np.float32)
        seq_source_inter_temp = np.zeros([itemnum+1], dtype=np.float32)
        seq_target_inter_temp = np.zeros([itemnum+1], dtype=np.float32)
#         ipdb.set_trace()
        # set the position-aware weight
        weight_mix = scale_withminmax(user_train_ti_mix[user], w_min, w_max, reweight_version)
        weight_source = scale_withminmax(user_train_ti_source[user], w_min, w_max, reweight_version)
        weight_target = scale_withminmax(user_train_ti_target[user], w_min, w_max, reweight_version)
        
                        
        index_tensor = torch.arange(len(user_train_mix_u))
        condition_mask = (user_train_mix_u >= random_source_min) & (user_train_mix_u < random_source_max)
        weight_mix[condition_mask] /= 2
        
        # generate the 
        seq_mix_inter[user_train_mix_u] = 1.0
        seq_source_inter[user_train_source_u] = 1.0
        seq_target_inter[user_train_target_u] = 1.0
        
        seq_mix_inter_temp[user_train_mix_u] = weight_mix
        seq_source_inter_temp[user_train_source_u] = weight_source
        seq_target_inter_temp[user_train_target_u] = weight_target
        
        return (user, seq_mix, seq_source, seq_target, pos_target, neg_target, user_train_mix_sequence_for_target_indices, user_train_source_sequence_for_target_indices, seq_mix_inter, seq_source_inter, seq_target_inter, seq_mix_inter_temp, seq_source_inter_temp, seq_target_inter_temp)

    
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))    
            
class WarpSampler_V13_final_please_Diff_TI(object):
    def __init__(self, random_min, random_max, random_source_min, random_source_max, user_train_mix, user_train_source, user_train_target, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, user_list, itemnum, itemnum_source, itemnum_target, w_min=0.1, w_max=1.0, reweight_version='AllOne', batch_size=64, maxlen=10, n_workers=1, sample_ratio=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_V13_final_please_Diff_TI, args=(random_min, 
                                                                random_max, 
                                                                random_source_min, 
                                                                random_source_max,
                                                                user_train_mix,
                                                                user_train_source,
                                                                user_train_target,
                                                                user_train_ti_mix, 
                                                                user_train_ti_source, 
                                                                user_train_ti_target, 
                                                                user_train_mix_sequence_for_target,
                                                                user_train_source_sequence_for_target, 
                                                                user_list,
                                                                itemnum,
                                                                w_min,
                                                                w_max,
                                                                reweight_version,
                                                                batch_size,
                                                                maxlen,
                                                                sample_ratio,
                                                                self.result_queue
                                                         )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()   
            
            
            
     
            
            
def data_partition(version, fname, dataset_name, maxlen):
    usernum = 0
    itemnum = 0
    user_train = {}
    user_valid = {}
    user_test = {}
    interval = 0
#     ipdb.set_trace()
    # assume user/item index starting from 1
    
    if fname == 'amazon_toy':
        with open('./Dataset/Amazon_toy/toy_log_file_final.pkl', 'rb') as f:
            toy_log_file_final = pkl.load(f)
            
        with open('./Dataset/Amazon_toy/mix_log_file_final.pkl', 'rb') as f:
            mix_log_file_final = pkl.load(f)

        with open('./Dataset/Amazon_toy/item_index_toy.pkl', 'rb') as f:
            item_index_toy = pkl.load(f)

        with open('./Dataset/Amazon_toy/item_index_mix.pkl', 'rb') as f:
            item_index_mix = pkl.load(f)

        with open('./Dataset/Amazon_toy/user_index_overleap.pkl', 'rb') as f:
            user_index_overleap = pkl.load(f)
        with open('./Dataset/Amazon_toy/toy_log_timestep_final.pkl', 'rb') as f:
            toy_log_timestep_final = pkl.load(f)
        
        with open('./Dataset/Amazon_toy/mix_log_timestep_final.pkl', 'rb') as f:
            mix_log_timestep_final = pkl.load(f)
        item_index_toy_array = np.load('./Dataset/Amazon_toy/item_index_toy.npy')

        interval = 37868

    elif fname == 'douban_book':
        with open('./Dataset/Douban_book/book_log_file_final.pkl', 'rb') as f:
            toy_log_file_final = pkl.load(f)
            
        with open('./Dataset/Douban_book/mix_log_file_final.pkl', 'rb') as f:
            mix_log_file_final = pkl.load(f)

        with open('./Dataset/Douban_book/item_index_book.pkl', 'rb') as f:
            item_index_toy = pkl.load(f)
            
        with open('./Dataset/Douban_book/item_index_mix.pkl', 'rb') as f:
            item_index_mix = pkl.load(f)

        with open('./Dataset/Douban_book/user_index_overleap.pkl', 'rb') as f:
            user_index_overleap = pkl.load(f)

        with open('./Dataset/Douban_book/book_log_timestep_final.pkl', 'rb') as f:
            toy_log_timestep_final = pkl.load(f)

        with open('./Dataset/Douban_book/mix_log_timestep_final.pkl', 'rb') as f:
            mix_log_timestep_final = pkl.load(f)

        item_index_toy_array = np.load('./Dataset/Douban_book/item_index_book.npy')

        interval = 33697
#     ipdb.set_trace()
    usernum = len(user_index_overleap.keys()) # 116254
    
    if fname == 'amazon_toy':
        user_train_toy_mix = {}
        user_train_toy_source = {}
        user_train_toy_target = {}
        user_valid_toy_target = {}
        user_test_toy_target = {}
        user_train_toy_mix_sequence_for_target = {}
        user_train_toy_source_sequence_for_target = {}
        
        user_train_ti_toy_mix = {}
        user_train_ti_toy_source = {}
        user_train_ti_toy_target = {}
        user_valid_ti_toy_target = {}
        user_test_ti_toy_target = {}
            
        position_mix = []
        position_source = []
#         ipdb.set_trace()
        itemnum = len(item_index_mix.keys())
        for k in range(1, len(user_index_overleap.keys()) + 1):
            v_mix_toy = copy.deepcopy(mix_log_file_final[k])
            v_toy = copy.deepcopy(toy_log_file_final[k])
            
            t_mix_toy = copy.deepcopy(mix_log_timestep_final[k])
            t_toy = copy.deepcopy(toy_log_timestep_final[k])

            toy_last_name = item_index_toy_array[(v_toy[-1] - 1)] # the name of the last interacted movie in Amazon Movie
            toy_last_id = item_index_mix[toy_last_name] # the name of the the last interacted movie in Amazon Mix
            toy_last_index = np.argwhere(np.array(v_mix_toy)==toy_last_id)[-1].item()
            user_mix_toy = v_mix_toy[:toy_last_index+1]
            user_ti_mix_toy = t_mix_toy[:toy_last_index+1]

            if len(user_mix_toy) < 3:
                ipdb.set_trace()

            user_train_toy_mix[k] = []
            user_train_toy_source[k] = []
            user_train_toy_target[k] = []
            user_valid_toy_target[k] = []
            user_test_toy_target[k] = []
            
            user_train_ti_toy_mix[k] = []
            user_train_ti_toy_source[k] = []
            user_train_ti_toy_target[k] = []
            user_valid_ti_toy_target[k] = []
            user_test_ti_toy_target[k] = []   
            for re_id in reversed(range(0,len(user_mix_toy))):
                if user_mix_toy[re_id] >= interval+1: # from 551942 to XXX, source
                    user_train_toy_source[k].append(user_mix_toy[re_id])
                    user_train_toy_mix[k].append(user_mix_toy[re_id])
                    user_train_ti_toy_source[k].append(user_ti_mix_toy[re_id])
                    user_train_ti_toy_mix[k].append(user_ti_mix_toy[re_id])
                elif user_mix_toy[re_id] <= interval: # from 1 to 551941, target
                    if len(user_test_toy_target[k]) == 0:
                        user_test_toy_target[k].append(user_mix_toy[re_id])
                        user_test_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                    elif len(user_valid_toy_target[k]) == 0:
                        user_valid_toy_target[k].append(user_mix_toy[re_id])
                        user_valid_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                    elif len(user_test_toy_target[k]) == 1 and len(user_valid_toy_target[k]) == 1:
                        user_train_toy_target[k].append(user_mix_toy[re_id])
                        user_train_toy_mix[k].append(user_mix_toy[re_id])
                        user_train_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                        user_train_ti_toy_mix[k].append(user_ti_mix_toy[re_id])
                        
            user_train_toy_mix[k].reverse()
            user_train_toy_source[k].reverse()
            user_train_toy_target[k].reverse()
            user_train_ti_toy_mix[k].reverse()
            user_train_ti_toy_source[k].reverse()
            user_train_ti_toy_target[k].reverse()
            
            
            pos_mix = len(user_train_toy_mix[k])-1
            pos_source = len(user_train_toy_source[k])-1
            mix_sequence_for_target_list = []
            source_sequence_for_target_list = []                
            for i in reversed(list(range(0, len(user_train_toy_mix[k])))):
                if user_train_toy_mix[k][i] >= interval+1:
                    pos_source = pos_source - 1
                elif user_train_toy_mix[k][i] <= interval:
                    mix_sequence_for_target_list.append(pos_mix-1)
                    source_sequence_for_target_list.append(pos_source)
                pos_mix = pos_mix - 1
                
            mix_sequence_for_target = mix_sequence_for_target_list[:-1]
            source_sequence_for_target = source_sequence_for_target_list[:-1]
            mix_sequence_for_target.reverse()
            source_sequence_for_target.reverse()
            
            user_train_toy_mix_sequence_for_target[k] = []
            user_train_toy_source_sequence_for_target[k] = []
            for x in mix_sequence_for_target:
                user_train_toy_mix_sequence_for_target[k].append(x - len(user_train_toy_mix[k]))
                    
            for x in source_sequence_for_target:
                user_train_toy_source_sequence_for_target[k].append(x - len(user_train_toy_source[k]))
            
#         ipdb.set_trace()
        return [user_train_toy_mix, user_train_toy_source, user_train_toy_target, user_valid_toy_target, user_test_toy_target, user_train_toy_mix_sequence_for_target, user_train_toy_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_toy_mix, user_train_ti_toy_source, user_train_ti_toy_target, user_valid_ti_toy_target, user_test_ti_toy_target]   

    elif fname == 'douban_book':
        user_train_toy_mix = {}
        user_train_toy_source = {}
        user_train_toy_target = {}
        user_valid_toy_target = {}
        user_test_toy_target = {}
        user_train_toy_mix_sequence_for_target = {}
        user_train_toy_source_sequence_for_target = {}
        
        user_train_ti_toy_mix = {}
        user_train_ti_toy_source = {}
        user_train_ti_toy_target = {}
        user_valid_ti_toy_target = {}
        user_test_ti_toy_target = {}
            
        position_mix = []
        position_source = []
#         ipdb.set_trace()
        itemnum = len(item_index_mix.keys())
        for k in range(1, len(user_index_overleap.keys()) + 1):
            v_mix_toy = copy.deepcopy(mix_log_file_final[k])
            v_toy = copy.deepcopy(toy_log_file_final[k])
            
            t_mix_toy = copy.deepcopy(mix_log_timestep_final[k])
            t_toy = copy.deepcopy(toy_log_timestep_final[k])

            toy_last_index = np.argwhere(np.array(v_mix_toy)==v_toy[-1])[-1].item()
            
            user_mix_toy = v_mix_toy[:toy_last_index+1]
            user_ti_mix_toy = t_mix_toy[:toy_last_index+1]

            if len(user_mix_toy) < 3:
                ipdb.set_trace()

            user_train_toy_mix[k] = []
            user_train_toy_source[k] = []
            user_train_toy_target[k] = []
            user_valid_toy_target[k] = []
            user_test_toy_target[k] = []
            
            user_train_ti_toy_mix[k] = []
            user_train_ti_toy_source[k] = []
            user_train_ti_toy_target[k] = []
            user_valid_ti_toy_target[k] = []
            user_test_ti_toy_target[k] = []   
            for re_id in reversed(range(0,len(user_mix_toy))):
                if user_mix_toy[re_id] >= interval+1: # from 551942 to XXX, source
                    user_train_toy_source[k].append(user_mix_toy[re_id])
                    user_train_toy_mix[k].append(user_mix_toy[re_id])
                    user_train_ti_toy_source[k].append(user_ti_mix_toy[re_id])
                    user_train_ti_toy_mix[k].append(user_ti_mix_toy[re_id])
                elif user_mix_toy[re_id] <= interval: # from 1 to 551941, target
                    if len(user_test_toy_target[k]) == 0:
                        user_test_toy_target[k].append(user_mix_toy[re_id])
                        user_test_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                    elif len(user_valid_toy_target[k]) == 0:
                        user_valid_toy_target[k].append(user_mix_toy[re_id])
                        user_valid_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                    elif len(user_test_toy_target[k]) == 1 and len(user_valid_toy_target[k]) == 1:
                        user_train_toy_target[k].append(user_mix_toy[re_id])
                        user_train_toy_mix[k].append(user_mix_toy[re_id])
                        user_train_ti_toy_target[k].append(user_ti_mix_toy[re_id])
                        user_train_ti_toy_mix[k].append(user_ti_mix_toy[re_id])
                        
            user_train_toy_mix[k].reverse()
            user_train_toy_source[k].reverse()
            user_train_toy_target[k].reverse()
            user_train_ti_toy_mix[k].reverse()
            user_train_ti_toy_source[k].reverse()
            user_train_ti_toy_target[k].reverse()
            
            
            pos_mix = len(user_train_toy_mix[k])-1
            pos_source = len(user_train_toy_source[k])-1
            mix_sequence_for_target_list = []
            source_sequence_for_target_list = []                
            for i in reversed(list(range(0, len(user_train_toy_mix[k])))):
                if user_train_toy_mix[k][i] >= interval+1:
                    pos_source = pos_source - 1
                elif user_train_toy_mix[k][i] <= interval:
                    mix_sequence_for_target_list.append(pos_mix-1)
                    source_sequence_for_target_list.append(pos_source)
                pos_mix = pos_mix - 1
                
            mix_sequence_for_target = mix_sequence_for_target_list[:-1]
            source_sequence_for_target = source_sequence_for_target_list[:-1]
            mix_sequence_for_target.reverse()
            source_sequence_for_target.reverse()
            
            user_train_toy_mix_sequence_for_target[k] = []
            user_train_toy_source_sequence_for_target[k] = []
            for x in mix_sequence_for_target:
                user_train_toy_mix_sequence_for_target[k].append(x - len(user_train_toy_mix[k]))
                    
            for x in source_sequence_for_target:
                user_train_toy_source_sequence_for_target[k].append(x - len(user_train_toy_source[k]))
            
        return [user_train_toy_mix, user_train_toy_source, user_train_toy_target, user_valid_toy_target, user_test_toy_target, user_train_toy_mix_sequence_for_target, user_train_toy_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_toy_mix, user_train_ti_toy_source, user_train_ti_toy_target, user_valid_ti_toy_target, user_test_ti_toy_target]   
    
    #calculate the auc
def compute_auc(scores):
    scores = -scores.detach().cpu().numpy()
    num_pos = 1
    score_neg = scores[num_pos:]
    num_hit = 0

    for i in range(num_pos):
        num_hit += len(np.where(score_neg < scores[i])[0])

    auc = num_hit / (num_pos * len(score_neg))
    return auc


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate_PDRec(model, dataset, args, user_list):
    with torch.no_grad():
        print('Start test...')
        [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_valid_ti_target, user_test_ti_target] = dataset

        random_min = 1
        random_max = interval + 1
        item_entries = np.arange(start=random_min, stop=random_max, step=1, dtype=int)
        print("The min in source domain is {} and the max in source domain is {}".format(random_min, random_max)) 

        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        NDCG_20 = 0.0
        NDCG_50 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        HT_20 = 0.0
        HT_50 = 0.0
        AUC = 0.0
        loss = 0.0
        valid_user = 0.0
        labels = torch.zeros(100, device=args.device)
        labels[0] = 1
                    
        for u in user_list:
            seq_target = np.zeros([args.maxlen], dtype=np.int32) # (200,)
            idx_target = args.maxlen - 1 #49

            seq_target[idx_target] = user_valid_target[u][0]
            idx_target -= 1       
            for i in reversed(user_train_target[u]):
                seq_target[idx_target] = i
                idx_target -= 1
                if idx_target == -1: break

            sample_pool = np.setdiff1d(item_entries, seq_target)
            item_idx = np.random.choice(sample_pool, args.num_samples, replace=False)
            item_idx[0] = user_test_target[u][0]
            predictions = model.predict(torch.tensor(u).cuda(), torch.tensor(seq_target).cuda().unsqueeze(0), torch.tensor(item_idx).cuda().unsqueeze(0))

            AUC += roc_auc_score(labels.cpu(), predictions[0].cpu())            

            loss_test = torch.nn.BCEWithLogitsLoss()(predictions[0].detach(), labels)

            loss += loss_test.item()
            predictions = -predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

#             AUC += compute_auc(predictions)
            if rank < 1:
                NDCG_1 += 1 / np.log2(rank + 2)
                HT_1 += 1
            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1
            if rank < 50:
                NDCG_50 += 1 / np.log2(rank + 2)
                HT_50 += 1
                
            if valid_user % 1000 == 0:
                print('process test user {}'.format(valid_user))
    print("The total number of user is:", valid_user)
    return NDCG_1 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_20 / valid_user, NDCG_50 / valid_user, HT_1 / valid_user, HT_5 / valid_user, HT_10 / valid_user, HT_20 / valid_user, HT_50 / valid_user, AUC / valid_user, loss / valid_user




def evaluate_T_DiffRec_TI(model, diffusion, dataset, args, random_min, random_max, random_source_min, random_source_max):   
    with torch.no_grad():
        print('Start test...')
        [user_train_mix, user_train_source, user_train_target, user_valid_target, user_test_target, user_train_mix_sequence_for_target, user_train_source_sequence_for_target, usernum, itemnum, interval, user_train_ti_mix, user_train_ti_source, user_train_ti_target, user_valid_ti_target, user_test_ti_target] = dataset
        print("The min in source domain is {} and the max in source domain is {}".format(random_min, random_max)) 
        item_entries = torch.arange(start=random_min, end=random_max, step=1, dtype=int, device='cuda')

        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        NDCG_20 = 0.0
        NDCG_50 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0
        HT_20 = 0.0
        HT_50 = 0.0
        
        AUC = 0.0
        valid_user = 0.0
        users = range(1, usernum + 1) # range(1, 116255)
        labels = torch.zeros(100, device='cuda')
        labels[0] = 1
        for u in users:
            if len(user_train_mix[u]) <= 1 or len(user_train_source[u]) <= 1 or len(user_train_target[u]) <= 1: 
                continue
                
            # init the tensor
            seq_target = torch.zeros([itemnum+1], dtype=torch.float32, device='cuda')
            seq_target_temp = torch.zeros([itemnum+1], dtype=torch.float32, device='cuda')
            # the interaction length
            user_train_target_this = user_train_target[u]+user_valid_target[u]
            user_all_target_this = user_train_target_this+user_test_target[u]
            user_train_ti_target_this = user_train_ti_target[u]+user_valid_ti_target[u]
            weight_target = scale_withminmax(user_train_ti_target_this, args.w_min, args.w_max, args.reweight_version)
            seq_target[user_train_target_this] = 1.0
            seq_target_temp[user_train_target_this] = torch.tensor(weight_target, dtype=torch.float32, device='cuda')
                
#             ipdb.set_trace()
            prediction = diffusion.p_sample(model, seq_target_temp.unsqueeze(0), args.sampling_steps, args.sampling_noise)
#             ipdb.set_trace()
            sample_pool = get_exclusive(item_entries, torch.tensor(user_all_target_this,device='cuda'))
            random_index = torch.randperm(sample_pool.shape[0])
            item_idx = sample_pool[random_index[:args.num_samples]]
            item_idx[0] = user_all_target_this[-1]
            score = torch.index_select(prediction, dim=1, index=item_idx).squeeze()

            AUC += roc_auc_score(labels.cpu(), score.cpu())            

            score = -score # - for 1st argsort DESC
            rank = score.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 1:
                NDCG_1 += 1 / np.log2(rank + 2)
                HT_1 += 1
            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2)
                HT_5 += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1
            if rank < 50:
                NDCG_50 += 1 / np.log2(rank + 2)
                HT_50 += 1
            if valid_user % 1000 == 0:
                print('process test user {}'.format(valid_user))
                
    return NDCG_1 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_20 / valid_user, NDCG_50 / valid_user, HT_1 / valid_user, HT_5 / valid_user, HT_10 / valid_user, HT_20 / valid_user, HT_50 / valid_user, AUC / valid_user
