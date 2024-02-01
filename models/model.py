import numpy as np
import torch
import ipdb
import torch.nn.functional as F
from torch import Tensor
import math
import os
import io
import copy
import time
import random
import copy    
class EarlyStopping_onetower:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, args, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.epoch = 100
        self.best_performance = None
        self.early_stop = False
        self.ndcg_max = None
        self.save_epoch = None
        self.delta = delta
        self.version = args.version
        self.dataset = args.dataset
        self.base_model = args.base_model

    def __call__(self, epoch, model, result_path, t_test):

        if self.ndcg_max is None:
            self.ndcg_max = t_test[2]
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
        elif t_test[2] < self.ndcg_max:
            self.counter += 1
            print(f'In the epoch: {epoch}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch>=self.epoch:
                self.early_stop = True
        else:
            self.best_performance = t_test
            self.save_epoch = epoch
            self.save_checkpoint(epoch, model, result_path, t_test)
            self.counter = 0

    def save_checkpoint(self, epoch, model, result_path, t_test):
        print(f'Validation loss in {epoch} decreased {self.ndcg_max:.4f} --> {t_test[2]:.4f}.  Saving model ...\n')
        with io.open(result_path + 'save_model.txt', 'a', encoding='utf-8') as file:
            file.write("NDCG@10 in epoch {} decreased {:.4f} --> {:.4f}, the HR@10 is {:.4f}, the AUC is {:.4f}, the loss_rec is {:.4f}. Saving model...\n".format(epoch, self.ndcg_max, t_test[2], t_test[7], t_test[10], t_test[11]))
        torch.save(model.state_dict(), os.path.join(result_path, 'checkpoint.pt')) 
        self.ndcg_max = t_test[2]

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    
class SASRec_Embedding(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec_Embedding, self).__init__()

        self.item_num = item_num # 3416
        self.dev = args.device #'cuda'

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE Embedding(200, 50)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) #Dropout(p=0.2)

        self.attention_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.attention_layers = torch.nn.ModuleList() # 2 layers of MultiheadAttention
        self.forward_layernorms = torch.nn.ModuleList() # 2 layers of LayerNorm
        self.forward_layers = torch.nn.ModuleList() # 2 layers of PointWiseFeedForward

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) #LayerNorm(torch.Size([50]), eps=1e-08, elementwise_affine=True)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate, batch_first=True) # MultiheadAttention((out_proj): NonDynamicallyQuantizableLinear(in_features=50, out_features=50, bias=True))
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8) # LayerNorm((50,), eps=1e-08, elementwise_affine=True)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
#         tt0 = time.time()
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5 # torch.Size([128, 200, 64])
        positions = torch.tile(torch.arange(0,log_seqs.shape[1]), [log_seqs.shape[0],1]).cuda() # torch.Size([128, 200])
            # add the position embedding
        seqs += self.pos_emb(positions) 
        seqs = self.emb_dropout(seqs) # torch.Size([128, 200, 64])

            # mask the noninteracted position
        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda() # (128,200)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality, 200
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device='cuda')) #(200,200)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs) #torch.Size([128, 200, 50])
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask) # torch.Size([128, 200, 50])
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs # torch.Size([128, 200, 50])

            seqs = self.forward_layernorms[i](seqs) # torch.Size([128, 200, 50])
            seqs = self.forward_layers[i](seqs) # torch.Size([128, 200, 50])
            seqs *=  ~timeline_mask.unsqueeze(-1) # torch.Size([128, 200, 50])

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs): # for training      
        log_feats = self.log2feats(log_seqs) # torch.Size([128, 200, 50]) user_ids hasn't been used yet

        return log_feats # pos_pred, neg_pred    
    


class GRU4Rec_withNeg_Dist(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4Rec_withNeg_Dist, self).__init__()
        
        self.source_item_emb = torch.nn.Embedding(item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)
        self.target_item_emb = torch.nn.Embedding(item_num+1, args.hidden_units, padding_idx=0) #Embedding(3417, 50, padding_idx=0)

        self.gru_source = torch.nn.GRU(args.hidden_units, args.hidden_units, batch_first=True)
        self.gru_target = torch.nn.GRU(args.hidden_units, args.hidden_units, batch_first=True)

        self.h0_source = torch.nn.Parameter(torch.zeros((1, 1, args.hidden_units), requires_grad=True))
        self.h0_target = torch.nn.Parameter(torch.zeros((1, 1, args.hidden_units), requires_grad=True))

        self.dev = args.device #'cuda'
                
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.temperature = args.temperature
        self.fname = args.dataset
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_list, log_seqs_all, soft_diff): # for training      
            # user_ids:(128,)
            # log_seqs:(128, 200)
            # pos_seqs:(128, 200)
            # neg_seqs:(128, 200)
#         ipdb.set_trace()
        neg_embs = []
        neg_logits = []
        source_log_embedding = self.source_item_emb(log_seqs)            
        source_log_feats, _ = self.gru_source(source_log_embedding, self.h0_source.tile(1,source_log_embedding.shape[0],1)) #2，121，100
        source_log_all_embedding = self.source_item_emb(log_seqs_all)
        source_log_all_feats, _ = self.gru_source(source_log_all_embedding, self.h0_source.tile(1,source_log_all_embedding.shape[0],1)) #2，121，100
        pos_embs = self.source_item_emb(pos_seqs) # torch.Size([128, 200, 50])
        soft_embs = self.source_item_emb(soft_diff) # torch.Size([128, 200, 50])
        for i in range(0,len(neg_list)):
            neg_embs.append(self.source_item_emb(neg_list[i])) # torch.Size([128, 200, 50])

        # get the l2 norm for the target domain recommendation
        source_log_feats_l2norm = torch.nn.functional.normalize(source_log_feats, p=2, dim=-1)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
        pos_logits = (source_log_feats_l2norm * pos_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        pos_logits = pos_logits * self.temperature

        for i in range(0,len(neg_list)):
            neg_embs_l2norm_i = torch.nn.functional.normalize(neg_embs[i], p=2, dim=-1)
            neg_logits_i = (source_log_feats_l2norm * neg_embs_l2norm_i).sum(dim=-1) # torch.Size([128, 200])
            neg_logits_i = neg_logits_i * self.temperature
            neg_logits.append(neg_logits_i)

        source_log_all_feats_l2norm = torch.nn.functional.normalize(source_log_all_feats, p=2, dim=-1)
        soft_embs_l2norm = torch.nn.functional.normalize(soft_embs, p=2, dim=-1)
        soft_logits = (source_log_all_feats_l2norm[:,-1,:].unsqueeze(1).expand(-1,soft_embs_l2norm.shape[1],-1) * soft_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        soft_logits = soft_logits * self.temperature

        return pos_logits, neg_logits, soft_logits # pos_pred, neg_pred
    

    def predict(self, user_ids, log_seqs, item_indices): # for inference
            # user_ids: (1,)
            # log_seqs: (1, 200)
            # item_indices: (101,)
#         ipdb.set_trace()
        source_log_embedding = self.source_item_emb(log_seqs)
        source_log_feats, _ = self.gru_source(source_log_embedding, self.h0_source.tile(1,source_log_embedding.shape[0],1)) #2，121，100

        item_embs = self.source_item_emb(item_indices) 
        # get the l2 norm for the target domain recommendation
        final_feat = source_log_feats[:, -1, :] # torch.Size([1, 50]) 
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=-1)
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1)

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) 
        logits = logits * self.temperature
            
            
        return logits # preds # (U, I)
    
    
    
    
class SASRec_V1_withNeg_Dist(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_V1_withNeg_Dist, self).__init__()

        self.sasrec_embedding_source = SASRec_Embedding(item_num, args)
        self.sasrec_embedding_target = SASRec_Embedding(item_num, args)
        self.dev = args.device #'cuda'
        
        
        self.leakyrelu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.temperature = args.temperature
        self.fname = args.dataset
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)



    def forward(self, user_ids, log_seqs, pos_seqs, neg_list, log_seqs_all, soft_diff): # for training      
        neg_embs = []
#         ipdb.set_trace()
        source_log_feats = self.sasrec_embedding_source(log_seqs) # torch.Size([128, 200, 50]) 
        source_log_all_feats = self.sasrec_embedding_source(log_seqs_all) # torch.Size([128, 200, 50]) 
        pos_embs = self.sasrec_embedding_source.item_emb(pos_seqs) # torch.Size([128, 200, 50])
        soft_embs = self.sasrec_embedding_source.item_emb(soft_diff) # torch.Size([128, 200, 50])
        for i in range(0,len(neg_list)):
            neg_embs.append(self.sasrec_embedding_source.item_emb(neg_list[i]))

        # get the l2 norm for the target domain recommendation
        source_log_feats_l2norm = torch.nn.functional.normalize(source_log_feats, p=2, dim=-1)
        pos_embs_l2norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
        pos_logits = (source_log_feats_l2norm * pos_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        pos_logits = pos_logits * self.temperature

        neg_logits = []
        for i in range(0,len(neg_list)):
            neg_embs_l2norm_i = torch.nn.functional.normalize(neg_embs[i], p=2, dim=-1)
            neg_logits_i = (source_log_feats_l2norm * neg_embs_l2norm_i).sum(dim=-1) # torch.Size([128, 200])
            neg_logits_i = neg_logits_i * self.temperature
            neg_logits.append(neg_logits_i)

        source_log_all_feats_l2norm = torch.nn.functional.normalize(source_log_all_feats, p=2, dim=-1)
        soft_embs_l2norm = torch.nn.functional.normalize(soft_embs, p=2, dim=-1)
        soft_logits = (source_log_all_feats_l2norm[:,-1,:].unsqueeze(1).expand(-1,soft_embs_l2norm.shape[1],-1) * soft_embs_l2norm).sum(dim=-1) # torch.Size([128, 200])
        soft_logits = soft_logits * self.temperature

        return pos_logits, neg_logits, soft_logits # pos_pred, neg_pred
    

    def predict(self, user_ids, log_seqs, item_indices): # for inference
#         ipdb.set_trace()
        source_log_feats = self.sasrec_embedding_source(log_seqs) # torch.Size([1, 200, 64])
        item_embs = self.sasrec_embedding_source.item_emb(item_indices) # torch.Size([1, 100, 64])
        # get the l2 norm for the target domain recommendation
        final_feat = source_log_feats[:, -1, :] # torch.Size([1, 64])
        final_feat_l2norm = torch.nn.functional.normalize(final_feat, p=2, dim=-1) # torch.Size([1, 64])
        item_embs_l2norm = torch.nn.functional.normalize(item_embs, p=2, dim=-1) # torch.Size([1, 100, 64])

        logits = item_embs_l2norm.matmul(final_feat_l2norm.unsqueeze(-1)).squeeze(-1) # torch.Size([1, 100])
        logits = logits * self.temperature # torch.Size([1, 100])
            
        return logits # torch.Size([1, 100])

    