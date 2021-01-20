#!/usr/bin/env python
# coding: utf-8

# In[435]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


import random
import math
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader


# In[436]:


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[437]:


#train = pd.read_csv('/home/research/hesu/KT/data/riii/train_10M.csv',
#                   usecols = [1,2,3,4,7,8,9],
#                   dtype={'timestamp':'int64',
#                         'used_id':'int32',
#                         'content_id':'int16',
#                         'content_type_id':'int8',
#                         'answered_correctly':'int8',
#                         'prior_question_elapsed_time':'float32',
#                         'prior_question_had_explanation':'boolean'})
train = pd.read_pickle('/Users/hesu/Documents/KT/riiid/user_seq/user_seq40.pkl')

train = train[train.content_type_id == False]

train = train.sort_values(['timestamp'],ascending=True).reset_index(drop=True)
train.head(10)


# In[438]:


question = pd.read_csv('/Users/hesu/Documents/KT/riiid/questions.csv')
question.head(10)


# In[439]:


train_ques = pd.merge(train, question, left_on='content_id',right_on='question_id', how='left')
train_ques.drop('content_id',axis=1,inplace=True)
train_ques.head(10)


# In[440]:


train_ques.tail(10)


# In[441]:


elapsed_mean = train_ques.prior_question_elapsed_time.mean()


# In[442]:


train_ques['prior_question_elapsed_time'].fillna(elapsed_mean, inplace=True)
train_ques['part'].fillna(4, inplace=True)


# In[443]:


train_ques.loc[:,'prior_question_elapsed_time'].value_counts()


# In[444]:


train_ques.loc[:,'part'].value_counts()


# In[445]:


import datetime
import time
def convert_time_to_yearMonthDay(timeStamp):
    timeStamp = timeStamp /1000.0
    timearr = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timearr)
    print(otherStyleTime)

convert_time_to_yearMonthDay(78091996556)


# In[446]:


def get_elapsed_time(ela):
    ela = ela // 1000
    if ela > 300:
        return 300
    else:
        return int(ela)


# In[ ]:





# In[447]:


train_ques['prior_question_elapsed_time'] = train_ques['prior_question_elapsed_time'].apply(lambda x: get_elapsed_time(x))


# In[448]:


train_ques.head(10)


# In[ ]:





# In[449]:


train_ques['timestamp'] = train_ques['timestamp'].astype(str)
train_ques['question_id'] = train_ques['question_id'].astype(str)
train_ques['part'] = train_ques['part'].astype(str)
train_ques['prior_question_elapsed_time'] = train_ques['prior_question_elapsed_time'].astype(str)
train_ques['answered_correctly'] = train_ques['answered_correctly'].astype(str)


# In[450]:


train_user = train_ques.groupby('user_id').agg({"question_id": ','.join, 
                                                "answered_correctly":','.join,
                                                "timestamp":','.join,
                                                "part":','.join,
                                                "prior_question_elapsed_time":','.join})


# In[451]:


train_user.head(10)


# In[452]:


train_user.shape


# In[453]:


type(train_user)


# In[454]:


train_user


# In[455]:


train_user.reset_index(inplace=True)


# In[456]:


train_user


# In[457]:


train_user = train_user.rename(columns={'question_id':'question_id_seq',
                            'answered_correctly':'answered_correctly_seq',
                             'timestamp':'timestamp_seq',
                             'part':'part_seq',
                             'prior_question_elapsed_time':'prior_question_elapsed_time_seq'})


# In[458]:




# In[459]:


def get_data_for_train_encode(train_user, seq_len):
    all_ques_seq = []
    all_ans_seq = []
    all_parts_seq = []
    all_ela_seq = []
    
    target_ques = []
    target_anss = []
    target_parts = []
    target_elas = []
    
    for row in train_user.itertuples():
        q_ids = getattr(row, 'question_id_seq').strip().split(',')
        ans_ids = getattr(row, 'answered_correctly_seq').strip().split(',')
        part_ids = getattr(row, 'part_seq').strip().split(',')
        ela_ids = getattr(row, 'prior_question_elapsed_time_seq').strip().split(',')
        
        assert len(q_ids) == len(ans_ids) == len(part_ids) == len(ela_ids)
        
        target_index = len(q_ids) - 1
        q_ids_seq = q_ids[:target_index]
        ans_ids_seq = ans_ids[:target_index]
        part_ids_seq = part_ids[:target_index]
        ela_ids_seq = ela_ids[:target_index]
        
        length = len(q_ids_seq)
        if length >= seq_len:
            q_ids_seq = q_ids_seq[-seq_len:]
            ans_ids_seq = ans_ids_seq[-seq_len:]
            part_ids_seq = part_ids_seq[-seq_len:]
            ela_ids_seq = ela_ids_seq[-seq_len:]  
                
            pad_counts = 0
        else:
            pad_counts = seq_len - length
            
        q_ids_seq = [int(float(e)) for e in q_ids_seq]
        ans_ids_seq = [int(float(e)) for e in ans_ids_seq]
        part_ids_seq = [int(float(e)) for e in part_ids_seq]
        ela_ids_seq = [int(float(e)) for e in ela_ids_seq]
            
        q_ids_seq = [13523]*pad_counts + q_ids_seq
        # question用13523表示padding位
        ans_ids_seq = [2]*pad_counts  + ans_ids_seq
        # ans用2表示padding位
        # ans因为是输入到decoder中，所以需要一个起始符号，这里选择3作为其实符号，也就是句子序列中的bos的作用
        part_ids_seq = [8]*pad_counts + part_ids_seq
        # part用8来表示padding位
        ela_ids_seq = [301]*pad_counts + ela_ids_seq
        # ela用301来表示padding位
#             print("q_ids length is:{}\n ans_ids length is:{}\n part length is:{}\n ela_ids length is:{}".format(len(q_ids_seq),len(ans_ids_seq),len(part_ids_seq),len(ela_ids_seq)))
        all_ques_seq.append(q_ids_seq)
        all_ans_seq.append(ans_ids_seq)
        all_parts_seq.append(part_ids_seq)
        all_ela_seq.append(ela_ids_seq)        
        
        target_ques.append([int(float(q_ids[-1]))])
        target_anss.append([int(float(ans_ids[-1]))])
        target_parts.append([int(float(part_ids[-1]))])
        target_elas.append([int(float(ela_ids[-1]))])


    return torch.LongTensor(all_ques_seq),        torch.LongTensor(all_ans_seq),        torch.LongTensor(all_parts_seq),        torch.LongTensor(all_ela_seq),        torch.LongTensor(target_ques),        torch.LongTensor(target_anss),        torch.LongTensor(target_parts),        torch.LongTensor(target_elas)
            
            


# In[ ]:





# In[460]:


class Rii_dataset_train(Dataset):
    def __init__(self,train_user):
        self.df = train_user
        self.ques_seq, self.ans_seq, self.parts_seq, self.ela_seq,        self.trg_que, self.trg_ans, self.trg_part, self.trg_ela = get_data_for_train_encode(self.df, 100)
    def __len__(self):
        return len(self.ques_seq)
    def __getitem__(self, index):
        return self.ques_seq[index], self.ans_seq[index], self.parts_seq[index], self.ela_seq[index],        self.trg_que[index], self.trg_ans[index], self.trg_part[index], self.trg_ela[index]


# In[461]:


test_df = pd.read_pickle('/home/research/hesu/KT/data/riii/valid_6.pkl')


# In[462]:


test_df = test_df.loc[test_df['content_type_id'] == 0].reset_index(drop=True)
test_df['prior_question_elapsed_time'].fillna(elapsed_mean, inplace=True)
test_df['prior_question_elapsed_time'] = test_df['prior_question_elapsed_time'].apply(lambda x: get_elapsed_time(x))


# In[463]:


test_df.head(10)


# In[464]:


question.head(10)


# In[465]:


test_df = pd.merge(test_df, question, left_on='content_id',right_on='question_id', how='left')


# In[466]:


test_df.head(10)


# In[ ]:





# In[467]:


test_df = pd.merge(test_df, train_user, on='user_id',how='left')


# In[468]:


test_df.head(10)


# In[469]:


test_df.dtypes


# In[470]:


test_df['question_id_seq'].fillna('13523', inplace=True)
test_df['question_id_seq'] = test_df['question_id_seq'].astype('str')


test_df['answered_correctly_seq'].fillna('2', inplace=True)
test_df['answered_correctly_seq'] = test_df['answered_correctly_seq'].astype('str')


test_df['part_seq'].fillna('8', inplace=True)
test_df['part_seq'] = test_df['part_seq'].astype('str')


test_df['prior_question_elapsed_time_seq'].fillna('301', inplace=True)
test_df['prior_question_elapsed_time_seq'] = test_df['prior_question_elapsed_time_seq'].astype('str')


# In[471]:


def pad_np(nums, pad_index):
    seq_size = 100
    
    if nums.size == 0:
        return np.array([0]*seq_size)

    if nums.size > seq_size:
        nums = nums[-seq_size:]
    else:
        pad_counts = seq_size - len(nums)
        nums = np.pad(nums,(pad_counts,0),'constant',constant_values=(pad_index,0))
        # (pad_counts, 0 )表示在左边填充pad_counts个数字，右边填充0个数字;
        # constant_values=(0,0)表示左边填充0， 右边也填充0
    return nums



def pad_seq(df):
    df['content_id'] = np.array(df['content_id'])

    
#     df['question_id_seq'] = df['question_id_seq'].apply(lambda x: np.array(x).astype(np.int16))
    df['question_id_seq'] = df['question_id_seq'].astype('str')
    df['question_id_seq'] = df['question_id_seq'].apply(lambda x: np.array(x.split(',')).astype(np.int16))
    df['question_id_seq_input'] = df.apply(lambda x: pad_np(x.question_id_seq, 13523), axis=1)
    
    df['answered_correctly_seq'] = df['answered_correctly_seq'].astype('str')
    df['answered_correctly_seq'] = df['answered_correctly_seq'].apply(lambda x: np.array(x.split(',')).astype(np.int16))
    df['answered_correctly_input'] = df.apply(lambda x: pad_np(x.answered_correctly_seq, 2), axis=1)

    df['part_seq'] = df['part_seq'].astype('str')
    df['part_seq'] = df['part_seq'].apply(lambda x: np.array(x.split(',')).astype(np.int16))
    df['part_seq_input'] = df.apply(lambda x: pad_np(x.part_seq, 8), axis=1)
    
    df['prior_question_elapsed_time_seq'] = df['prior_question_elapsed_time_seq'].astype('str')
    df['prior_question_elapsed_time_seq'] = df['prior_question_elapsed_time_seq'].apply(lambda x: np.array(x.split(',')).astype(np.int16))
    df['prior_question_elapsed_time_seq_input'] = df.apply(lambda x: pad_np(x.prior_question_elapsed_time_seq, 301), axis=1)
    
    return df
    


# In[472]:


test_df['question_id_seq']


# In[ ]:





# In[ ]:





# In[473]:


test_df = pad_seq(test_df)


# In[474]:


test_df.head(2)


# In[475]:


test_df.columns


# In[476]:


class Rii_dataset_test(Dataset):
    def __init__(self,test_user):
        self.df = test_user
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        ques_seq = torch.from_numpy(self.df.at[index, 'question_id_seq_input']).long()
        ans_seq = torch.from_numpy(self.df.at[index, 'answered_correctly_input']).long()
        parts_seq = torch.from_numpy(self.df.at[index, 'part_seq_input']).long()
        ela_seq = torch.from_numpy(self.df.at[index, 'prior_question_elapsed_time_seq_input']).long()
        
        trg_que = torch.LongTensor([self.df.at[index,'content_id']])
        trg_ans = torch.LongTensor([self.df.at[index,'answered_correctly']])
        trg_part = torch.LongTensor([self.df.at[index,'part']])
        trg_ela = torch.LongTensor([self.df.at[index,'prior_question_elapsed_time']])

        return ques_seq, ans_seq, parts_seq, ela_seq, trg_que, trg_ans, trg_part, trg_ela


# In[ ]:





# In[ ]:





# ## Model

# In[477]:


class Encoder(nn.Module):
    def __init__(self, 
                 que_num,
                 part_num,
                 ela_num,
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.que_embedding = nn.Embedding(que_num, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.part_embedding = nn.Embedding(part_num, hid_dim)
        self.ela_embedding = nn.Embedding(ela_num, hid_dim)
        self.ans_embedding = nn.Embedding(3, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hid_dim, 2)
        self.trg_linear = nn.Linear(hid_dim, hid_dim)
        
        self.output_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.avgpool = nn.AvgPool1d(max_length)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
        
    def forward(self, src_que,src_ans,src_part,src_ela,src_mask, trg_que, trg_part, trg_ela,trg_src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src_que.shape[0]
        src_len = src_que.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos的维度是[batch_size, src_len]，其中每个一维的都是都是[1,100]，
        # 其中unsqueeze(0)的作用是将tensor由[seq_len]维度变成[batch_size, seq_len]维
        
        que_emb = self.que_embedding(src_que)
        part_emb = self.part_embedding(src_part)
        ela_emb = self.ela_embedding(src_ela)
        ans_emb = self.ans_embedding(src_ans)
        tok_emb = que_emb+part_emb+ela_emb+ans_emb
        
        src = self.dropout((tok_emb * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        encoder_output = src

        trg_que_emb = self.que_embedding(trg_que)
        trg_part_emb = self.part_embedding(trg_part)
        trg_ela_emb = self.ela_embedding(trg_ela)
        
        trg_emb = trg_que_emb+trg_part_emb+trg_ela_emb
        trg_linear = self.trg_linear(trg_emb)
        
#        print("encoder_output shape is:{}\ntrg_linear shape is:{}".format(encoder_output.shape, trg_linear.shape))
        attention_output, _ = self.output_attention(trg_linear, encoder_output, encoder_output, trg_src_mask)
#         print("src shape:{}\ntrg_que_emb shape:{}\ntrg_part_emb shape:{}\n".format(src.shape, trg_que_emb.shape, trg_part_emb.shape))
        #src = [batch size, src len, hid dim]
#        print("attention output shape is:{}".format(attention_output.shape))
        
        output_pool = self.avgpool(attention_output.permute(0,2,1)).permute(0,2,1)
        
        output = self.output_layer(output_pool)
        
        return output
        


# In[478]:


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
#         print("In encoder Q shape is:{}\t K shape is:{}\t V shape is:{}\t mask shape is:{}".format(src.shape,\
#                                                                                                    src.shape,src.shape,src_mask.shape))
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src


# In[479]:


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
#         print("Q shape is:{}\t K shape is:{}\t V shape is:{}\t energy shape is:{}\tmask shapeis:{}".format(Q.shape,\
#                                                                                 K.shape, V.shape, energy.shape,mask.shape))
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention        


# In[480]:


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x


# # Seq2Seq

# In[481]:


class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 src_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = src_pad_idx
        self.device = device
        
    def make_trg_src_mask(self, src):
        # 这个是trg和src中的每一个计算attention分布时用的mask
        #src = [batch size, src len]
        
        trg_src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return trg_src_mask
    
    def make_src_mask(self, src):
        # 这个是encoder部分，只能看见当前que前面que信息的mask矩阵，上三角mask矩阵
        #src = [batch size, trg len]
        
        src_pad_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #src_pad_mask = [batch size, 1, 1, src len]
        
        src_len = src.shape[1]
        
        src_sub_mask = torch.tril(torch.ones((src_len, src_len), device = self.device)).bool()
        
        #src_sub_mask = [src len, src len]
            
        src_mask = src_pad_mask & src_sub_mask
        
        #src_mask = [batch size, 1, src len, src len]
        
        return src_mask

    def forward(self, src_que,src_ans,src_part,src_ela,trg_que, trg_part,trg_ela):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
            
        src_mask = self.make_src_mask(src_que)
        trg_src_mask = self.make_src_mask(src_que)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_src_mask = [batch size, 1, trg len, trg len]
        
        output = self.encoder(src_que,src_ans,src_part,src_ela,src_mask, trg_que, trg_part, trg_ela,trg_src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
#        print("The model output shape is:{}".format(output.shape))
        return output       


# In[482]:


que_num = 13524
ans_num = 3
part_num = 9
ela_num = 302

HID_DIM = 256
ENC_LAYERS = 3
ENC_HEADS = 8
ENC_PF_DIM = 512
ENC_DROPOUT = 0.1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

enc = Encoder(que_num,part_num,ela_num,
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)


# In[483]:


src_pad_que_idx = 13523
trg_pad_ans_idx = 2


# In[484]:


model = Seq2Seq(enc, src_pad_que_idx, device).to(device)


# In[485]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[486]:


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# In[487]:


model.apply(initialize_weights)


# In[488]:


LEARNING_RATE = 5e-4

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


# In[489]:


criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_ans_idx)


# ## Train

# In[490]:


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    total_num = 0
    right_num = 0
    
    for i, batch in tqdm(enumerate(iterator)):
        
        batch = tuple(t.to(device) for t in batch)
        
        src_que, src_ans, src_part, src_ela, trg_que, trg_ans, trg_part, trg_ela = batch
        
        optimizer.zero_grad()
#        print("src_que is:{}\nsrc_ans is:{}\ntrg_que is:{}\ntrg_ans is:{}".format(src_que.shape, src_ans.shape,trg_que.shape,trg_ans.shape))
        
        output = model(src_que, src_ans, src_part, src_ela, trg_que, trg_part, trg_ela)
        # 由于decoder预测时是错位预测，也就是用trg[t-1]去预测trg[t]，所以输入到decoder模型中的trg缺少最后一个样本的结果 
        
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
        
        output = output.squeeze(1)
        output_dim = output.shape[-1]
            
        preds = F.softmax(output, dim=-1)
        
        output = output.contiguous().view(-1, output_dim)
        trg_ans = trg_ans.contiguous().view(-1)
        # contiguous()用于判定tensor是否是连续的
        
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg_ans)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
        optimizer.step()
        
        epoch_loss += loss.item()
        
        preds_ind = torch.max(preds, dim=1)[1]
        right_num += (preds_ind == trg_ans).sum().item()
        total_num += len(trg_ans)
        
    return epoch_loss / len(iterator),right_num / total_num


# In[ ]:





# In[491]:


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    total_num = 0
    right_num = 0
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            batch = tuple(t.to(device) for t in batch)

            src_que, src_ans, src_part, src_ela, trg_que, trg_ans, trg_part, trg_ela = batch
            output = model(src_que, src_ans, src_part, src_ela, trg_que, trg_part, trg_ela)

            output = output.squeeze(1)
            output_dim = output.shape[-1]
            
            preds = F.softmax(output, dim=-1)
        
            output = output.contiguous().view(-1, output_dim)
            trg_ans = trg_ans.contiguous().view(-1)
            # contiguous()用于判定tensor是否是连续的
        
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg_ans)
    
            epoch_loss += loss.item()
        
            preds_ind = torch.max(preds, dim=1)[1]
            right_num += (preds_ind == trg_ans).sum().item()
            total_num += len(trg_ans)
                        

        
    return epoch_loss / len(iterator),right_num / total_num



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



train_dataset = Rii_dataset_train(train_user)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataset = Rii_dataset_test(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


N_EPOCHS = 10
CLIP = 1

best_test_loss = float('inf')


for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, acc_train = train(model, train_dataloader, optimizer, criterion, CLIP)
    test_loss, acc_test = evaluate(model, test_dataloader, criterion)
    end_time = time.time()
    
    print("At epoch-{}\tThe training loss is:{}\nTrain accuracy is:{}".format(epoch, train_loss, acc_train))
    print("At epoch-{}\tThe test loss is:{}\nTest accuracy is:{}".format(epoch, test_loss, acc_test))

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best-100-model.pt')




