from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt,pca_compression
import preprompt
from utils import process
import pdb
import aug
import os
import tqdm
import argparse
from downprompt import downprompt,prefeatureprompt
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("MDGPT")
import torch.nn.functional as F
parser.add_argument('--dataset', type=str, default="Photo", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3_computers.pkl', help='save ckpt name')
parser.add_argument('--val_name', type=str, default='noval_graphcl_BZR.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
# parser.add_argument('--local_rank', type=str, help='local rank for dist')      
args = parser.parse_args()

# world_size = torch.cuda.device_count()
# local_rank = args.local_rank
# dist_backend = 'nccl'
# dist.init_process_group(backend=dist_backend)
print('-' * 100)
print(args)
print('-' * 100)

# dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))

from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# training params

# idx_train = torch.load("data/fewshot/0/idx.pt").type(torch.long).cuda()
batch_size = 100
nb_epochs = 10000
patience = 50
lr = 0.00001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = True
useMLP =False
class_num = 2
LP = False


nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset

# labels = torch.LongTensor(labels)
# dataset = TUDataset(root='data', name='BZR',use_node_attr=True)                                                                                                
# loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)


# dataset = MyDataset(root='data',name1='BZR',name2='COX2',use_node_attr=True)

device = torch.device("cuda")

# model = model.to(device)

# model.load_state_dict(torch.load(args.save_name))

best = 1e9
firstbest = 0

    # lr = 0.01
dataset1 = Planetoid(root='data', name='Cora')                                                                                               
loader1 = DataLoader(dataset1)

dataset2 = Planetoid(root='data', name='Citeseer')                                                                                         
loader2 = DataLoader(dataset2)

dataset3 = Planetoid(root='data', name='Pubmed')                                                                                         
loader3 = DataLoader(dataset3)



dataset4 = Amazon(root='data',name='Photo')
loader4 = DataLoader(dataset4)

dataset5 = Amazon(root='data',name='Computers')
loader5 = DataLoader(dataset5)

dataset6 = Coauthor(root='data',name='CS')
loader6 = DataLoader(dataset6)

dataset7 = Coauthor(root='data',name='Physics')
loader7 = DataLoader(dataset7)

cnt_wait = 0

# print(dataset1)
# print(loader1)
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
unify_dim =50
for step, (data2,data3,data4,data1,data5,data6) in enumerate(zip(loader1,loader2,loader3,loader4,loader5,loader6)):
    print(step,data1,data2)
    # features11,adj1= process.process_tu(data1,data1.x.shape[1])
    features22,adj2= process.process_tu(data2,data2.x.shape[1])
    features33,adj3= process.process_tu(data3,data3.x.shape[1])
    features44,adj4= process.process_tu(data4,data4.x.shape[1])
    features55,adj5= process.process_tu(data5,data5.x.shape[1])
    features66,adj6= process.process_tu(data6,data6.x.shape[1])
#        # features1 = model.texttoken1(torch.FloatTensor(features11).cuda())
#        # features2 = model.texttoken2(torch.FloatTensor(features22).cuda())
    # features1 = pca_compression(features11,k=unify_dim)
    features2 = pca_compression(features22,k=unify_dim)
    features3 = pca_compression(features33,k=unify_dim)
    features4 = pca_compression(features44,k=unify_dim)
    features5 = pca_compression(features55,k=unify_dim)
    features6 = pca_compression(features66,k=unify_dim)

    features2 = torch.FloatTensor(features2).cuda()
    features3 = torch.FloatTensor(features3).cuda()
    features4 = torch.FloatTensor(features4).cuda()
    features5 = torch.FloatTensor(features5).cuda()
    features6 = torch.FloatTensor(features6).cuda()
#        # print("texttoken1",model.texttoken1.weight)
#        setshape1 = adj1.shape[0]
#        setshape2 = adj2.shape[0]
    adj = process.combine_dataset(adj2,adj3,adj4,adj5)
    # print(adj)
    # print(adj.shape)
    negetive_sample = preprompt.prompt_pretrain_sample(adj,50)
    # negetive_sample2 = preprompt.prompt_pretrain_sample(adj2,50)
    # negetive_sample3 = preprompt.prompt_pretrain_sample(adj3,50)
    # negetive_sample4 = preprompt.prompt_pretrain_sample(adj4,50)
    # negetive_sample5 = preprompt.prompt_pretrain_sample(adj5,50)
    # negetive_sample = torch.cat((torch.Tensor(negetive_sample2).cuda(),torch.Tensor(negetive_sample3).cuda(),torch.Tensor(negetive_sample4).cuda(),torch.Tensor(negetive_sample5).cuda()),dim=0).cuda()

adj2 = process.normalize_adj(adj2 + sp.eye(adj2.shape[0]))
adj3 = process.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
adj4 = process.normalize_adj(adj4 + sp.eye(adj4.shape[0]))
adj5 = process.normalize_adj(adj5 + sp.eye(adj5.shape[0]))
adj6 = process.normalize_adj(adj6 + sp.eye(adj6.shape[0]))
if sparse:
    sp_adj4 = process.sparse_mx_to_torch_sparse_tensor(adj4)
    sp_adj2 = process.sparse_mx_to_torch_sparse_tensor(adj2)
    sp_adj3 = process.sparse_mx_to_torch_sparse_tensor(adj3)
    sp_adj5 = process.sparse_mx_to_torch_sparse_tensor(adj5)
    sp_adj6 = process.sparse_mx_to_torch_sparse_tensor(adj6)
    
# else:
#     adj1 = adj1.todense()
#     adj2 = adj2.todense()
# if not sparse:
#     adj1 = torch.FloatTensor(adj1[np.newaxis])
#     adj2 = torch.FloatTensor(adj2[np.newaxis])
model = PrePrompt(unify_dim, hid_units, nonlinearity,negetive_sample,3,0.1,args.combinetype)
    
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
if torch.cuda.is_available():
    print('Using CUDA')
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    model = model.cuda()
    features4 = features4.cuda()
    features2 = features2.cuda()
    features3 = features3.cuda()
    features5 = features5.cuda()
    features6 = features6.cuda()
    if sparse:
        sp_adj4 = sp_adj4.cuda()
        sp_adj2 = sp_adj2.cuda()
        sp_adj3 = sp_adj3.cuda()
        sp_adj5 = sp_adj5.cuda()
        sp_adj6 = sp_adj6.cuda()
    # else:
    #     adj1 = adj1.cuda()
    #     adj2 = adj2.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

for epoch in range(nb_epochs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    regloss = 0
    # model.texttoken1.reset_parameters()
    # model.texttoken2.reset_parameters()


    # negetive_sample2 = preprompt.prompt_pretrain_sample(adj2,50)
    # print(negetive_sample1)
    # nb_nodes = features.shape[0]  # node number
    # ft_size = features.shape[1]  # node features dim
    '''
    # ------------------------------------------------------------
    # edge node mask subgraph
    # ------------------------------------------------------------
    '''
    # print("Begin Aug:[{}]".format(args.aug_type))
    # if args.aug_type == 'edge':
    '''
    # ------------------------------------------------------------
    '''

    # best = 1e9
    model.train()
    optimiser.zero_grad()
    loss = model( features2,features3,features4,features5,
                sp_adj2 if sparse else adj2,sp_adj3 if sparse else adj3,sp_adj4 if sparse else adj4,sp_adj5 if sparse else adj5,
                sparse, None, None, None)
    loss.backward()
    optimiser.step()
    # loss.backward()
    # optimiser.step()
    print('Loss:[{:.4f}]'.format(loss.item()))
    if loss < best:
        firstbest = 1
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break
    print('Loading {}th epoch'.format(best_t))


model = PrePrompt(unify_dim, hid_units, nonlinearity,1,3,0.1,args.combinetype)

print('#'*50)
print('Downastream dataset is ',args.dataset)

if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
    dataset = Planetoid(root='data', name=args.dataset)                                                                                         
if args.dataset == 'Computers' or args.dataset =='Photo':
    dataset = Amazon(root='data', name=args.dataset) 
if args.dataset == 'Reddit':
    dataset = Reddit(root='data/Reddit') 
print(dataset)
loader = DataLoader(dataset)
for data in loader:
    print(data)
    features,adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    features = pca_compression(features,k=unify_dim)
    print('pca')
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    print("adj")
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_adj = sp_adj.cuda()
    features = torch.FloatTensor(features).cuda()
    print(features.shape)
    idx_test = range(data.y.shape[0]-1000,data.y.shape[0])
    labels = data.y
    data=np.array(data.y)

    np.unique(data)

    nb_classes=len(np.unique(data))
    print(nb_classes)
    

model = model.cuda()

model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None,LP)
downstreamlrlist = [0.001]
acclist = torch.FloatTensor(100,).cuda()

for downstreamlr in downstreamlrlist:
    
    print(labels.shape)
    # val_lbls = torch.argmax(labels[idx_val.cpu()], dim=1).cuda()
    test_lbls = labels[idx_test].cuda()
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    print('-' * 100)

    for shotnum in range(1,2):
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum",shotnum)
        for i in tqdm(range(100)):
            log = downprompt(model.texttoken1.weight.detach(),model.texttoken2.weight.detach(),model.texttoken3.weight.detach(),model.texttoken4.weight.detach(),model.pretext1.weight.detach(),model.pretext2.weight.detach(),model.pretext3.weight.detach(),model.pretext4.weight.detach(),hid_units, nb_classes,args.combinetype,unify_dim).cuda()
    

            idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
            pretrain_embs = embeds[0, idx_train]
            # print(embeds.shape)
            test_embs = embeds[0, idx_test]
            # print(test_embs.shape)
            train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
            # print("true",i,train_lbls)
            # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
            opt = torch.optim.Adam([
                {'params': log.parameters()}
            ], lr=downstreamlr)
            # opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            pat_steps = 0
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            for _ in range(400):
                log.train()
                opt.zero_grad()
                # print(features)
                # preval_embs = embeds[0, idx_val]
                
                # logits = log(pretrain_embs)
                logits = log(features,sp_adj,sparse,model.gcn,idx_train,pretrain_embs,train_lbls,1).float().cuda()
                loss = xent(logits, train_lbls)


                
                # print('loss' ,loss)
                # print("predict=",torch.argmax(logits, dim=1))
                # print("lbels",train_lbls)
                # print("predict=",torch.argmax(logits, dim=1))
                # print("train acc",torch.sum(torch.argmax(logits, dim=1)== train_lbls).float() / train_lbls.shape[0])
                if loss < best:
                    best = loss
                    # best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!')
                    break
                
                loss.backward(retain_graph=True)
                opt.step()
            logits = log(features,sp_adj,sparse,model.gcn,idx_test,test_embs)
            # print("logits",logits)
            # print(log.a)
            preds = torch.argmax(logits, dim=1).cuda()
            # print('preds:',preds)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            # print('acc:[{:.4f}]'.format(acc))
            tot += acc
        print('-' * 100)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / 100))
        accs = torch.stack(accs)
        print('Mean:[{:.4f}]'.format(accs.mean().item()))
        print('Std :[{:.4f}]'.format(accs.std().item()))
        print('-' * 100)
        row = [shotnum,lr,downstreamlr,hid_units,accs.mean().item(),accs.std().item()]
        out = open("data/NIPS24_{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
