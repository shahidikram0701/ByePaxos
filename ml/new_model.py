# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dGaDzt63x339RlvEVq4n0npY9NnQIOD3
"""

#from google.colab import drive
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# Commented out IPython magic to ensure Python compatibility.

#drive.mount('/content/drive')
# navigate to current directory
#
# %cd drive/MyDrive/CS598AWG/
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
"""Dataset Class"""

class TraceDataset(Dataset):
    def __init__(self, buffers):
        self.buffers = buffers
        self.data = [None]*9
        self.mapping = [None]*9
        self.reverse_mapping = [None]*9
        self.output = [None]*9
        self.min_arrival = 5.05484710e+04
        self.max_arrival = 6.58189520e+04
        self.increment = 0.1/ (self.max_arrival - self.min_arrival)

        self.SERVER_NAMES = ["amd159", "c220g5-110531", "clnode241", "pc421"]

        self.device = device

    def load_data(self):
        for batch in range(1,10):
            #f = open("./label_normalized/server-{}.json".format(batch), "r")
            #f = open("./true_normalized-{}".format(i), "r")
            d = torch.tensor(pd.read_csv("./true_normalized-{}.csv".format(batch), header = None).to_numpy(), dtype = torch.float32).to(self.device)
            ssize = torch.tensor(pd.read_csv("./true_normalized-{}-ssize.csv".format(batch), header = None).to_numpy(), dtype =int).to(self.device)
            
            self.data[batch-1], self.output[batch-1] = self.create_tensor(d, ssize)
            #print( self.create_data_index_match(self.data[batch-1], ssize))
            self.mapping[batch-1], self.reverse_mapping[batch-1] = self.create_data_index_match(self.data[batch-1], ssize)
            print("finish batch")
            #self.create_tensor(data)
    def create_data_index_match(self, data, ssize):
        data_size = ssize[0].item()
        index_mapping = list()
        reverse_mapping = list()

        index_mapping.append(torch.tensor(list(range(data_size)), dtype = int).to(self.device))
        reverse_mapping.append(torch.tensor(list(range(data_size)), dtype = int).to(self.device))
        
        index_mapping_dict = dict()
        for i in range(data_size):
            index_mapping_dict[self.get_shared_req_id(data[0][i])] = i
        for i in range(1,4):
            indexes = torch.zeros(data_size, dtype=int).to(self.device)
            reverse_indexes = torch.zeros(data_size, dtype=int).to(self.device)
            for j in range(data_size):
                indexes[j] = index_mapping_dict[self.get_shared_req_id(data[i][j])]
                reverse_indexes[index_mapping_dict[self.get_shared_req_id(data[i][j])]] = j
            index_mapping.append(indexes)
            reverse_mapping.append(reverse_indexes)
        return (index_mapping, reverse_mapping)

    
    def create_tensor(self, data, ssize):
        l = list()
        o = list()
        prev  = 0
        for i in range(4):
            l.append(data[prev:prev+ssize[i].item()])
            o.append(torch.zeros(ssize[i].item()).to(self.device))
            prev += ssize[i].item()
        return l, o
            
    def get_data(self):
        return self.data    
    def get_min_time(self, data):
        l = list()
        for i in range(4):
            l.append(data[i][0, 3])
        return min(l)
    def get_shared_req_id(self, req):
        return (req[1].item(), req[4].item())
    def server_name_to_ind(self, name):
        return self.SERVER_NAMES.index(name)
    def compute_new_labels(self, models, batch):
        data = self.data[batch]
        mapping = self.mapping[batch]
        #print(mapping)
        reverse_mapping = self.reverse_mapping[batch]

        # print(mapping)
        # print(reverse_mapping)

        buffers = [list([set() for j in range(self.buffers)]) for i in range(4)]

        uncommitted_set = set(list(range(mapping[0].size()[0])))

        #ground_truth = [dict() for i in range(4)]
        ground_truth = [list for i in range(4)]

        #req_label = [dict() for i in range(4)]

        start_time = self.get_min_time(data)
        increment = self.increment
        max_time = increment + start_time

        model_index = [0 for i in range(4)]

        c = 0
        # for i in range(4):
        #     print(models[i](data[i]).argmax(dim = 1))
        #     exit(0)
        req_labels = [models[i](data[i]).argmax(dim = 1) for i in range(4)]
            
        
        actual_label = [torch.zeros(mapping[0].size()[0], dtype = int).to(self.device) for i in range(4)]
        for i in range(4):
            actual_label[i] = req_labels[i].detach().clone()        
        # for i in range(4):
        #     print((actual_label[i] > 19).sum())
        filter_req = [torch.ones(mapping[0].size()[0], dtype = int).to(self.device) for i in range(4)]


        while len(uncommitted_set) != 0:
            for i in range(4):
                #change
                
                #check if there still is index left
                #check if time within max time
                while model_index[i] < data[i].size()[0] and data[i][model_index[i], 3].item() < max_time:
                    # print(len(buffers[i]))
                    # print(req_labels[i][model_index[i]].item())
                    buffers[i][req_labels[i][model_index[i]].item()].add(mapping[i][model_index[i]].item())
                    # print(data[i][model_index[i], 3].item())
                    
                    model_index[i] += 1
            # print(max_time)
            #     #print("finished while")
            # #print("finished for")
            # print(len(uncommitted_set))
            #find intersect
            # for i in range(4):
            #     compute_len = list()
            #     for j in range(self.buffers):
            #         compute_len.append(len(buffers[i][j]))
            #     print(compute_len)
            # for i in range(4):
            #     print(list(buffers[i][0]))
            inter = None
            for i in range(4):
                if inter is None:
                    inter =  buffers[i][0]
                else:
                    inter = inter.intersection(buffers[i][0])
            #print(len(inter))
            uncommitted_set = uncommitted_set.difference(inter)
            # for i in range(4):
            #     for req in inter:
            #         if  actual_label[i][reverse_mapping[i][req].item()] >= self.buffers:
            #             print("FAIL")
            #update
            for i in range(4):
                remain = buffers[i][0].difference(inter)
                for req in remain:
                    #filter out req that might have been canceled
                    if req in uncommitted_set:
                        ind = reverse_mapping[i][req].item()
                        actual_label[i][ind] += 1
                        #print(req, ind, actual_label[i][ind])
                        if actual_label[i][ind].item() >= self.buffers:

                            #filter from all data
                            for j in range(4):
                                filter_req[j][reverse_mapping[j][req].item()] = 0

                            uncommitted_set.remove(req)
                        else:
                            buffers[i][1].add(req)
                buffers[i].pop(0)
                buffers[i].append(set())
            
            max_time += increment
        # for i in range(4):
        #     print(filter_req[i].sum(), filter_req[i].size())
        filter_ind = [filter_req[i].nonzero().squeeze(1) for i in range(4)]
        # for i in range(4):
        #     print(filter_ind[i].size())
        non_filter_ind = []
        data_filtered = [torch.index_select(data[i], 0, filter_ind[i]).to(self.device) for i in range(4)]
        
        # for i in range(4):
        #     print((torch.index_select(actual_label[i], 0, filter_ind[i]) > 19).sum() )
        one_hot = [F.one_hot(torch.index_select(actual_label[i], 0, filter_ind[i]).to(self.device), num_classes = self.buffers) for i in range(4)]
        #print("Done")
        # print(actual_label[0])
        # print(data_filtered[0])
        return one_hot, data_filtered, filter_ind[0].size()[0], filter_req[0].size()[0]

"""Classifier"""

class BufferClassifier(torch.nn.Module):
    def __init__(self, buffer_size):
        super().__init__()
        self.buffer_size = buffer_size
        self.clientid = torch.nn.Embedding(3,10).to(device)
        self.clientid.apply(self.init_weights)
        self.lastrequested = torch.nn.Embedding(4,10).to(device)
        self.lastrequested.apply(self.init_weights)

        self.req_arrival = torch.nn.Linear(1, 10).to(device)
        self.req_arrival.apply(self.init_weights)

        self.sequencenumber = torch.nn.Linear(1,10).to(device)
        self.sequencenumber.apply(self.init_weights)

        self.timeatclient = torch.nn.Linear(1,10).to(device)
        self.timeatclient.apply(self.init_weights)

        self.timeclientlast = torch.nn.Linear(1,10).to(device)
        self.timeclientlast.apply(self.init_weights)

        self.timelast = torch.nn.Linear(1, 10).to(device)
        self.timelast.apply(self.init_weights)

        self.memory = torch.nn.Linear(11, 10).to(device)
        self.memory.apply(self.init_weights)

        self.cpuutil = torch.nn.Linear(1, 10).to(device)
        self.cpuutil.apply(self.init_weights)

        # past window from client side
        self.past_window = torch.nn.RNN(1, 10, batch_first = True, dropout = 0.5).to(device) 
        #self.past_window.apply(self.init_weights)
        # past history of each server
        self.history = list()
        for i in range(4):
            self.history.append(torch.nn.RNN(1, 10, batch_first = True, dropout = 0.5).to(device))

        self.hidden = torch.nn.Linear(140, buffer_size)
        self.hidden.apply(self.init_weights)
        
        self.feedforward = torch.nn.Linear(buffer_size, buffer_size)
        self.feedforward.apply(self.init_weights)
    def init_weights(self, module):
        module.weight.data.zero_()
    def forward(self, data):
        #print(data)
        
        input = torch.zeros([data.size()[0],140], dtype=torch.float32).to(device)
        # print(input.size())
        # print(data.size())
        # print(data[:,1].size())
        # print(data[:,1].long().size())
        input[:,:10] = self.clientid(data[:,1].long())
        #print(data[:,2].long().size())
        #print(self.lastrequested(data[:,2].long()).size())
        input[:,10:20] = self.lastrequested(data[:,2].long())
        # print(data[:3].size())
        # print(data[:3].dtype)
        # print(torch.unsqueeze(data[:,3], 1).size())
        # print(self.req_arrival.parameters)
        #print(torch.unsqueeze(data[:,3], 1).dtype)
        #print(self.req_arrival.dtype)
        #print(torch.unsqueeze(data[:,3], 1).size())
        input[:,20:30] = self.req_arrival(torch.unsqueeze(data[:,3], 1))
        input[:,30:40] = self.sequencenumber(torch.unsqueeze(data[:,4], 1))
        input[:,40:50] = self.timeatclient(torch.unsqueeze(data[:,5], 1))
        input[:,50:60] = self.timeclientlast(torch.unsqueeze(data[:,6], 1))
        input[:,60:70] = self.timelast(torch.unsqueeze(data[:,7], 1))
        input[:,70:80] = self.memory(data[:,8:19])
        input[:,80:90] = self.cpuutil(torch.unsqueeze(data[:,19], 1))
        # print(data[:, 20:30].unsqueeze(2).size())
        # print(self.past_window(data[:, 20:30].unsqueeze(2))[1].squeeze(0).size())
        # exit(0)
        input[:,90:100] = self.past_window(data[:, 20:30].unsqueeze(2))[1].squeeze(0)
        
        input[:,100:110] = self.history[0](data[:, 30:40].unsqueeze(2))[1].squeeze(0)
        input[:,110:120] = self.history[0](data[:, 40:50].unsqueeze(2))[1].squeeze(0)
        input[:,120:130] = self.history[0](data[:, 50:60].unsqueeze(2))[1].squeeze(0)
        input[:,130:140] = self.history[0](data[:, 60:70].unsqueeze(2))[1].squeeze(0)
        
        h_out = self.hidden(input)
        to_ff = F.relu(h_out)
        # print(to_ff.size())
        
        output = self.feedforward(F.dropout(to_ff))
        output = F.softmax(output, dim = 1)

        # print(output.size())

        # exit(0)
        
        return output
        #forward

"""Train Loop"""

def accuracy(output, labels):
    # print("output shape:", output.shape)
    # print("labels shape:", labels.shape)
    predictions = torch.argmax(output, 1)
    actuals = torch.argmax(labels, 1)
    # print("predictions shape:", predictions.shape)
    # print("actuals shape:", actuals.shape)
    # compare predictions to true label
    correct = np.squeeze(predictions.eq(actuals.view_as(predictions)))
    # print("predict labels:", predictions)
    # print("actual labels:", actuals)
    # print("correct results:", correct)
    
    return correct.sum().item() , correct.size()[0], predictions.sum().item()

def train(models, data_set, buffer_size):  # try RNN next
    optimizers = [torch.optim.Adam(models[i].parameters(), lr = 0.03) for i in range(4)]
    criterions = [nn.MSELoss() for i in range(4)]
    epochs = 20
    for i in range(4):
        models[i] = models[i].to(device)

    def train_epoch():
        num_correct = 0
        num_total = 0
        epoch_loss = 0
        filter_req_weight = 0.1
        buffer_weight = 0.0000001
        model_params = [[None]*3 for i in range(4)]
        for model in models:
            model.eval()
        
        #gather labels with alg
        actual_labels_batch = [[None]*9 for i in range(4)]
        data_filtered_batch = [[None]*9 for i in range(4)]

        total_actual = [None]*4
        total_data_filtered = [None]*4

        total_filter_c = 0
        total_data_c = 0

        with torch.no_grad():
            for j in range(0,9):
                actual_labels, data_filtered, filter_count, data_count = data_set.compute_new_labels(models, j)
                
                
                for i in range(4):
                    actual_labels_batch[i][j] = actual_labels[i]
                    data_filtered_batch[i][j] = data_filtered[i]
                total_filter_c += filter_count
                total_data_c += data_count
            for i in range(4):
                total_actual[i] = torch.cat(actual_labels_batch[i]).to(device)
                total_data_filtered[i] = torch.cat(data_filtered_batch[i]).to(device)
        
        #start train
        for model in models:
            model.train()
        
        num_correct_list = list()
        num_total_list = list()
        for i in range(4):
            outputs = models[i](total_data_filtered[i])
            num_correct, num_total, pred_sum = accuracy(outputs, total_actual[i])
            num_correct_list.append(num_correct)
            num_total_list.append(num_total)
            mse_loss = criterions[i](outputs.float(), total_actual[i].float())
            buffer_loss = torch.tensor(pred_sum / buffer_size * buffer_weight, dtype=torch.float32).to(device)
            filter_loss = torch.tensor(total_data_filtered[i].size()[0] * filter_req_weight * (1-total_filter_c/total_data_c) , dtype=torch.float32).to(device)

            print("Model {} Individual Losses:".format(i))
            print(mse_loss.item(), buffer_loss.item(), filter_loss.item())
            mse_loss += buffer_loss + filter_loss
            print("Model {} Total Losses: {}".format(i, mse_loss.item()))
            print("Model {} Accuracy: {}".format(i, num_correct/ num_total))

            optimizers[i].zero_grad()
            mse_loss.backward()
            optimizers[i].step()
            
        print("Accuracy all: {}".format(sum(num_correct_list ) / sum(num_total_list)))

    for e in range(epochs):
        print("epoch:", e)
        train_epoch()

BUFFER_SIZE = 20
models = [BufferClassifier(BUFFER_SIZE) for i in range(4)]
ds = TraceDataset(BUFFER_SIZE)
ds.load_data()
train(models, ds, BUFFER_SIZE)

# """Run Train Loop"""

# train(models, ds, BUFFER_SIZE)