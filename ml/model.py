import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
class TraceDataset(Dataset):
    def __init__(self, buffers):
        self.buffers = buffers
        self.data = [None]*9
        self.mapping = [None]*9
        self.reverse_mapping = [None]*9
        self.output = [None]*9
        self.min_arrival = 5.05484710e+04
        self.max_arrival = 6.58189520e+04
        self.increment = 0.1 / (max_arrival - min_arrival)

        self.SERVER_NAMES = ["amd159", "c220g5-110531", "clnode241", "pc421"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        for batch in range(1,10):
            #f = open("./label_normalized/server-{}.json".format(batch), "r")
            #f = open("./true_normalized-{}".format(i), "r")
            d = torch.tensor(pd.read_csv("./true_normalized-{}".format(i), header = None).to_numpy()).to(self.device)
            ssize = torch.tensor(pd.read_csv("./true_normalized-{}-ssize".format(i), header = None).to_numpy()).to(self.device)
            
            self.data[batch-1], self.output[batch-1] = self.create_tensor(d, ssize)
            self.mapping[batch-1], self.reverse_mapping[batch-1] = self.create_data_index_match(self.data[batch-1], ssize)
            
            #self.create_tensor(data)
    def create_data_index_match(data, ssize):
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
                indexes[j] = index_mapping[self.get_shared_req_id(data[i][j])]
                reverse_indexes[index_mapping[self.get_shared_req_id(data[i][j])]] = j
            index_mapping.append(indexes)
            reverse_mapping.append(reverse_indexes)
        return index_mapping, reverse_mapping

    
    def create_tensor(self, data, ssize):
        l = list()
        o = list()
        prev  = 0
        for i in range(4):
            l.append(data[prev:prev+ssize[i].item()])
            o.append(torch.zeros(ssize[i].item()).to(self.device))
            prev += ssize[i].item()buffer_offset = 0
        return l, o
    
            

    def get_data(self):
        return self.data    
    def get_min_time(self, data):
        l = list()
        for i in range(4):
            l.append(data[i][0, 3])
        return min(l)
    def get_shared_req_id(self, req):
        return (req[1], req[4])
    def server_name_to_ind(self, name):
        return self.SERVER_NAMES.index(name)
    def compute_new_labels(self, models, batch):
        data = self.data[batch]
        mapping = self.mapping[batch]
        reverse_mapping = self.reverse_mapping[batch]

        buffers = [list([set() for j in range(self.buffers)]) for i in range(4)]

        uncommitted_set = set(list(range(mapping.size()[0])))

        #ground_truth = [dict() for i in range(4)]
        ground_truth = [list for i in range(4)]

        #req_label = [dict() for i in range(4)]

        start_time = self.get_min_time(data)
        increment = self.increment
        max_time = increment + start_time

        model_index = [0 for i in range(4)]

        c = 0

        req_labels = [req_labels.append(models[i](data[i]).argmax(dim = 1)) for i in range(4)]
            
        
        actual_label = [torch.zeros(mapping.size()[0], dtype = int).to(self.device) for i in range(4)]
        for i in range(4):
            actual_label[i] = req_labels[i].detach().clone()        
 
        filter_req = [torch.ones(mapping.size()[0], dtype = int).to(self.device) for i in range(4)]


        while len(uncommitted_set) != 0:
            for i in range(4):
                #change
                
                #check if there still is index left
                #check if time within max time
                while model_index[i] < data[i].size()[0] and data[i][3].item() < max_time:
                    buffers[i][req_labels[i][model_index[i]].item()].add(mapping[model_index[i]])
                    
                    model_index[i] += 1
            
            #find intersect
            inter = None
            for i in range(4):
                if inter is None:
                    inter =  buffers[i][0]
                else:
                    inter = inter.intersection(buffers[i][0])
            
            uncommitted_set = uncommitted_set.difference(inter)

            #update
            for i in range(4):
                remain = buffers[i][0].difference(inter)
                for req in remain:
                    ind = reverse_mapping[i][req]
                    actual_label[i, ind] += 1
                    if actual_label[i][ind].item() == self.buffers:
                        filter_req[i][ind] == 0
                    else:
                        buffers[i][1].add(req)
                buffers[i].pop(0)
                buffers[i].append(set())
            
            max_time += increment
        fitern_ind = [filter_req[i].nonzero().squeeze(1) for i in range(4)]
        one_hot = [F.one_hot(torch.index_select(actual_label[i], 0, filter_ind[i]).to(self.device), num_classes = self.buffers) for i in range(4)]

        return one_hot, filter_ind




    def compute_labels(self, models_dict, batch):
        #print(self.data)
        data = self.data[batch]

        buffers = dict()

        #create buffers initial
        for server_name in self.SERVER_NAMES:
            buffers[server_name] = list()
            for i in range(self.buffers):
                buffers[server_name].append(set())
        buffer_offset = 0

        uncommitted_set = set()


        
        for d in data[self.server_name_to_ind(list(models_dict.keys())[0])]:
            uncommitted_set.add(self.get_shared_req_id(d))
        committed_set = set()

        ground_truth = dict()
        for model_name in models_dict:
            ground_truth[model_name] = dict()

        req_label = dict()
        for model_name in models_dict:
            req_label[model_name] = dict()
        
        start_time = self.get_min_time(data)
        increment = 0.1
        max_time = increment

        model_index = dict()
        for model_name in models_dict:
            model_index[model_name] = 0
        
        c =0

        

        while len(uncommitted_set) != 0:
            
            for model_name, model in models_dict.items():
               # print(data[model_name][model_index[model_name]]["RequestArrivalTime"].tolist())
                # print(model_index[model_name] < len(data[model_name]))
                # print(data[model_name][model_index[model_name]]["RequestArrivalTime"].tolist() <= start_time + max_time) 
                while model_index[model_name] < len(data[model_name]) and data[model_name][model_index[model_name]]["RequestArrivalTime"].tolist() <= start_time + max_time:
                    # maybe turn into tensors here?
                    label = torch.argmax(model(data[model_name][model_index[model_name]])).item()
                    #print(label)
                    curr_req = data[model_name][model_index[model_name]]
                    req_label[model_name][self.get_shared_req_id(curr_req)] = label
                    buffers[model_name][buffer_offset + label].add(self.get_shared_req_id(data[model_name][model_index[model_name]]))
                    
                    model_index[model_name] += 1
            
            #find intersections of first buffer
            inter = None
            for model_name in models_dict:
                if inter is None:
                    inter = buffers[model_name][buffer_offset]
                else:
                    inter = inter.intersection(buffers[model_name][buffer_offset])
            
            #update ground truth
            committed_set.update(inter)
            uncommitted_set = uncommitted_set.difference(inter)
            for request in inter:
                for model_name in req_label:
                    ground_truth[model_name][request] = req_label[model_name][request]

            #push rest to next buffer
            for model_name in models_dict:
                remain = buffers[model_name][buffer_offset].difference(inter)
                for req in remain:
                    req_label[model_name][req] += 1
                buffers[model_name][buffer_offset+1].update(remain)
                buffers[model_name].append(set())
            
            #update offest
            buffer_offset += 1
            
            max_time += increment

            #print(max_time)
            # if c == 100:
            #     print(ground_truth)
            #     exit(0)
            # c += 1
        return ground_truth
    def compute_label_tensor(self, ground_truth):
        labels = list()
        for i in range(9):
            l_dict = dict()
            for key in self.data[i]:
                l = list()
                # print(len(self.data))
                # print(key)
                # print
                # print(self.data[i].keys())
                for req in self.data[i][key]:
                    l.append(ground_truth[i][key][self.get_shared_req_id(req)])
                    if l[-1] > 9:
                        l[-1] = 9
                t = list()
                for j in range(10):
                    t.append(l.count(j))
                #print("{} : ")
                l_dict[key] = F.one_hot(torch.tensor(l), num_classes = 10)
            labels.append(l_dict)
        return labels
    
            


# client network
# 
# server network
#

class BufferClassifier(torch.nn.Module):
    def __init__(self, ip):
        super().__init__()

        self.ip = ip
        # past window from client side
        self.past_window = torch.nn.RNN(1, 10) 
        #self.past_window.apply(self.init_weights)
        # past history of each server
        self.history = list()
        for i in range(4):
            self.history.append(torch.nn.RNN(1, 10))
            #self.history[i].apply(self.init_weights)

        self.memory = torch.nn.Linear(11, 10)
        self.memory.apply(self.init_weights)

        self.cpuutil = torch.nn.Linear(1, 10)
        self.cpuutil.apply(self.init_weights)

        self.timelast = torch.nn.Linear(1, 10)
        self.timelast.apply(self.init_weights)

        self.timeclientlast = torch.nn.Linear(1,10)
        self.timeclientlast.apply(self.init_weights)

        self.timeatclient = torch.nn.Linear(1,10)
        self.timeatclient.apply(self.init_weights)

        self.sequencenumber = torch.nn.Linear(1,10)
        self.sequencenumber.apply(self.init_weights)

        self.clientid = torch.nn.Linear(4,10)
        self.clientid.apply(self.init_weights)

        self.lastrequested = torch.nn.Linear(4,10)
        self.lastrequested.apply(self.init_weights)

        self.hidden = torch.nn.Linear(13, 13)
        self.hidden.apply(self.init_weights)
        
        self.feedforward = torch.nn.Linear(13, 1)
        self.feedforward.apply(self.init_weights)
    def init_weights(self, module):
        module.weight.data.zero_()
    def forward(self, data):
        #print(data)
        
        input = torch.zeros([10,13], dtype=torch.float32)

        if data["PastWindowData"].size()[-1] != 0:
            input[:,0] = self.past_window(data["PastWindowData"].unsqueeze(1))[1].squeeze(0)
        if len(data["History"].keys()) != 0:
            for i in range(4):
                if self.ip[i] in data["History"]:
                    input[:,i+1] = self.history[i](data["History"][self.ip[i]].unsqueeze(1))[1].squeeze(0)
        input[:,5] = self.memory(data["Memory"])
        #print(data["CPUUtil"].size())
        input[:,6] = self.cpuutil(data["CPUUtil"].unsqueeze(0))
        if data["TimeSinceLastRequest"].size()[-1] != 0:
            input[:,7] = self.timelast(data["TimeSinceLastRequest"])
        if data["TimeSinceThisClientsLastRequest"].size()[-1] != 0:
            input[:,8] = self.timeclientlast(data["TimeSinceThisClientsLastRequest"])
        input[:,9] = self.timeatclient(data["TimeAtClient"].unsqueeze(0))
        # print(data["ClientId"].unsqueeze(1).size())
        # print(data["ClientId"].to(torch.float32).unsqueeze(1))
        input[:,10] = self.clientid(data["ClientId"].to(torch.float32).unsqueeze(0))
        if data["LastRequestBy"].size()[-1] != 0:
            input[:,11] = self.lastrequested(data["LastRequestBy"].to(torch.float32).unsqueeze(0))
        input[:,12] = self.sequencenumber(data["SequenceNumber"].to(torch.float32).unsqueeze(0))
        
        h_out = self.hidden(input)
        to_ff = F.relu(h_out)
        #print(to_ff.size())
        output = self.feedforward(to_ff)
        output = F.softmax(output, dim = 0)

        
        return output
        #forward


if __name__ == "__main__":
    #training loop
    models ={
        "amd159": None,
        "c220g5-110531": None,
        "clnode241": None,
        "pc421": None
    }
    ip = ["128.110.219.70", "128.105.144.137", "130.127.134.5", "155.98.38.21"]
    for key in models:
        model = BufferClassifier(ip)
        models[key] = model
    ds = TraceDataset(10)
    ds.load_data()
    #train here
    epochs = 20

    for i in range(epochs):
        print("Starting epoch {}".format(i))
        print("Calculating Labels")
        for key in models:
            models[key].eval()
        ground_truth = list()

        with torch.no_grad():
            for i in range(0,9):
                gt = ds.compute_labels(models, i)
                ground_truth.append(gt)
                print("Finished GT {}".format(i))
        print("finished ground truth")
        data = ds.get_data()
        print("turning data to tensor")
        labels = ds.compute_label_tensor(ground_truth)
        #exit(0)
        print("Starting training")
        for key in models:
            models[key].train()

        for j in range(9):
            inputs = data[j]
            outputs = labels[j]
            print("Starting Batch {}".format(j))
            for key in inputs:
                size = len(inputs[key])
                losses = torch.zeros(10)

                correct = 0
                label_total = 0
                for d in range(size):
                    o = models[key](inputs[key][d]).squeeze(1)
                    e = outputs[key][d]
                    losses += torch.abs(o-e)
                    label_total += torch.argmax(e).item()
                    if torch.argmax(o).item() == torch.argmax(e).item():
                        correct += 1
               
                models[key].zero_grad()
                total_loss = torch.sum(losses) + torch.tensor(label_total/9)
                total_loss.backward()
                print("Model {} Accuracy: {} Loss: {}".format(key, correct/size, total_loss))
        print("Done with epoch")
                
# processing 1 at a time
# create singular tensor representation of all data
# i
# need push onto gpu
# padded sequence in torch with a mask
# normalize
# zero the initial -> softmax would be [0.1, 0.1, 0.1 ...]

