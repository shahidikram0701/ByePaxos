import torch
from torch.utils.data import Dataset
import json

class TraceDataset(Dataset):
    def __init__(self, buffers, model):
        self.files = files
        self.buffers = buffers
        self.data = [None]*9
    def compute_starting_time(data):
        l = list()
        for server_name in data:
            l.append(data[server_name][0]["RequestArrivalTime"])
        return int(min(l) / 60) * 60
    def load_data(){
        for batch in range(1,10):
            f = open("./label_normalized/server-{}".format(batch), "r")
            data = json.load(f)
            f.close()
            self.data[batch-1] = self.create_tensor(data)
    }
    def create_tensor(data):
        for key in data:
            for req in data[key]:
                for k in req:
                    if k != "History":
                        req[k] = torch.tensor(req[k])
                    else:
                        for server in req[k]:
                            req[k][server] = torch.tensor(req[k][server])
             
    def get_min_time(data):
        l = list()
        for key in data:
            l.append(data[key][0]["RequestArrivalTime"].tolist())
        return int(min(l) / 5)*5
    def get_shared_req_id(req):
        return (".".join(req["ClientId"].tolist()), req["SequenceNumber"].tolist())
    def compute_labels(models_dict, batch):
        
        data = self.data[batch]

        buffers = dict()

        #create buffers initial
        for server_name in data:
            buffers[server_name] = list()
            for i in range(self.buffers):
                buffers[server_name].append(set())
        buffer_offset = 0

        uncommitted_set = set()
        for d in data[model.keys()[0]]:
            uncommitted_set.add(self.get_shared_req_id(d))
        committed_set = set()

        ground_truth = dict()

        start_time = self.get_min_time(data)
        max_time = 5

        model_index = dict()
        for model_name in models_dict:
            model_index[model_name] = 0
        while len(uncommitted_set) != 0:
            for model_name, model in models_dict.items():
                while model_index[model_name] < len(data[model_name]) and data[model_name][model_index[model_name]]["RequestArrivalTime"].tolist() <= start_time + max_time:
                    # maybe turn into tensors here?
                    label = torch.argmax(model(data[model_name][model_index[model_name]])).item()
                    buffers[model_name][buffer_offset + label].add(self.get_shared_req_id(data[model_name][model_index[model_name]]))
                    model_index[model_name] += 1
            
            #find intersections of first buffer
            inter = None
            for model_name in models_dict:
                if inter = None:
                    inter = buffers[model_name][buffer_offset]
                else:
                    inter = inter.intersection(buffers[model_name][buffer_offset])
            
            #update ground truth
            committed_set.update(inter)
            uncommitted_set = uncommitted_set.difference(inter)
            for request in inter:
                ground_truth[request] = buffer_offset

            #push rest to next buffer
            for model_name in models_dict:
                buffers[model_name][buffer_offset+1].update(buffers[model_name][buffer_offset].difference(inter))
            
            #update offest
            buffer_offset += 1
            max_time += 5
        return ground_truth
            


            

class BufferClassifier(torch.nn.Module):
    def __init__(self):
        # past window from client side
        self.past_window = torch.nn.RNN(1, 10) 
        # past history of each server
        self.history = torch.nn.RNN(1, 10)

        self.memory = torch.nn.Linear(11, 10)

        self.cpuutil = torch.nn.Linear(1, 10)

        self.timelast = torch.nn.Linear(1, 10)
        
        self.timeclientlast = torch.nn.Linear(1,10)

        self.timeatclient = torch.nn.Linear(1,10)


        self.clientid = torch.nn.Linear(4,10)

        self.lastrequested = torch.nn.Linear(4,10)

        self.hidden = torch.nn.Linear(12, 12)

        self.feedforward = torch.nn.Linear(12, 1)
    
    def forward(data):
        input = torch.zeros([10,9], dtype=torch.float64)

        input[:,0] = self.past_window(data)



