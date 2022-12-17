import json
import numpy as np
SERVER_NAMES = ["amd159", "c220g5-110531", "clnode241", "pc421"]
CLIENT_ID = ["155.98.38.97", "128.110.96.31", "155.98.38.46"]
SERVER_IP = ["128.110.219.70", "128.105.144.137", "130.127.134.5", "155.98.38.21"]
def generate_row(row, name):
    l = list()
    l.append(SERVER_NAMES.index(name))
    print(len(l))
    l.append(CLIENT_ID.index(".".join(str(v) for v  in row["ClientId"])))
    print(len(l))
    if len(row["LastRequestBy"]) == 0:
        l.append(len(CLIENT_ID))
    else:
        l.append(CLIENT_ID.index(".".join(str(v) for v  in row["LastRequestBy"])))
    print(len(l))
    l.append(row["RequestArrivalTime"])
    print(len(l))
    l.append(row["SequenceNumber"])
    print(len(l))
    l.append(row["TimeAtClient"])
    print(len(l))
    if len(row["TimeSinceThisClientsLastRequest"]) == 0:
        l.append(0)
    else:
        l.append(row["TimeSinceThisClientsLastRequest"][0])
    print(len(l))
    if len(row["TimeSinceLastRequest"]) == 0:
        l.append(0)
    else:
        l.append(row["TimeSinceLastRequest"][0])
    print(len(l))
    l += row["Memory"]
    print(len(l))
    l.append(row["CPUUtil"])
    print(len(l))
    if len(row["PastWindowData"]) < 10:
        for i in range(10-len(row["PastWindowData"])):
            l.append(0)
    l += row["PastWindowData"]
    print(len(l))
    for server_ip in SERVER_IP:
        if server_ip in row["History"]:

            if len(row["History"][server_ip]) < 10:
                for i in range(10-len(row["History"][server_ip])):
                    l.append(0)
            l += row["History"][server_ip]
        else:
            for i in range(10):
                l.append(0)
    print(len(l))
    exit(0)
    return np.array(l)
    #return l
def normalize_data(data, min_max_arrival):
    for i in range(3, 20):
        mi = np.amin(data[:,i])
        ma = np.amax(data[:,i])
        data[:,i] = (data[:,i] - mi) / (ma - mi)
        if i == 3:
            min_max_arrival["min"].append(mi)
            min_max_arrival["max"].append(ma)
    mi = np.amin(data[:,20:])
    ma = np.amax(data[:,20:])
    data[:,20:] = (data[:,20:] - mi) / (ma - mi)
    return data
def normalize_all_data(data_l):
    mi_s = None
    ma_s = None
    for d in data_l:
        mi = np.zeros(21)
        ma = np.zeros(21)
        mi[3:20] = np.amin(d[:,3:20], axis = 0)
        ma[3:20] = np.amax(d[:,3:20], axis = 0)
        mi[20] = np.amin(data[:,20:])
        ma[20] = np.amax(data[:,20:])
        if mi_s is not None:
            mi_s = np.minimum(mi, mi_s)
        else:
            mi_s = mi
        if ma_s is not None:
            ma_s = np.maximum(ma, ma_s)
        else:
            ma_s = ma
    for d in data_l:
        d[:, 3:20] = (d[:, 3:20] - mi_s[3:20]) / (ma_s[3:20] - mi_s[3:20])
        d[:, 20:] = (d[:, 20:] - mi_s[20]) / (ma_s[20] - mi_s[20])
    #
    return data_l

if __name__ == "__main__":
    
    data_all = []
    server_sizes = []
    for b in range(1,10):
        f = open("./label_normalized/server-{}.json".format(b), 'r')
        data_raw = json.load(f)
        total_len = 0
        server_size = []
        for server_name in SERVER_NAMES:
            server_size.append(len(data_raw[server_name]))
            total_len += len(data_raw[server_name])
        server_sizes.append(server_size)
        data = np.zeros((total_len, 70))
        c = 0
        for server_name in SERVER_NAMES:
            for i in range(len(data_raw[server_name])):
                data[c] = generate_row(data_raw[server_name][i], server_name)
                c += 1
        data_all.append(data)
        # data = normalize_data(data, min_max_arrival)

        # np.savetxt("true_normalized-{}.csv".format(b), data, delimiter=",")
    data_all = normalize_all_data(data_all)
    for i in range(len(data_all)):
        #print(data_all[i][-1])
        #np.savetxt("true_normalized-{}.csv".format(i+1),data_all[i], delimiter=",")
        np.savetxt("true_normalized-{}-ssize.csv".format(i+1), np.array(server_sizes[i],dtype = int), delimiter = ",")
    
    