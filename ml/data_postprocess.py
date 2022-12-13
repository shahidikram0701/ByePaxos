import json
import numpy as np
SERVER_NAMES = ["amd159", "c220g5-110531", "clnode241", "pc421"]
CLIENT_ID = ["155.98.38.97", "128.110.96.31", "155.98.38.46"]
SERVER_IP = ["128.110.219.70", "128.105.144.137", "130.127.134.5", "155.98.38.21"]
def generate_row(row, name):
    l = list()
    l.append(SERVER_NAMES.index(name))
    l.append(CLIENT_ID.index(".".join(str(v) for v  in row["ClientId"])))
    if len(row["LastRequestBy"]) == 0:
        l.append(len(CLIENT_ID))
    else:
        l.append(CLIENT_ID.index(".".join(str(v) for v  in row["LastRequestBy"])))
    l.append(row["RequestArrivalTime"])

    l.append(row["SequenceNumber"])
    l.append(row["TimeAtClient"])
    if len(row["TimeSinceThisClientsLastRequest"]) == 0:
        l.append(0)
    else:
        l.append(row["TimeSinceThisClientsLastRequest"][0])
    if len(row["TimeSinceLastRequest"]) == 0:
        l.append(0)
    else:
        l.append(row["TimeSinceLastRequest"][0])
    
    l += row["Memory"]
    l.append(row["CPUUtil"])

    if len(row["PastWindowData"]) < 10:
        for i in range(10-len(row["PastWindowData"])):
            l.append(0)
    l += row["PastWindowData"]
    for server_ip in SERVER_IP:
        if server_ip in row["History"]:

            if len(row["History"][server_ip]) < 10:
                for i in range(10-len(row["History"][server_ip])):
                    l.append(0)
            l += row["History"][server_ip]
        else:
            for i in range(10):
                l.append(0)

    return np.array(l)
    #return l
def normalize_data(data):
    for i in range(4, 20):
        mi = np.amin(data[:,i])
        ma = np.amax(data[:,i])
        data[:,i] = (data[:,i] - mi) / (ma - mi)
    mi = min(np.amin(data[:,3]), np.amin(data[:,20:]))
    ma = max(np.amax(data[:,3]), np.amax(data[:,20:]))
    data[:,3] = (data[:,3] - mi) / (ma - mi)
    data[:,20:] = (data[:,20:] - mi) / (ma - mi)
    return data


if __name__ == "__main__":
    f = open("./label_normalized/server-1.json", 'r')
    data_raw = json.load(f)
    total_len = 0
    for server_name in SERVER_NAMES:
        total_len += len(data_raw[server_name])
    data = np.zeros((total_len, 70))
    c = 0
    for server_name in SERVER_NAMES:
        for i in range(len(data_raw[server_name])):
            data[c] = generate_row(data_raw[server_name][i], server_name)
            c += 1
    data = normalize_data(data)
   
    np.savetxt("true_normalized.csv", data, delimiter=",")
    
    