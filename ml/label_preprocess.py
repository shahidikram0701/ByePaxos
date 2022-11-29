import json
import datetime

SERVER_DICT={
    "amd159": 0,
    "c220g5-110531": 1,
    "clnode241": 2,
    "pc421": 0
}
memory_keys = ["total", "available", "percent", "used", "free", "active", "inactive", "buffers", "cached", "shared", "slab"]
def convert_time(time_string, server_name, is_server):
    normalized_time = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S.%f')
    if is_server:
        normalized_time -= datetime.timedelta(hours=SERVER_DICT[server_name])
    normalized_seconds = (normalized_time - normalized_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return normalized_seconds
def convert_rtt(time_string):
    temp = datetime.datetime.strptime(time_string, '%H:%M:%S.%f')
    return (temp - temp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
def convert_to_data(data, server_name):
    server_time = convert_time(data["RequestArrivalTime"], server_name, True)
    client_time = convert_time(data["TimeAtClient"], None, False)
    time_since_client_last = []
    if data["TimeSinceThisClientsLastRequest"] != "None":
        time_since_client_last.append(convert_rtt(data["TimeSinceThisClientsLastRequest"]))
    last_requested_by = []
    if data["LastRequestBy"] != "None":
        last_requested_by = list(map(int, data["LastRequestBy"].split('.')))
    time_since_last_request = []
    if data["TimeSinceLastRequest"] != "None":
        time_since_last_request.append(convert_rtt(data["TimeSinceLastRequest"]))
    memory = []
    for k in memory_keys:
        memory.append(data["Memory"][k])
    past_window = []
    for past_req in data["PastWindowData"]:
        past_window.append(convert_rtt(past_req["rtt"]))
    history = dict()
    for client in data["History"]:
        l = list()
        for req in data["History"][client]:
            l.append(convert_rtt(req["rtt"]))

        history[client] = l
    return {
        "RequestArrivalTime": server_time,
        "ClientId": list(map(int, data["ClientId"].split('.'))),
        "SequenceNumber": int(data["SequenceNumber"]),
        "TimeAtClient": client_time,
        "TimeSinceThisClientsLastRequest":  time_since_client_last,
        "TimeSinceLastRequest": time_since_last_request,
        "LastRequestBy": last_requested_by,
        "Memory": memory,
        "CPUUtil": float(data["CPUUtil"]),
        "PastWindowData": past_window,
        "History":history

    }
def filter_request_log(data, seen):
    if data["LogType"] == "RequestLog" and (data["ClientId"], data["SequenceNumber"]) in seen:
        return True
    else:
        return False
def localize_time(data, start_time):
    for server_name in data:
        for info in data[server_name]:
            data_time = datetime.datetime.strptime(info["RequestArrivalTime"], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=SERVER_DICT[server_name])
            info["RequestArrivalTime"] = (data_time-start_time).total_seconds()
def find_seen(data):
    server_req = dict()

    for server_name in SERVER_DICT:
        server_req[server_name] = set()
        for d in data[server_name]:
            server_req[server_name].add((d["ClientId"], d["SequenceNumber"]))
    inter = None
    for server_name in SERVER_DICT:
        if inter == None:
            inter = server_req[server_name]
        else:
            inter = inter.intersection(server_req[server_name])
    return inter
def normalize_data(batch):
    data = dict()
    initial_date_times = list()

    raw_data = dict()
    for server_name in SERVER_DICT:
        f = open("./data_processed/server-{}-{}.json".format(server_name, batch), 'r')
        raw_json = json.load(f)
        f.close()
        raw_data[server_name] = raw_json
    seen = find_seen(raw_data)

    for server_name in raw_data:

        conv_func = lambda lst: convert_to_data(lst, server_name)
        filter_func = lambda lst: filter_request_log(lst, seen)

        data[server_name] = list(map(conv_func, list(filter(filter_func, raw_data[server_name]))))
        #initial_date_times.append(datetime.datetime.strptime(data[server_name][0]["RequestArrivalTime"], '%Y-%m-%d %H:%M:%S'))
    # min_time = min(initial_date_times)
    # localize_time(data, min_time)
    f = open("./label_normalized/server-{}.json".format(batch), "w")
    json.dump(data, f, indent = 4)
    f.close()

if __name__ == "__main__":
    for i in range(1,10):
        normalize_data(i)
        print(i)

    
    
    
