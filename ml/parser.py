import re
import pickle
import json
import argparse

CLIENT_JSON_ATTRIBUTES = ["Memory"]
SERVER_JSON_ATTRIBUTES = ["PastWindowData", "Memory", "History"]

def parse_server(file_name):
    server_logs = list()

    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        
        log_attributes = dict()
        
        if "ReplicaRequestLog" in line:
            log_attributes["LogType"] = "ReplicaRequestLog"
        elif "RequestLog" in line:
            log_attributes["LogType"] = "RequestLog"

        attributes = re.findall("\((\w+), ([^\);]+)\);", line)
        
        for attribute in attributes:
            if attribute[0] in SERVER_JSON_ATTRIBUTES:
                log_attributes[attribute[0]] = json.loads(attribute[1].replace("\'", "\""))
            else:
                log_attributes[attribute[0]] = attribute[1]

        if "RequestLog" in line or "ReplicaRequestLog" in line:
            server_logs.append(log_attributes)
        
    f.close()

    f = open(file_name + ".json", "w")
    f.write(json.dumps(server_logs, indent = 4))

    f.close()

def parse_client(file_name):
    client_logs = list()

    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        if "ResponseLog" in line:
            log_attributes = dict()
            attributes = re.findall("\((\w+), ([^\);]+)\);", line)
            for attribute in attributes:
                if attribute[0] == CLIENT_JSON_ATTRIBUTES:
                    log_attributes[attribute[0]] = json.loads(attribute[1].replace("\'", "\""))
                else:
                    log_attributes[attribute[0]] = attribute[1]
            client_logs.append(log_attributes)

    f.close()

    f = open(file_name + ".json", "w")
    f.write(json.dumps(client_logs, indent = 4))

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_names", action = "append", help = "file names", required = True)
    parser.add_argument("-p", "--base_path")
    parser.add_argument("-t", "--type", help = "client or server", required = True)

    args = parser.parse_args()

    s = "./data/"
    if args.base_path is not None:
        s = args.base_path
        if s[-1] != '/':
            s += "/"

    for f_name in args.file_names:
        if args.type == "client":
            parse_client(s + f_name)
        elif args.type == "server":
            parse_server(s + f_name)
        else:
            print("-t can only be \"client\" or \"server\"")
            exit(0)