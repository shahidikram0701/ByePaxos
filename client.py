import logging
import socket
import uuid
import pickle
import datetime
import os
import time
import psutil
import sys

import grpc
from proto import helloworld_pb2
from proto import helloworld_pb2_grpc

selfId = ""

pastWindowData = []
history = {}

def sayHello(stub, seq, replica):
    global selfId
    global pastWindowData

    timeAtClient = datetime.datetime.now()
    requestId = str(uuid.uuid1())

    seqNum = str(seq)

    ''' System conditions '''
    cpu_percentage = psutil.cpu_percent()
    memory_conditions = dict(psutil.virtual_memory()._asdict())

    cpu_util_tuple = "(CPUUtil, " + str(cpu_percentage) + ")"
    memory_conditions_tuple = "(Memory, " + str(memory_conditions) + ")"

    response = stub.SayHello(
        helloworld_pb2.HelloRequest(
            clientId = selfId, 
            requestId = requestId, 
            timeAtClient = timeAtClient.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], 
            pastWindowData = pickle.dumps(pastWindowData, 0).decode(),
            sequenceNumber = seqNum,
            history = pickle.dumps(history, 0).decode()
        )
    )

    
    if(requestId == response.requestId):
        '''
        logging.info(
            "[ " + requestId + " ]" +
            "[ ServerTime ]" + 
            response.serverTime
        )
        '''
        responseReceivedTime = datetime.datetime.now()

        rtt = str(responseReceivedTime - timeAtClient)
        '''
        logging.info(
            "[ " + requestId + " ]" +
            "[ RTT ]" + 
            rtt
        )
        '''
        pastWindowData.append({"requestId": requestId, "rtt": rtt})

        if(replica not in history):
            history[replica] = []

        history[replica].append({"requestId": requestId, "rtt": rtt})

        if(len(history[replica]) > 10):
            history[replica].pop(0)
        
        
        serverTimeTuple = "(ServerTime, " + response.serverTime + ")"
        requestIdTuple = "(RequestId, " + requestId + ")"
        rttTuple = "(RTT, " + rtt + ")"
        requestSentAtTuple = "(RequestSentAt, " + timeAtClient.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ")"
        responseReceivedTimeTuple = "(ResponseReceivedAt, " + responseReceivedTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ")"
        sequenceNumberTuple = "(SequenceNumber, " + seqNum + ")"  

        logging.info(
            "[ResponseLog]"
            + requestIdTuple + "; "
            + sequenceNumberTuple + "; "
            + serverTimeTuple + "; "
            + requestSentAtTuple + "; "
            + responseReceivedTimeTuple + "; "
            + rttTuple + "; "
            + cpu_util_tuple + "; "
            + memory_conditions_tuple + "; "
        )

        if(len(pastWindowData) > 10):
            logging.info(
                "[ " + requestId + " ]" +
                "[ DeletingPastEntry ]" + 
                str(pastWindowData.pop(0))
            )
    
    return("RTT completed successfully for request id: " + requestId)

def run(port, seq):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    replicas = ["128.110.219.70", "128.105.144.137", "130.127.134.5", "155.98.38.21"]

    for replica in replicas:
        print("Seding request to " + replica)
        with grpc.insecure_channel(replica + ":" + port) as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            sayHello(stub, seq, replica)


    # For local dev

    # print("Seding request to 127.0.0.1")
    # with grpc.insecure_channel('127.0.0.1:' + port) as channel:
    #     stub = helloworld_pb2_grpc.GreeterStub(channel)
    #     sayHello(stub, seq, "127.0.0.1")


if __name__ == '__main__':
    path = "logs/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    port = "50060"
    logfilename = sys.argv[1]
    logfilepath = "logs/" + logfilename
    logging.basicConfig(
        filename=logfilepath,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    selfId = s.getsockname()[0]
    s.close()
    t_end = time.time() + 60 * 20
    seq = 0
    while time.time() < t_end:
        run(port, seq)
        seq += 1