import logging
import socket
import uuid
import pickle
import datetime
import os
import time

import grpc
from proto import helloworld_pb2
from proto import helloworld_pb2_grpc

selfId = ""

pastWindowData = []

def sayHello(stub, seq):
    global selfId
    global pastWindowData

    timeAtClient = datetime.datetime.now()
    requestId = str(uuid.uuid1())

    seqNum = str(seq)

    response = stub.SayHello(
        helloworld_pb2.HelloRequest(
            clientId = selfId, 
            requestId = requestId, 
            timeAtClient = timeAtClient.strftime('%Y-%m-%d %H:%M:%S'), 
            pastWindowData = pickle.dumps(pastWindowData, 0).decode(),
            sequenceNumber = seqNum
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
        
        serverTimeTuple = "(ServerTime, " + response.serverTime + ")"
        requestIdTuple = "(RequestId, " + requestId + ")"
        rttTuple = "(RTT, " + rtt + ")"
        requestSentAtTuple = "(RequestSentAt, " + timeAtClient.strftime('%Y-%m-%d %H:%M:%S') + ")"
        responseReceivedTimeTuple = "(ResponseReceivedAt, " + responseReceivedTime.strftime('%Y-%m-%d %H:%M:%S') + ")"
        sequenceNumberTuple = "(SequenceNumber, " + seqNum + ")"  

        logging.info(
            "[ResponseLog]"
            + requestIdTuple + "; "
            + sequenceNumberTuple + "; "
            + serverTimeTuple + "; "
            + requestSentAtTuple + "; "
            + responseReceivedTimeTuple + "; "
            + rttTuple + ";"
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

    print("Seding request to 128.110.217.82")
    with grpc.insecure_channel('128.110.217.82:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub, seq)
    
    print("Seding request to 128.105.145.255")
    with grpc.insecure_channel('128.105.145.255:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub, seq)
    
    print("Seding request to 130.127.133.136")
    with grpc.insecure_channel('130.127.133.136:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub, seq)

    print("Seding request to 155.98.38.24")
    with grpc.insecure_channel('155.98.38.24:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub, seq) 


    # For local dev

    # print("Seding request to 127.0.0.1")
    # with grpc.insecure_channel('127.0.0.1:' + port) as channel:
    #     stub = helloworld_pb2_grpc.GreeterStub(channel)
    #     sayHello(stub, seq)


if __name__ == '__main__':
    path = "logs/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    port = "50059"
    logging.basicConfig(
        filename="logs/client.log",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    selfId = s.getsockname()[0]
    s.close()
    t_end = time.time() + 1
    seq = 0
    while time.time() < t_end:
        run(port, seq)
        seq += 1