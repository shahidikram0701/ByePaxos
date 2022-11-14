import logging
import socket
import uuid
import pickle
import datetime
import os

import grpc
from proto import helloworld_pb2
from proto import helloworld_pb2_grpc

selfId = ""

pastWindowData = []

def sayHello(stub):
    global selfId
    global pastWindowData

    timeAtClient = datetime.datetime.now()
    requestId = str(uuid.uuid1())

    response = stub.SayHello(
        helloworld_pb2.HelloRequest(
            clientId = selfId, 
            requestId = requestId, 
            timeAtClient = timeAtClient.strftime('%Y-%m-%d %H:%M:%S'), 
            pastWindowData = pickle.dumps(pastWindowData, 0).decode()
        )
    )

    if(requestId == response.requestId):
        logging.info(
            "[ " + requestId + " ]" +
            "[ ServerTime ]" + 
            response.serverTime
        )
        responseReceivedTime = datetime.datetime.now()

        rtt = str(responseReceivedTime - timeAtClient)
        logging.info(
            "[ " + requestId + " ]" +
            "[ RTT ]" + 
            rtt
        )
        pastWindowData.append({"requestId": requestId, "rtt": rtt})
        if(len(pastWindowData) > 10):
            logging.info(
                "[ " + requestId + " ]" +
                "[ DeletingPastEntry ]" + 
                str(pickle.dumps(pastWindowData.pop(0), 0))
            )
    
    return("RTT completed successfully for request id: " + requestId)

def run(port):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel('128.110.217.82:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub)
    
    with grpc.insecure_channel('128.105.145.255:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub)
    
    with grpc.insecure_channel('130.127.133.136:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub)

    with grpc.insecure_channel('155.98.38.24:' + port) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        sayHello(stub) 

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
    selfId = s.getsockname()[0] + ":" + port
    s.close()
    i = 0
    while(i < 10):
        run(port)
        i += 1