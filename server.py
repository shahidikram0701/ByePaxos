from concurrent import futures
import logging
from rwmutex import RWLock
import pickle

import grpc
from proto import helloworld_pb2
from proto import helloworld_pb2_grpc
from datetime import datetime
import os


class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def __init__(self):
        self.clientPastRequests = {}
        self.lastRequestArrivalTime = {}
        self.lock1 = RWLock()
        self.lock2= RWLock()

    def SayHello(self, request, context):
        clientId = request.clientId
        requestId = request.requestId
        timeAtClient = request.timeAtClient
        pastWindowData = request.pastWindowData
        seqNum = request.sequenceNumber

        print("[ " + seqNum + " ]" + "Received request from " + clientId + " with requestId: " + requestId)

        serverTime = datetime.now()

        sequenceNumberTuple = "(SequenceNumber, " + seqNum + ")"  
        requestArrivalTimeTuple = "(RequestArrivalTime, " + serverTime.strftime('%Y-%m-%d %H:%M:%S') + ")"
        clientIdTuple = "(ClientId, " + clientId + ")"
        timeAtClientTuple = "(TimeAtClient, " + timeAtClient + ")"
        pastWindowDataTuple = "(PastWindowData, " + str(pickle.loads(pastWindowData.encode())) + ")"
        timeSinceClientsLastReqTuple = "(TimeSinceThisClientsLastRequest, "
        with self.lock1.read:
            if clientId in self.clientPastRequests:
                timeSinceClientsLastReqTuple += str(serverTime - self.clientPastRequests[clientId]) + ")"
            else:
                timeSinceClientsLastReqTuple += "None)"

        lastReqByTuple = "(LastRequestBy, "
        timeSinceLastReqTuple = "(TimeSincesLastRequest, "
        with self.lock2.read:
            if self.lastRequestArrivalTime:
                lastReqByTuple += self.lastRequestArrivalTime["clientId"] + ")"
                timeSinceLastReqTuple += str(serverTime - self.lastRequestArrivalTime["time"]) + ")"
            else:
                lastReqByTuple += "None)"
                timeSinceLastReqTuple += "None)"
                
        requestIdTuple = "(RequestId, " + requestId + ")"
        
        logging.info("[RequestLog]"
            + requestIdTuple + "; " 
            + sequenceNumberTuple + "; " 
            + requestArrivalTimeTuple + "; "
            + clientIdTuple + "; "
            + timeAtClientTuple + "; "
            + pastWindowDataTuple + "; "
            + timeSinceClientsLastReqTuple + "; "
            + lastReqByTuple + "; "
            + timeSinceLastReqTuple + ";"
        )
        '''
        logging.info(
            "[ " + requestId + " ]" + 
            "[ RequestArrivalTime ]" + 
            serverTime.strftime('%Y-%m-%d %H:%M:%S')
        )
        logging.info(
            "[ " + requestId + " ]" + 
            "[ ClientId ]" + 
            clientId
        )
        logging.info(
            "[ " + requestId + " ]" + 
            "[ TimeAtClient ]" + 
            timeAtClient
        )
        # print("PastWindowData", pickle.loads(pastWindowData.encode()))
        logging.info(
            "[ " + requestId + " ]" + 
            "[ PastWindowData ]" + 
            str(pickle.loads(pastWindowData.encode()))
        )
        with self.lock1.read:
            if clientId in self.clientPastRequests:
                logging.info(
                    "[ " + requestId + " ]" +
                    "[ TimeSinceThisClientsLastRequest ]" +
                    str(serverTime - self.clientPastRequests[clientId])
                )
        '''
        with self.lock1.write:
            self.clientPastRequests[clientId] = serverTime
        '''
        with self.lock2.read:
            if self.lastRequestArrivalTime:
                logging.info(
                    "[ " + requestId + " ]" +
                    "[ LastRequestBy ]" + 
                    self.lastRequestArrivalTime["clientId"]
                )
                logging.info(
                    "[ " + requestId + " ]" +
                    "[ TimeSincesLastRequest ]" +
                    str(serverTime - self.lastRequestArrivalTime["time"])
                )
        '''
        with self.lock2.write:
            self.lastRequestArrivalTime["clientId"] = clientId
            self.lastRequestArrivalTime["time"] = serverTime
        
        return helloworld_pb2.HelloReply(requestId = requestId, serverTime = serverTime.strftime('%Y-%m-%d %H:%M:%S'), requestTime = timeAtClient)


def serve():
    port = '50059'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    logging.info("Server started, listening on " + port)
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    path = "logs/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    logging.basicConfig(
        filename="logs/server.log",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    serve()