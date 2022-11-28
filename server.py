from concurrent import futures
import logging
from rwmutex import RWLock
import pickle
import sys

import grpc
from proto import helloworld_pb2
from proto import helloworld_pb2_grpc
from datetime import datetime
import os
import psutil


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
        history = request.history

        print("[ " + seqNum + " ]" + "Received request from " + clientId + " with requestId: " + requestId)

        serverTime = datetime.now()

        sequenceNumberTuple = "(SequenceNumber, " + seqNum + ")"  
        requestArrivalTimeTuple = "(RequestArrivalTime, " + serverTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ")"
        clientIdTuple = "(ClientId, " + clientId + ")"
        timeAtClientTuple = "(TimeAtClient, " + timeAtClient + ")"
        pastWindowDataTuple = "(PastWindowData, " + str(pickle.loads(pastWindowData.encode())) + ")"
        historyTuple = "(History, " + str(pickle.loads(history.encode())) + ")"
        timeSinceClientsLastReqTuple = "(TimeSinceThisClientsLastRequest, "
        with self.lock1.read:
            if clientId in self.clientPastRequests:
                timeSinceClientsLastReqTuple += str(serverTime - self.clientPastRequests[clientId]) + ")"
            else:
                timeSinceClientsLastReqTuple += "None)"

        lastReqByTuple = "(LastRequestBy, "
        timeSinceLastReqTuple = "(TimeSinceLastRequest, "
        with self.lock2.read:
            if self.lastRequestArrivalTime:
                lastReqByTuple += self.lastRequestArrivalTime["clientId"] + ")"
                timeSinceLastReqTuple += str(serverTime - self.lastRequestArrivalTime["time"]) + ")"
            else:
                lastReqByTuple += "None)"
                timeSinceLastReqTuple += "None)"
                
        requestIdTuple = "(RequestId, " + requestId + ")"
        
        ''' System conditions '''
        cpu_percentage = psutil.cpu_percent()
        memory_conditions = dict(psutil.virtual_memory()._asdict())

        cpu_util_tuple = "(CPUUtil, " + str(cpu_percentage) + ")"
        memory_conditions_tuple = "(Memory, " + str(memory_conditions) + ")"


        logging.info("[RequestLog]"
            + requestIdTuple + "; " 
            + sequenceNumberTuple + "; " 
            + requestArrivalTimeTuple + "; "
            + clientIdTuple + "; "
            + timeAtClientTuple + "; "
            + pastWindowDataTuple + "; "
            + timeSinceClientsLastReqTuple + "; "
            + lastReqByTuple + "; "
            + timeSinceLastReqTuple + "; "
            + historyTuple + "; "
            + cpu_util_tuple + "; "
            + memory_conditions_tuple + "; "
        )
        '''
        logging.info(
            "[ " + requestId + " ]" + 
            "[ RequestArrivalTime ]" + 
            serverTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
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
        
        return helloworld_pb2.HelloReply(requestId = requestId, serverTime = serverTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], requestTime = timeAtClient)

    def SayHelloReplica(self, request, context):
        requestId = request.requestId
        history = request.history
        pastWindowData = request.pastWindowData
        timeAtSender = request.timeAtSender
        senderId = request.replicaId

        serverTime = datetime.now()

        requestIdTuple = "(RequestId, " + requestId + ")"
        requestArrivalTimeTuple = "(RequestArrivalTime, " + serverTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ")"
        historyTuple = "(History, " + str(pickle.loads(history.encode())) + ")"
        pastWindowDataTuple = "(PastWindowData, " + str(pickle.loads(pastWindowData.encode())) + ")"
        timeAtSenderTuple = "(TimeAtSenderTuple, " + timeAtSender + ")"
        senderIdTuple = "(SenderId, " + str(senderId) + ")"

        ''' System conditions '''
        cpu_percentage = psutil.cpu_percent()
        memory_conditions = dict(psutil.virtual_memory()._asdict())

        cpu_util_tuple = "(CPUUtil, " + str(cpu_percentage) + ")"
        memory_conditions_tuple = "(Memory, " + str(memory_conditions) + ")"

        logging.info("[ReplicaRequestLog]"
            + str(requestIdTuple) + "; "
            + str(senderIdTuple) + "; " 
            + str(requestArrivalTimeTuple) + "; "
            + str(pastWindowDataTuple) + "; "
            + str(historyTuple) + "; "
            + str(timeAtSenderTuple) + "; "
            + str(cpu_util_tuple) + "; "
            + str(memory_conditions_tuple) + "; "
        )

        return helloworld_pb2.HelloReplyReplica(
            requestId = requestId,
            timeAtReceiver = serverTime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        )

def serve():
    port = '50060'
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

    logfilename = sys.argv[1]
    logfilepath = "logs/" + logfilename
    
    logging.basicConfig(
        filename=logfilepath,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    serve()