# ByePaxos

Research on Learned consensus to achieve total ordering in a Replicated State Machine that eventually eliminates the need for Paxos.

## server.py
This script sets up grpc servers which mimics replicas for read/write operations. The server recieves requests and logs the various properties of the request which is then used to train the ML models.

## client.py
This script mimics clients in the system that make concurrent requests to the replica nodes.

## replica_client.py
This script is used for replicas to ping each other to gauge the network conditions around the replicas. This is there to gather features to see if and how much of the network conditions around the replica affects the total ordering.

## new_model.py
This script is used to create the models and train them. The file contains the dataset class, classifier model class, and the train loop. The shifting algorithm is in the dataset class. To the a couple parameters such as BUFFER_SIZE and increment can be changed.

## model.py
This is deprecate and shouldn't be used.

## parser.py
The parser script takes in a couple command line arguments and then creates the json output representation of the data

## label_preprocess.py
This script takes the output of the parser.py script and then make the json data numeric

## data_postprocess.py
This script takes the output of label_preprocess.py and then create actuall csv normalized numerical representation of all the features. The resulting data is used as is when training the model

## pull_log.sh
This bash file is used to help us streamline the process of pulling data from the cloudlab

## data_pipeline.sh
The data pipeline basically runs the parser.py through the raw log data pulled from cloudlab servers.

## plot.py
This is just a utility file that create accuracy plots.

### Authors:
Shahid Ikram
Ricky Hsu
