# ByePaxos

Research on Learned consensus to achieve total ordering in a replicated state machine without Paxos.

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
