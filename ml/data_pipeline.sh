#!/bin/bash
export OUTPUT_DIR=./data_processed

mkdir -p $OUTPUT_DIR

declare clients=(apt031 pc446 pc497)
declare servers=(amd159 c220g5-110531 clnode241 pc421)

for i in ${clients[@]}; do
    for j in {1..9}; do
        python parser.py -t client -o $OUTPUT_DIR -f client-$i-$j.log
    done
done

for i in ${servers[@]}; do
    for j in {1..9}; do
        python parser.py -t server -o $OUTPUT_DIR -f server-$i-$j.log
        python parser.py -t server -o $OUTPUT_DIR -f interreplica-$i-$j.log
    done
done
# python parser.py -t client -f client-apt031.log -f client-pc446.log -f client-pc497.log
# python parser.py -t server -f  server-amd159.log -f server-c220g5-110531.log -f server-clnode241.log -f server-pc421.log