#!/bin/bash
python parser.py -t client -f client-apt031.log -f client-pc446.log -f client-pc497.log
python parser.py -t server -f  server-amd159.log -f server-c220g5-110531.log -f server-clnode241.log -f server-pc421.log