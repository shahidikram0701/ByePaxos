export CLOUDLAB_USER=rickyhh2
export PRIVATE_KEY_PATH=../../../final_temp
export OUTPUT_DIR=./data

mkdir -p data

declare -A pairs=( [pc446.emulab.net]=client-pc446 [pc497.emulab.net]=client-pc497 [apt031.apt.emulab.net]=client-apt031)
for i in "${!pairs[@]}"; do
  j=${pairs[$i]}
  scp -i $PRIVATE_KEY_PATH -P 22 $CLOUDLAB_USER@$i:/users/shahidi3/byepaxos/logs/client.log $OUTPUT_DIR/$j.log
done

declare -A pairs=([pc421.emulab.net]=server-pc421 [c220g5-110531.wisc.cloudlab.us]=server-c220g5-110531 [amd159.utah.cloudlab.us]=server-amd159 [clnode241.clemson.cloudlab.us]=server-clnode241)
for i in "${!pairs[@]}"; do
  j=${pairs[$i]}
  scp -i $PRIVATE_KEY_PATH -P 22 $CLOUDLAB_USER@$i:/users/shahidi3/byepaxos/logs/server.log $OUTPUT_DIR/$j.log
done