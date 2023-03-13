#!/bin/sh

set -ue

unset TEMP

# NOTE: This requires GNU getopt.  On Mac OS X and FreeBSD, you have to install this
# separately; see below.
TEMP=$(getopt --options dcgm:t: --longoptions debug,force-cpu,force-gpu,model:,test-data: \
              --name $(basename $0) -- "$@")
# usage() { echo "Usage: $0 [-d] [-c|-g] [-m model_file | -t test_file] 1>&2; exit 1; }

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

# Note the quotes around '$TEMP': they are essential!
eval set -- "$TEMP"

USE_GPU=auto
MODEL_FILE="./models/java-large-model/model_iter52.release"
TEST_FILE="./data/java-large/java-large.test.c2s"
DEBUG_OPTS=""
 
while true; do
  case "$1" in
    -d | --debug ) DEBUG_OPTS="-m pdb"; shift ;;
    -c | --force-cpu ) USE_GPU=false; shift ;;
    -g | --force-gpu ) USE_GPU=true; shift ;;
    -m | --model ) MODEL_FILE="$2"; shift 2 ;;
    -t | --test-data ) TEST_FILE="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done


if [ x"$USE_GPU" = x"auto" ]; then
  if which nvidia-smi > /dev/null && nvidia-smi > /dev/null; then
    USE_GPU="true"
  else
    USE_GPU="false"
  fi
fi

GPU_FLAGS=""
if [ x"$USE_GPU" = x"true" ]; then
  GPU_FLAGS="--gpus=all --group-add video"
fi

cd "$(dirname $0)"
docker run -u $(id -u):$(id -g) $GPU_FLAGS -it -v $(pwd):/code2seq -w /code2seq --rm code2seq:1 \
 env TF_FORCE_GPU_ALLOW_GROWTH=true \
 python3 $DEBUG_OPTS code2seq.py --load "$MODEL_FILE" --predict_c2s $TEST_FILE
exit
 python3 $DEBUG_OPTS code2seq.py --load "$MODEL_FILE" --batch_java_src java-src/paths.txt
 python3 $DEBUG_OPTS code2seq.py --load "$MODEL_FILE" --test "$TEST_FILE"
 python3 $DEBUG_OPTS code2seq.py --load "$MODEL_FILE" --predict
