#!/bin/sh
set -ue

cd $(dirname "$0")
curl https://s3.amazonaws.com/code2seq/model/java-large/java-large-model.tar.gz | tar xzf -
