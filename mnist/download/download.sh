#!/bin/sh

mkdir -d mnist/download
cd mnist/download
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output train-images.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output train-labels.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --output test-images.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --output test-labels.gz
gzip -d *.gz
