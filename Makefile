SHELL := /bin/bash

update:
	go get -u
	go mod tidy

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

mnist:
	go run cmd/mnist/main.go --dir ./testdata

mnist1:
	go run cmd/mnist/main.go --dir ./testdata --epochs 1

mnistdl:
	curl -s -o testdata/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	curl -s -o testdata/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	curl -s -o testdata/t10k-images-idx3-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -s -o testdata/t10k-labels-idx1-ubyte.gz  http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# https://github.com/zalandoresearch/fashion-mnist
fashiondl:
	curl -s -o testdata/train-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
	curl -s -o testdata/train-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
	curl -s -o testdata/t10k-images-idx3-ubyte.gz  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
	curl -s -o testdata/t10k-labels-idx1-ubyte.gz  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

cbow:
	go run cmd/cbow/main.go

rnnlm10:
	go run cmd/rnnlm/main.go --dir ./testdata --epochs 10

lstm10:
	go run cmd/rnnlm_lstm/main.go --dir ./testdata --epochs 10

rnnlmgen0:
	go run cmd/rnnlm_gen/main.go --dir ./testdata --epochs 0

rnnlmgen1:
	go run cmd/rnnlm_gen/main.go --dir ./testdata --epochs 1

seq2seq:
	go run cmd/seq2seq/main.go --dir ./testdata

ptbdl:
	curl -s -o testdata/ptb.train.txt https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
	curl -s -o testdata/ptb.test.txt  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
	curl -s -o testdata/ptb.valid.txt https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt

additiondl:
	curl -s -o testdata/addition.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/master/dataset/addition.txt
