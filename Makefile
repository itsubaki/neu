SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /vendor/ | grep -v /build/ | grep -v /cmd/) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

lint:
	golangci-lint run

update:
	go get -u
	go mod tidy

mnist:
	go run cmd/mnist/main.go --dir ./testdata

mnist1:
	go run cmd/mnist/main.go --dir ./testdata --epochs 1

cbow:
	go run cmd/cbow/main.go

cbow_negative_sampling:
	go run cmd/cbow_negative_sampling/main.go

rnnlm:
	go run cmd/rnnlm/main.go --dir ./testdata --corpus-size 1000

rnnlm_lstm:
	go run cmd/rnnlm_lstm/main.go --dir ./testdata --corpus-size 1000

rnnlm_gru:
	go run cmd/rnnlm_gru/main.go --dir ./testdata --corpus-size 1000

rnnlm_gen:
	go run cmd/rnnlm_gen/main.go --dir ./testdata --epochs 0

seq2seq:
	go run cmd/seq2seq/main.go --dir ./testdata --data-size 10

seq2seq_attention:
	go run cmd/seq2seq_attention/main.go --dir ./testdata --data-size 10

datasetdl: mnistdl ptbdl additiondl datedl

mnistdl:
	curl -fs -o testdata/train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	curl -fs -o testdata/train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
	curl -fs -o testdata/t10k-images-idx3-ubyte.gz  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
	curl -fs -o testdata/t10k-labels-idx1-ubyte.gz  https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# https://github.com/zalandoresearch/fashion-mnist
fashiondl:
	curl -fs -o testdata/train-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
	curl -fs -o testdata/train-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
	curl -fs -o testdata/t10k-images-idx3-ubyte.gz  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
	curl -fs -o testdata/t10k-labels-idx1-ubyte.gz  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

ptbdl:
	curl -fs -o testdata/ptb.train.txt https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
	curl -fs -o testdata/ptb.test.txt  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
	curl -fs -o testdata/ptb.valid.txt https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt

additiondl:
	curl -fs -o testdata/addition.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/master/dataset/addition.txt

datedl:
	curl -fs -o testdata/date.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/master/dataset/date.txt
