package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/itsubaki/neu"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/optimizer"
)

func main() {
	var dir string
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.Parse()

	// data
	train, test, err := mnist.Load(dir)
	if err != nil {
		fmt.Printf("load mnist: %v", err)
		os.Exit(1)
	}

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// hyper-parameter
	batchSize := 100
	iter := 10000

	// init
	rand.Seed(time.Now().Unix())
	n := neu.New(&neu.Config{
		InputSize:     784, // 24 * 24
		HiddenSize:    50,
		OutputSize:    10, // 0 ~ 9
		BatchSize:     batchSize,
		WeightInitStd: 0.01,
		Optimizer:     &optimizer.SGD{LearningRate: 0.1},
	})

	// learning
	for i := 0; i < iter; i++ {
		mask := neu.Random(len(train.Image), batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		grads := n.Gradient(xbatch, tbatch)
		n.Optimize(grads)

		if i%(iter/batchSize) == 0 {
			loss := n.Loss(xbatch, tbatch)
			acc := n.Accuracy(xbatch, tbatch)
			mask := neu.Random(len(test.Image), batchSize)
			tacc := n.Accuracy(matrix.Batch(xt, mask), matrix.Batch(tt, mask))

			fmt.Printf("%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", i, loss, acc, tacc)
		}
	}
}
