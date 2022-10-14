package main

import (
	"flag"
	"fmt"
	"math/rand"
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
	train, test := mnist.Must(mnist.Load(dir))

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
		mask := neu.Random(train.N, batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		grads := n.Gradient(xbatch, tbatch)
		n.Optimize(grads)

		if i%(iter/batchSize) == 0 {
			loss := n.Loss(xbatch, tbatch)
			acc := n.Accuracy(xbatch, tbatch)
			mask := neu.Random(test.N, batchSize)
			tacc := n.Accuracy(matrix.Batch(xt, mask), matrix.Batch(tt, mask))
			fmt.Printf("%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", i, loss, acc, tacc)

			p := n.Predict(xbatch)
			fmt.Printf("predict: %v\n", p.Argmax()[:10])
			fmt.Printf("label  : %v\n", tbatch.Argmax()[:10])
		}
	}
}
