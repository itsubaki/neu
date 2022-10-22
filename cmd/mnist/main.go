package main

import (
	"flag"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
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

	// init
	rand.Seed(time.Now().Unix())
	m := model.New(&model.Config{
		InputSize:         784, // 24 * 24
		HiddenSize:        []int{50, 50, 50},
		OutputSize:        10, // 0 ~ 9
		WeightDecayLambda: 1e-6,
		WeightInit:        model.He,
		Optimizer:         &optimizer.AdaGrad{LearningRate: 0.01},
	})

	// training
	batchSize := 100
	iter := 1000

	for i := 0; i < iter+1; i++ {
		// batch
		mask := neu.Random(train.N, batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		// update
		grads := n.Gradient(xbatch, tbatch)
		n.Optimize(grads)

		if i%(iter/batchSize) == 0 {
			// train data
			loss := n.Loss(xbatch, tbatch)
			acc := neu.Accuracy(n.Predict(xbatch), tbatch)

			// test data
			mask := neu.Random(test.N, batchSize)
			xtbatch := matrix.Batch(xt, mask)
			ttbatch := matrix.Batch(tt, mask)
			yt := n.Predict(xtbatch)
			tacc := neu.Accuracy(yt, ttbatch)

			// print
			fmt.Printf("%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", i, loss, acc, tacc)
			fmt.Printf("predict: %v\n", yt.Argmax()[:20])
			fmt.Printf("label  : %v\n", ttbatch.Argmax()[:20])
			fmt.Println()
		}
	}

	fmt.Printf("test_acc=%v\n", neu.Accuracy(n.Predict(xt), tt))
}
