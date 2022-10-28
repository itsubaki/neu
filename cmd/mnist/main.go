package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

// go run cmd/mnist/main.go --iter 12000
func main() {
	var dir string
	var iter, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&iter, "iter", 1200, "")
	flag.IntVar(&batchSize, "batchsize", 100, "")
	flag.Parse()

	// data
	train, test := mnist.Must(mnist.Load(dir))

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// init
	rand.Seed(time.Now().Unix())
	m := model.NewMLP(&model.MLPConfig{
		InputSize:  784, // 24 * 24
		HiddenSize: []int{50, 50, 50},
		OutputSize: 10, // 0 ~ 9
		WeightInit: weight.He,
	})

	// training
	tr := trainer.Trainer{
		Model:     m,
		Optimizer: &optimizer.AdaGrad{LearningRate: 0.01},
	}

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      x,
		TrainLabel: t,
		Test:       xt,
		TestLabel:  tt,
		Iter:       iter,
		BatchSize:  batchSize,
		Verbose: func(i int, m trainer.Model, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix) {
			loss := m.Loss(xbatch, tbatch)
			acc := trainer.Accuracy(m.Predict(xbatch), tbatch)
			yt := m.Predict(xtbatch)
			tacc := trainer.Accuracy(yt, ttbatch)

			fmt.Printf("%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", i, loss, acc, tacc)
			fmt.Printf("predict: %v\n", yt.Argmax()[:20])
			fmt.Printf("label  : %v\n", ttbatch.Argmax()[:20])
			fmt.Println()
		},
	})

	fmt.Printf("elapsed=%v\n", time.Since(now))
}
