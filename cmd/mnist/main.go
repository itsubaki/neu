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
	m := model.NewMulti(&model.MultiConfig{
		InputSize:         784, // 24 * 24
		HiddenSize:        []int{50, 50, 50},
		OutputSize:        10, // 0 ~ 9
		WeightDecayLambda: 1e-6,
		WeightInit:        model.He,
		Optimizer:         &optimizer.AdaGrad{LearningRate: 0.01},
	})

	// training
	t_acc, acc := trainer.Train(m, x, t, xt, tt, 1000, 100, true)
	fmt.Printf("train_acc=%v, test_acc=%v\n", t_acc, acc)
}
