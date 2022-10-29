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

// go run cmd/mnist/main.go
func main() {
	var dir string
	var epochs, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 1, "")
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
	tr := &trainer.Trainer{
		Model: m,
		Optimizer: &optimizer.AdaGrad{
			LearningRate: 0.01,
			Hooks:        []optimizer.Hook{weight.Decay(1e-6)},
		},
	}

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      x,
		TrainLabel: t,
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, m trainer.Model) {
			if j%(train.N/batchSize/10) != 0 {
				return
			}

			// batch
			mask := trainer.Random(train.N, batchSize)
			xbatch := matrix.Batch(x, mask)
			tbatch := matrix.Batch(t, mask)

			maskt := trainer.Random(test.N, batchSize)
			xtbatch := matrix.Batch(xt, maskt)
			ttbatch := matrix.Batch(tt, maskt)

			// loss, accuracy
			loss := m.Forward(xbatch, tbatch)
			acc := trainer.Accuracy(m.Predict(xbatch), tbatch)
			yt := m.Predict(xtbatch)
			tacc := trainer.Accuracy(yt, ttbatch)

			// print
			fmt.Printf("%4d,%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", epoch, j, loss, acc, tacc)
			fmt.Printf("predict: %v\n", yt.Argmax()[:20])
			fmt.Printf("label  : %v\n", ttbatch.Argmax()[:20])
			fmt.Println()
		},
	})

	fmt.Printf("elapsed=%v\n", time.Since(now))
}
