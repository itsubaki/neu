package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/neu/dataset/mnist"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

// go run cmd/mnist/main.go
func main() {
	// flags
	var dir string
	var epochs, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 10, "")
	flag.IntVar(&batchSize, "batchsize", 100, "")
	flag.Parse()

	// load mnist dataset
	train, test := mnist.Must(mnist.Load(dir))

	// train data
	x := matrix.New(mnist.Normalize(train.Image)...) // 60000 * 784
	t := matrix.New(mnist.OneHot(train.Label)...)    // 60000 * 10

	// test data
	xt := matrix.New(mnist.Normalize(test.Image)...) // 10000 * 784
	tt := matrix.New(mnist.OneHot(test.Label)...)    // 10000 * 10

	// model
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         mnist.Width * mnist.Height, // 24 * 24 = 784
		OutputSize:        mnist.Labels,               // 0 ~ 9
		HiddenSize:        []int{50, 50, 50},
		WeightInit:        weight.He,
		BatchNormMomentum: 0.9,
	})

	// print layer
	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// training
	tr := trainer.New(m, &optimizer.AdaGrad{
		LearningRate: 0.01,
		Hooks: []optimizer.Hook{
			hook.WeightDecay(1e-6),
		},
	})

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      x,
		TrainLabel: t,
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, loss float64, m trainer.Model) {
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

			// accuracy
			acc := trainer.Accuracy(m.Predict(xbatch), tbatch)
			y := m.Predict(xtbatch)
			acct := trainer.Accuracy(y, ttbatch)

			// print
			fmt.Printf("%3d,%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", epoch, j, loss, acc, acct)
			fmt.Printf("predict: %v\n", y.Argmax()[:20])
			fmt.Printf("label  : %v\n", ttbatch.Argmax()[:20])
			fmt.Println()
		},
	})

	fmt.Printf("elapsed=%v\n", time.Since(now))
}
