package trainer_test

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/itsubaki/neu"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/winit"
)

func ExampleTrain() {
	// data
	train, test := mnist.Must(mnist.Load("../testdata"))

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// init
	rand.Seed(1)
	m := neu.New(&neu.Config{
		InputSize:         784, // 24 * 24
		HiddenSize:        []int{100},
		OutputSize:        10, // 0 ~ 9
		WeightDecayLambda: 1e-6,
		WeightInit:        winit.He,
		Optimizer:         &optimizer.AdaGrad{LearningRate: 0.01},
	})

	// training
	trainer.Train(&trainer.Input{
		Model:     m,
		X:         x,
		T:         t,
		XT:        xt,
		TT:        tt,
		Iter:      20,
		BatchSize: 20,
		Verbose:   false,
		Func:      func(i int, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix) {},
	})

	fmt.Printf("test_acc=%v\n", trainer.Accuracy(m.Predict(xt), tt))

	// Output:
	// test_acc=0.787
}

func ExampleTrain_verbose() {
	// data
	train, test := mnist.Must(mnist.Load("../testdata"))

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// init
	rand.Seed(1)
	m := neu.New(&neu.Config{
		InputSize:         784, // 24 * 24
		HiddenSize:        []int{100},
		OutputSize:        10, // 0 ~ 9
		WeightDecayLambda: 1e-6,
		WeightInit:        winit.He,
		Optimizer:         &optimizer.AdaGrad{LearningRate: 0.01},
	})

	// training
	trainer.Train(&trainer.Input{
		Model:     m,
		X:         x,
		T:         t,
		XT:        xt,
		TT:        tt,
		Iter:      20,
		BatchSize: 20,
		Verbose:   true,
		Func: func(i int, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix) {
			fmt.Printf("%d, ", i)
		},
	})

	fmt.Printf("test_acc=%v\n", trainer.Accuracy(m.Predict(xt), tt))

	// Output:
	// 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, test_acc=0.7976
}

func ExampleAccuracy() {
	// data
	y0 := matrix.New([]float64{0, 1}, []float64{1, 0}, []float64{1, 0})
	y1 := matrix.New([]float64{0, 1}, []float64{1, 0}, []float64{0, 1})
	y2 := matrix.New([]float64{0, 1}, []float64{0, 1}, []float64{0, 1})
	y3 := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	fmt.Println(trainer.Accuracy(y0, t))
	fmt.Println(trainer.Accuracy(y1, t))
	fmt.Println(trainer.Accuracy(y2, t))
	fmt.Println(trainer.Accuracy(y3, t))

	// Output:
	// 0
	// 0.3333333333333333
	// 0.6666666666666666
	// 1
}

func ExampleRandom() {
	x := matrix.New([]float64{0, 1}, []float64{0, 2}, []float64{0, 3}, []float64{0, 4})

	rand.Seed(1) // for test
	r1 := trainer.Random(len(x), 1)
	r2 := trainer.Random(len(x), 2)
	r3 := trainer.Random(len(x), 3)
	r4 := trainer.Random(len(x), 4)

	sort.Ints(r1)
	sort.Ints(r2)
	sort.Ints(r3)
	sort.Ints(r4)

	fmt.Println(r1)
	fmt.Println(r2)
	fmt.Println(r3)
	fmt.Println(r4)

	// Output:
	// [1]
	// [1 3]
	// [0 1 2]
	// [0 1 2 3]
}
