package trainer_test

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/trainer"
)

var _ trainer.Model = (*TestModel)(nil)

type TestModel struct{}

func (m *TestModel) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix { return matrix.New() }
func (m *TestModel) Loss(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix { return matrix.New() }
func (m *TestModel) Gradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	return make(map[string]matrix.Matrix)
}
func (m *TestModel) Optimize(grads map[string]matrix.Matrix) {}

func ExampleTrain() {
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	trainer.Train(&trainer.Input{
		Model:      &TestModel{},
		Train:      x,
		TrainLabel: t,
		Test:       x,
		TestLabel:  t,
		Iter:       10,
		BatchSize:  1,
	})

	// Output:
}

func ExampleTrain_verbose() {
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	trainer.Train(&trainer.Input{
		Model:      &TestModel{},
		Train:      x,
		TrainLabel: t,
		Test:       x,
		TestLabel:  t,
		Iter:       10,
		BatchSize:  1,
		Verbose: func(i int, m trainer.Model, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix) {
			fmt.Printf("%2v: %T\n", i, m)
		},
	})

	// Output:
	// 	0: *trainer_test.TestModel
	// 10: *trainer_test.TestModel
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