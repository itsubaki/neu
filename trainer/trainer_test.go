package trainer_test

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
)

var _ trainer.Model = (*Test)(nil)

type Test struct{}

func (m *Test) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix { return matrix.New() }
func (m *Test) Forward(x, t matrix.Matrix) matrix.Matrix                  { return matrix.New() }
func (m *Test) Backward(x, t matrix.Matrix) matrix.Matrix                 { return matrix.New() }
func (m *Test) Layers() []model.Layer                                     { return make([]model.Layer, 0) }
func (m *Test) Params() [][]matrix.Matrix                                 { return [][]matrix.Matrix{} }
func (m *Test) Grads() [][]matrix.Matrix                                  { return [][]matrix.Matrix{} }

func ExampleTrainer_Fit() {
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	tr := &trainer.Trainer{
		Model:     &Test{},
		Optimizer: &optimizer.SGD{},
	}

	tr.Fit(&trainer.Input{
		Train:      x,
		TrainLabel: t,
		Epochs:     3,
		BatchSize:  1,
		Verbose: func(epoch, j int, m trainer.Model) {
			fmt.Printf("%v,%v: %T\n", epoch, j, m)
		},
	})

	// Output:
	// 0,0: *trainer_test.Test
	// 0,1: *trainer_test.Test
	// 0,2: *trainer_test.Test
	// 1,0: *trainer_test.Test
	// 1,1: *trainer_test.Test
	// 1,2: *trainer_test.Test
	// 2,0: *trainer_test.Test
	// 2,1: *trainer_test.Test
	// 2,2: *trainer_test.Test

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

func ExampleShuffle() {
	x := matrix.New([]float64{0, 1}, []float64{0, 2}, []float64{0, 3}, []float64{0, 4})
	t := matrix.New([]float64{1, 0}, []float64{2, 0}, []float64{3, 0}, []float64{4, 0})

	rand.Seed(1234) // for test
	fmt.Println(trainer.Shuffle(x, t))
	fmt.Println(trainer.Shuffle(x, t))
	fmt.Println(trainer.Shuffle(x, t))
	fmt.Println(x, t)

	// Output:
	// [[0 4] [0 2] [0 3] [0 1]] [[4 0] [2 0] [3 0] [1 0]]
	// [[0 2] [0 3] [0 1] [0 4]] [[2 0] [3 0] [1 0] [4 0]]
	// [[0 2] [0 4] [0 3] [0 1]] [[2 0] [4 0] [3 0] [1 0]]
	// [[0 1] [0 2] [0 3] [0 4]] [[1 0] [2 0] [3 0] [4 0]]

}
