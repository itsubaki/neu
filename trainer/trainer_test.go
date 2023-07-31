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

var _ trainer.Model = (*TestModel)(nil)

type TestModel struct{}

func (m *TestModel) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix { return matrix.New() }
func (m *TestModel) Forward(x, t matrix.Matrix) matrix.Matrix                  { return matrix.New([]float64{1}) }
func (m *TestModel) Backward() matrix.Matrix                                   { return matrix.New() }
func (m *TestModel) Layers() []model.Layer                                     { return make([]model.Layer, 0) }
func (m *TestModel) Params() [][]matrix.Matrix                                 { return [][]matrix.Matrix{} }
func (m *TestModel) Grads() [][]matrix.Matrix                                  { return [][]matrix.Matrix{} }

func (m *TestModel) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}

func ExampleTrainer_Fit() {
	tr := trainer.New(&TestModel{}, &optimizer.SGD{
		LearningRate: 0.1,
	})

	tr.Fit(&trainer.Input{
		Train:      matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1}),
		TrainLabel: matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1}),
		Epochs:     3,
		BatchSize:  1,
		Verbose: func(epoch, j int, loss float64, m trainer.Model) {
			fmt.Printf("%v,%v: %T\n", epoch, j, m)
		},
	})

	// Output:
	// 0,0: *trainer_test.TestModel
	// 0,1: *trainer_test.TestModel
	// 0,2: *trainer_test.TestModel
	// 1,0: *trainer_test.TestModel
	// 1,1: *trainer_test.TestModel
	// 1,2: *trainer_test.TestModel
	// 2,0: *trainer_test.TestModel
	// 2,1: *trainer_test.TestModel
	// 2,2: *trainer_test.TestModel

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

	s := rand.NewSource(1)
	r1 := trainer.Random(len(x), 1, s)
	r2 := trainer.Random(len(x), 2, s)
	r3 := trainer.Random(len(x), 3, s)
	r4 := trainer.Random(len(x), 4, s)
	r5 := trainer.Random(1, 1)

	sort.Ints(r1)
	sort.Ints(r2)
	sort.Ints(r3)
	sort.Ints(r4)

	fmt.Println(r1)
	fmt.Println(r2)
	fmt.Println(r3)
	fmt.Println(r4)
	fmt.Println(r5)

	// Output:
	// [1]
	// [1 3]
	// [0 1 2]
	// [0 1 2 3]
	// [0]
}

func ExampleShuffle() {
	x := matrix.New([]float64{0, 1}, []float64{0, 2}, []float64{0, 3}, []float64{0, 4})
	t := matrix.New([]float64{1, 0}, []float64{2, 0}, []float64{3, 0}, []float64{4, 0})

	s := rand.NewSource(1234)
	fmt.Println(trainer.Shuffle(x, t, s))
	fmt.Println(trainer.Shuffle(x, t, s))
	fmt.Println(trainer.Shuffle(x, t, s))
	fmt.Println(x, t)

	fmt.Println(trainer.Shuffle(matrix.New([]float64{0}), matrix.New([]float64{1})))

	// Output:
	// [[0 4] [0 2] [0 3] [0 1]] [[4 0] [2 0] [3 0] [1 0]]
	// [[0 2] [0 3] [0 1] [0 4]] [[2 0] [3 0] [1 0] [4 0]]
	// [[0 2] [0 4] [0 3] [0 1]] [[2 0] [4 0] [3 0] [1 0]]
	// [[0 1] [0 2] [0 3] [0 4]] [[1 0] [2 0] [3 0] [4 0]]
	// [[0]] [[1]]

}
