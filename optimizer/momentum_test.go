package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleMomentum() {
	m := &TestModel{
		params: append(make([][]matrix.Matrix, 0), []matrix.Matrix{
			matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6}),
		}),
		grads: append(make([][]matrix.Matrix, 0), []matrix.Matrix{
			matrix.New([]float64{2, 4, 6}, []float64{8, 10, 12}),
		}),
	}

	for _, lr := range []float64{0.0, 0.5, 1.0} {
		opt := optimizer.Momentum{LearningRate: lr, Momentum: 1.0, Hooks: []optimizer.Hook{weight.Decay(0.0)}}
		fmt.Println(opt.Update(m)[0][0])
	}
	fmt.Println()

	opt := optimizer.Momentum{LearningRate: 0.5, Momentum: 1.0}
	fmt.Println(opt.Update(m)[0][0])
	fmt.Println(opt.Update(m)[0][0])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]
	//
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]

}
