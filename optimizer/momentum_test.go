package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
)

func ExampleMomentum() {
	m := &TestModel{
		params: [][]matrix.Matrix{{{{1, 2, 3}, {4, 5, 6}}}},
		grads:  [][]matrix.Matrix{{{{2, 4, 6}, {8, 10, 12}}}},
	}

	for _, lr := range []float64{0.0, 0.5, 1.0} {
		opt := optimizer.Momentum{LearningRate: lr, Momentum: 1.0}
		fmt.Println(opt.Update(m)[0][0])
	}
	fmt.Println()

	opt := optimizer.Momentum{LearningRate: 0.5, Momentum: 1.0, Hooks: []optimizer.Hook{hook.WeightDecay(0.0)}}
	fmt.Println(opt.Update(m)[0][0])
	fmt.Println(opt.Update(m)[0][0])
	fmt.Println(opt.Update(m)[0][0])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]
	//
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]
	// [[-2 -4 -6] [-8 -10 -12]]

}
