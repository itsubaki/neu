package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
)

func ExampleSGD() {
	m := &TestModel{
		params: append(make([][]matrix.Matrix, 0), []matrix.Matrix{{{1, 2, 3}, {4, 5, 6}}}),
		grads:  append(make([][]matrix.Matrix, 0), []matrix.Matrix{{{2, 4, 6}, {8, 10, 12}}}),
	}

	opt := optimizer.SGD{Hooks: []optimizer.Hook{hook.WeightDecay(0.0)}}
	opt.LearningRate = 0.0
	fmt.Println(opt.Update(m)[0][0])

	opt.LearningRate = 0.5
	fmt.Println(opt.Update(m)[0][0])

	opt.LearningRate = 1.0
	fmt.Println(opt.Update(m)[0][0])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]

}
