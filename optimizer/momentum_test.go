package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

func ExampleMomentum() {
	params, grads := make(map[string]matrix.Matrix), make(map[string]matrix.Matrix)

	params["W"] = matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	grads["W"] = matrix.New([]float64{2, 4, 6}, []float64{8, 10, 12})

	for _, lr := range []float64{0.0, 0.5, 1.0} {
		opt := optimizer.Momentum{LearningRate: lr, Momentum: 1.0}
		fmt.Println(opt.Update(params, grads)["W"])
	}

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]
}
