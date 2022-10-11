package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

func ExampleSGD() {
	params, grads := make(map[string]matrix.Matrix), make(map[string]matrix.Matrix)

	params["W"] = matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	grads["W"] = matrix.New([]float64{2, 4, 6}, []float64{8, 10, 12})

	opt := optimizer.SGD{}
	opt.LearningRate = 0.0
	fmt.Println(opt.Update(params, grads)["W"])

	opt.LearningRate = 0.5
	fmt.Println(opt.Update(params, grads)["W"])

	opt.LearningRate = 1.0
	fmt.Println(opt.Update(params, grads)["W"])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0 0 0] [0 0 0]]
	// [[-1 -2 -3] [-4 -5 -6]]

}