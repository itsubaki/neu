package hook_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer/hook"
)

func ExampleWeightDecay() {
	params := append(make([][]matrix.Matrix, 0), []matrix.Matrix{
		matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6}),
	})
	grads := append(make([][]matrix.Matrix, 0), []matrix.Matrix{
		matrix.New([]float64{2, 4, 6}, []float64{8, 10, 12}),
	})

	fmt.Println(hook.WeightDecay(0.0)(params, grads))
	fmt.Println(hook.WeightDecay(0.5)(params, grads))
	fmt.Println(hook.WeightDecay(1.0)(params, grads))

	// Output:
	// [[[[2 4 6] [8 10 12]]]]
	// [[[[2.5 5 7.5] [10 12.5 15]]]]
	// [[[[3 6 9] [12 15 18]]]]
}
