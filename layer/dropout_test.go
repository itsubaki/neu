package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleDropout() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})

	drop := layer.Dropout{Ratio: 0.5, TrainFlag: true}
	fmt.Println(drop.Forward(x, nil))
	fmt.Println(drop.Backward(x))

	drop2 := layer.Dropout{Ratio: 0.5, TrainFlag: false}
	fmt.Println(drop2.Forward(x, nil))

	// Output:
	// [[0 -0.5] [-2 0]]
	// [[0 -0.5] [-2 0]] []
	// [[0.5 -0.25] [-1 1.5]]
}
