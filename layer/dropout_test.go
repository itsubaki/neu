package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleDropout() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})

	drop := layer.Dropout{Ratio: 0.5, TrainFlag: true}
	drop.Forward(x, nil)

	drop.TrainFlag = false
	fmt.Println(drop.Forward(x, nil))
	drop.Backward(x)

	// Output:
	// [[0.5 -0.25] [-1 1.5]]
}
