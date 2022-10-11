package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleReLU() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})
	relu := layer.ReLU{}

	fmt.Println(relu.Forward(x, nil))
	fmt.Println(relu.Backward(x))

	// Output:
	// [[1 -0] [-0 3]]
	// [[1 -0] [-0 3]] []
}
