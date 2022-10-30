package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleReLU() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})
	relu := &layer.ReLU{}

	fmt.Println(relu)
	fmt.Println(relu.Forward(x, nil))
	fmt.Println(relu.Backward(x))

	// Output:
	// *layer.ReLU
	// [[1 -0] [-0 3]]
	// [[1 -0] [-0 3]] []
}

func ExampleReLU_Params() {
	relu := &layer.ReLU{}

	relu.SetParams(make([]matrix.Matrix, 0))
	relu.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(relu.Params())
	fmt.Println(relu.Grads())

	// Output:
	// []
	// []
}
