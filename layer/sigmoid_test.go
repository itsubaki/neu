package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleSigmoid() {
	x := matrix.New([]float64{0.0})
	sigmoid := &layer.Sigmoid{}

	fmt.Println(sigmoid.Forward(x, nil))
	fmt.Println(sigmoid.Backward(x))

	// Output:
	// [[0.5]]
	// [[0]] []
}

func ExampleSigmoid_Params() {
	sigmoid := &layer.Sigmoid{}

	sigmoid.SetParams(make([]matrix.Matrix, 0))
	sigmoid.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(sigmoid)
	fmt.Println(sigmoid.Params())
	fmt.Println(sigmoid.Grads())

	// Output:
	// *layer.Sigmoid
	// []
	// []
}
