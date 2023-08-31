package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMeanSquaredError() {
	l := &layer.MeanSquaredError{}
	fmt.Println(l)

	// forward
	x := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)
	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	loss := l.Forward(x, t)
	fmt.Println(loss)

	// backward
	dout := matrix.New([]float64{1})
	gx0, gx1 := l.Backward(dout)
	for _, r := range gx0 {
		fmt.Println(r)
	}

	for _, r := range gx1 {
		fmt.Println(r)
	}

	// Output:
	// *layer.MeanSquaredError
	// [[0.019500000000000007 0.11949999999999998]]
	// [0.1 0.05 -0.4 0 0.05 0.1 0 0.1 0 0]
	// [0.1 0.05 -0.9 0 0.05 0.1 0 0.6 0 0]
	// [-0.1 -0.05 0.4 -0 -0.05 -0.1 -0 -0.1 -0 -0]
	// [-0.1 -0.05 0.9 -0 -0.05 -0.1 -0 -0.6 -0 -0]
}

func ExampleMeanSquaredError_Params() {
	l := &layer.MeanSquaredError{}

	l.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// []
	// []
}
