package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAffine() {
	affine := &layer.Affine{
		W: matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6}),
		B: matrix.New([]float64{0.1, 0.2, 0.3}),
	}
	fmt.Println(affine)

	// forward
	x := matrix.New([]float64{1.0, 0.5})
	fmt.Println(affine.Forward(x, nil))

	// backward
	fmt.Println(affine.Backward(x))

	// grads
	for _, g := range affine.Grads() {
		fmt.Println(g)
	}

	// Output:
	// *layer.Affine: W(2, 3), B(1, 3): 9
	// [[0.30000000000000004 0.7 1.1]]
	// [[0.25 0.4]] []
	// [[1 0.5] [0.5 0.25]]
	// [[1 0.5]]
}

func ExampleAffine_Params() {
	affine := &layer.Affine{}
	affine.SetParams(make([]matrix.Matrix, 2)...)

	fmt.Println(affine.Params())
	fmt.Println(affine.Grads())

	// Output:
	// [[] []]
	// [[] []]
}
