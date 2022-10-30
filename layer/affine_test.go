package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAffine() {
	// weight
	W := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B := matrix.New([]float64{0.1, 0.2, 0.3})

	// data
	x := matrix.New([]float64{1.0, 0.5})

	// layer
	affine := &layer.Affine{
		W: W,
		B: B,
	}

	fmt.Println(affine)
	fmt.Println(affine.Forward(x, nil))
	fmt.Println(affine.Backward(x))

	// Output:
	// *layer.Affine: W(2, 3), B(1, 3): 9
	// [[0.30000000000000004 0.7 1.1]]
	// [[0.25 0.4]] []

}

func ExampleAffine_Params() {
	affine := &layer.Affine{}

	affine.SetParams(make([]matrix.Matrix, 2))
	affine.SetGrads(make([]matrix.Matrix, 2))

	fmt.Println(affine.Params())
	fmt.Println(affine.Grads())

	// Output:
	// [[] []]
	// [[] []]
}
