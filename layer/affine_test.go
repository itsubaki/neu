package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAffine() {
	W := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B := matrix.New([]float64{0.1, 0.2, 0.3})
	X := matrix.New([]float64{1.0, 0.5})

	affine := layer.Affine{
		W: W,
		B: B,
	}

	A := affine.Forward(X, nil)
	fmt.Println(A)
	fmt.Println(affine.Backward(A))

	// Output:
	// [[0.30000000000000004 0.7 1.1]]
	// [[0.79 1]] []

}

func ExampleSumAxis1() {
	x := matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	fmt.Println(layer.SumAxis1(x))

	// Output:
	// [5 7 9]
}
