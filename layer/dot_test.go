package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleDot() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	dot := &layer.Dot{}
	for _, r := range dot.Forward(A, B) {
		fmt.Println(r)
	}
	fmt.Println()

	dx, dy := dot.Backward(matrix.New([]float64{1, 0}, []float64{0, 1}))
	for _, r := range dx {
		fmt.Println(r)
	}
	fmt.Println()

	for _, r := range dy {
		fmt.Println(r)
	}

	// Output:
	// [19 22]
	// [43 50]
	//
	// [5 7]
	// [6 8]
	//
	// [1 3]
	// [2 4]
}

func ExampleDot_Params() {
	dot := &layer.Dot{}

	dot.SetParams(make([]matrix.Matrix, 0))
	dot.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(dot)
	fmt.Println(dot.Params())
	fmt.Println(dot.Grads())

	// Output:
	// *layer.Dot
	// []
	// []
}
