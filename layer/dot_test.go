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

	dot := &layer.Dot{W: B}
	fmt.Println(dot)
	fmt.Println()

	for _, r := range dot.Forward(A, nil) {
		fmt.Println(r)
	}
	fmt.Println()

	dx, _ := dot.Backward(matrix.New([]float64{1, 0}, []float64{0, 1}))
	for _, r := range dx {
		fmt.Println(r)
	}
	fmt.Println()

	// Output:
	// *layer.Dot: W(2, 2): 4
	//
	// [19 22]
	// [43 50]
	//
	// [5 7]
	// [6 8]
	//

}

func ExampleDot_Params() {
	dot := &layer.Dot{}
	dot.SetParams(make([]matrix.Matrix, 1)...)

	fmt.Println(dot.Params())
	fmt.Println(dot.Grads())

	// Output:
	// [[]]
	// [[]]
}
