package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMatMul() {
	W := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)
	matmul := &layer.MatMul{W: W}
	fmt.Println(matmul)
	fmt.Println()

	// forward
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matmul.Forward(A, nil) {
		fmt.Println(r)
	}
	fmt.Println()

	// backward
	dx, _ := matmul.Backward(matrix.New([]float64{1, 0}, []float64{0, 1}))
	for _, r := range dx {
		fmt.Println(r)
	}
	fmt.Println()

	// Output:
	// *layer.MatMul: W(2, 2): 4
	//
	// [19 22]
	// [43 50]
	//
	// [5 7]
	// [6 8]
	//

}

func ExampleMatMul_Params() {
	matmul := &layer.MatMul{}
	matmul.SetParams(make([]matrix.Matrix, 1)...)

	fmt.Println(matmul.Params())
	fmt.Println(matmul.Grads())

	// Output:
	// [[]]
	// [[]]
}
