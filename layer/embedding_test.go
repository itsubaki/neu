package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleEmbedding() {
	embed := &layer.Embedding{W: matrix.New(
		[]float64{0.0, 0.1, 0.2},
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.2, 0.3, 0.4},
		[]float64{0.3, 0.4, 0.5},
		[]float64{0.4, 0.5, 0.6},
		[]float64{0.5, 0.6, 0.7},
	)}
	fmt.Println(embed)

	// forward
	// p138
	x := matrix.New(
		[]float64{0},
		[]float64{2},
		[]float64{0},
		[]float64{4},
	)
	fmt.Println(embed.Forward(x, nil))

	// backward
	dh := matrix.New(
		[]float64{9.0, 9.1, 9.2},
		[]float64{9.1, 9.2, 9.3},
		[]float64{9.2, 9.3, 9.4},
		[]float64{9.3, 9.4, 9.5},
	)

	embed.Backward(dh)
	fmt.Println(embed.DW)

	// Output:
	// *layer.Embedding: W(6, 3): 18
	// [[0 0.1 0.2] [0.2 0.3 0.4] [0 0.1 0.2] [0.4 0.5 0.6]]
	// [[18.2 18.4 18.6] [0 0 0] [9.1 9.2 9.3] [0 0 0] [9.3 9.4 9.5] [0 0 0]]
}

func ExampleEmbedding_Params() {
	l := &layer.Embedding{}

	l.SetParams(make([]matrix.Matrix, 2)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// [[]]
	// [[]]
}
