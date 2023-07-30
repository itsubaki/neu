package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleEmbeddingDot() {
	// p150
	l := &layer.EmbeddingDot{
		Embedding: layer.Embedding{
			W: matrix.New(
				[]float64{0, 1, 2},
				[]float64{3, 4, 5},
				[]float64{6, 7, 8},
				[]float64{9, 10, 11},
				[]float64{12, 13, 14},
				[]float64{15, 16, 17},
				[]float64{18, 17, 18},
			),
		},
	}
	fmt.Println(l)

	// forward
	h := matrix.New(
		[]float64{0, 1, 2},
		[]float64{3, 4, 5},
		[]float64{6, 7, 8},
	)

	idx := matrix.New(
		[]float64{0, 3, 1},
	)

	out := l.Forward(h, idx)
	for _, r := range out {
		fmt.Println(r)
	}
	fmt.Println()

	// backward
	dout := matrix.New([]float64{1})
	dh, _ := l.Backward(dout)
	for _, r := range dh {
		fmt.Println(r)
	}

	// Output:
	// *layer.EmbeddingDot: W(7, 3): 21
	// [5 122 86]
	//
	// [0 1 2]
	// [9 10 11]
	// [3 4 5]
}

func ExampleEmbeddingDot_Params() {
	l := &layer.EmbeddingDot{}
	l.SetParams(make([]matrix.Matrix, 2)...)

	fmt.Println(l)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// *layer.EmbeddingDot: W(0, 0): 0
	// [[]]
	// [[]]
}
