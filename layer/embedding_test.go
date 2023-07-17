package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleEmbedding() {
	// weight
	W := matrix.New(
		[]float64{0.1, 0.3, 0.5},
		[]float64{0.2, 0.4, 0.6},
		[]float64{0.7, 0.8, 0.9},
	)

	// data
	x := matrix.New([]float64{0, 2})

	// layer
	embed := layer.Embedding{W: W}

	fmt.Println(embed.W)
	fmt.Println(embed.DW)
	fmt.Println(embed.Forward(x, nil))
	fmt.Println()

	dout := matrix.New(
		[]float64{0.1, 0.3, 0.5},
		[]float64{0.2, 0.4, 0.6},
		[]float64{0.7, 0.8, 0.9},
	)
	embed.Backward(dout)
	fmt.Println(embed.DW)

	// Output:
	// [[0.1 0.3 0.5] [0.2 0.4 0.6] [0.7 0.8 0.9]]
	// []
	// [[0.1 0.3 0.5] [0.7 0.8 0.9]]
	//
	// [[0.1 0.3 0.5] [0 0 0] [0.2 0.4 0.6]]
}
