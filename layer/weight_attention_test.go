package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAttentionWeight() {
	at := &layer.AttentionWeight{
		Softmax: &layer.Softmax{},
	}
	fmt.Println(at)

	// forward
	hs := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{4, 5, 6},
			{4, 5, 6},
		},
	}
	h := matrix.New(
		[]float64{1, 2, 3},
		[]float64{2, 2, 2},
	)
	fmt.Println(at.Forward(hs, h))

	// Output:
	// *layer.AttentionWeight
	// [[1.12535162055095e-07 0.9999998874648379] [0.8807970779778823 0.11920292202211755]]
}

func ExampleAttentionWeight_Params() {
	at := &layer.AttentionWeight{}
	at.SetParams(matrix.New())

	fmt.Println(at.Params())
	fmt.Println(at.Grads())

	// Output:
	// []
	// []
}
