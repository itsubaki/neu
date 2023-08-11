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

	// backward
	da := matrix.Matrix{
		// (T, N) (2, 2)
		{2, 4},
		{2, 4},
	}
	dhs, dh := at.Backward(da)
	fmt.Println(dhs)
	fmt.Println(dh)

	// Output:
	// *layer.AttentionWeight
	// [[1.12535162055095e-07 0.9999998874648379] [0.8807970779778823 0.11920292202211755]]
	// [[[-2.2507029878186457e-07 -4.5014059756372915e-07 -6.752108963455937e-07] [4.5014059768533343e-07 4.5014059768533343e-07 4.5014059768533343e-07]] [[-0.20998717080701268 -0.41997434161402536 -0.629961512421038] [0.41997434161402614 0.41997434161402614 0.41997434161402614]]]
	// [[-0.8399489082983495 -1.0499363041756609 -1.2599237000529724] [0.8399495835092476 1.0499369793865596 1.2599243752638714]]
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

func ExampleExpand() {
	ds := matrix.Matrix{ // (T, N) (2, 2)
		{2, 4}, // (1, 2)
		{2, 4}, // (1, 2)
	}

	T, N, H := 2, 2, 3
	for _, m := range layer.Expand(ds, T, N, H) {
		for _, r := range m {
			fmt.Println(r)
		}
	}

	// Output:
	// [2 2 2]
	// [4 4 4]
	// [2 2 2]
	// [4 4 4]
}
