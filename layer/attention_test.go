package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAttention() {
	at := &layer.Attention{
		AttentionWeight: &layer.AttentionWeight{
			Softmax: &layer.Softmax{},
		},
		WeightSum: &layer.WeightSum{},
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
		// (N, H)
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
	// *layer.Attention
	// [[3.5231884244466913 4.403985614959736 5.28478280547278] [4.476811237947822 5.596014047434778 6.7152168569217325]]
	// [[[-1.8005623649265908e-06 -3.6011247298531815e-06] [2.000003826195054 4.00000360112473]] [[1.7615941559557682 3.5231883119115364] [0.23840584404423598 0.4768116880884711]]]
	// [[-2.025632674825926e-06 -4.051265360309993e-06 -6.07689804579406e-06] [8.102530758336002e-06 1.0128163447920002e-05 1.2153796137504003e-05]]
}

func ExampleAttention_Params() {
	at := &layer.Attention{}
	at.SetParams(matrix.New())

	fmt.Println(at.Params())
	fmt.Println(at.Grads())

	// Output:
	// []
	// []
}
