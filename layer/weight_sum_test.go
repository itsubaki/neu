package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleWeightSum() {
	ws := &layer.WeightSum{}

	// forward
	hs := []matrix.Matrix{
		// (T, N, H)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}
	a := []matrix.Matrix{
		{{2, 4}}, // (T, 1, N)
	}
	fmt.Println(ws.Forward(hs, a))

	// backward
	dc := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	dhs, da := ws.Backward(dc)
	fmt.Println(dhs)
	fmt.Println(da)

	// Output:
	// [[18 24 30]]
	// [[[2 4 6] [16 20 24]]]
	// [[14 77]]
}

func ExampleWeightSum_Params() {
	ws := &layer.WeightSum{}
	ws.SetParams()

	fmt.Println(ws.Params())
	fmt.Println(ws.Grads())

	// Output:
	// []
	// []
}

func ExampleWeightSum_state() {
	ws := &layer.WeightSum{}
	ws.SetState(matrix.New())
	ws.ResetState()

	// Output:
}
