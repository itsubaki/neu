package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleWeightSum() {
	ws := &layer.WeightSum{}
	fmt.Println(ws)

	// forward
	hs := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}
	a := matrix.Matrix{
		// (T, N) (2, 2)
		{2, 4},
		{2, 4},
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
	// *layer.WeightSum
	// [[4 8 12] [32 40 48]]
	// [[[2 4 6] [16 20 24]] [[2 4 6] [16 20 24]]]
	// [[14 77] [14 77]]
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

func ExampleTimeSum() {
	hs := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	sum := layer.TimeSum(hs)
	for _, r := range sum {
		fmt.Println(r)
	}

	// Output:
	// [2 4 6]
	// [8 10 12]
}

func Example_transpose_broadcast() {
	a := matrix.New([]float64{1, 2, 3}) // (1 ,N)
	ar := a.T().Broadcast(3, 4)         // (1, N) -> (N, 1) -> (N, H)
	for _, r := range ar {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1 1]
	// [2 2 2 2]
	// [3 3 3 3]
}
