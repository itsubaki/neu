package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeGRU() {
	// (D, H) = (3, 1)
	gru := &layer.TimeGRU{
		Wx: matrix.New(
			// (D, 3H) = (3, 3)
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.1, 0.2, 0.3},
		),
		Wh: matrix.New(
			// (H, 3H) = (1, 3)
			[]float64{0.1, 0.2, 0.3},
		),
		B: matrix.New(
			// (1, 3H) = (1, 3)
			[]float64{0, 0, 0},
		),
	}
	fmt.Println(gru)

	// forward
	xs := []matrix.Matrix{
		// (T, N, D) = (1, 2, 3)
		{
			// (N, D) = (2, 3)
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	hs := gru.Forward(xs, nil)
	for i := range hs {
		fmt.Print(hs[i].Dim()) // (N, H) = (2, 1)
		fmt.Println(":", hs[i])
	}

	// backward
	dhs := []matrix.Matrix{
		{
			// (N, H) = (2, 1)
			{0.1},
			{0.3},
		},
	}
	dxs := gru.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dim()) // (T, N, D) = (1, 2, 3)
		fmt.Println(":", dxs[i])
	}

	fmt.Println(gru.DH())

	// Output:
	// *layer.TimeGRU: Wx(3, 3), Wh(1, 3), B(1, 3): 15
	// 2 1: [[0.09171084600490446] [0.18295102825645382]]
	// 2 3: [[0.019946542181883273 0.019946542181883273 0.019946542181883273] [0.05601681471011035 0.05601681471011035 0.05601681471011035]]
	// [[0.056873464301738536] [0.16710562310833357]]
}

func ExampleTimeGRU_Params() {
	gru := &layer.TimeGRU{}

	gru.SetParams(make([]matrix.Matrix, 3)...)
	fmt.Println(gru.Params())
	fmt.Println(gru.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}

func ExampleTimeGRU_SetState() {
	gru := &layer.TimeGRU{}
	gru.SetState(matrix.New())
	gru.SetState(matrix.New(), matrix.New())
	gru.ResetState()

	// Output:
}
