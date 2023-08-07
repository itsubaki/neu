package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeLSTM() {
	// (D, H) = (3, 1)
	lstm := &layer.TimeLSTM{
		Wx: matrix.New(
			// (D, 4H) = (3, 4)
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
		),
		Wh: matrix.New(
			// (H, 4H) = (1, 4)
			[]float64{0.1, 0.2, 0.3, 0.4},
		),
		B: matrix.New(
			// (4H, 1) = (4, 1)
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
		),
	}
	fmt.Println(lstm)

	// forward
	xs := []matrix.Matrix{
		// (T, N, D) = (1, 2, 3)
		{
			// (N, D) = (2, 3)
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	hs := lstm.Forward(xs, nil)
	for i := range hs {
		fmt.Print(hs[i].Dimension()) // (N, H) = (2, 1)
		fmt.Println(":", hs[i])
	}

	// backward
	dhs := []matrix.Matrix{
		{
			// (N, H)
			{0.1},
			{0.3},
		},
	}
	dxs := lstm.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dimension()) // (T, N, D) = (1, 2, 3)
		fmt.Println(":", dxs[i])
	}

	fmt.Println(lstm.DH())

	// Output:
	// *layer.TimeLSTM: Wx(3, 4), Wh(1, 4), B(4, 1): 20
	// 2 1: [[0.03637115410746597] [0.08514639710269788]]
	// 2 3: [[0.007122885825224865 0.007122885825224865 0.007122885825224865] [0.027245030881980627 0.027245030881980627 0.027245030881980627]]
	// [[0.007122885825224865] [0.027245030881980627]]
}

func ExampleTimeLSTM_Params() {
	lstm := &layer.TimeLSTM{}
	lstm.SetParams(make([]matrix.Matrix, 3)...)

	fmt.Println(lstm.Params())
	fmt.Println(lstm.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}

func ExampleTimeLSTM_state() {
	lstm := &layer.TimeLSTM{}
	lstm.SetState(matrix.New())
	lstm.ResetState()

	// Output:
}
