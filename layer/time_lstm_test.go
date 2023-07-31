package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeLSTM() {
	lstm := &layer.TimeLSTM{
		Wx: matrix.New(
			// (D, 4H)
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
		),
		Wh: matrix.New(
			// (D, 4H)
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
			[]float64{0.1, 0.2, 0.3, 0.4},
		),
		B: matrix.New(
			// (4H, 1)
			[]float64{0},
			[]float64{0},
			[]float64{0},
			[]float64{0},
		),
	}
	fmt.Println(lstm)

	// forward
	xs := []matrix.Matrix{
		matrix.New(
			// (N, D)
			[]float64{0.1, 0.2},
			[]float64{0.3, 0.4},
		),
	}

	hs := lstm.Forward(xs, nil)
	for i := range hs {
		fmt.Print(hs[i].Dimension())
		fmt.Println(":", hs[i])
	}
	fmt.Println()

	// backward
	dhs := []matrix.Matrix{
		matrix.New(
			[]float64{0.1},
			[]float64{0.3},
		),
	}
	dxs := lstm.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dimension())
		fmt.Println(":", dxs[i])
	}
	fmt.Println()

	// Output:
	// *layer.TimeLSTM: Wx(3, 4), Wh(3, 4), B(4, 1): 28
	// 2 1: [[0.016588561630723222] [0.04366773042879063]]
	//
	// 2 3: [[0.006062040384277095 0.006062040384277095 0.006062040384277095] [0.02240814455580229 0.02240814455580229 0.02240814455580229]]
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
