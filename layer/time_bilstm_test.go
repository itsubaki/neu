package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeBiLSTM() {
	lstm := &layer.TimeBiLSTM{
		F: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
		B: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
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
		fmt.Print(hs[i].Dim()) // (N, 2H) = (2, 2)
		fmt.Println(":", hs[i])
	}

	// backward
	dhs := []matrix.Matrix{
		{
			// (N, 2H)
			{0.1, 0.2},
			{0.3, 0.4},
		},
	}
	dxs := lstm.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dim()) // (T, N, D) = (1, 2, 3)
		fmt.Println(":", dxs[i])
	}

	// Output:
	// *layer.TimeBiLSTM: Wx(3, 4), Wh(1, 4), B(1, 4): 40
	// 2 2: [[0.03637115410746597 0.03637115410746597] [0.08514639710269788 0.08514639710269788]]
	// 2 3: [[0.021368657475674596 0.021368657475674596 0.021368657475674596] [0.06357173872462146 0.06357173872462146 0.06357173872462146]]
}

func ExampleTimeBiLSTM_Params() {
	lstm := &layer.TimeBiLSTM{
		F: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
		B: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
	}

	lstm.SetParams(make([]matrix.Matrix, 6)...)
	fmt.Println(lstm.Params())
	fmt.Println(lstm.Grads())

	// Output:
	// [[] [] [] [] [] []]
	// [[] [] [] [] [] []]
}

func ExampleTimeBiLSTM_SetState() {
	lstm := &layer.TimeBiLSTM{
		F: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
		B: &layer.TimeLSTM{
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
				// (1, 4H) = (1, 4)
				[]float64{0, 0, 0, 0},
			),
		},
	}
	lstm.SetState(matrix.New())
	lstm.SetState(matrix.New(), matrix.New())
	lstm.ResetState()

	// Output:
}
