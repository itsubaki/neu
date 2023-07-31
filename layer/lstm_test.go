package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleLSTM() {
	lstm := &layer.LSTM{
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
	x := matrix.New(
		// (N, D)
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)
	h := matrix.New(
		// (N, H)
		[]float64{1},
		[]float64{1},
	)
	c := matrix.New(
		// (N, H)
		[]float64{1},
		[]float64{2},
	)

	hNext, cNext := lstm.Forward(x, h, c)
	fmt.Println(hNext)
	fmt.Println(cNext)

	// backward
	dhNext := matrix.New(
		[]float64{1},
		[]float64{1},
	)
	dcNext := matrix.New(
		[]float64{1},
		[]float64{1},
	)

	dx, dhPrev, dcPrev := lstm.Backward(dhNext, dcNext)
	fmt.Println(dx)
	fmt.Println(dhPrev)
	fmt.Println(dcPrev)

	// Output:
	// *layer.LSTM: Wx(3, 4), Wh(3, 4), B(4, 1): 28
	// [[0.4083993715020527] [0.6230325858147007]]
	// [[0.7311121273610465] [1.382257865851477]]
	// [[0.2782747345210347 0.2782747345210347 0.2782747345210347] [0.2890431140697133 0.2890431140697133 0.2890431140697133]]
	// [[0.2782747345210347 0.2782747345210347 0.2782747345210347] [0.2890431140697133 0.2890431140697133 0.2890431140697133]]
	// [[0.7558896314112273] [0.6422382397686938]]
}

func ExampleLSTM_Params() {
	lstm := &layer.LSTM{}
	lstm.SetParams(make([]matrix.Matrix, 3)...)

	fmt.Println(lstm.Params())
	fmt.Println(lstm.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}
