package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleGRU() {
	gru := &layer.GRU{
		Wx: matrix.New(
			// (D, 3H)
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.1, 0.2, 0.3},
		),
		Wh: matrix.New(
			// (H, 3H)
			[]float64{0.1, 0.2, 0.3},
		),
		B: matrix.New(
			// (3H, 1)
			[]float64{0},
			[]float64{0},
			[]float64{0},
		),
	}
	fmt.Println(gru)

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
	hNext := gru.Forward(x, h)
	fmt.Println(hNext)

	// backward
	dhNext := matrix.New(
		[]float64{1},
		[]float64{1},
	)
	dx, dhprev := gru.Backward(dhNext)
	fmt.Println(dx)
	fmt.Println(dhprev)

	// Output:
	// *layer.GRU: Wx(3, 3), Wh(1, 3), B(3, 1): 15
	// [[0.6435151783703015] [0.7197594876758177]]
	// [[0.19802226281185067 0.19802226281185067 0.19802226281185067] [0.17355749298151332 0.17355749298151332 0.17355749298151332]]
	// [[0.5336723979435917] [0.5151767843512871]]
}

func ExampleGRU_Params() {
	gru := &layer.GRU{}

	gru.SetParams(make([]matrix.Matrix, 3)...)
	fmt.Println(gru.Params())
	fmt.Println(gru.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}
