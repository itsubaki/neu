package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleRNN() {
	x := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)
	hPrev := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)
	dhNext := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)

	rnn := &layer.RNN{
		Wx: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		Wh: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		B:  matrix.New([]float64{0}, []float64{0}),
	}
	fmt.Println(rnn)
	fmt.Println()

	// forward
	hNext := rnn.Forward(x, hPrev)
	fmt.Println(hPrev.Dimension())
	fmt.Println(hNext.Dimension())
	fmt.Println(hNext)
	fmt.Println()

	// backward
	dx, dhPrev := rnn.Backward(dhNext)
	fmt.Println(dhNext.Dimension())
	fmt.Println(dhPrev.Dimension())
	fmt.Println(dx)
	fmt.Println(dhPrev)

	// Output:
	// *layer.RNN: Wx(3, 3), Wh(3, 3), B(2, 1): 20
	//
	// 2 3
	// 2 3
	// [[0.1194272985343859 0.23549574953849797 0.3452140341355209] [0.23549574953849797 0.4462436102487797 0.6169093028770649]]
	//
	// 2 3
	// 2 3
	// [[0.12691349563884896 0.12691349563884896 0.12691349563884896] [0.1853190205870099 0.1853190205870099 0.1853190205870099]]
	// [[0.12691349563884896 0.12691349563884896 0.12691349563884896] [0.1853190205870099 0.1853190205870099 0.1853190205870099]]
}

func ExampleRNN_Params() {
	rnn := &layer.RNN{}
	rnn.SetParams(make([]matrix.Matrix, 3)...)

	fmt.Println(rnn.Params())
	fmt.Println(rnn.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}
