package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleRNN() {
	rnn := &layer.RNN{
		Wx: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		Wh: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		B:  matrix.New([]float64{0}, []float64{0}),
	}
	fmt.Println(rnn)

	// forward
	x := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)
	hPrev := matrix.New(
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	)

	hNext := rnn.Forward(x, hPrev)
	fmt.Print(hNext.Dimension())
	fmt.Println(":", hNext)

	// backward
	dhNext := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
	)

	dx, dhPrev := rnn.Backward(dhNext)
	fmt.Print(dx.Dimension())
	fmt.Println(":", dx)
	fmt.Print(dhPrev.Dimension())
	fmt.Println(":", dhPrev)

	// grads
	for _, g := range rnn.Grads() {
		fmt.Println(g)
	}

	// Output:
	// *layer.RNN: Wx(3, 3), Wh(3, 3), B(2, 1): 20
	// 2 3: [[0.0599281035291435 0.1194272985343859 0.1780808681173302] [0.1194272985343859 0.23549574953849797 0.3452140341355209]]
	// 2 3: [[0.13653941943561718 0.13653941943561718 0.13653941943561718] [0.23725954436226937 0.23725954436226937 0.23725954436226937]]
	// 2 3: [[0.13653941943561718 0.13653941943561718 0.13653941943561718] [0.23725954436226937 0.23725954436226937 0.23725954436226937]]
	// [[0.09868042705690408 0.1330597526412117 0.16117270672769926] [0.13821662689192143 0.19055616512647933 0.23426268639181055] [0.1777528267269388 0.24805257761174693 0.3073526660559219]]
	// [[0 0 0] [0 0 0] [0 0 0]]
	// [[0.3953619983501736 0.5749641248526761 0.730899796641113]]
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
