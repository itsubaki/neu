package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeRNN() {
	rnn := &layer.TimeRNN{
		Wx: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		Wh: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		B:  matrix.New([]float64{0}, []float64{0}),
	}
	fmt.Println(rnn)

	// forward
	xs := []matrix.Matrix{
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	hs := rnn.Forward(xs, nil)
	for i := range hs {
		fmt.Print(hs[i].Dimension())
		fmt.Println(":", hs[i])
	}

	// backward
	dhs := []matrix.Matrix{
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	dxs := rnn.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dimension())
		fmt.Println(":", dxs[i])
	}
	fmt.Println()

	// grads
	for _, g := range rnn.Grads() {
		fmt.Println(g)
	}

	// Output:
	// *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(2, 1): 20
	// 2 3: [[0.0599281035291435 0.1194272985343859 0.1780808681173302] [0.1194272985343859 0.23549574953849797 0.3452140341355209]]
	// 2 3: [[0.13653941943561718 0.13653941943561718 0.13653941943561718] [0.23725954436226937 0.23725954436226937 0.23725954436226937]]
	//
	// [[0.09868042705690408 0.1330597526412117 0.16117270672769926] [0.13821662689192143 0.19055616512647933 0.23426268639181055] [0.1777528267269388 0.24805257761174693 0.3073526660559219]]
	// [[0 0 0] [0 0 0] [0 0 0]]
	// [[0.3953619983501736 0.5749641248526761 0.730899796641113]]
}

func ExampleTimeRNN_time2() {
	rnn := &layer.TimeRNN{
		Wx: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		Wh: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}, []float64{0.1, 0.2, 0.3}),
		B:  matrix.New([]float64{0}, []float64{0}),
	}
	fmt.Println(rnn)
	fmt.Println()

	// forward
	xs := []matrix.Matrix{
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	hs := rnn.Forward(xs, nil)
	for i := range hs {
		fmt.Print(hs[i].Dimension())
		fmt.Println(":", hs[i])
	}
	fmt.Println()

	// backward
	dhs := []matrix.Matrix{
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
		{
			{0.1, 0.2, 0.3},
			{0.3, 0.4, 0.5},
		},
	}

	dxs := rnn.Backward(dhs)
	for i := range dxs {
		fmt.Print(dxs[i].Dimension())
		fmt.Println(":", dxs[i])
	}
	fmt.Println()

	// Output:
	// *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(2, 1): 20
	//
	// 2 3: [[0.0599281035291435 0.1194272985343859 0.1780808681173302] [0.1194272985343859 0.23549574953849797 0.3452140341355209]]
	// 2 3: [[0.0954521402060995 0.1891806346377357 0.2795841294978488] [0.1877594308589768 0.3627312769661758 0.515389479517456]]
	//
	// 2 3: [[0.2137321004822292 0.2137321004822292 0.2137321004822292] [0.35233531505495563 0.35233531505495563 0.35233531505495563]]
	// 2 3: [[0.131442260696387 0.131442260696387 0.131442260696387] [0.2085725262009533 0.2085725262009533 0.2085725262009533]]
}

func ExampleTimeRNN_Params() {
	rnn := &layer.TimeRNN{}
	rnn.SetParams(make([]matrix.Matrix, 3)...)

	fmt.Println(rnn.Params())
	fmt.Println(rnn.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}

func ExampleTimeRNN_state() {
	rnn := &layer.TimeRNN{}
	rnn.SetState(matrix.New())
	rnn.ResetState()

	// Output:
}
