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
		[]float64{1, 2, 3},
		[]float64{1, 2, 3},
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
	fmt.Println()

	// grads
	for _, g := range rnn.Grads() {
		fmt.Println(g)
	}

	// Output:
	// *layer.RNN: Wx(3, 3), Wh(3, 3), B(2, 1): 20
	// 2 3: [[0.5783634130445059 0.8667839288498187 0.9625869800912907] [0.616909302877065 0.8936977272038726 0.9737493633257944]]
	// 2 3: [[0.02321074997030762 0.02321074997030762 0.02321074997030762] [0.04245886376535411 0.04245886376535411 0.04245886376535411]]
	// 2 3: [[0.02321074997030762 0.02321074997030762 0.02321074997030762] [0.04245886376535411 0.04245886376535411 0.04245886376535411]]
	//
	// [[0.06240301970665113 0.02913023710062935 0.009974615786153376] [0.08764066469187826 0.0421561244100081 0.01476801383004575] [0.11287830967710537 0.055182011719386845 0.019561411873938124]]
	// [[0.25237644985227115 0.13025887309378748 0.04793398043892374] [0.5047528997045423 0.26051774618757495 0.09586796087784748] [0.7571293495568134 0.3907766192813624 0.1438019413167712]]
	// [[0.25237644985227115 0.13025887309378748 0.04793398043892374]]
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
