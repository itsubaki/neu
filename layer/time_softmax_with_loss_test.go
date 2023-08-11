package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeSoftmaxWithLoss() {
	l := &layer.TimeSoftmaxWithLoss{}
	fmt.Println(l)

	// forward
	xs := []matrix.Matrix{
		{
			{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
			{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
		},
	}
	ts := []matrix.Matrix{
		{
			{2},
			{2},
		},
	}
	loss := l.Forward(xs, ts)
	fmt.Println(loss)

	// backward
	dout := []matrix.Matrix{{{1}}}
	dx := l.Backward(dout)
	for _, m := range dx {
		for _, r := range m {
			fmt.Println(r)
		}
	}

	// Output:
	// *layer.TimeSoftmaxWithLoss
	// [[[2.0694934853340516]]]
	// [0.04916164744936827 0.04676400561076958 -0.41894614614756587 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.04916164744936827 0.044483298144480495 0.044483298144480495]
	// [0.04916164744936827 0.04676400561076958 -0.45083835255063176 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.08105385385243416 0.044483298144480495 0.044483298144480495]

}

func ExampleTimeSoftmaxWithLoss_Params() {
	l := &layer.TimeSoftmaxWithLoss{}

	l.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// []
	// []
}

func ExampleTimeSoftmaxWithLoss_state() {
	l := &layer.TimeSoftmaxWithLoss{}
	l.SetState(matrix.New())
	l.ResetState()

	// Output:
}

func ExampleOneHot() {
	ts := []matrix.Matrix{
		{
			{0, 1, 2},
			{3, 4, 5},
		},
		{
			{0, 1, 2},
			{3, 4, 5},
		},
	}

	onehot := layer.OneHot(ts, 6)
	for _, m := range onehot {
		for _, r := range m {
			fmt.Println(r)
		}
		fmt.Println()
	}

	// Output:
	// [1 0 0 0 0 0]
	// [0 1 0 0 0 0]
	// [0 0 1 0 0 0]
	// [0 0 0 1 0 0]
	// [0 0 0 0 1 0]
	// [0 0 0 0 0 1]
	//
	// [1 0 0 0 0 0]
	// [0 1 0 0 0 0]
	// [0 0 1 0 0 0]
	// [0 0 0 1 0 0]
	// [0 0 0 0 1 0]
	// [0 0 0 0 0 1]
}
