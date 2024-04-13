package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleTimeDropout() {
	xs := []matrix.Matrix{{{1.0, -0.5}, {-2.0, 3.0}}}
	s := rand.Const(1)

	drop := &layer.TimeDropout{}
	fmt.Println(drop)

	drop.Ratio = 0.0
	fmt.Println(drop.Forward(xs, nil, layer.Opts{Train: true, Source: s}))
	fmt.Println(drop.Backward(xs))
	fmt.Println(drop.Forward(xs, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(xs))
	fmt.Println()

	drop.Ratio = 0.5
	fmt.Println(drop.Forward(xs, nil, layer.Opts{Train: true, Source: s}))
	fmt.Println(drop.Backward(xs))
	fmt.Println(drop.Forward(xs, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(xs))

	// Output:
	// *layer.TimeDropout: Ratio(0)
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	//
	// [[[2 -1] [-0 0]]]
	// [[[2 -1] [-0 0]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[2 -1] [-0 0]]]
}

func ExampleTimeDropout_Params() {
	drop := &layer.TimeDropout{Ratio: 0.5}

	drop.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(drop.Params())
	fmt.Println(drop.Grads())

	// Output:
	// []
	// []
}

func ExampleTimeDropout_SetState() {
	dropout := &layer.TimeDropout{}
	dropout.SetState(matrix.New())
	dropout.ResetState()

	// Output:
}
