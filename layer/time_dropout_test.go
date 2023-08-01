package layer_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeDropout() {
	xs := []matrix.Matrix{
		matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0}),
	}

	s := rand.NewSource(1)
	drop := &layer.TimeDropout{Ratio: 0}
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
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[1 -0.5] [-2 3]]]
	//
	// [[[0 -1] [-0 0]]]
	// [[[0 -1] [-0 0]]]
	// [[[1 -0.5] [-2 3]]]
	// [[[0 -1] [-0 0]]]
}

func ExampleTimeDropout_Params() {
	drop := &layer.TimeDropout{Ratio: 0.5}
	drop.SetParams(make([]matrix.Matrix, 0)...)

	fmt.Println(drop)
	fmt.Println(drop.Params())
	fmt.Println(drop.Grads())

	// Output:
	// *layer.TimeDropout: Ratio(0.5)
	// []
	// []
}

func ExampleTimeDropout_state() {
	dropout := &layer.TimeDropout{}
	dropout.SetState(matrix.New())
	dropout.ResetState()

	// Output:
}
