package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleDropout() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})
	s := rand.Const(1)

	drop := &layer.Dropout{}
	fmt.Println(drop)

	drop.Ratio = 0.0
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true, Source: s}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))
	fmt.Println()

	drop.Ratio = 1.0
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true, Source: s}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))
	fmt.Println()

	drop.Ratio = 0.5
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true, Source: s}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))

	// Output:
	// *layer.Dropout: Ratio(0)
	// [[1 -0.5] [-2 3]]
	// [[1 -0.5] [-2 3]] []
	// [[1 -0.5] [-2 3]]
	// [[1 -0.5] [-2 3]] []
	//
	// [[0 -0] [-0 0]]
	// [[0 -0] [-0 0]] []
	// [[0 -0] [-0 0]]
	// [[0 -0] [-0 0]] []
	//
	// [[1 -0] [-0 0]]
	// [[1 -0] [-0 0]] []
	// [[0.5 -0.25] [-1 1.5]]
	// [[1 -0] [-0 0]] []
}

func ExampleDropout_Params() {
	drop := &layer.Dropout{Ratio: 0.5}

	drop.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(drop.Params())
	fmt.Println(drop.Grads())

	// Output:
	// []
	// []
}
