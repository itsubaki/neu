package layer_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleDropout() {
	x := matrix.New([]float64{1.0, -0.5}, []float64{-2.0, 3.0})

	rand.Seed(1)
	drop := layer.Dropout{Ratio: 0}
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))
	fmt.Println()

	drop.Ratio = 1.0
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))
	fmt.Println()

	drop.Ratio = 0.5
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(drop.Backward(x))
	fmt.Println(drop.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(drop.Backward(x))

	// Output:
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
	// [[0 -0] [-2 3]]
	// [[0 -0] [-2 3]] []
	// [[0.5 -0.25] [-1 1.5]]
	// [[0 -0] [-2 3]] []
}

func ExampleDropout_Params() {
	dropout := layer.Dropout{}

	dropout.SetParams(make([]matrix.Matrix, 0))
	dropout.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(dropout.Params())
	fmt.Println(dropout.Grads())

	// Output:
	// []
	// []
}
