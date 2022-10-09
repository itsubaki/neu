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
	drop := layer.Dropout{Ratio: 0.5, TrainFlag: true}
	fmt.Println(drop.Forward(x, nil))

	drop.TrainFlag = false
	fmt.Println(drop.Forward(x, nil))
	fmt.Println(drop.Backward(x))

	// Output:
	// [[1 -0.5] [-2 0]]
	// [[0.5 -0.25] [-1 1.5]]
	// [[1 -0.5] [-2 0]] []
}
