package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleSigmoidWithLoss() {
	x := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)
	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)

	l := &layer.SigmoidWithLoss{}
	loss := l.Forward(x, t)
	dx, _ := l.Backward(matrix.New([]float64{1.0}))

	fmt.Println(loss)
	fmt.Println(dx)

	// Output:
	// [[7.130183898637823]]
	// [[0.26248959373947 0.25624869824210517 -0.1771718468871023 0.25 0.25624869824210517 0.26248959373947 0.25 0.26248959373947 0.25 0.25] [0.26248959373947 0.25624869824210517 -0.23751040626053 0.25 0.25624869824210517 0.26248959373947 0.25 0.3228281531128977 0.25 0.25]]
}

func ExampleSigmoidWithLoss_Params() {
	l := &layer.SigmoidWithLoss{}
	l.SetParams(make([]matrix.Matrix, 0)...)

	fmt.Println(l)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// *layer.SigmoidWithLoss
	// []
	// []
}
