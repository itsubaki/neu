package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleSoftmaxWithLoss() {
	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	x := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)

	softmax := &layer.SoftmaxWithLoss{}
	loss := softmax.Forward(x, t)

	fmt.Println(loss)

	dx, _ := softmax.Backward(nil)
	for _, r := range dx {
		fmt.Println(r)
	}

	// Output:
	// [[2.0694934853340516]]
	// [0.04916164744936827 0.04676400561076958 -0.41894614614756587 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.04916164744936827 0.044483298144480495 0.044483298144480495]
	// [0.04916164744936827 0.04676400561076958 -0.45083835255063176 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.08105385385243416 0.044483298144480495 0.044483298144480495]

}

func ExampleCrossEntropyError() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC3%E5%88%B7%E3%81%BE%E3%81%A7

	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	y := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)

	fmt.Println(layer.CrossEntropyError(y, t))

	// Output:
	// 1.406704775046942

}
