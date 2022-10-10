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
	// [[0.43748779560473416 0.6443964695898473]]
	// [0.26248959373947 0.25624869824210517 -0.1771718468871023 0.25 0.25624869824210517 0.26248959373947 0.25 0.26248959373947 0.25 0.25]
	// [0.26248959373947 0.25624869824210517 -0.23751040626053 0.25 0.25624869824210517 0.26248959373947 0.25 0.3228281531128977 0.25 0.25]
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
	// [0.510825457099338 2.302584092994546]

}
