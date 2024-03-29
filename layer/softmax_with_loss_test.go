package layer_test

import (
	"fmt"
	"math"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleSoftmaxWithLoss() {
	l := &layer.SoftmaxWithLoss{}
	fmt.Println(l)

	// forward
	x := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)
	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	loss := l.Forward(x, t)
	fmt.Println(loss)

	// backward
	dout := matrix.New([]float64{1})
	dx, _ := l.Backward(dout)
	for _, r := range dx {
		fmt.Println(r)
	}

	// Output:
	// *layer.SoftmaxWithLoss
	// [[2.0694934853340516]]
	// [0.04916164744936827 0.04676400561076958 -0.41894614614756587 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.04916164744936827 0.044483298144480495 0.044483298144480495]
	// [0.04916164744936827 0.04676400561076958 -0.45083835255063176 0.044483298144480495 0.04676400561076958 0.04916164744936827 0.044483298144480495 0.08105385385243416 0.044483298144480495 0.044483298144480495]
}

func ExampleLoss() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC3%E5%88%B7%E3%81%BE%E3%81%A7

	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	y := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)

	fmt.Println(layer.Loss(y, t))

	// Output:
	// 1.406704775046942
}

func Example_lossLabel() {
	t := []int{2, 2}
	y := [][]float64{
		{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	}

	f := func(y [][]float64, t []int) float64 {
		var sum float64
		for i := range y {
			sum = sum + math.Log(y[i][t[i]]+1e-7)
		}

		return -1.0 * sum / float64(len(y))
	}

	fmt.Println(f(y, t))

	// Output:
	// 1.406704775046942

}

func ExampleSoftmaxWithLoss_Params() {
	l := &layer.SoftmaxWithLoss{}

	l.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// []
	// []
}
