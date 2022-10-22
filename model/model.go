package model

import (
	"math"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Layer      = (*layer.Add)(nil)
	_ Layer      = (*layer.Mul)(nil)
	_ Layer      = (*layer.ReLU)(nil)
	_ Layer      = (*layer.Sigmoid)(nil)
	_ Layer      = (*layer.Affine)(nil)
	_ Layer      = (*layer.SoftmaxWithLoss)(nil)
	_ Optimizer  = (*optimizer.SGD)(nil)
	_ Optimizer  = (*optimizer.Momentum)(nil)
	_ WeightInit = Xavier
	_ WeightInit = He
	_ WeightInit = Std(0.01)
)

type Layer interface {
	Forward(x, y matrix.Matrix) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type Optimizer interface {
	Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix
}

type WeightInit func(prevNodeNum int) float64

var (
	Xavier = func(prevNodeNum int) float64 { return math.Sqrt(1.0 / float64(prevNodeNum)) }
	He     = func(prevNodeNum int) float64 { return math.Sqrt(2.0 / float64(prevNodeNum)) }
	Std    = func(std float64) func(_ int) float64 { return func(_ int) float64 { return std } }
)

func Accuracy(y, t matrix.Matrix) float64 {
	count := func(x, y []int) int {
		var c int
		for i := range x {
			if x[i] == y[i] {
				c++
			}
		}

		return c
	}

	ymax := y.Argmax()
	tmax := t.Argmax()

	c := count(ymax, tmax)
	return float64(c) / float64(len(ymax))
}
