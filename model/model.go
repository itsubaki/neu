package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Layer      = (*layer.Add)(nil)
	_ Layer      = (*layer.Affine)(nil)
	_ Layer      = (*layer.Dot)(nil)
	_ Layer      = (*layer.Dropout)(nil)
	_ Layer      = (*layer.Mul)(nil)
	_ Layer      = (*layer.ReLU)(nil)
	_ Layer      = (*layer.Sigmoid)(nil)
	_ Layer      = (*layer.SoftmaxWithLoss)(nil)
	_ Optimizer  = (*optimizer.AdaGrad)(nil)
	_ Optimizer  = (*optimizer.Momentum)(nil)
	_ Optimizer  = (*optimizer.SGD)(nil)
	_ WeightInit = weight.Std(0.01)
	_ WeightInit = weight.He
	_ WeightInit = weight.Xavier
	_ WeightInit = weight.Glorot
)

type Layer interface {
	Forward(x, y matrix.Matrix) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type Optimizer interface {
	Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix
}

type WeightInit func(prevNodeNum int) float64
