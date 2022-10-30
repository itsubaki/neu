package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Layer      = (*layer.Add)(nil)
	_ Layer      = (*layer.Affine)(nil)
	_ Layer      = (*layer.BatchNorm)(nil)
	_ Layer      = (*layer.Dot)(nil)
	_ Layer      = (*layer.Dropout)(nil)
	_ Layer      = (*layer.Mul)(nil)
	_ Layer      = (*layer.ReLU)(nil)
	_ Layer      = (*layer.Sigmoid)(nil)
	_ Layer      = (*layer.SoftmaxWithLoss)(nil)
	_ WeightInit = weight.Std(0.01)
	_ WeightInit = weight.He
	_ WeightInit = weight.Xavier
	_ WeightInit = weight.Glorot
)

type Layer interface {
	Params() []matrix.Matrix
	SetParams(p []matrix.Matrix)
	Grads() []matrix.Matrix
	SetGrads(g []matrix.Matrix)
	Forward(x, y matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type WeightInit func(prevNodeNum int) float64
