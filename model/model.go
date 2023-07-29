package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Affine)(nil)
	_ Layer = (*layer.BatchNorm)(nil)
	_ Layer = (*layer.Dot)(nil)
	_ Layer = (*layer.Dropout)(nil)
	_ Layer = (*layer.EmbeddingDot)(nil)
	_ Layer = (*layer.Embedding)(nil)
	_ Layer = (*layer.Mul)(nil)
	_ Layer = (*layer.ReLU)(nil)
	_ Layer = (*layer.RNN)(nil)
	_ Layer = (*layer.Sigmoid)(nil)
	_ Layer = (*layer.SigmoidWithLoss)(nil)
	_ Layer = (*layer.SoftmaxWithLoss)(nil)
)

var (
	_ WeightInit = weight.Std(0.01)
	_ WeightInit = weight.He
	_ WeightInit = weight.Xavier
	_ WeightInit = weight.Glorot
)

// Layer is an interface that represents a layer.
type Layer interface {
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
	Forward(x, y matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

// WeightInit is an interface that represents a weight initializer.
type WeightInit func(prevNodeNum int) float64
