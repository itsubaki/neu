package neu

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Mul)(nil)
	_ Layer = (*layer.ReLU)(nil)
	_ Layer = (*layer.Sigmoid)(nil)
	_ Layer = (*layer.Affine)(nil)
	_ Layer = (*layer.SoftmaxWithLoss)(nil)
)

var (
	_ Optimizer = (*optimizer.SGD)(nil)
	_ Optimizer = (*optimizer.Momentum)(nil)
	_ Optimizer = (*optimizer.AdaGrad)(nil)
	_ Optimizer = (*optimizer.Adam)(nil)
)

type Layer interface {
	Forward(x, y matrix.Matrix) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type Optimizer interface {
	Update(params, grads *map[string][]float64)
}
