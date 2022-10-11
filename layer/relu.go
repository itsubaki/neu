package layer

import "github.com/itsubaki/neu/math/matrix"

type ReLU struct {
	mask matrix.Matrix
}

func (l *ReLU) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.mask = mask(x, func(x float64) bool { return x <= 0 })
	return x.Mul(l.mask)
}

func (l *ReLU) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.mask)
	return dx, matrix.New()
}
