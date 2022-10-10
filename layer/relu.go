package layer

import "github.com/itsubaki/neu/math/matrix"

type ReLU struct {
	mask [][]bool
}

func (l *ReLU) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.mask = newMask(x, func(x float64) bool { return x <= 0 })
	return mask(x, l.mask)
}

func (l *ReLU) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := mask(dout, l.mask)
	return dx, matrix.New()
}
