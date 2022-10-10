package layer

import "github.com/itsubaki/neu/math/matrix"

type Mul struct {
	x matrix.Matrix
	y matrix.Matrix
}

func (l *Mul) Forward(x, y matrix.Matrix) matrix.Matrix {
	l.x, l.y = x, y
	return matrix.FuncWith(l.x, l.y, func(a, b float64) float64 { return a * b })
}

func (l *Mul) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.FuncWith(dout, l.y, func(a, b float64) float64 { return a * b })
	dy := matrix.FuncWith(dout, l.x, func(a, b float64) float64 { return a * b })
	return dx, dy
}
