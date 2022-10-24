package layer

import "github.com/itsubaki/neu/math/matrix"

type Mul struct {
	x matrix.Matrix
	y matrix.Matrix
}

func (l *Mul) Forward(x, y matrix.Matrix, opts ...Opts) matrix.Matrix {
	l.x, l.y = x, y
	return l.x.Mul(l.y)
}

func (l *Mul) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.y)
	dy := dout.Mul(l.x)
	return dx, dy
}
