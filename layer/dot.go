package layer

import "github.com/itsubaki/neu/math/matrix"

type Dot struct {
	x matrix.Matrix
	w matrix.Matrix
}

func (l *Dot) Forward(x, w matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	l.w = w
	return matrix.Dot(l.x, l.w)
}

func (l *Dot) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.w.T())
	dy := matrix.Dot(l.x.T(), dout)
	return dx, dy
}
