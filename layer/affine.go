package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB matrix.Matrix
}

func (l *Affine) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W).Add(l.B) // x.W + b
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)
	l.DB = dout.SumAxis0()
	return dx, matrix.New()
}
