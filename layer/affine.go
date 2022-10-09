package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB []float64
}

func (l *Affine) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W).Add(l.B) // x.W + b
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)
	l.DB = matrix.SumAxis1(dout)

	return dx, matrix.New()
}
