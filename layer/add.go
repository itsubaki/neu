package layer

import "github.com/itsubaki/neu/math/matrix"

type Add struct{}

func (l *Add) Forward(x, y matrix.Matrix) matrix.Matrix {
	return x.Add(y)
}

func (l *Add) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mulf64(1.0)
	dy := dout.Mulf64(1.0)
	return dx, dy
}
