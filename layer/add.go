package layer

import "github.com/itsubaki/neu/math/matrix"

type Add struct{}

func (l *Add) Forward(x, y matrix.Matrix) matrix.Matrix {
	return x.Add(y)
}

func (l *Add) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	n, m := dout.Shape()
	one := matrix.Fill(1.0, n, m)

	dx := dout.Mul(one)
	dy := dout.Mul(one)
	return dx, dy
}
