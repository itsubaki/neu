package layer

import "github.com/itsubaki/neu/math/matrix"

type Add struct{}

func (l *Add) Forward(x, y matrix.Matrix) matrix.Matrix {
	return x.Add(y)
}

func (l *Add) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Func(func(v float64) float64 { return v * 1.0 })
	dy := dout.Func(func(v float64) float64 { return v * 1.0 })
	return dx, dy
}
