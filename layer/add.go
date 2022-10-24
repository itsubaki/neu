package layer

import "github.com/itsubaki/neu/math/matrix"

type Add struct{}

func (l *Add) Forward(x, y matrix.Matrix, opts ...Opts) matrix.Matrix {
	return x.Add(y)
}

func (l *Add) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Func(x1)
	dy := dout.Func(x1)
	return dx, dy
}

func x1(v float64) float64 { return 1.0 * v }
