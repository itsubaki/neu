package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type Sigmoid struct {
	out matrix.Matrix
}

func (l *Sigmoid) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.out = x.Func(func(v float64) float64 { return activation.Sigmoid(v) })
	return l.out
}

func (l *Sigmoid) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.out.Func(func(v float64) float64 { return -1.0*v + 1.0 })).Mul(l.out) // dout * (-1.0 * out + 1.0) * out
	return dx, matrix.New()
}
