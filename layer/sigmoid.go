package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type Sigmoid struct {
	out matrix.Matrix
}

func (l *Sigmoid) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.out = x.Func(activation.Sigmoid)
	return l.out
}

func (l *Sigmoid) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.FuncWith(dout, l.out, func(do, o float64) float64 { return do * (1.0 - o) * o }) // dout * (1.0 - out) * out
	return dx, matrix.New()
}
