package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type Sigmoid struct {
	out matrix.Matrix
}

func (l *Sigmoid) Forward(x, _ matrix.Matrix) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for i := range x {
		out = append(out, activation.Sigmoid(x[i]))
	}

	l.out = out
	return l.out
}

func (l *Sigmoid) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	n, m := dout.Shape()
	one := matrix.Fill(1.0, n, m)
	dx := dout.Mul(one.Sub(l.out)).Mul(l.out)

	return dx, matrix.New()
}
