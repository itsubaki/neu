package layer

import "github.com/itsubaki/neu/math/matrix"

type Sigmoid struct {
	out matrix.Matrix
}

func (l *Sigmoid) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.out = matrix.Sigmoid(x)
	return l.out
}

func (l *Sigmoid) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.out.Mulf64(-1.0).Addf64(1)).Mul(l.out) // dout * (-1.0 * out + 1.0) * out
	return dx, matrix.New()
}
