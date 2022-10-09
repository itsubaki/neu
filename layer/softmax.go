package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
)

type SoftmaxWithLoss struct {
	loss []float64
	y    matrix.Matrix
	t    matrix.Matrix
}

func (l *SoftmaxWithLoss) Forward(x, t matrix.Matrix) matrix.Matrix {
	y := make(matrix.Matrix, 0)
	for i := range x {
		y = append(y, activation.Sigmoid(x[i]))
	}

	ls := make([]float64, 0)
	for i := range y {
		ls = append(ls, loss.CrossEntropyError(y[i], t[i]))
	}

	l.t = t
	l.y = y
	l.loss = ls
	return matrix.New(l.loss)
}

func (l *SoftmaxWithLoss) Backward(_ matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	n, m := l.t.Shape()

	den := matrix.Fill(1.0/float64(n), n, m)
	dx := l.y.Sub(l.t).Mul(den)
	return dx, matrix.New()
}
