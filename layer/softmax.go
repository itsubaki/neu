package layer

import (
	"github.com/itsubaki/neu/math/matrix"
)

type SoftmaxWithLoss struct {
	loss []float64
	y    matrix.Matrix
	t    matrix.Matrix
}

func (l *SoftmaxWithLoss) Forward(x, t matrix.Matrix) matrix.Matrix {
	l.t = t
	l.y = matrix.Sigmoid(x)
	l.loss = matrix.CrossEntropyError(l.y, t)
	return matrix.New(l.loss)
}

func (l *SoftmaxWithLoss) Backward(_ matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	size, _ := l.t.Shape()
	dx := l.y.Sub(l.t).Mulf64(1.0 / float64(size)) // (y - t)/batch_size
	return dx, matrix.New()
}
