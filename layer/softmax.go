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
	l.t = t
	l.y = matrix.Func(x, activation.Sigmoid)
	l.loss = crossEntropyError(l.y, l.t)
	return matrix.New(l.loss)
}

func (l *SoftmaxWithLoss) Backward(_ matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	size, _ := l.t.Dimension()
	dx := matrix.FuncWith(l.y, l.t, func(y, t float64) float64 { return (y - t) / float64(size) }) // (y - t)/batch_size
	return dx, matrix.New()
}

func crossEntropyError(y, t matrix.Matrix) []float64 {
	out := make([]float64, 0)
	for i := range y {
		out = append(out, loss.CrossEntropyError(y[i], t[i]))
	}

	return out
}
