package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
)

type SoftmaxWithLoss struct {
	t    matrix.Matrix
	y    matrix.Matrix
	loss float64
}

func (l *SoftmaxWithLoss) Forward(x, t matrix.Matrix, opts ...Opts) matrix.Matrix {
	l.t = t
	l.y = Softmax(x)
	l.loss = CrossEntropyError(l.y, l.t)
	return matrix.New([]float64{l.loss})
}

func (l *SoftmaxWithLoss) Backward(_ matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	size, _ := l.t.Dimension()
	dx := matrix.FuncWith(l.y, l.t, func(y, t float64) float64 { return (y - t) / float64(size) }) // (y - t)/batch_size
	return dx, matrix.New()
}

func CrossEntropyError(y, t matrix.Matrix) float64 {
	list := make([]float64, 0)
	for i := range y {
		list = append(list, loss.CrossEntropyError(y[i], t[i]))
	}

	var sum float64
	for _, e := range list {
		sum = sum + e
	}

	return sum / float64(len(y))
}

func Softmax(x matrix.Matrix) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for _, r := range x {
		out = append(out, activation.Softmax(r))
	}

	return out
}
