package layer

import (
	"fmt"

	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
)

// SoftmaxWithLoss is a layer that performs a softmax and a cross-entropy loss.
type SoftmaxWithLoss struct {
	t matrix.Matrix
	y matrix.Matrix
}

func (l *SoftmaxWithLoss) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *SoftmaxWithLoss) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *SoftmaxWithLoss) SetParams(p ...matrix.Matrix) {}
func (l *SoftmaxWithLoss) String() string               { return fmt.Sprintf("%T", l) }

func (l *SoftmaxWithLoss) Forward(x, t matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.y, l.t = softmax(x), t
	loss := Loss(l.y, l.t)
	return matrix.New([]float64{loss})
}

func (l *SoftmaxWithLoss) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	s := float64(len(l.t))
	dx := l.y.Sub(l.t).Mul(dout).MulC(1.0 / s) // (y - t) * dout / size
	return dx, nil
}

func Loss(y, t matrix.Matrix) float64 {
	list := make([]float64, len(y))
	for i := range y {
		list[i] = loss.CrossEntropyError(y[i], t[i])
	}

	var sum float64
	for _, e := range list {
		sum = sum + e
	}

	return sum / float64(len(y))
}
