package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

// SigmoidWithLoss is a layer that performs a sigmoid and a cross-entropy loss.
type SigmoidWithLoss struct {
	t matrix.Matrix
	y matrix.Matrix
}

func (l *SigmoidWithLoss) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *SigmoidWithLoss) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *SigmoidWithLoss) SetParams(p ...matrix.Matrix) {}

func (l *SigmoidWithLoss) Forward(x, t matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.t = t
	l.y = matrix.Func(x, activation.Sigmoid)

	// loss = Loss(y, t) + Loss(1 - y, 1 - t)
	one := matrix.One(l.y.Dimension())
	loss := Loss(l.y, l.t) + Loss(one.Sub(l.y), one.Sub(l.t))
	return matrix.New([]float64{loss})
}

func (l *SigmoidWithLoss) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	size, _ := l.t.Dimension()
	dx := l.y.Sub(l.t).Mul(dout).MulC(1.0 / float64(size)) // (y - t) * dout / size
	return dx, nil
}

func (l *SigmoidWithLoss) String() string {
	return fmt.Sprintf("%T", l)
}
