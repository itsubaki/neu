package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

// SigmoidWithLoss is a layer that performs a sigmoid and a cross-entropy loss.
type SigmoidWithLoss struct {
	y, t matrix.Matrix
}

func (l *SigmoidWithLoss) Params() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *SigmoidWithLoss) Grads() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *SigmoidWithLoss) SetParams(p ...matrix.Matrix) {
	// noop
}

func (l *SigmoidWithLoss) String() string {
	return fmt.Sprintf("%T", l)
}

func (l *SigmoidWithLoss) Forward(x, t matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.y, l.t = matrix.F(x, activation.Sigmoid), t

	// loss = Loss(y, t) + Loss(1 - y, 1 - t)
	loss := Loss(l.y, l.t) + Loss(matrix.SubC(1, l.y), matrix.SubC(1, l.t))
	return matrix.New([]float64{loss})
}

func (l *SigmoidWithLoss) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := l.y.Sub(l.t).Mul(dout).MulC(1.0 / float64(len(l.t))) // (y - t) * dout / size
	return dx, nil
}
