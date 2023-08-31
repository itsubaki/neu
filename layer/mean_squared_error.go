package layer

import (
	"fmt"

	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
)

type MeanSquaredError struct {
	x, t matrix.Matrix
}

func (l *MeanSquaredError) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *MeanSquaredError) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *MeanSquaredError) SetParams(p ...matrix.Matrix) {}
func (l *MeanSquaredError) String() string               { return fmt.Sprintf("%T", l) }

func (l *MeanSquaredError) Forward(x, t matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x, l.t = x, t

	out := make([]float64, len(t))
	for i := range t {
		out[i] = loss.MeanSquaredError(x[i], t[i])
	}

	return matrix.New(out)
}

func (l *MeanSquaredError) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	gx0 := l.x.Sub(l.t).Mul(dout).MulC(2.0 / float64(len(l.t))) // (x - t) * dout * (2.0 / size)
	return gx0, gx0.MulC(-1.0)
}
