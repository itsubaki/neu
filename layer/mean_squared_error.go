package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type MeanSquaredError struct {
	target, q matrix.Matrix
}

func (l *MeanSquaredError) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *MeanSquaredError) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *MeanSquaredError) SetParams(p ...matrix.Matrix) {}
func (l *MeanSquaredError) String() string               { return fmt.Sprintf("%T", l) }

func (l *MeanSquaredError) Forward(target, q matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.target, l.q = target, q

	mse := l.target.Sub(l.q).Pow2().Sum() / float64(l.target.Size()) // sum((x - t)**2)/size
	return matrix.New([]float64{mse})
}

func (l *MeanSquaredError) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	gx0 := l.target.Sub(l.q).Mul(dout).MulC(2.0 / float64(l.target.Size())) // (x - t) * dout * (2.0 / size)
	return gx0, gx0.MulC(-1.0)
}
