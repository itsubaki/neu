package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// ReLU is a layer that performs an element-wise ReLU.
type ReLU struct {
	mask matrix.Matrix
}

func (l *ReLU) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *ReLU) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *ReLU) SetParams(p ...matrix.Matrix) {}

func (l *ReLU) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.mask = matrix.Mask(x, func(x float64) bool { return x > 0 })
	return x.Mul(l.mask)
}

func (l *ReLU) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.mask)
	return dx, matrix.New()
}

func (l *ReLU) String() string {
	return fmt.Sprintf("%T", l)
}
