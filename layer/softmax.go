package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

// Softmax is a layer that performs a softmax.
type Softmax struct {
	out matrix.Matrix
}

func (l *Softmax) Params() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Softmax) Grads() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Softmax) SetParams(p ...matrix.Matrix) {
	// noop
}

func (l *Softmax) String() string {
	return fmt.Sprintf("%T", l)
}

func (l *Softmax) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.out = softmax(x)
	return l.out
}

func (l *Softmax) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := l.out.Mul(dout)                // (N, H)
	sum := matrix.New(dx.SumAxis1()).T() // (N, H) -> (1, N) -> (N, 1)
	dx = dx.Sub(l.out.Mul(sum))          // (N, H)
	return dx, nil
}

func softmax(x matrix.Matrix) matrix.Matrix {
	out := make(matrix.Matrix, len(x))
	for i, r := range x {
		out[i] = activation.Softmax(r)
	}

	return out
}
