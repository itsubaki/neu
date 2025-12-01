package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// Dropout is a layer that performs a dropout.
type Dropout struct {
	Ratio float64
	mask  matrix.Matrix
}

func (l *Dropout) Params() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Dropout) Grads() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Dropout) SetParams(p ...matrix.Matrix) {
	// noop
}

func (l *Dropout) String() string {
	return fmt.Sprintf("%T: Ratio(%v)", l, l.Ratio)
}

func (l *Dropout) Forward(x, _ matrix.Matrix, opts ...Opts) matrix.Matrix {
	if len(opts) > 0 && opts[0].Train {
		a, b := x.Dim()
		rnd := matrix.Rand(a, b, opts[0].Source)

		l.mask = matrix.Mask(rnd, func(x float64) bool { return x > l.Ratio })
		return x.Mul(l.mask)
	}

	return x.MulC(1.0 - l.Ratio) // x * (1.0 - ratio)
}

func (l *Dropout) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.mask)
	return dx, nil
}
