package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// Mul is a layer that performs an element-wise multiplication.
type Mul struct {
	x, y matrix.Matrix
}

func (l *Mul) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Mul) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *Mul) SetParams(p ...matrix.Matrix) {}
func (l *Mul) String() string               { return fmt.Sprintf("%T", l) }

func (l *Mul) Forward(x, y matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x, l.y = x, y
	return l.x.Mul(l.y)
}

func (l *Mul) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.y)
	dy := dout.Mul(l.x)
	return dx, dy
}
