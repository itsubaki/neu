package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// Dot is a layer that performs a dot product.
type Dot struct {
	W  matrix.Matrix // params
	DW matrix.Matrix // grads
	x  matrix.Matrix
}

func (l *Dot) Params() []matrix.Matrix      { return []matrix.Matrix{l.W} }
func (l *Dot) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW} }
func (l *Dot) SetParams(p ...matrix.Matrix) { l.W = p[0] }
func (l *Dot) String() string {
	a, b := l.W.Dim()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}

func (l *Dot) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W)
}

func (l *Dot) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	dW := matrix.Dot(l.x.T(), dout)
	l.DW = dW
	return dx, nil
}
