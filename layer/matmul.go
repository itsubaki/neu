package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// MatMul is a layer that performs a matrix product.
type MatMul struct {
	W  matrix.Matrix // params
	DW matrix.Matrix // grads
	x  matrix.Matrix
}

func (l *MatMul) Params() []matrix.Matrix      { return []matrix.Matrix{l.W} }
func (l *MatMul) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW} }
func (l *MatMul) SetParams(p ...matrix.Matrix) { l.W = p[0] }
func (l *MatMul) String() string {
	a, b := l.W.Dim()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}

func (l *MatMul) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	return matrix.MatMul(l.x, l.W)
}

func (l *MatMul) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.MatMul(dout, l.W.T())
	dW := matrix.MatMul(l.x.T(), dout)
	l.DW = dW
	return dx, nil
}
