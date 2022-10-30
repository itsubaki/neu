package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type Dot struct {
	x matrix.Matrix
	w matrix.Matrix
}

func (l *Dot) Params() []matrix.Matrix     { return make([]matrix.Matrix, 0) }
func (l *Dot) SetParams(p []matrix.Matrix) {}
func (l *Dot) Grads() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Dot) SetGrads(g []matrix.Matrix)  {}

func (l *Dot) Forward(x, w matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	l.w = w
	return matrix.Dot(l.x, l.w)
}

func (l *Dot) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.w.T())
	dy := matrix.Dot(l.x.T(), dout)
	return dx, dy
}

func (l *Dot) String() string {
	return fmt.Sprintf("%T", l)
}
