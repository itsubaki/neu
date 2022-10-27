package layer

import "github.com/itsubaki/neu/math/matrix"

type Mul struct {
	x matrix.Matrix
	y matrix.Matrix
}

func (l *Mul) Params() []matrix.Matrix     { return make([]matrix.Matrix, 0) }
func (l *Mul) SetParams(p []matrix.Matrix) {}
func (l *Mul) Grads() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Mul) SetGrads(g []matrix.Matrix)  {}

func (l *Mul) Forward(x, y matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x, l.y = x, y
	return l.x.Mul(l.y)
}

func (l *Mul) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.y)
	dy := dout.Mul(l.x)
	return dx, dy
}
