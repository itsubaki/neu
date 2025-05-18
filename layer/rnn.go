package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type RNN struct {
	Wx, Wh, B    matrix.Matrix // params
	DWx, DWh, DB matrix.Matrix // grads
	x, h, hNext  matrix.Matrix // cache
}

func (l *RNN) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *RNN) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DB} }
func (l *RNN) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *RNN) String() string {
	a, b := l.Wx.Dim()
	c, d := l.Wh.Dim()
	e, f := l.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *RNN) Forward(x, h matrix.Matrix, _ ...Opts) matrix.Matrix {
	t := matrix.MatMul(h, l.Wh).Add(matrix.MatMul(x, l.Wx)).Add(l.B) // h(N, H).Wh(H, H) + x(N, D).Wx(D, H) + b(1, H)
	hNext := matrix.F(t, activation.Tanh)

	l.x, l.h, l.hNext = x, h, hNext // cache
	return l.hNext
}

func (l *RNN) Backward(dhNext matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dt := dhNext.Mul(matrix.F(l.hNext, dTanh)) // dt = dhNext * (1 - hNext**2)
	dx := matrix.MatMul(dt, l.Wx.T())          // dot(dt(N, H), Wx.T(H, D)) -> dx(N, D)
	dh := matrix.MatMul(dt, l.Wh.T())          // dot(dt(N, H), Wh.T(H, H)) -> dh(N, H)

	l.DWx = matrix.MatMul(l.x.T(), dt) // dot(x.T(D, N), dt(N, H)) -> (D, H)
	l.DWh = matrix.MatMul(l.h.T(), dt) // dot(hPrev.T(H, N), dt(N, H)) -> (H, H)
	l.DB = matrix.New(dt.SumAxis0())   // sum(dt(N, H), axis=0) -> (1, H)
	return dx, dh
}

// dtanh returns 1 - a**2
func dTanh(y float64) float64 { return 1 - y*y }
