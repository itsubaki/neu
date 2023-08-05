package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type LSTM struct {
	Wx, Wh, B    matrix.Matrix // params
	DWx, DWh, DB matrix.Matrix // grads
	x, h, c      matrix.Matrix // cache
	i, f, g, o   matrix.Matrix // cache
	cNext        matrix.Matrix // cache
}

func (l *LSTM) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *LSTM) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DB} }
func (l *LSTM) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }

func (l *LSTM) Forward(x, h, c matrix.Matrix, _ ...Opts) (matrix.Matrix, matrix.Matrix) {
	A := matrix.Dot(x, l.Wx).Add(matrix.Dot(h, l.Wh)).Add(l.B) // A(N, 4H) = x(N, D).Wx(D, 4H) + h(N, H).Wh(H, 4H) + b(4*H, 1)
	_, H := h.Dimension()                                      // h(N, H)

	f, g, i, o := matrix.New(), matrix.New(), matrix.New(), matrix.New()
	for _, r := range A {
		f = append(f, r[:H])
		g = append(g, r[H:2*H])
		i = append(i, r[2*H:3*H])
		o = append(o, r[3*H:])
	}

	f = matrix.Func(f, activation.Sigmoid) // (N, H)
	g = matrix.Func(g, activation.Tanh)    // (N, H)
	i = matrix.Func(i, activation.Sigmoid) // (N, H)
	o = matrix.Func(o, activation.Sigmoid) // (N, H)

	// next
	cNext := f.Mul(c).Add(g.Mul(i))                     // f * cPrev + g * i
	hNext := o.Mul(matrix.Func(cNext, activation.Tanh)) // o * tanh(cNext)

	// cache
	l.x, l.h, l.c = x, h, c
	l.i, l.f, l.g, l.o = i, f, g, o
	l.cNext = cNext

	return hNext, cNext
}

func (l *LSTM) Backward(dhNext, dcNext matrix.Matrix) (matrix.Matrix, matrix.Matrix, matrix.Matrix) {
	tanhcNext := matrix.Func(l.cNext, activation.Tanh) // tanh(cNext)
	dt := matrix.Func(tanhcNext, dtanh)                // 1 - tanh(cNext)**2
	ds := dcNext.Add(dhNext.Mul(l.o).Mul(dt))          // dhNext + (dhNext * o) * (1 - tanh(cNext)**2)

	df := ds.Mul(l.c)           // ds * cPrev
	dg := ds.Mul(l.i)           // ds * i
	di := ds.Mul(l.g)           // ds * g
	do := dhNext.Mul(tanhcNext) // dhNext * tanh(cNext)

	df = df.Mul(matrix.Func(l.f, dsigmoid)) // df = df * f * (1 - f)
	dg = dg.Mul(matrix.Func(l.g, dtanh))    // dg = dg * (1 - g**2)
	di = di.Mul(matrix.Func(l.i, dsigmoid)) // di = di * i * (1 - i)
	do = do.Mul(matrix.Func(l.o, dsigmoid)) // do = do * o * (1 - o)

	dA := matrix.HStack(df, dg, di, do) // (N, 4H)

	// grads
	l.DWx = matrix.Dot(l.x.T(), dA)
	l.DWh = matrix.Dot(l.h.T(), dA)
	l.DB = dA.SumAxis0()

	// prev
	dx := matrix.Dot(dA, l.Wx.T())     // (N, D)
	dhPrev := matrix.Dot(dA, l.Wh.T()) // (N, H)
	dcPrev := ds.Mul(l.f)              // (N, H)

	return dx, dhPrev, dcPrev
}

func (l *LSTM) String() string {
	a, b := l.Wx.Dimension()
	c, d := l.Wh.Dimension()
	e, f := l.B.Dimension()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}
