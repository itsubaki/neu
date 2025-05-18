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
	f, g, i, o   matrix.Matrix // cache
	cNext        matrix.Matrix // cache
}

func (l *LSTM) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *LSTM) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DB} }
func (l *LSTM) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *LSTM) String() string {
	a, b := l.Wx.Dim()
	c, d := l.Wh.Dim()
	e, f := l.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *LSTM) Forward(x, h, c matrix.Matrix, _ ...Opts) (matrix.Matrix, matrix.Matrix) {
	A := matrix.MatMul(x, l.Wx).Add(matrix.MatMul(h, l.Wh)).Add(l.B) // (N, 4H) = x(N, D).Wx(D, 4H) + h(N, H).Wh(H, 4H) + b(1, 4H)
	AH := matrix.Split(A, len(h[0]))                                 // (4, N, H)

	f := matrix.F(AH[0], activation.Sigmoid) // (N, H)
	g := matrix.F(AH[1], activation.Tanh)    // (N, H)
	i := matrix.F(AH[2], activation.Sigmoid) // (N, H)
	o := matrix.F(AH[3], activation.Sigmoid) // (N, H)

	// next
	cNext := f.Mul(c).Add(g.Mul(i))                  // f * cPrev + g * i
	hNext := o.Mul(matrix.F(cNext, activation.Tanh)) // o * tanh(cNext)

	// cache
	l.x, l.h, l.c = x, h, c
	l.f, l.g, l.i, l.o = f, g, i, o
	l.cNext = cNext

	return hNext, cNext
}

func (l *LSTM) Backward(dhNext, dcNext matrix.Matrix) (matrix.Matrix, matrix.Matrix, matrix.Matrix) {
	tanh := matrix.F(l.cNext, activation.Tanh) // tanh(cNext)
	dt := matrix.F(tanh, dTanh)                // 1 - tanh(cNext)**2
	ds := dcNext.Add(dhNext.Mul(l.o).Mul(dt))  // dcNext + (dhNext * o) * (1 - tanh(cNext)**2)

	df := ds.Mul(l.c)      // ds * cPrev
	dg := ds.Mul(l.i)      // ds * i
	di := ds.Mul(l.g)      // ds * g
	do := dhNext.Mul(tanh) // dhNext * tanh(cNext)

	df = df.Mul(matrix.F(l.f, dSigmoid)) // df = df * f * (1 - f)
	dg = dg.Mul(matrix.F(l.g, dTanh))    // dg = dg * (1 - g**2)
	di = di.Mul(matrix.F(l.i, dSigmoid)) // di = di * i * (1 - i)
	do = do.Mul(matrix.F(l.o, dSigmoid)) // do = do * o * (1 - o)

	dA := matrix.HStack(df, dg, di, do) // (N, 4H)

	// grads
	l.DWx = matrix.MatMul(l.x.T(), dA)
	l.DWh = matrix.MatMul(l.h.T(), dA)
	l.DB = matrix.New(dA.SumAxis0())

	// prev
	dx := matrix.MatMul(dA, l.Wx.T())     // (N, D)
	dhPrev := matrix.MatMul(dA, l.Wh.T()) // (N, H)
	dcPrev := ds.Mul(l.f)                 // (N, H)

	return dx, dhPrev, dcPrev
}
