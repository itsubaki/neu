package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type GRU struct {
	Wx, Wh, B      matrix.Matrix // params
	DWx, DWh, DB   matrix.Matrix // grads
	x, hprev, hhat matrix.Matrix // cache
	z, r           matrix.Matrix // cache
}

func (l *GRU) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *GRU) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DB} }
func (l *GRU) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *GRU) String() string {
	a, b := l.Wx.Dim()
	c, d := l.Wh.Dim()
	e, f := l.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *GRU) Forward(x, h matrix.Matrix, _ ...Opts) matrix.Matrix {
	H := len(l.Wh)               // (H, 3H)
	WxH := matrix.Split(l.Wx, H) // (3, D, H)
	WhH := matrix.Split(l.Wh, H) // (3, H, H)
	BH := matrix.Split(l.B, H)   // (3, H)

	Wxz, Wxr, Wxh := WxH[0], WxH[1], WxH[2] // (D, H)
	Whz, Whr, Whh := WhH[0], WhH[1], WhH[2] // (H, H)
	Bz, Br, Bh := BH[0], BH[1], BH[2]       // (1, H)

	l.z = matrix.F(matrix.Dot(x, Wxz).Add(matrix.Dot(h, Whz)).Add(Bz), activation.Sigmoid)          // z = sigmoid(x.Wxz + h.Whz + bz)
	l.r = matrix.F(matrix.Dot(x, Wxr).Add(matrix.Dot(h, Whr)).Add(Br), activation.Sigmoid)          // r = sigmoid(x.Wxr + h.Whr + br)
	l.hhat = matrix.F(matrix.Dot(x, Wxh).Add(matrix.Dot(h.Mul(l.r), Whh)).Add(Bh), activation.Tanh) // hhat = tanh(x.Wxh + (h * r).Whh + bh)
	l.x, l.hprev = x, h

	hnext := matrix.SubC(1, l.z).Mul(l.hprev).Add(l.z.Mul(l.hhat)) // (1 - z) * hprev + z * hhat
	return hnext
}

func (l *GRU) Backward(dhnext matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	H := len(l.Wh)               // (H, 3H)
	WxH := matrix.Split(l.Wx, H) // (3, D, H)
	WhH := matrix.Split(l.Wh, H) // (3, H, H)

	Wxz, Wxr, Wxh := WxH[0], WxH[1], WxH[2]
	Whz, Whr, Whh := WhH[0], WhH[1], WhH[2]

	// dh
	dhhat := dhnext.Mul(l.z)                  // dhhat = dhnext * z
	dhprev := dhnext.Mul(matrix.SubC(1, l.z)) // dhprev = dhnext * (1 - z)

	// tanh
	dt := dhhat.Mul(matrix.F(l.hhat, dTanh))     // dt = dhhat * (1 - hhat**2)
	dbh := matrix.New(dt.SumAxis0())             // dbh = sum(dt, axis=0)
	dWhh := matrix.Dot(l.r.Mul(l.hprev).T(), dt) // dWhh = (r * hprev).T.dt
	dhr := matrix.Dot(dt, Whh.T())               // dhr = dt.Whh.T
	dWxh := matrix.Dot(l.x.T(), dt)              // dWxh = x.T.dt
	dx := matrix.Dot(dt, Wxh.T())                // dx = dt.Wxh.T
	dhprev = dhprev.Add(dhr.Mul(l.r))            // dhprev = dhprev + dhr * r

	// gate(z)
	dz := dhnext.Mul(l.hhat).Sub(dhnext.Mul(l.hprev)) // dz = dhnext * hhat - dhnext * hprev
	dtz := dz.Mul(matrix.F(l.z, dSigmoid))            // dtz = dz * z * (1 - z)
	dbz := matrix.New(dtz.SumAxis0())                 // dbz = sum(dtz, axis=0)
	dWhz := matrix.Dot(l.hprev.T(), dtz)              // dWhz = hprev.T.dtz
	dhprev = dhprev.Add(matrix.Dot(dtz, Whz.T()))     // dhprev = dhprev + dtz.Whz.T
	dWxz := matrix.Dot(l.x.T(), dtz)                  // dWxz = x.T.dtz
	dx = dx.Add(matrix.Dot(dt, Wxz.T()))              // dx = dx + dtz.Wxz.T

	// gate(r)
	dr := dhr.Mul(l.hprev)                        // dr = dhr * hprev
	dtr := dr.Mul(matrix.F(l.r, dSigmoid))        // dtr = dr * r * (1 - r)
	dbr := matrix.New(dtr.SumAxis0())             // dbr = sum(dtr, axis=0)
	dWhr := matrix.Dot(l.hprev.T(), dtr)          // dWhr = hprev.T.dtr
	dhprev = dhprev.Add(matrix.Dot(dtr, Whr.T())) // dhprev = dhprev + dtr.Whr.T
	dWxr := matrix.Dot(l.x.T(), dtr)              // dWzr = x.T.dtr
	dx = dx.Add(matrix.Dot(dtr, Wxr.T()))         // dx = dx + dtr.Wxr.T

	// grads
	l.DWx = matrix.HStack(dWxz, dWxr, dWxh)
	l.DWh = matrix.HStack(dWhz, dWhr, dWhh)
	l.DB = matrix.HStack(dbz, dbr, dbh)
	return dx, dhprev
}
