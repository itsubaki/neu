package layer

import (
	"github.com/itsubaki/neu/math/matrix"
)

type BatchNorm struct {
	Gamma, Beta   matrix.Matrix
	Momentum      float64
	batchSize     int
	xc, xn        matrix.Matrix
	std, mu, va   matrix.Matrix
	dGamma, dBeta matrix.Matrix
}

func (l *BatchNorm) Forward(x, _ matrix.Matrix, opts ...Opts) matrix.Matrix {
	if l.mu == nil {
		l.mu = matrix.Zero(1, len(x[0]))
		l.va = matrix.Zero(1, len(x[0]))
	}

	var xn matrix.Matrix
	if len(opts) > 0 && opts[0].Train {
		// (x - mu) / sqrt(var + eps)
		mu := MeanAxis0(x)                  // mean(x, axis=0)
		xc := x.Sub(Broadcast(mu, len(x)))  // x - mu
		va := MeanAxis0(xc.Pow2())          // mean(xc**2, axis=0)
		std := va.Sqrt(1e-7)                // sqrt(var + 1e-7)
		xn = xc.Div(Broadcast(std, len(x))) // xc/std

		// for backword
		l.batchSize = len(x)
		l.xc = xc
		l.xn = xn
		l.std = std
		l.mu = l.mu.FuncWith(mu, func(a, b float64) float64 { return l.Momentum*a + (1-l.Momentum)*b })
		l.va = l.va.FuncWith(va, func(a, b float64) float64 { return l.Momentum*a + (1-l.Momentum)*b })
	} else {
		// (x - mu) / sqrt(var + eps)
		xc := x.Sub(Broadcast(l.mu, len(x))) // x - mu
		std := l.va.Sqrt(1e-7)               // sqrt(var + 1e-7)
		xn = xc.Div(Broadcast(std, len(x)))  // xc/std
	}

	// gamma * xn + beta
	return xn.Mul(Broadcast(l.Gamma, len(x))).Add(Broadcast(l.Beta, len(x)))
}

func (l *BatchNorm) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	// dBeta, dGamma, ... but not used.
	l.dBeta = SumAxis0(dout)
	l.dGamma = SumAxis0(l.xn.Mul(Broadcast(dout, len(l.xn))))

	// dxn, dxc
	dxn := l.Gamma.Mul(dout)
	dxc := dxn.Div(l.std)

	// dstd, dvar
	dstd := SumAxis0(dxn.Mul(l.xc).Div(l.std.Mul(l.std))).Func(func(v float64) float64 { return -1.0 * v }) // -1.0 * Sum((dxn * xc) / (std * std))
	dvar := dstd.Div(l.std).Func(func(v float64) float64 { return 0.5 * v })                                // 0.5 * (dstd / std)

	// dxc, dmu
	xcdvar := l.xc.Mul(Broadcast(dvar, len(l.xc)))                                            // xc * dvar
	xcdvar2 := xcdvar.Func(func(v float64) float64 { return 2.0 / float64(l.batchSize) * v }) // 2.0/batchSize * xc * dvar
	dxc = dxc.Add(xcdvar2)                                                                    // dxc = dxc + 2.0/batchSize * xc * dvar
	dmu := SumAxis0(dxc)

	// dx
	dx := dxc.Sub(dmu.Func(func(v float64) float64 { return v / float64(l.batchSize) })) // dxc - dmu / batchSize
	return dx, matrix.New()
}

func MeanAxis0(m matrix.Matrix) matrix.Matrix {
	return SumAxis0(m).Func(func(v float64) float64 { return v / float64(len(m)) })
}
