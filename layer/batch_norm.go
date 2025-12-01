package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// BatchNorm is a layer that performs batch normalization.
type BatchNorm struct {
	Gamma, Beta         matrix.Matrix // params
	DGamma, DBeta       matrix.Matrix // grads
	Momentum            float64
	batchSize           int
	xc, xn, std, mu, va matrix.Matrix
}

func (l *BatchNorm) Params() []matrix.Matrix {
	return []matrix.Matrix{l.Gamma, l.Beta}
}

func (l *BatchNorm) Grads() []matrix.Matrix {
	return []matrix.Matrix{l.DGamma, l.DBeta}
}

func (l *BatchNorm) SetParams(p ...matrix.Matrix) {
	l.Gamma, l.Beta = p[0], p[1]
}

func (l *BatchNorm) String() string {
	a, b := l.Gamma.Dim()
	c, d := l.Beta.Dim()
	return fmt.Sprintf("%T: G(%v, %v), B(%v, %v): %v", l, a, b, c, d, a*b+c*d)
}

func (l *BatchNorm) Forward(x, _ matrix.Matrix, opts ...Opts) matrix.Matrix {
	if l.mu == nil {
		_, d := x.Dim()
		l.mu = matrix.Zero(1, d)
		l.va = matrix.Zero(1, d)
	}

	var xn matrix.Matrix
	if len(opts) > 0 && opts[0].Train {
		// var, std, norm
		mu := matrix.New(x.MeanAxis0())         // mean(x, axis=0)
		xc := x.Sub(mu)                         // x - mu
		va := matrix.New(xc.Pow2().MeanAxis0()) // mean(xc**2, axis=0)
		std := va.Sqrt(1e-7)                    // sqrt(var + 1e-7)
		xn = xc.Div(std)                        // (x - mu) / sqrt(var + eps)

		// cache
		l.batchSize = len(x)
		l.xc = xc
		l.xn = xn
		l.std = std
		l.mu = matrix.F2(l.mu, mu, func(a, b float64) float64 { return l.Momentum*a + (1-l.Momentum)*b }) // l.mu = momentum * l.mu + (1 - momentum) * mu
		l.va = matrix.F2(l.va, va, func(a, b float64) float64 { return l.Momentum*a + (1-l.Momentum)*b }) // l.va = momentum * l.va + (1 - momentum) * va
	} else {
		xc := x.Sub(l.mu)      // x - mu
		std := l.va.Sqrt(1e-7) // sqrt(var + 1e-7)
		xn = xc.Div(std)       // (x - mu) / sqrt(var + eps)
	}

	return xn.Mul(l.Gamma).Add(l.Beta) // xn * gamma + beta
}

func (l *BatchNorm) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	// DBeta, DGamma
	l.DGamma = matrix.New(l.xn.Mul(dout).SumAxis0()) // sum(xn * dout)
	l.DBeta = matrix.New(dout.SumAxis0())            // sum(dout)

	// dxn, dxc
	dxn := dout.Mul(l.Gamma) // dout * gamma
	dxc := dxn.Div(l.std)    // dxn / std

	// dstd, dvar
	dxnc := dxn.Mul(l.xc).Div(l.std.Mul(l.std))    // (dxn * xc) / (std * std)
	dstd := matrix.New(dxnc.SumAxis0()).MulC(-1.0) // -1.0 * Sum((dxn * xc) / (std * std))
	dvar := dstd.Div(l.std).MulC(0.5)              // 0.5 * (dstd / std)

	// dxc, dmu
	xcdv := l.xc.Mul(dvar).MulC(2.0 / float64(l.batchSize)) // 2.0/batchSize * xc * dvar
	dxc = dxc.Add(xcdv)                                     // dxc = dxc + 2.0/batchSize * xc * dvar
	dmu := matrix.New(dxc.SumAxis0())                       // dmu = sum(dxc)

	// dx
	dx := dxc.Sub(dmu.MulC(1.0 / float64(l.batchSize))) // dxc - dmu / batchSize
	return dx, nil
}
