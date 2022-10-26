package layer

import "github.com/itsubaki/neu/math/matrix"

type Dropout struct {
	Ratio float64
	mask  matrix.Matrix
}

func (l *Dropout) Forward(x, _ matrix.Matrix, opts ...Opts) matrix.Matrix {
	if len(opts) > 0 && opts[0].Train {
		l.mask = matrix.Mask(matrix.Rand(x.Dimension()), func(x float64) bool { return x > l.Ratio })
		return x.Mul(l.mask)
	}

	return x.MulC(1.0 - l.Ratio) // x * (1.0 - ratio)
}

func (l *Dropout) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(l.mask)
	return dx, matrix.New()
}
