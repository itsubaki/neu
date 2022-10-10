package layer

import "github.com/itsubaki/neu/math/matrix"

type Dropout struct {
	Ratio     float64
	TrainFlag bool
	mask      [][]bool
}

func (l *Dropout) Forward(x, _ matrix.Matrix) matrix.Matrix {
	if l.TrainFlag {
		l.mask = newMask(matrix.Rand(x.Dimension()), func(x float64) bool { return x > l.Ratio })
		return mask(x, l.mask)
	}

	return x.Func(func(v float64) float64 { return v * (1.0 - l.Ratio) })
}

func (l *Dropout) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := mask(dout, l.mask)
	return dx, matrix.New()
}
