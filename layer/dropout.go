package layer

import "github.com/itsubaki/neu/math/matrix"

type Dropout struct {
	Ratio     float64
	TrainFlag bool
	mask      [][]bool
}

func (l *Dropout) Forward(x, _ matrix.Matrix) matrix.Matrix {
	if l.TrainFlag {
		l.mask = mask(x, func(x float64) bool { return x > l.Ratio })
		return matrix.Mask(x, l.mask)
	}

	return x.Mulf64(1.0 - l.Ratio)
}

func (l *Dropout) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Mask(dout, l.mask)
	return dx, matrix.New()
}
