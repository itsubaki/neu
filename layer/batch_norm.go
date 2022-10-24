package layer

import "github.com/itsubaki/neu/math/matrix"

type BatchNorm struct {
}

func (l *BatchNorm) Forward(x, y matrix.Matrix, opts ...Opts) matrix.Matrix {
	return nil
}

func (l *BatchNorm) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	return nil, nil
}
