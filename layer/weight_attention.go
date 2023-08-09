package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type AttentionWeight struct {
	Softmax *Softmax
	hs, hr  []matrix.Matrix
}

func (l *AttentionWeight) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *AttentionWeight) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *AttentionWeight) SetParams(p ...matrix.Matrix) {}

func (l *AttentionWeight) Forward(hs []matrix.Matrix, h matrix.Matrix) matrix.Matrix {
	T := len(hs)                         // (T, N, H)
	l.hs, l.hr = hs, matrix.Repeat(h, T) // (T, N, H)
	t := TimeMul(hs, l.hr)               // (T, N, H)

	c := make(matrix.Matrix, T) // (T, N)
	for i := 0; i < T; i++ {
		c[i] = t[i].SumAxis1()[0] // (1, N)
	}

	return l.Softmax.Forward(c, nil) // (T, N)
}

func (l *AttentionWeight) Backward(da matrix.Matrix) ([]matrix.Matrix, []matrix.Matrix) {
	T, N, H := len(l.hs), len(l.hs[0]), len(l.hs[0][0])

	ds, _ := l.Softmax.Backward(da) // (T, N)
	dt := reshape(ds, T, N, H)      // (T, N, H)
	dhs := TimeMul(dt, l.hr)
	dhr := TimeMul(dt, l.hs)

	dh := make([]matrix.Matrix, N) // (N, H)
	for i := 0; i < T; i++ {
		dh[i] = dhr[i].SumAxis0()
	}

	return dhs, dh
}

func (l *AttentionWeight) String() string {
	return fmt.Sprintf("%T", l)
}

func reshape(ds matrix.Matrix, T, N, H int) []matrix.Matrix {
	return nil
}
