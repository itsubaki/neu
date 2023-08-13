package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
	"github.com/itsubaki/neu/math/vector"
)

type AttentionWeight struct {
	Softmax *Softmax
	hs, hr  []matrix.Matrix
}

func (l *AttentionWeight) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *AttentionWeight) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *AttentionWeight) SetParams(p ...matrix.Matrix) {}
func (l *AttentionWeight) String() string               { return fmt.Sprintf("%T", l) }

func (l *AttentionWeight) Forward(hs []matrix.Matrix, h matrix.Matrix) matrix.Matrix {
	l.hs = hs
	l.hr = tensor.Repeat(h, len(hs))

	T := len(hs)                         // (T, N, H)
	l.hs, l.hr = hs, tensor.Repeat(h, T) // (N, H) -> (T, N, H)
	t := tensor.Mul(hs, l.hr)            // (T, N, H)

	c := make(matrix.Matrix, T) // (T, N)
	for i := 0; i < T; i++ {
		c[i] = t[i].SumAxis1() // (N, H) -> (1, N)
	}

	return l.Softmax.Forward(c, nil) // (T, N)
}

func (l *AttentionWeight) Backward(da matrix.Matrix) ([]matrix.Matrix, matrix.Matrix) {
	T, N, H := len(l.hs), len(l.hs[0]), len(l.hs[0][0])

	ds, _ := l.Softmax.Backward(da) // (T, N)
	dt := Expand(ds, T, N, H)       // (T, N, H)
	dhs := tensor.Mul(dt, l.hr)     // (T, N, H)
	dhr := tensor.Mul(dt, l.hs)     // (T, N, H)
	dh := tensor.Sum(dhr)           // (N, H)

	return dhs, dh // (T, N, H), (N, H)
}

func Expand(ds matrix.Matrix, T, N, H int) []matrix.Matrix {
	out := make([]matrix.Matrix, T)
	for i := 0; i < T; i++ {
		out[i] = matrix.New(vector.T(ds[i])...).Broadcast(N, H) // (T, N) -> (T, (1, N)) -> (T, (N, 1)) -> (T, (N, H))
	}

	return out
}
