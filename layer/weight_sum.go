package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type WeightSum struct {
	hs, ar []matrix.Matrix
}

func (l *WeightSum) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *WeightSum) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *WeightSum) SetParams(p ...matrix.Matrix) {}

func (l *WeightSum) Forward(hs, a []matrix.Matrix) []matrix.Matrix {
	T, N, H := len(hs), len(hs[0]), len(hs[0][0])
	ar := make([]matrix.Matrix, T)
	for i := 0; i < T; i++ {
		ar[i] = a[i].T().Broadcast(N, H) // (1, N) -> (N, 1) -> (N, H)
	}
	l.hs, l.ar = hs, ar // (T, N, H) (T, N, H)

	c := make([]matrix.Matrix, T)
	for i := 0; i < T; i++ {
		t := hs[i].Mul(ar[i]) // (N, H)
		c[i] = t.SumAxis0()   // (1, H)
	}

	return c // (T, N)
}

func (l *WeightSum) Backward(dc matrix.Matrix) ([]matrix.Matrix, []matrix.Matrix) {
	T := len(l.hs)
	dt := matrix.Repeat(dc, T) // (N, H) -> (T, N, H)
	dar := TimeMul(dt, l.hs)   // (T, N, H)
	dhs := TimeMul(dt, l.ar)   // (T, N, H)

	da := make([]matrix.Matrix, T)
	for i := 0; i < T; i++ {
		da[i] = dar[i].SumAxis1() // (1, N)
	}

	return dhs, da // (T, N, H), (T, N)
}

func (l *WeightSum) String() string {
	return fmt.Sprintf("%T", l)
}

func TimeMul(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = x[i].Mul(y[i])
	}

	return out
}
