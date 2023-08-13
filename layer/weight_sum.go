package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
	"github.com/itsubaki/neu/math/vector"
)

type WeightSum struct {
	hs, ar []matrix.Matrix
}

func (l *WeightSum) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *WeightSum) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *WeightSum) SetParams(p ...matrix.Matrix) {}
func (l *WeightSum) String() string               { return fmt.Sprintf("%T", l) }

func (l *WeightSum) Forward(hs []matrix.Matrix, a matrix.Matrix) matrix.Matrix {
	T, N, H := len(hs), len(hs[0]), len(hs[0][0])

	ar := make([]matrix.Matrix, T)
	for i := 0; i < T; i++ {
		ar[i] = matrix.New(vector.T(a[i])...).Broadcast(N, H) // (1, N) -> (N, 1) -> (N, H)
	}

	l.hs, l.ar = hs, ar                   // (T, N, H) (T, N, H)
	return tensor.Sum(tensor.Mul(hs, ar)) // (T, N, H) -> (N, H)
}

func (l *WeightSum) Backward(dc matrix.Matrix) ([]matrix.Matrix, matrix.Matrix) {
	T := len(l.hs)

	dt := tensor.Repeat(dc, T)  // (N, H) -> (T, N, H)
	dar := tensor.Mul(dt, l.hs) // (T, N, H)
	dhs := tensor.Mul(dt, l.ar) // (T, N, H)

	da := make(matrix.Matrix, T)
	for i := 0; i < T; i++ {
		da[i] = dar[i].SumAxis1() // (N, H) -> (1, N)
	}

	return dhs, da // (T, N, H), (T, N)
}
