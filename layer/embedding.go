package layer

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
)

type Embedding struct {
	W   matrix.Matrix // params
	DW  matrix.Matrix // grads
	idx []int
}

func (l *Embedding) Params() []matrix.Matrix      { return []matrix.Matrix{l.W} }
func (l *Embedding) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW} }
func (l *Embedding) SetParams(p ...matrix.Matrix) { l.W = p[0] }

func (l *Embedding) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	idx := make([]int, 0)
	for _, i := range x[0] {
		idx = append(idx, int(i))
	}
	l.idx = idx

	out := matrix.New()
	for _, i := range l.idx {
		out = append(out, l.W[i])
	}

	return out
}

func (l *Embedding) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	l.DW = matrix.Zero(len(l.W), len(l.W[0]))

	for i, v := range l.idx {
		// NOTE: l.DW[v] = vector.Add(l.DW[v], dout[v]) ?
		l.DW[v] = vector.Add(l.DW[v], dout[i])
	}

	return matrix.New(), matrix.New()
}
