package layer

import (
	"fmt"

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
func (l *Embedding) String() string {
	a, b := l.W.Dim()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}

func (l *Embedding) Forward(idx, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.idx = make([]int, len(idx)) // idx(N, 1)
	for i := range idx {
		l.idx[i] = int(idx[i][0])
	}

	out := matrix.New()
	for _, i := range l.idx {
		out = append(out, l.W[i]) // W(13, 16)
	}

	return out // (128, 16)
}

func (l *Embedding) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	l.DW = matrix.ZeroLike(l.W) // DW(V, D) (13, 16)
	for i, v := range l.idx {   // idx(N, 1) (128, 1)
		l.DW[v] = vector.Add(l.DW[v], dout[i]) // dout(N, D) (128, 16)
	}

	return nil, nil
}
