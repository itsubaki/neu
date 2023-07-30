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

func (l *Embedding) Forward(idx, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.idx = vector.Int(idx[0])

	out := matrix.New()
	for _, i := range l.idx {
		out = append(out, l.W[i])
	}

	return out
}

func (l *Embedding) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	l.DW = matrix.Zero(len(l.W), len(l.W[0]))

	for i, v := range l.idx {
		l.DW[v] = vector.Add(l.DW[v], dout[i])
	}

	return nil, nil
}

func (l *Embedding) String() string {
	a, b := l.W.Dimension()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}
