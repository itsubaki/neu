package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type EmbeddingDot struct {
	Embedding Embedding
	H, W      matrix.Matrix
}

func (l *EmbeddingDot) Params() []matrix.Matrix      { return []matrix.Matrix{l.Embedding.W} }
func (l *EmbeddingDot) Grads() []matrix.Matrix       { return []matrix.Matrix{l.Embedding.DW} }
func (l *EmbeddingDot) SetParams(p ...matrix.Matrix) { l.Embedding.W = p[0] }
func (l *EmbeddingDot) String() string {
	a, b := l.Embedding.W.Dim()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}

func (l *EmbeddingDot) Forward(h, idx matrix.Matrix, _ ...Opts) matrix.Matrix {
	targetW := l.Embedding.Forward(idx, nil)
	out := targetW.Mul(h).SumAxis1()

	l.H, l.W = h, targetW
	return matrix.New(out)
}

func (l *EmbeddingDot) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	h, targetW := l.H, l.W
	dout = matrix.Reshape(dout, len(dout), 1)

	dtargetW := h.Mul(dout)        // Broadcast
	dh := targetW.Mul(dout)        // Broadcast
	l.Embedding.Backward(dtargetW) //
	return dh, nil
}
