package optimizer

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer/hook"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.CBOW)(nil)
	_ Model = (*model.RNNLM)(nil)
	_ Model = (*model.LSTMLM)(nil)
	_ Model = (*model.RNNLMGen)(nil)
	_ Model = (*model.Seq2Seq)(nil)
)

var (
	_ Hook = hook.WeightDecay(0.1)
	_ Hook = hook.GradsClipping(1.0)
)

type Model interface {
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
	SetParams(p [][]matrix.Matrix)
}

type Hook func(params, grads [][]matrix.Matrix) [][]matrix.Matrix

func ZeroLike(param [][]matrix.Matrix) [][]matrix.Matrix {
	z := make([][]matrix.Matrix, len(param))
	for i := range param {
		z[i] = make([]matrix.Matrix, len(param[i]))
		for j := range param[i] {
			z[i][j] = matrix.Zero(param[i][j].Dim())
		}
	}

	return z
}
