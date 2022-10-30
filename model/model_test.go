package model_test

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/model"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
)

type Model interface {
	Forward(x, t matrix.Matrix) matrix.Matrix
	Layers() []model.Layer
}

func numericalGrads(m Model, x, t matrix.Matrix) [][]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return m.Forward(x, t)[0][0]
	}

	grad := func(f func(x ...float64) float64, x matrix.Matrix) matrix.Matrix {
		out := make(matrix.Matrix, 0)
		for _, r := range x {
			out = append(out, numerical.Gradient(f, r))
		}

		return out
	}

	// gradient
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layers() {
		g := make([]matrix.Matrix, 0)
		for _, p := range l.Params() {
			g = append(g, grad(lossW, p))
		}

		grads = append(grads, g)
	}

	return grads
}
