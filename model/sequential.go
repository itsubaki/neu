package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
)

type Sequential struct {
	Layer []Layer
}

func (m *Sequential) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		x = l.Forward(x, nil, opts...)
	}

	return x
}

func (m *Sequential) Loss(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	y := m.Predict(x, opts...)
	return m.Layer[len(m.Layer)-1].Forward(y, t, opts...)
}

func (m *Sequential) Gradient(x, t matrix.Matrix) [][]matrix.Matrix {
	// forward
	m.Loss(x, t, layer.Opts{Train: true})

	// backward
	dout := matrix.New([]float64{1})
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout, _ = m.Layer[i].Backward(dout)
	}

	// gradient
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *Sequential) NumericalGradient(x, t matrix.Matrix) [][]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return m.Loss(x, t, layer.Opts{Train: true})[0][0]
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
	for _, l := range m.Layer {
		g := make([]matrix.Matrix, 0)
		for _, p := range l.Params() {
			g = append(g, grad(lossW, p))
		}

		grads = append(grads, g)
	}

	return grads
}

func (m *Sequential) Optimize(opt Optimizer, grads [][]matrix.Matrix) {
	// params
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		params = append(params, l.Params())
	}

	// update
	updated := opt.Update(params, grads)
	for i, l := range m.Layer {
		l.SetParams(updated[i])
	}
}
