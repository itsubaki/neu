package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
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

func (m *Sequential) Forward(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	y := m.Predict(x, opts...)
	return m.Layer[len(m.Layer)-1].Forward(y, t, opts...)
}

func (m *Sequential) Backward(x, t matrix.Matrix) matrix.Matrix {
	dout := matrix.New([]float64{1})
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout, _ = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *Sequential) Optimize(opt Optimizer) [][]matrix.Matrix {
	updated := opt.Update(m.Params(), m.Grads())
	for i, l := range m.Layer {
		l.SetParams(updated[i])
	}

	return updated
}

func (m *Sequential) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		params = append(params, l.Params())
	}

	return params
}

func (m *Sequential) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		grads = append(grads, l.Grads())
	}

	return grads
}
