package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type Sequential struct {
	Layer []Layer
}

func NewSequential(layer ...Layer) *Sequential {
	return &Sequential{
		Layer: layer,
	}
}

func (m *Sequential) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		x = l.Forward(x, nil, opts...)
	}

	return x
}

func (m *Sequential) Forward(x, t matrix.Matrix) matrix.Matrix {
	y := m.Predict(x, layer.Opts{Train: true})
	return m.Layer[len(m.Layer)-1].Forward(y, t, layer.Opts{Train: true})
}

func (m *Sequential) Backward(x, t matrix.Matrix) matrix.Matrix {
	dout := matrix.New([]float64{1})
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout, _ = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *Sequential) Layers() []Layer {
	return m.Layer
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
