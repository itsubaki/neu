package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type Sequential struct {
	Layer  []Layer
	Source randv2.Source
}

func NewSequential(layer []Layer, s randv2.Source) *Sequential {
	return &Sequential{
		Layer:  layer,
		Source: s,
	}
}

func (m *Sequential) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		x = l.Forward(x, nil, opts...)
	}

	return x
}

func (m *Sequential) Forward(x, t matrix.Matrix) matrix.Matrix {
	opts := layer.Opts{Train: true, Source: m.Source}
	y := m.Predict(x, opts)
	return m.Layer[len(m.Layer)-1].Forward(y, t, opts)
}

func (m *Sequential) Backward() matrix.Matrix {
	dout := matrix.New([]float64{1})
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout, _ = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *Sequential) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
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

func (m *Sequential) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}
