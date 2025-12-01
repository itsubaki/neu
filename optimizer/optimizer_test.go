package optimizer_test

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

var _ optimizer.Model = (*TestModel)(nil)

type TestModel struct {
	params [][]matrix.Matrix
	grads  [][]matrix.Matrix
}

func (m *TestModel) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	return nil
}

func (m *TestModel) Forward(x, t matrix.Matrix) matrix.Matrix {
	return nil
}

func (m *TestModel) Backward(x, t matrix.Matrix) matrix.Matrix {
	return nil
}

func (m *TestModel) Layers() []model.Layer {
	return []model.Layer{&layer.ReLU{}}
}

func (m *TestModel) Params() [][]matrix.Matrix {
	return m.params
}

func (m *TestModel) Grads() [][]matrix.Matrix {
	return m.grads
}

func (m *TestModel) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}
