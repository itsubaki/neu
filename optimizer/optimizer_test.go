package optimizer_test

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

var _ optimizer.Model = (*Test)(nil)

type Test struct {
	params [][]matrix.Matrix
	grads  [][]matrix.Matrix
}

func (m *Test) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix { return matrix.New() }
func (m *Test) Forward(x, t matrix.Matrix) matrix.Matrix                  { return matrix.New() }
func (m *Test) Backward(x, t matrix.Matrix) matrix.Matrix                 { return matrix.New() }
func (m *Test) Layers() []model.Layer                                     { return make([]model.Layer, 0) }
func (m *Test) Params() [][]matrix.Matrix                                 { return m.params }
func (m *Test) Grads() [][]matrix.Matrix                                  { return m.grads }
