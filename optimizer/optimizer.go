package optimizer

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
	_ Hook  = weight.Decay(0.1)
)

type Model interface {
	Layers() []model.Layer
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
}

type Hook func(params, grads [][]matrix.Matrix) [][]matrix.Matrix
