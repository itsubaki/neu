package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

var (
	_ TimeLayer = (*layer.TimeAffine)(nil)
	_ TimeLayer = (*layer.TimeRNN)(nil)
)

type TimeLayer interface {
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
	SetState(h ...matrix.Matrix)
	ResetState()
	Forward(xs, ys []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix
	Backward(dout []matrix.Matrix) []matrix.Matrix
}
