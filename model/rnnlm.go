package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

type RNNLMConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
	WeightInit  WeightInit
}

type RNNLM struct {
	Layer  []TimeLayer
	Source randv2.Source
}

func NewRNNLM(c *RNNLMConfig, s ...randv2.Source) *RNNLM {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layer
	// TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss
	layers := []TimeLayer{
		&layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		&layer.TimeRNN{
			Wx:       matrix.Randn(D, H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, H),
			Stateful: true,
		},
		&layer.TimeAffine{
			W: matrix.Randn(H, V, s[0]).MulC(c.WeightInit(H)),
			B: matrix.Zero(1, V),
		},
		&layer.TimeSoftmaxWithLoss{},
	}

	return &RNNLM{
		Layer:  layers,
		Source: s[0],
	}
}

func (m *RNNLM) Predict(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		xs = l.Forward(xs, nil, opts...)
	}

	return xs
}

func (m *RNNLM) Forward(xs, ts []matrix.Matrix) []matrix.Matrix {
	opts := layer.Opts{Train: true, Source: m.Source}
	ys := m.Predict(xs, opts)
	return m.Layer[len(m.Layer)-1].Forward(ys, ts, opts)
}

func (m *RNNLM) Backward() []matrix.Matrix {
	dout := []matrix.Matrix{{{1}}}
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *RNNLM) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}

func (m *RNNLM) Layers() []TimeLayer {
	return m.Layer
}

func (m *RNNLM) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		params = append(params, l.Params())
	}

	return params
}

func (m *RNNLM) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *RNNLM) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}

func (m *RNNLM) ResetState() {
	for _, l := range m.Layer {
		l.ResetState()
	}
}
