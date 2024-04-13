package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

type CBOWNegativeSamplingConfig struct {
	CBOWConfig
	Corpus     []int
	WindowSize int
	SampleSize int
	Power      float64
}

type CBOWNegativeSampling struct {
	Embedding []Layer
	Loss      Layer
	s         randv2.Source
}

func NewCBOWNegativeSampling(c CBOWNegativeSamplingConfig, s ...randv2.Source) *CBOWNegativeSampling {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	// size
	V, H := c.VocabSize, c.HiddenSize

	// Weight
	Win := matrix.Randn(V, H, s[0]).MulC(0.01)
	Wout := matrix.Randn(V, H, s[0]).MulC(0.01)

	// layer
	embed := make([]Layer, 2*c.WindowSize)
	for i := 0; i < len(embed); i++ {
		embed[i] = &layer.Embedding{W: Win}
	}
	loss := layer.NewNegativeSamplingLoss(Wout, c.Corpus, c.Power, c.SampleSize, s[0])

	return &CBOWNegativeSampling{
		Embedding: embed,
		Loss:      loss,
		s:         s[0],
	}
}

func (m *CBOWNegativeSampling) Predict(x matrix.Matrix, _ ...layer.Opts) matrix.Matrix {
	h := matrix.Zero(1, 1)
	for i, l := range m.Embedding {
		h = l.Forward(matrix.Column(x, i), nil).Add(h)
	}
	h = h.MulC(1.0 / float64(len(m.Embedding)))

	return h
}

func (m *CBOWNegativeSampling) Forward(contexts, target matrix.Matrix) matrix.Matrix {
	h := m.Predict(contexts)
	return m.Loss.Forward(h, target)
}

func (m *CBOWNegativeSampling) Backward() matrix.Matrix {
	dout, _ := m.Loss.Backward(matrix.New([]float64{1}))
	dout = dout.MulC(1.0 / float64(len(m.Embedding)))
	for _, l := range m.Embedding {
		l.Backward(dout)
	}

	return nil
}

func (m *CBOWNegativeSampling) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}

func (m *CBOWNegativeSampling) Layers() []Layer {
	return append(m.Embedding, m.Loss)
}

func (m *CBOWNegativeSampling) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layers() {
		params = append(params, l.Params())
	}

	return params
}

func (m *CBOWNegativeSampling) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layers() {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *CBOWNegativeSampling) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}
