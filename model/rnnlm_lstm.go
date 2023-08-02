package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type LSTMLMConfig struct {
	VocabSize    int
	WordVecSize  int
	HiddenSize   int
	WeightInit   WeightInit
	DropoutRatio float64
}

type LSTMLM struct {
	Layer  []TimeLayer
	Source rand.Source
}

func NewLSTMLM(c *LSTMLMConfig, s ...rand.Source) *LSTMLM {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layer
	layers := []TimeLayer{
		&layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeLSTM{
			Wx:       matrix.Randn(D, 4*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeLSTM{
			Wx:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeAffine{
			W: matrix.Randn(D, V, s[0]).MulC(c.WeightInit(H)),
			B: matrix.Zero(1, V),
		},
		&layer.TimeSoftmaxWithLoss{},
	}

	return &LSTMLM{
		Layer:  layers,
		Source: s[0],
	}
}

func (m *LSTMLM) Predict(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		xs = l.Forward(xs, nil, opts...)
	}

	return xs
}

func (m *LSTMLM) Forward(xs, ts []matrix.Matrix) matrix.Matrix {
	opts := layer.Opts{Train: true, Source: m.Source}
	ys := m.Predict(xs, opts)
	return m.Layer[len(m.Layer)-1].Forward(ys, ts, opts)[0]
}

func (m *LSTMLM) Backward() []matrix.Matrix {
	dout := []matrix.Matrix{matrix.New([]float64{1})}
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *LSTMLM) Layers() []TimeLayer {
	return m.Layer
}

func (m *LSTMLM) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		params = append(params, l.Params())
	}

	return params
}

func (m *LSTMLM) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *LSTMLM) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}

func (m *LSTMLM) ResetState() {
	for _, l := range m.Layer {
		l.ResetState()
	}
}
