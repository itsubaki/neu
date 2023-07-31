package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type RNNLMConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
	WeightInit  WeightInit
}

type RNNLM struct {
	Layer []TimeLayer
}

func NewRNNLM(c *RNNLMConfig, s ...rand.Source) *RNNLM {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layers
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
		Layer: layers,
	}
}

func (m *RNNLM) Predict(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		xs = l.Forward(xs, nil, opts...)
	}

	return xs
}

func (m *RNNLM) Forward(xs, ts []matrix.Matrix) matrix.Matrix {
	ys := m.Predict(xs, layer.Opts{Train: true})
	return m.Layer[len(m.Layer)-1].Forward(ys, ts, layer.Opts{Train: true})[0]
}

func (m *RNNLM) Backward() []matrix.Matrix {
	dout := []matrix.Matrix{matrix.New([]float64{1})}
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout = m.Layer[i].Backward(dout)
	}

	return dout
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