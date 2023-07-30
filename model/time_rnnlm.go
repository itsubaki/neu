package model

import (
	"math"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type TimeRNNLMConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
}

type TimeRNNLM struct {
	Layer []TimeLayer
}

func NewTimeRNNLM(c *TimeRNNLMConfig, s ...rand.Source) *TimeRNNLM {
	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layers
	// TimeEmbedding -> TimeRNN -> TimeAffine -> TimeSoftmaxWithLoss
	layers := []TimeLayer{
		&layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		&layer.TimeRNN{
			Wx:       matrix.Randn(D, H, s[0]).MulC(math.Sqrt(float64(D))),
			Wh:       matrix.Randn(H, H, s[0]).MulC(math.Sqrt(float64(H))),
			B:        matrix.Zero(1, H),
			Stateful: true,
		},
		&layer.TimeAffine{
			W: matrix.Randn(H, V, s[0]).MulC(math.Sqrt(float64(H))),
			B: matrix.Zero(1, V),
		},
		&layer.TimeSoftmaxWithLoss{},
	}

	return &TimeRNNLM{
		Layer: layers,
	}
}

func (m *TimeRNNLM) Forward(xs, ts []matrix.Matrix) matrix.Matrix {
	for _, l := range m.Layer[:len(m.Layer)-1] {
		xs = l.Forward(xs, nil)
	}

	return m.Layer[len(m.Layer)-1].Forward(xs, ts)[0]
}

func (m *TimeRNNLM) Backward() []matrix.Matrix {
	dout := []matrix.Matrix{matrix.New([]float64{1})}
	for i := len(m.Layer) - 1; i > -1; i-- {
		dout = m.Layer[i].Backward(dout)
	}

	return dout
}

func (m *TimeRNNLM) Layers() []TimeLayer {
	return m.Layer
}

func (m *TimeRNNLM) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		params = append(params, l.Params())
	}

	return params
}

func (m *TimeRNNLM) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.Layer {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *TimeRNNLM) ResetState() {
	for _, l := range m.Layer {
		l.ResetState()
	}
}
