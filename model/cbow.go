package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

type CBOWConfig struct {
	VocabSize  int
	HiddenSize int
}

type CBOW struct {
	Win0, Win1, Wout Layer
	Loss             Layer
	Source           randv2.Source
}

func NewCBOW(c *CBOWConfig, s ...randv2.Source) *CBOW {
	if len(s) == 0 {
		s = append(s, rand.MustNewSource())
	}

	// size
	V, H := c.VocabSize, c.HiddenSize

	// model
	return &CBOW{
		Win0:   &layer.Dot{W: matrix.Randn(V, H, s[0]).MulC(0.01)},
		Win1:   &layer.Dot{W: matrix.Randn(V, H, s[0]).MulC(0.01)},
		Wout:   &layer.Dot{W: matrix.Randn(H, V, s[0]).MulC(0.01)},
		Loss:   &layer.SoftmaxWithLoss{},
		Source: s[0],
	}
}

func (m *CBOW) Predict(xs []matrix.Matrix) matrix.Matrix {
	c0, c1 := matrix.New(), matrix.New()
	for _, c := range xs {
		c0, c1 = append(c0, c[0]), append(c1, c[1])
	}

	h0 := m.Win0.Forward(c0, nil)
	h1 := m.Win1.Forward(c1, nil)
	h := h0.Add(h1).MulC(0.5)
	score := m.Wout.Forward(h, nil)
	return score
}

func (m *CBOW) Forward(contexts []matrix.Matrix, target matrix.Matrix) matrix.Matrix {
	score := m.Predict(contexts)
	return m.Loss.Forward(score, target)
}

func (m *CBOW) Backward() {
	dout, _ := m.Loss.Backward(matrix.New([]float64{1}))
	da, _ := m.Wout.Backward(dout)
	da = da.MulC(0.5)
	m.Win1.Backward(da)
	m.Win0.Backward(da)
}

func (m *CBOW) Summary() []string {
	return []string{
		fmt.Sprintf("%T", m),
		m.Win0.String(),
		m.Win1.String(),
		m.Wout.String(),
		m.Loss.String(),
	}
}

func (m *CBOW) Layers() []Layer {
	return []Layer{
		m.Win0,
		m.Win1,
		m.Wout,
		m.Loss,
	}
}

func (m *CBOW) Params() [][]matrix.Matrix {
	return [][]matrix.Matrix{
		m.Win0.Params(),
		m.Win1.Params(),
		m.Wout.Params(),
	}
}

func (m *CBOW) Grads() [][]matrix.Matrix {
	return [][]matrix.Matrix{
		m.Win0.Grads(),
		m.Win1.Grads(),
		m.Wout.Grads(),
	}
}

func (m *CBOW) SetParams(p [][]matrix.Matrix) {
	m.Win0.SetParams(p[0]...)
	m.Win1.SetParams(p[1]...)
	m.Wout.SetParams(p[2]...)
}
