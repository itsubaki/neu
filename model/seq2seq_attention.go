package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

type AttentionSeq2Seq struct {
	Encoder *AttentionEncoder
	Decoder *AttentionDecoder
	Softmax *layer.TimeSoftmaxWithLoss
	Source  randv2.Source
}

func NewAttentionSeq2Seq(c *RNNLMConfig, s ...randv2.Source) *AttentionSeq2Seq {
	if len(s) == 0 {
		s = append(s, rand.MustNewSource())
	}

	return &AttentionSeq2Seq{
		Encoder: NewAttentionEncoder(c, s[0]),
		Decoder: NewAttentionDecoder(c, s[0]),
		Softmax: &layer.TimeSoftmaxWithLoss{},
		Source:  s[0],
	}
}

func (m *AttentionSeq2Seq) Forward(xs, ts []matrix.Matrix) []matrix.Matrix {
	dxs, dts := ts[:len(ts)-1], ts[1:]
	h := m.Encoder.Forward(xs)
	score := m.Decoder.Forward(dxs, h)
	loss := m.Softmax.Forward(score, dts)
	return loss
}

func (m *AttentionSeq2Seq) Backward() {
	dout := []matrix.Matrix{{{1}}}
	dscore := m.Softmax.Backward(dout)
	dh := m.Decoder.Backward(dscore)
	m.Encoder.Backward(dh)
}

func (m *AttentionSeq2Seq) Generate(xs []matrix.Matrix, startID, length int) []int {
	h := m.Encoder.Forward(xs)
	sampeld := m.Decoder.Generate(h, startID, length)
	return sampeld
}

func (m *AttentionSeq2Seq) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	s = append(s, m.Encoder.Summary()...)
	s = append(s, m.Decoder.Summary()...)
	s = append(s, m.Softmax.String())
	return s
}

func (m *AttentionSeq2Seq) Layers() []TimeLayer {
	layers := make([]TimeLayer, 0)
	layers = append(layers, m.Encoder.Layers()...)
	layers = append(layers, m.Decoder.Layers()...)
	layers = append(layers, m.Softmax)
	return layers
}

func (m *AttentionSeq2Seq) Params() [][]matrix.Matrix {
	return [][]matrix.Matrix{
		m.Encoder.Params(),
		m.Decoder.Params(),
	}
}

func (m *AttentionSeq2Seq) Grads() [][]matrix.Matrix {
	return [][]matrix.Matrix{
		m.Encoder.Grads(),
		m.Decoder.Grads(),
	}
}

func (m *AttentionSeq2Seq) SetParams(p [][]matrix.Matrix) {
	m.Encoder.SetParams(p[0]...)
	m.Decoder.SetParams(p[1]...)
}
