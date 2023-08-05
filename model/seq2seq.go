package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type Seq2SeqConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
	WeightInit  WeightInit
}

type Seq2Seq struct {
	Encoder *Encoder
	Decoder *Decoder
	Softmax *layer.TimeSoftmaxWithLoss
}

func NewSeq2Seq(c *Seq2SeqConfig, s ...rand.Source) *Seq2Seq {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layer
	encoder := NewEncoder(&EncoderConfig{
		VocabSize:   V,
		WordVecSize: D,
		HiddenSize:  H,
		WeightInit:  c.WeightInit,
	})

	decoder := NewDecoder(&DecoderConfig{
		VocabSize:   V,
		WordVecSize: D,
		HiddenSize:  H,
		WeightInit:  c.WeightInit,
	})

	return &Seq2Seq{
		Encoder: encoder,
		Decoder: decoder,
		Softmax: &layer.TimeSoftmaxWithLoss{},
	}
}

func (m *Seq2Seq) Forward(xs, ts []matrix.Matrix, opts ...layer.Opts) float64 {
	dxs, dts := []matrix.Matrix{ts[1]}, ts[1:]     // dxs(1, 128, 1), dts(4, 128, 1)
	h := m.Encoder.Forward(xs, opts...)            // h(128, 128)
	score := m.Decoder.Forward(dxs, h, opts...)    // score(1, 128, 13) NOTE: dts.T != score.T
	loss := m.Softmax.Forward(score, dts, opts...) // (1, 1, 1)
	return loss[0][0][0]
}

func (m *Seq2Seq) Backward() {
	dout := []matrix.Matrix{matrix.New([]float64{1})} // (1, 1, 1)
	dout = m.Softmax.Backward(dout)                   // (1, 128, 13)
	dh := m.Decoder.Backward(dout)                    // (128, 128)
	m.Encoder.Backward(dh)                            // (0, 0, 0)
}

func (m *Seq2Seq) Generate(xs []matrix.Matrix, startID, length int) []int {
	h := m.Encoder.Forward(xs)                        // xs(7, 1, 1), h(1, 128)
	sampeld := m.Decoder.Generate(h, startID, length) //
	return sampeld
}

func (m *Seq2Seq) Params() [][]matrix.Matrix {
	params := make([][]matrix.Matrix, 2)
	params[0] = m.Encoder.Params()
	params[1] = m.Decoder.Params()
	return params
}

func (m *Seq2Seq) Grads() [][]matrix.Matrix {
	grads := make([][]matrix.Matrix, 2)
	grads[0] = m.Encoder.Grads()
	grads[1] = m.Decoder.Grads()
	return grads
}

func (m *Seq2Seq) SetParams(p [][]matrix.Matrix) {
	m.Encoder.SetParams(p[0]...)
	m.Decoder.SetParams(p[1]...)
}
