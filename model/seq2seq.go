package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type Seq2SeqConfig struct {
	RNNLMConfig
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

func (m *Seq2Seq) Forward(xs, ts matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	decoxs, decots := Split(ts)
	h := m.Encoder.Forward([]matrix.Matrix{xs}, opts...)
	score := m.Decoder.Forward([]matrix.Matrix{decoxs}, h, opts...)
	loss := m.Softmax.Forward(score, []matrix.Matrix{decots}, opts...)
	return loss
}

func (m *Seq2Seq) Backward(dout []matrix.Matrix) []matrix.Matrix {
	dout = m.Softmax.Backward(dout)
	dh := m.Decoder.Backward(dout)
	dout = m.Encoder.Backward(dh)
	return dout
}

func (m *Seq2Seq) Generate(xs []matrix.Matrix, startID, length int) []int {
	h := m.Encoder.Forward(xs)
	sampeld := m.Decoder.Generate(h, startID, length)
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

func Split(x matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	xs, ts := matrix.New(), matrix.New()
	for _, r := range x {
		xs, ts = append(xs, []float64{r[len(r)-1]}), append(ts, r[1:])
	}

	return xs, ts
}
