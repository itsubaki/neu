package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/rand"
)

type PeekySeq2Seq struct {
	Seq2Seq
}

func NewPeekySeq2Seq(c *RNNLMConfig, s ...randv2.Source) *PeekySeq2Seq {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	return &PeekySeq2Seq{
		Seq2Seq: Seq2Seq{
			Encoder: NewEncoder(c, s[0]),
			Decoder: NewPeekyDecoder(c, s[0]),
			Softmax: &layer.TimeSoftmaxWithLoss{},
			Source:  s[0],
		},
	}
}

func (m *PeekySeq2Seq) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	s = append(s, m.Encoder.Summary()...)
	s = append(s, m.Decoder.Summary()...)
	s = append(s, m.Softmax.String())
	return s
}
