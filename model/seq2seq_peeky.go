package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
)

type PeekySeq2Seq struct {
	Seq2Seq
}

func NewPeekySeq2Seq(c *RNNLMConfig, s ...rand.Source) *PeekySeq2Seq {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
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
