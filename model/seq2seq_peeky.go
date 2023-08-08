package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
)

type PeekySeq2Seq struct {
	Seq2Seq
}

func NewPeekySeq2Seq(c *Seq2SeqConfig, s ...rand.Source) *PeekySeq2Seq {
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
	}, s[0])

	decoder := NewPeekyDecoder(&DecoderConfig{
		VocabSize:   V,
		WordVecSize: D,
		HiddenSize:  H,
		WeightInit:  c.WeightInit,
	}, s[0])

	return &PeekySeq2Seq{
		Seq2Seq: Seq2Seq{
			Encoder: encoder,
			Decoder: decoder,
			Softmax: &layer.TimeSoftmaxWithLoss{},
			Source:  s[0],
		},
	}
}
