package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
	"github.com/itsubaki/neu/math/vector"
)

type RNNLMGen struct {
	GRULM
}

func NewRNNLMGen(c *LSTMLMConfig, s ...randv2.Source) *RNNLMGen {
	return &RNNLMGen{
		GRULM: *NewGRULM(c, s...),
	}
}

func (g *RNNLMGen) Generate(startID int, skipIDs []int, length int) []int {
	wordIDs := make([]int, 0)

	x := startID
	for {
		if len(wordIDs) >= length {
			break
		}

		// predict
		xs := []matrix.Matrix{matrix.New([]float64{float64(x)})}
		score := tensor.Flatten(g.Predict(xs))
		p := activation.Softmax(score)

		// sample
		sampled := vector.Choice(p, g.Source)
		if vector.Contains(sampled, skipIDs) {
			continue
		}

		x = sampled
		wordIDs = append(wordIDs, x)
	}

	return wordIDs
}

func (m *RNNLMGen) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
