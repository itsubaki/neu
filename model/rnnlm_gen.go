package model

import (
	"math/rand"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type RNNLMGen struct {
	LSTMLM
}

func NewRNNLMGen(c *LSTMLMConfig, s ...rand.Source) *RNNLMGen {
	return &RNNLMGen{
		LSTMLM: *NewLSTMLM(c, s...),
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
		score := Flatten(g.Predict(xs))
		p := activation.Softmax(score)

		// sample
		sampled := Choice(p, g.Source)
		if Contains(sampled, skipIDs) {
			continue
		}
		wordIDs = append(wordIDs, sampled)

		// next
		x = sampled
	}

	return wordIDs
}
