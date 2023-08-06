package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

type RNNMLGen struct {
	RNNLM
}

func NewRNNLMGen(c *RNNLMConfig, s ...rand.Source) *RNNMLGen {
	return &RNNMLGen{
		RNNLM: *NewRNNLM(c, s...),
	}
}

func (g *RNNMLGen) Generate(startID int, skipIDs []int, length int) []int {
	wordIDs := []int{startID}

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

func Flatten(m []matrix.Matrix) []float64 {
	flatten := make([]float64, 0)
	for _, s := range m {
		flatten = append(flatten, matrix.Flatten(s)...)
	}

	return flatten
}

func Contains[T comparable](v T, s []T) bool {
	for _, ss := range s {
		if v == ss {
			return true
		}
	}

	return false
}

func Choice(p []float64, s ...rand.Source) int {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	cumsum := make([]float64, len(p))
	var sum float64
	for i, prob := range p {
		sum += prob
		cumsum[i] = sum
	}

	r := rand.New(s[0]).Float64()
	for i, prop := range cumsum {
		if r <= prop {
			return i
		}
	}

	return -1
}
