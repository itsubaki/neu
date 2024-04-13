package agent

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/vector"
)

type AlphaAgent struct {
	Epsilon float64
	Alpha   float64
	Qs      []float64
	Source  randv2.Source
}

func (a *AlphaAgent) GetAction() int {
	rng := randv2.New(a.Source)
	if a.Epsilon > rng.Float64() {
		return rng.IntN(len(a.Qs))
	}

	return vector.Argmax(a.Qs)
}

func (a *AlphaAgent) Update(action int, reward float64) {
	a.Qs[action] += (reward - a.Qs[action]) * a.Alpha
}
