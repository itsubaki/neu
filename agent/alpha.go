package agent

import (
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type AlphaAgent struct {
	Epsilon float64
	Alpha   float64
	Qs      []float64
	RNG     *rand.Rand
}

func (a *AlphaAgent) GetAction() int {
	if a.Epsilon > a.RNG.Float64() {
		return a.RNG.Intn(len(a.Qs))
	}

	return vector.Argmax(a.Qs)
}

func (a *AlphaAgent) Update(action int, reward float64) {
	a.Qs[action] += (reward - a.Qs[action]) * a.Alpha
}
