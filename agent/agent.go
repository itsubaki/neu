package agent

import (
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type Agent struct {
	Epsilon float64
	Qs      []float64
	Ns      []float64
	RNG     *rand.Rand
}

func (a *Agent) GetAction() int {
	if a.Epsilon > a.RNG.Float64() {
		return a.RNG.Intn(len(a.Qs))
	}

	return vector.Argmax(a.Qs)
}

func (a *Agent) Update(action int, reward float64) {
	a.Ns[action]++
	a.Qs[action] += (reward - a.Qs[action]) / a.Ns[action]
}
