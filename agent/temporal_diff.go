package agent

import (
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type TemporalDiffAgent struct {
	Gamma         float64
	Alpha         float64
	ActionSize    int
	RandomActions RandomActions
	Pi            map[string]RandomActions
	V             map[string]float64
	Source        rand.Source
}

func (a *TemporalDiffAgent) GetAction(state string) int {
	if _, ok := a.Pi[state]; !ok {
		a.Pi[state] = a.RandomActions
	}
	probs := make([]float64, a.ActionSize)
	for i, p := range a.Pi[state] {
		probs[i] = p
	}

	return vector.Choice(probs, a.Source)
}

func (a *TemporalDiffAgent) Eval(state string, reward float64, next string, done bool) {
	var nextV float64
	if !done {
		nextV = a.V[next]
	}

	target := reward + a.Gamma*nextV
	a.V[state] += a.Alpha * (target - a.V[state])
}
