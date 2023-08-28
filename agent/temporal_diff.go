package agent

import (
	"fmt"
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

func (a *TemporalDiffAgent) GetAction(state fmt.Stringer) int {
	s := state.String()
	if _, ok := a.Pi[s]; !ok {
		a.Pi[s] = a.RandomActions
	}

	probs := make([]float64, a.ActionSize)
	for i, p := range a.Pi[s] {
		probs[i] = p
	}

	return vector.Choice(probs, a.Source)
}

func (a *TemporalDiffAgent) Eval(state fmt.Stringer, reward float64, next fmt.Stringer, done bool) {
	var nextV float64
	if !done {
		nextV = a.V[next.String()]
	}

	target := reward + a.Gamma*nextV
	a.V[state.String()] += a.Alpha * (target - a.V[state.String()])
}
