package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type QLearningAgent struct {
	Gamma         float64
	Alpha         float64
	Epsilon       float64
	ActionSize    int
	RandomActions RandomActions
	Pi            map[string]RandomActions
	B             map[string]RandomActions
	Q             map[string]float64
	Source        rand.Source
}

func (a *QLearningAgent) GetAction(state fmt.Stringer) int {
	s := state.String()
	if _, ok := a.B[s]; !ok {
		a.B[s] = a.RandomActions
	}

	probs := make([]float64, a.ActionSize)
	for i, p := range a.B[s] {
		probs[i] = p
	}

	return vector.Choice(probs, a.Source)
}

func (a *QLearningAgent) Update(state fmt.Stringer, action int, reward float64, next fmt.Stringer, done bool) {
	nextqmax, s, n := 0.0, state.String(), next.String()
	if !done {
		nextqs := make([]float64, 0)
		for i := 0; i < a.ActionSize; i++ {
			key := StateAction{State: n, Action: i}.String()
			if _, ok := a.Q[key]; !ok {
				a.Q[key] = 0
			}

			nextqs = append(nextqs, a.Q[key])
		}

		nextqmax = vector.Max(nextqs)
	}

	target := reward + a.Gamma*nextqmax
	key := StateAction{State: s, Action: action}.String()
	a.Q[key] += a.Alpha * (target - a.Q[key])

	a.Pi[s] = greedyProps(a.Q, s, 0, a.ActionSize)
	a.B[s] = greedyProps(a.Q, s, a.Epsilon, a.ActionSize)
}