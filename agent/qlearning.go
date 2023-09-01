package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type QLearningAgent struct {
	Gamma      float64
	Alpha      float64
	Epsilon    float64
	ActionSize int
	Q          map[string]float64
	Source     rand.Source
}

func (a *QLearningAgent) GetAction(state fmt.Stringer) int {
	rng := rand.New(a.Source)
	if a.Epsilon > rng.Float64() {
		return rng.Intn(a.ActionSize)
	}

	qs := qstate(a.Q, state.String(), a.ActionSize)
	return vector.Argmax(qs)
}

func (a *QLearningAgent) Update(state fmt.Stringer, action int, reward float64, next fmt.Stringer, done bool) {
	nextqmax, s, n := 0.0, state.String(), next.String()
	if !done {
		nextqs := qstate(a.Q, n, a.ActionSize)
		nextqmax = vector.Max(nextqs)
	}

	target := reward + a.Gamma*nextqmax
	key := StateAction{State: s, Action: action}.String()
	a.Q[key] += a.Alpha * (target - a.Q[key])
}
