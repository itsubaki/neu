package agent

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/vector"
)

type QLearningAgent struct {
	Gamma      float64
	Alpha      float64
	Epsilon    float64
	ActionSize int
	Q          DefaultMap[float64]
	Source     randv2.Source
}

func (a *QLearningAgent) GetAction(state fmt.Stringer) int {
	g := randv2.New(a.Source)
	if a.Epsilon > g.Float64() {
		return g.IntN(a.ActionSize)
	}

	qs := qstate(a.Q, state.String(), a.ActionSize)
	return vector.Argmax(qs)
}

func (a *QLearningAgent) Update(state fmt.Stringer, action int, reward float64, next fmt.Stringer, done bool) {
	var nextqmax float64
	if !done {
		nextqs := qstate(a.Q, next.String(), a.ActionSize)
		nextqmax = vector.Max(nextqs)
	}

	target := reward + a.Gamma*nextqmax
	s := StateAction{State: state.String(), Action: action}.String()
	a.Q[s] += a.Alpha * (target - a.Q[s])
}
