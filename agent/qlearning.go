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
	RNG        *rand.Rand
}

func (a *QLearningAgent) GetAction(state fmt.Stringer) int {
	if a.Epsilon > a.RNG.Float64() {
		return a.RNG.Intn(a.ActionSize)
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
