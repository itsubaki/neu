package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type MonteCarloAgent struct {
	Gamma          float64
	Epsilon        float64
	Alpha          float64
	ActionSize     int
	DefaultActions RandomActions
	Pi             DefaultMap[RandomActions]
	Q              DefaultMap[float64]
	Memory         []Memory
	Source         rand.Source
}

func (a *MonteCarloAgent) GetAction(state fmt.Stringer) int {
	probs := a.Pi.Get(state, a.DefaultActions).Probs()
	return vector.Choice(probs, a.Source)
}

func (a *MonteCarloAgent) Add(state fmt.Stringer, action int, reward float64) {
	a.Memory = append(a.Memory, NewMemory(state, action, reward, false))
}

func (a *MonteCarloAgent) Reset() {
	a.Memory = a.Memory[:0]
}

func (a *MonteCarloAgent) Update() {
	var G float64
	for i := len(a.Memory) - 1; i > -1; i-- {
		state, action, reward := a.Memory[i].State, a.Memory[i].Action, a.Memory[i].Reward

		G = a.Gamma*G + reward
		s := StateAction{State: state, Action: action}.String()
		a.Q[s] += a.Alpha * (G - a.Q[s])

		a.Pi[state] = greedyProbs(a.Q, state, a.Epsilon, a.ActionSize)
	}
}

func greedyProbs(Q DefaultMap[float64], state string, epsilon float64, actionSize int) RandomActions {
	qs := qstate(Q, state, actionSize)
	max := vector.Argmax(qs)

	probs := make(RandomActions)
	for i := 0; i < actionSize; i++ {
		probs[i] = epsilon / float64(actionSize)
	}

	probs[max] += 1 - epsilon
	return probs
}

func qstate(Q DefaultMap[float64], state string, actionSize int) []float64 {
	qs := make([]float64, 0)
	for i := 0; i < actionSize; i++ {
		qs = append(qs, Q.Get(StateAction{State: state, Action: i}, 0.0))
	}

	return qs
}

type StateAction struct {
	State  string
	Action int
}

func (s StateAction) String() string {
	return fmt.Sprintf("%s: %v", s.State, s.Action)
}
