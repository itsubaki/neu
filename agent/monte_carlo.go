package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type StateAction struct {
	State  string
	Action int
}

func (s StateAction) String() string {
	return fmt.Sprintf("%s: %v", s.State, s.Action)
}

type MonteCarloAgent struct {
	Gamma         float64
	Epsilon       float64
	Alpha         float64
	ActionSize    int
	RandomActions RandomActions
	Pi            map[string]RandomActions
	Q             map[string]float64
	Memory        []Memory
	Source        rand.Source
}

func (a *MonteCarloAgent) GetAction(state fmt.Stringer) int {
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

func (a *MonteCarloAgent) Add(state fmt.Stringer, action int, reward float64) {
	a.Memory = append(a.Memory, Memory{State: state.String(), Action: action, Reward: reward})
}

func (a *MonteCarloAgent) Reset() {
	a.Memory = a.Memory[:0]
}

func (a *MonteCarloAgent) Update() {
	var G float64
	for i := len(a.Memory) - 1; i > -1; i-- {
		state, action, reward := a.Memory[i].State, a.Memory[i].Action, a.Memory[i].Reward

		G = a.Gamma*G + reward
		key := StateAction{State: state, Action: action}.String()
		a.Q[key] += a.Alpha * (G - a.Q[key])
		a.Pi[state] = greedyProps(a.Q, state, a.Epsilon, a.ActionSize)
	}
}

func greedyProps(Q map[string]float64, state string, epsilon float64, actionSize int) RandomActions {
	qs := make([]float64, 0)
	for i := 0; i < actionSize; i++ {
		key := StateAction{State: state, Action: i}.String()
		if _, ok := Q[key]; !ok {
			Q[key] = 0
		}

		qs = append(qs, Q[key])
	}
	max := vector.Argmax(qs)

	probs := make(RandomActions)
	for i := 0; i < actionSize; i++ {
		probs[i] = epsilon / float64(actionSize)
	}

	probs[max] += 1 - epsilon
	return probs
}
