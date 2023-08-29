package agent

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/itsubaki/neu/math/vector"
)

type RandomActions map[int]float64

type Memory struct {
	State  string
	Action int
	Reward float64
	Done   bool
}

type RandomAgent struct {
	Gamma         float64
	ActionSize    int
	RandomActions RandomActions
	Pi            map[string]RandomActions
	V             map[string]float64
	Counts        map[string]int
	Memory        []Memory
	Source        rand.Source
}

func (a *RandomAgent) GetAction(state fmt.Stringer) int {
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

func (a *RandomAgent) Add(state fmt.Stringer, action int, reward float64) {
	a.Memory = append(a.Memory, Memory{State: state.String(), Action: action, Reward: reward})
}

func (a *RandomAgent) Reset() {
	a.Memory = a.Memory[:0]
}

func (a *RandomAgent) Eval() {
	var G float64
	for i := len(a.Memory) - 1; i > -1; i-- {
		state, reward := a.Memory[i].State, a.Memory[i].Reward

		G = a.Gamma*G + reward
		a.Counts[state]++
		a.V[state] += (G - a.V[state]) / float64(a.Counts[state])
	}
}

func SortedKeys(m map[string]float64) []string {
	keys := make([]string, 0)
	for k := range m {
		keys = append(keys, k)
	}

	sort.Strings(keys)
	return keys
}