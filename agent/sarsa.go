package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type SarsaAgent struct {
	Gamma          float64
	Alpha          float64
	Epsilon        float64
	ActionSize     int
	DefaultActions RandomActions
	Pi             map[string]RandomActions
	Q              map[string]float64
	Memory         *Deque
	Source         rand.Source
}

func (a *SarsaAgent) GetAction(state fmt.Stringer) int {
	probs := make([]float64, a.ActionSize)
	for i, p := range Get(a.Pi, state, a.DefaultActions) {
		probs[i] = p
	}

	return vector.Choice(probs, a.Source)
}

func (a *SarsaAgent) Reset() {
	a.Memory = NewDeque(a.Memory.Size())
}

func (a *SarsaAgent) Update(state fmt.Stringer, action int, reward float64, done bool) {
	a.Memory.Append(Memory{State: state.String(), Action: action, Reward: reward, Done: done})
	if a.Memory.Len() < 2 {
		return
	}

	m0, m1 := a.Memory.Get(0), a.Memory.Get(1)
	var nextq float64
	if !m0.Done {
		nextq = a.Q[StateAction{State: m1.State, Action: m1.Action}.String()]
	}

	target := m0.Reward + a.Gamma*nextq
	s := StateAction{State: m0.State, Action: m0.Action}.String()
	a.Q[s] += a.Alpha * (target - a.Q[s])

	a.Pi[m0.State] = greedyProbs(a.Q, m0.State, a.Epsilon, a.ActionSize)
}
