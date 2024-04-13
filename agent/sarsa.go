package agent

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/vector"
)

type SarsaAgent struct {
	Gamma          float64
	Alpha          float64
	Epsilon        float64
	ActionSize     int
	DefaultActions RandomActions
	Pi             DefaultMap[RandomActions]
	Q              DefaultMap[float64]
	Memory         *Deque[Memory]
	Source         randv2.Source
}

func (a *SarsaAgent) GetAction(state fmt.Stringer) int {
	probs := a.Pi.Get(state, a.DefaultActions).Probs()
	return vector.Choice(probs, a.Source)
}

func (a *SarsaAgent) Reset() {
	a.Memory = NewDeque[Memory](a.Memory.Size())
}

func (a *SarsaAgent) Update(state fmt.Stringer, action int, reward float64, done bool) {
	a.Memory.Add(NewMemory(state, action, reward, done))
	if a.Memory.Len() < 2 {
		return
	}

	m0, m1 := a.Memory.Get(0), a.Memory.Get(1)
	var nextq float64
	if !m0.Done {
		s := StateAction{State: m1.State, Action: m1.Action}.String()
		nextq = a.Q[s]
	}

	target := m0.Reward + a.Gamma*nextq
	s := StateAction{State: m0.State, Action: m0.Action}.String()
	a.Q[s] += a.Alpha * (target - a.Q[s])

	a.Pi[m0.State] = greedyProbs(a.Q, m0.State, a.Epsilon, a.ActionSize)
}
