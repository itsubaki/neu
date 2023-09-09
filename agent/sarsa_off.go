package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type SarsaOffPolicyAgent struct {
	Gamma          float64
	Alpha          float64
	Epsilon        float64
	ActionSize     int
	DefaultActions RandomActions
	Pi             DefaultMap[RandomActions]
	B              DefaultMap[RandomActions]
	Q              DefaultMap[float64]
	Memory         *Deque[Memory]
	Source         rand.Source
}

func (a *SarsaOffPolicyAgent) GetAction(state fmt.Stringer) int {
	probs := a.B.Get(state, a.DefaultActions).Probs()
	return vector.Choice(probs, a.Source)
}

func (a *SarsaOffPolicyAgent) Reset() {
	a.Memory = NewDeque[Memory](a.Memory.Size())
}

func (a *SarsaOffPolicyAgent) Update(state fmt.Stringer, action int, reward float64, done bool) {
	a.Memory.Add(NewMemory(state, action, reward, done))
	if a.Memory.Len() < 2 {
		return
	}

	m0, m1 := a.Memory.Get(0), a.Memory.Get(1)
	nextq, rho := 0.0, 1.0
	if !m0.Done {
		s := StateAction{State: m1.State, Action: m1.Action}.String()
		nextq = a.Q[s]
		rho = a.Pi[m1.State][m1.Action] / a.B[m1.State][m1.Action]
	}

	target := (m0.Reward + a.Gamma*nextq) * rho
	s := StateAction{State: m0.State, Action: m0.Action}.String()
	a.Q[s] += a.Alpha * (target - a.Q[s])

	a.Pi[m0.State] = greedyProbs(a.Q, m0.State, 0, a.ActionSize)
	a.B[m0.State] = greedyProbs(a.Q, m0.State, a.Epsilon, a.ActionSize)
}
