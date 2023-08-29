package agent

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type SarsaOffPolicyAgent struct {
	Gamma         float64
	Alpha         float64
	Epsilon       float64
	ActionSize    int
	RandomActions RandomActions
	Pi            map[string]RandomActions
	B             map[string]RandomActions
	Q             map[string]float64
	Memory        *Deque
	Source        rand.Source
}

func (a *SarsaOffPolicyAgent) GetAction(state fmt.Stringer) int {
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

func (a *SarsaOffPolicyAgent) Reset() {
	a.Memory = NewDeque(a.Memory.Size())
}

func (a *SarsaOffPolicyAgent) Update(state fmt.Stringer, action int, reward float64, done bool) {
	a.Memory.Append(Memory{State: state.String(), Action: action, Reward: reward, Done: done})
	if a.Memory.Len() < 2 {
		return
	}

	m0, m1 := a.Memory.Get(0), a.Memory.Get(1)
	nextq, rho := 0.0, 1.0
	if !m0.Done {
		next := StateAction{State: m1.State, Action: m1.Action}.String()
		nextq = a.Q[next]
		rho = a.Pi[m1.State][m1.Action] / a.B[m1.State][m1.Action]
	}

	target := (m0.Reward + a.Gamma*nextq) * rho
	s := StateAction{State: m0.State, Action: m0.Action}.String()
	a.Q[s] += a.Alpha * (target - a.Q[s])

	a.Pi[m0.State] = greedyProps(a.Q, m0.State, 0, a.ActionSize)
	a.B[m0.State] = greedyProps(a.Q, m0.State, a.Epsilon, a.ActionSize)
}
