package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleTemporalDiffAgent() {
	e := env.NewGridWorld()
	a := &agent.TemporalDiffAgent{
		Gamma:          0.9,
		Alpha:          0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		V:              make(map[string]float64),
		Source:         rand.Const(1),
	}

	episodes := 1000
	for i := 0; i < episodes; i++ {
		state := e.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := e.Step(action)
			a.Eval(state, reward, next, done)

			if done {
				break
			}

			state = next
		}
	}

	for _, k := range agent.SortedKeys(a.V) {
		fmt.Printf("%s: %.2f\n", k, a.V[k])
	}

	// Output:
	// (0, 0): 0.07
	// (0, 1): 0.13
	// (0, 2): 0.05
	// (1, 0): -0.01
	// (1, 2): -0.68
	// (1, 3): -0.60
	// (2, 0): -0.08
	// (2, 1): -0.15
	// (2, 2): -0.43
	// (2, 3): -0.60
}
