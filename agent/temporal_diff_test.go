package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleTemporalDiffAgent() {
	env := env.NewGridWorld()
	a := agent.TemporalDiffAgent{
		Gamma:          0.9,
		Alpha:          0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		V:              make(map[string]float64),
		Source:         rand.NewSource(1),
	}

	episodes := 1000
	for i := 0; i < episodes; i++ {
		state := env.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := env.Step(action)
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
	// (0, 0): 0.04
	// (0, 1): 0.14
	// (0, 2): 0.27
	// (1, 0): -0.02
	// (1, 2): -1.06
	// (1, 3): -0.68
	// (2, 0): -0.08
	// (2, 1): -0.17
	// (2, 2): -0.46
	// (2, 3): -0.85
}
