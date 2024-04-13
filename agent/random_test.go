package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleRandomAgent() {
	e := env.NewGridWorld()
	a := &agent.RandomAgent{
		Gamma:          0.9,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		V:              make(map[string]float64),
		Counts:         make(map[string]int),
		Memory:         make([]agent.Memory, 0),
		Source:         rand.Const(1),
	}

	episodes := 1000
	for i := 0; i < episodes; i++ {
		state := e.Reset()
		a.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := e.Step(action)
			a.Add(state, action, reward)

			if done {
				a.Eval()
				break
			}

			state = next
		}
	}

	for _, k := range agent.SortedKeys(a.V) {
		fmt.Println(k, a.V[k])
	}

	// Output:
	// (0, 0) 0.019763603081288356
	// (0, 1) 0.07803773078469887
	// (0, 2) 0.1844500459964045
	// (1, 0) -0.023319112315680166
	// (1, 2) -0.484607672112952
	// (1, 3) -0.3464933845764332
	// (2, 0) -0.10313829730561434
	// (2, 1) -0.22467848343570673
	// (2, 2) -0.4547704293279075
	// (2, 3) -0.760130052613597
}
