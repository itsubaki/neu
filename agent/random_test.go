package agent_test

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleRandomAgent() {
	env := env.NewGridWorld()
	a := agent.RandomAgent{
		Gamma:         0.9,
		ActionSize:    4,
		RandomActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:            make(map[string]agent.RandomActions),
		V:             make(map[string]float64),
		Counts:        make(map[string]int),
		Memory:        make([]agent.Memory, 0),
		Source:        rand.NewSource(1),
	}

	episodes := 1000
	for i := 0; i < episodes; i++ {
		state := env.Reset()
		a.Reset()

		for {
			action := a.GetAction(state.String())
			next, reward, done := env.Step(action)
			a.Add(state.String(), action, reward)

			if done {
				a.Eval()
				break
			}

			state = next
		}
	}

	keys := make([]string, 0)
	for k := range a.V {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		fmt.Println(k, a.V[k])
	}

	// Output:
	// (0, 0) 0.028141156021612355
	// (0, 1) 0.09403279135233088
	// (0, 2) 0.1930212924088801
	// (1, 0) -0.02420641778873053
	// (1, 2) -0.5224524749248304
	// (1, 3) -0.3712960430522996
	// (2, 0) -0.0963549289338684
	// (2, 1) -0.21550648132838401
	// (2, 2) -0.43460634403475157
	// (2, 3) -0.7648298236773389
}
