package agent_test

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleSarsaAgent() {
	e := env.NewGridWorld()
	a := &agent.SarsaAgent{
		Gamma:          0.9,
		Alpha:          0.8,
		Epsilon:        0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		Q:              make(map[string]float64),
		Memory:         agent.NewDeque[agent.Memory](2),
		Source:         rand.Const(1),
	}

	episodes := 10000
	for range episodes {
		state := e.Reset()
		a.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := e.Step(action)
			a.Update(state, action, reward, done)

			if done {
				a.Update(state, -1, 0, false)
				break
			}

			state = next
		}
	}

	for _, k := range agent.SortedKeys(a.Q) {
		s := strings.Split(k, ": ")
		move, _ := strconv.Atoi(s[1])
		fmt.Printf("%s %-6s: %.4f\n", s[0], e.ActionMeaning[move], a.Q[k])
	}

	// Output:
	// (0, 0) UP    : 0.7160
	// (0, 0) DOWN  : 0.3935
	// (0, 0) LEFT  : 0.6322
	// (0, 0) RIGHT : 0.8100
	// (0, 1) UP    : 0.7571
	// (0, 1) DOWN  : 0.7528
	// (0, 1) LEFT  : 0.4538
	// (0, 1) RIGHT : 0.9000
	// (0, 2) UP    : 0.9000
	// (0, 2) DOWN  : 0.8100
	// (0, 2) LEFT  : 0.6762
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.7290
	// (1, 0) DOWN  : 0.3271
	// (1, 0) LEFT  : 0.5255
	// (1, 0) RIGHT : 0.6079
	// (1, 2) UP    : 0.9000
	// (1, 2) DOWN  : 0.3905
	// (1, 2) LEFT  : 0.7233
	// (1, 2) RIGHT : -0.1058
	// (1, 3) UP    : 0.9997
	// (1, 3) DOWN  : 0.0000
	// (1, 3) LEFT  : -0.0336
	// (1, 3) RIGHT : -0.8000
	// (2, 0) UP    : 0.6561
	// (2, 0) DOWN  : 0.3115
	// (2, 0) LEFT  : 0.3696
	// (2, 0) RIGHT : 0.4583
	// (2, 1) UP    : 0.2455
	// (2, 1) DOWN  : 0.3161
	// (2, 1) LEFT  : 0.5346
	// (2, 1) RIGHT : 0.3008
	// (2, 2) UP    : -0.1420
	// (2, 2) DOWN  : 0.0495
	// (2, 2) LEFT  : 0.1471
	// (2, 2) RIGHT : 0.0431
	// (2, 3) UP    : -0.2669
	// (2, 3) DOWN  : -0.1597
	// (2, 3) LEFT  : 0.2008
	// (2, 3) RIGHT : 0.0000
}
