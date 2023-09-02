package agent_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleSarsaAgent() {
	env := env.NewGridWorld()
	a := &agent.SarsaAgent{
		Gamma:          0.9,
		Alpha:          0.8,
		Epsilon:        0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		Q:              make(map[string]float64),
		Memory:         agent.NewDeque[agent.Memory](2),
		Source:         rand.NewSource(1),
	}

	episodes := 10000
	for i := 0; i < episodes; i++ {
		state := env.Reset()
		a.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := env.Step(action)
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
		fmt.Printf("%s %-6s: %.4f\n", s[0], env.ActionMeaning[move], a.Q[k])
	}

	// Output:
	// (0, 0) UP    : 0.4938
	// (0, 0) DOWN  : 0.3980
	// (0, 0) LEFT  : 0.5072
	// (0, 0) RIGHT : 0.7697
	// (0, 1) UP    : 0.5558
	// (0, 1) DOWN  : 0.7522
	// (0, 1) LEFT  : 0.5122
	// (0, 1) RIGHT : 0.8945
	// (0, 2) UP    : 0.8989
	// (0, 2) DOWN  : 0.8096
	// (0, 2) LEFT  : 0.6849
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.6510
	// (1, 0) DOWN  : 0.3653
	// (1, 0) LEFT  : 0.3668
	// (1, 0) RIGHT : 0.3637
	// (1, 2) UP    : 0.9000
	// (1, 2) DOWN  : 0.3071
	// (1, 2) LEFT  : 0.7579
	// (1, 2) RIGHT : -0.1647
	// (1, 3) UP    : 0.9997
	// (1, 3) DOWN  : 0.0000
	// (1, 3) LEFT  : -0.0331
	// (1, 3) RIGHT : 0.0000
	// (2, 0) UP    : 0.6220
	// (2, 0) DOWN  : 0.4408
	// (2, 0) LEFT  : 0.4706
	// (2, 0) RIGHT : 0.4306
	// (2, 1) UP    : 0.2574
	// (2, 1) DOWN  : 0.3115
	// (2, 1) LEFT  : 0.5228
	// (2, 1) RIGHT : -0.0869
	// (2, 2) UP    : -0.0407
	// (2, 2) DOWN  : 0.0979
	// (2, 2) LEFT  : 0.4150
	// (2, 2) RIGHT : -0.1596
	// (2, 3) UP    : -0.1133
	// (2, 3) DOWN  : 0.0000
	// (2, 3) LEFT  : 0.0000
	// (2, 3) RIGHT : 0.0000
}
