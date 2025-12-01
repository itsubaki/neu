package agent_test

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleMonteCarloAgent() {
	e := env.NewGridWorld()
	a := &agent.MonteCarloAgent{
		Gamma:          0.9,
		Epsilon:        0.1,
		Alpha:          0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		Q:              make(map[string]float64),
		Memory:         make([]agent.Memory, 0),
		Source:         rand.Const(1),
	}

	episodes := 10000
	for range episodes {
		state := e.Reset()
		a.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := e.Step(action)
			a.Add(state, action, reward)

			if done {
				a.Update()
				break
			}

			state = next
		}
	}

	for _, k := range agent.SortedKeys(a.Q) {
		s := strings.Split(k, ": ")
		move, _ := strconv.Atoi(s[1])
		fmt.Printf("%s %-6s: %.2f\n", s[0], e.ActionMeaning[move], a.Q[k])
	}

	// Output:
	// (0, 0) UP    : 0.71
	// (0, 0) DOWN  : 0.60
	// (0, 0) LEFT  : 0.73
	// (0, 0) RIGHT : 0.79
	// (0, 1) UP    : 0.81
	// (0, 1) DOWN  : 0.79
	// (0, 1) LEFT  : 0.71
	// (0, 1) RIGHT : 0.88
	// (0, 2) UP    : 0.89
	// (0, 2) DOWN  : 0.80
	// (0, 2) LEFT  : 0.80
	// (0, 2) RIGHT : 1.00
	// (1, 0) UP    : 0.71
	// (1, 0) DOWN  : 0.57
	// (1, 0) LEFT  : 0.64
	// (1, 0) RIGHT : 0.63
	// (1, 2) UP    : 0.89
	// (1, 2) DOWN  : 0.63
	// (1, 2) LEFT  : 0.70
	// (1, 2) RIGHT : -0.13
	// (1, 3) UP    : 1.00
	// (1, 3) DOWN  : 0.25
	// (1, 3) LEFT  : 0.45
	// (1, 3) RIGHT : -0.10
	// (2, 0) UP    : 0.62
	// (2, 0) DOWN  : 0.55
	// (2, 0) LEFT  : 0.54
	// (2, 0) RIGHT : 0.59
	// (2, 1) UP    : 0.24
	// (2, 1) DOWN  : 0.07
	// (2, 1) LEFT  : 0.37
	// (2, 1) RIGHT : 0.66
	// (2, 2) UP    : 0.74
	// (2, 2) DOWN  : 0.40
	// (2, 2) LEFT  : 0.50
	// (2, 2) RIGHT : 0.01
	// (2, 3) UP    : -0.24
	// (2, 3) DOWN  : -0.00
	// (2, 3) LEFT  : 0.02
	// (2, 3) RIGHT : -0.15
}
