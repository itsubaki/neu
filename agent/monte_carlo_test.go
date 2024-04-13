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
	for i := 0; i < episodes; i++ {
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
	// (0, 0) UP    : 0.70
	// (0, 0) DOWN  : 0.63
	// (0, 0) LEFT  : 0.73
	// (0, 0) RIGHT : 0.75
	// (0, 1) UP    : 0.81
	// (0, 1) DOWN  : 0.80
	// (0, 1) LEFT  : 0.69
	// (0, 1) RIGHT : 0.86
	// (0, 2) UP    : 0.89
	// (0, 2) DOWN  : 0.77
	// (0, 2) LEFT  : 0.80
	// (0, 2) RIGHT : 1.00
	// (1, 0) UP    : 0.68
	// (1, 0) DOWN  : 0.57
	// (1, 0) LEFT  : 0.61
	// (1, 0) RIGHT : 0.64
	// (1, 2) UP    : 0.88
	// (1, 2) DOWN  : 0.63
	// (1, 2) LEFT  : 0.78
	// (1, 2) RIGHT : -0.11
	// (1, 3) UP    : 1.00
	// (1, 3) DOWN  : -0.14
	// (1, 3) LEFT  : 0.30
	// (1, 3) RIGHT : -0.10
	// (2, 0) UP    : 0.61
	// (2, 0) DOWN  : 0.56
	// (2, 0) LEFT  : 0.54
	// (2, 0) RIGHT : 0.56
	// (2, 1) UP    : 0.51
	// (2, 1) DOWN  : 0.21
	// (2, 1) LEFT  : 0.45
	// (2, 1) RIGHT : 0.64
	// (2, 2) UP    : 0.71
	// (2, 2) DOWN  : 0.48
	// (2, 2) LEFT  : 0.42
	// (2, 2) RIGHT : -0.09
	// (2, 3) UP    : -0.20
	// (2, 3) DOWN  : -0.20
	// (2, 3) LEFT  : -0.04
	// (2, 3) RIGHT : -0.23
}
