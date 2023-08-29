package agent_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleMonteCarloAgent() {
	env := env.NewGridWorld()
	a := agent.MonteCarloAgent{
		Gamma:         0.9,
		Epsilon:       0.1,
		Alpha:         0.1,
		ActionSize:    4,
		RandomActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:            make(map[string]agent.RandomActions),
		Q:             make(map[string]float64),
		Memory:        make([]agent.Memory, 0),
		Source:        rand.NewSource(1),
	}

	episodes := 10000
	for i := 0; i < episodes; i++ {
		state := env.Reset()
		a.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := env.Step(action)
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
		fmt.Printf("%s %-6s: %.2f\n", s[0], env.ActionMeaning[move], a.Q[k])
	}

	// Output:
	// (0, 0) UP    : 0.64
	// (0, 0) DOWN  : 0.58
	// (0, 0) LEFT  : 0.28
	// (0, 0) RIGHT : 0.79
	// (0, 1) UP    : 0.73
	// (0, 1) DOWN  : 0.28
	// (0, 1) LEFT  : 0.70
	// (0, 1) RIGHT : 0.88
	// (0, 2) UP    : 0.86
	// (0, 2) DOWN  : 0.75
	// (0, 2) LEFT  : 0.80
	// (0, 2) RIGHT : 1.00
	// (1, 0) UP    : 0.71
	// (1, 0) DOWN  : 0.53
	// (1, 0) LEFT  : 0.47
	// (1, 0) RIGHT : 0.22
	// (1, 2) UP    : 0.89
	// (1, 2) DOWN  : 0.66
	// (1, 2) LEFT  : 0.55
	// (1, 2) RIGHT : -0.10
	// (1, 3) UP    : 0.98
	// (1, 3) DOWN  : 0.00
	// (1, 3) LEFT  : 0.15
	// (1, 3) RIGHT : -0.02
	// (2, 0) UP    : 0.61
	// (2, 0) DOWN  : 0.46
	// (2, 0) LEFT  : 0.44
	// (2, 0) RIGHT : 0.49
	// (2, 1) UP    : 0.38
	// (2, 1) DOWN  : 0.09
	// (2, 1) LEFT  : 0.46
	// (2, 1) RIGHT : 0.61
	// (2, 2) UP    : 0.74
	// (2, 2) DOWN  : 0.54
	// (2, 2) LEFT  : 0.37
	// (2, 2) RIGHT : -0.16
	// (2, 3) UP    : -0.18
	// (2, 3) DOWN  : 0.00
	// (2, 3) LEFT  : 0.00
	// (2, 3) RIGHT : -0.01
}
