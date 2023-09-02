package agent_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleSarsaOffPolicyAgent() {
	env := env.NewGridWorld()
	a := &agent.SarsaOffPolicyAgent{
		Gamma:          0.9,
		Alpha:          0.8,
		Epsilon:        0.1,
		ActionSize:     4,
		DefaultActions: agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
		Pi:             make(map[string]agent.RandomActions),
		B:              make(map[string]agent.RandomActions),
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
	// (0, 0) UP    : 0.0254
	// (0, 0) DOWN  : 0.0537
	// (0, 0) LEFT  : 0.0710
	// (0, 0) RIGHT : 0.8649
	// (0, 1) UP    : 0.1514
	// (0, 1) DOWN  : 0.1759
	// (0, 1) LEFT  : 0.1171
	// (0, 1) RIGHT : 0.9667
	// (0, 2) UP    : 0.9730
	// (0, 2) DOWN  : 0.9269
	// (0, 2) LEFT  : 0.1602
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.5462
	// (1, 0) DOWN  : 0.0451
	// (1, 0) LEFT  : 0.0413
	// (1, 0) RIGHT : 0.0090
	// (1, 2) UP    : 0.9730
	// (1, 2) DOWN  : 0.0280
	// (1, 2) LEFT  : 0.0096
	// (1, 2) RIGHT : -0.1081
	// (1, 3) UP    : 1.0000
	// (1, 3) DOWN  : 0.2193
	// (1, 3) LEFT  : 0.8196
	// (1, 3) RIGHT : -0.0865
	// (2, 0) UP    : 0.1737
	// (2, 0) DOWN  : 0.1080
	// (2, 0) LEFT  : 0.0727
	// (2, 0) RIGHT : 0.8900
	// (2, 1) UP    : 0.0023
	// (2, 1) DOWN  : 0.0057
	// (2, 1) LEFT  : 0.1262
	// (2, 1) RIGHT : 0.9207
	// (2, 2) UP    : 0.9467
	// (2, 2) DOWN  : 0.6805
	// (2, 2) LEFT  : 0.0030
	// (2, 2) RIGHT : 0.0031
	// (2, 3) UP    : -0.1081
	// (2, 3) DOWN  : 0.0031
	// (2, 3) LEFT  : 0.0125
	// (2, 3) RIGHT : 0.0033
}
