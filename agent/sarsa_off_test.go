package agent_test

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleSarsaOffPolicyAgent() {
	e := env.NewGridWorld()
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
	// (0, 0) UP    : 0.0175
	// (0, 0) DOWN  : 0.2770
	// (0, 0) LEFT  : 0.5972
	// (0, 0) RIGHT : 0.3044
	// (0, 1) UP    : 0.1440
	// (0, 1) DOWN  : 0.1110
	// (0, 1) LEFT  : 0.0107
	// (0, 1) RIGHT : 0.8173
	// (0, 2) UP    : 0.9730
	// (0, 2) DOWN  : 0.3039
	// (0, 2) LEFT  : 0.9236
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.6087
	// (1, 0) DOWN  : 0.0328
	// (1, 0) LEFT  : 0.0166
	// (1, 0) RIGHT : 0.0174
	// (1, 2) UP    : 0.9727
	// (1, 2) DOWN  : 0.1290
	// (1, 2) LEFT  : 0.1149
	// (1, 2) RIGHT : -0.1081
	// (1, 3) UP    : 1.0000
	// (1, 3) DOWN  : 0.6852
	// (1, 3) LEFT  : 0.7043
	// (1, 3) RIGHT : -0.0867
	// (2, 0) UP    : 0.0838
	// (2, 0) DOWN  : 0.0973
	// (2, 0) LEFT  : 0.0936
	// (2, 0) RIGHT : 0.0914
	// (2, 1) UP    : 0.1000
	// (2, 1) DOWN  : 0.0551
	// (2, 1) LEFT  : 0.4486
	// (2, 1) RIGHT : 0.7586
	// (2, 2) UP    : 0.9428
	// (2, 2) DOWN  : 0.1647
	// (2, 2) LEFT  : 0.3839
	// (2, 2) RIGHT : 0.2200
	// (2, 3) UP    : -0.1081
	// (2, 3) DOWN  : 0.0032
	// (2, 3) LEFT  : 0.7448
	// (2, 3) RIGHT : 0.1050
}
