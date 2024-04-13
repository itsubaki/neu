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
	for i := 0; i < episodes; i++ {
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
	// (0, 0) UP    : 0.0392
	// (0, 0) DOWN  : 0.0294
	// (0, 0) LEFT  : 0.0825
	// (0, 0) RIGHT : 0.9415
	// (0, 1) UP    : 0.1644
	// (0, 1) DOWN  : 0.0989
	// (0, 1) LEFT  : 0.7034
	// (0, 1) RIGHT : 0.9727
	// (0, 2) UP    : 0.9730
	// (0, 2) DOWN  : 0.7868
	// (0, 2) LEFT  : 0.3601
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.8772
	// (1, 0) DOWN  : 0.0449
	// (1, 0) LEFT  : 0.0249
	// (1, 0) RIGHT : 0.1231
	// (1, 2) UP    : 0.9717
	// (1, 2) DOWN  : 0.0294
	// (1, 2) LEFT  : 0.0286
	// (1, 2) RIGHT : -0.1081
	// (1, 3) UP    : 1.0000
	// (1, 3) DOWN  : 0.2980
	// (1, 3) LEFT  : 0.9221
	// (1, 3) RIGHT : -0.1081
	// (2, 0) UP    : 0.7060
	// (2, 0) DOWN  : 0.1899
	// (2, 0) LEFT  : 0.0973
	// (2, 0) RIGHT : 0.0888
	// (2, 1) UP    : 0.0137
	// (2, 1) DOWN  : 0.0042
	// (2, 1) LEFT  : 0.4360
	// (2, 1) RIGHT : 0.6891
	// (2, 2) UP    : 0.9112
	// (2, 2) DOWN  : 0.0142
	// (2, 2) LEFT  : 0.0029
	// (2, 2) RIGHT : 0.0530
	// (2, 3) UP    : -0.0182
	// (2, 3) DOWN  : 0.0171
	// (2, 3) LEFT  : 0.0591
	// (2, 3) RIGHT : 0.0065
}
