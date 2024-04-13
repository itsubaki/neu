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
	// (0, 0) UP    : 0.4435
	// (0, 0) DOWN  : 0.4794
	// (0, 0) LEFT  : 0.5081
	// (0, 0) RIGHT : 0.8100
	// (0, 1) UP    : 0.6910
	// (0, 1) DOWN  : 0.7147
	// (0, 1) LEFT  : 0.5726
	// (0, 1) RIGHT : 0.9000
	// (0, 2) UP    : 0.8856
	// (0, 2) DOWN  : 0.8097
	// (0, 2) LEFT  : 0.6566
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.6862
	// (1, 0) DOWN  : 0.4138
	// (1, 0) LEFT  : 0.4232
	// (1, 0) RIGHT : 0.4689
	// (1, 2) UP    : 0.9000
	// (1, 2) DOWN  : 0.0828
	// (1, 2) LEFT  : 0.7719
	// (1, 2) RIGHT : -0.1973
	// (1, 3) UP    : 0.9920
	// (1, 3) DOWN  : 0.0000
	// (1, 3) LEFT  : 0.6116
	// (1, 3) RIGHT : -0.8000
	// (2, 0) UP    : 0.4778
	// (2, 0) DOWN  : 0.4032
	// (2, 0) LEFT  : 0.4687
	// (2, 0) RIGHT : 0.3780
	// (2, 1) UP    : 0.4054
	// (2, 1) DOWN  : 0.3012
	// (2, 1) LEFT  : 0.4845
	// (2, 1) RIGHT : 0.0498
	// (2, 2) UP    : -0.1938
	// (2, 2) DOWN  : 0.0000
	// (2, 2) LEFT  : 0.3481
	// (2, 2) RIGHT : -0.1597
	// (2, 3) UP    : -0.2669
	// (2, 3) DOWN  : 0.0000
	// (2, 3) LEFT  : 0.0000
	// (2, 3) RIGHT : 0.0000
}
