package agent_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleQLearningAgent() {
	env := env.NewGridWorld()
	a := agent.QLearningAgent{
		Gamma:      0.9,
		Alpha:      0.8,
		Epsilon:    0.1,
		ActionSize: 4,
		Props:      []float64{0.25, 0.25, 0.25, 0.25},
		Q:          make(map[string]float64),
		Source:     rand.NewSource(1),
	}

	episodes := 10000
	for i := 0; i < episodes; i++ {
		state := env.Reset()

		for {
			action := a.GetAction(state)
			next, reward, done := env.Step(action)
			a.Update(state, action, reward, next, done)

			if done {
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
	// (0, 0) UP    : 0.7290
	// (0, 0) DOWN  : 0.6561
	// (0, 0) LEFT  : 0.7290
	// (0, 0) RIGHT : 0.8100
	// (0, 1) UP    : 0.8100
	// (0, 1) DOWN  : 0.8100
	// (0, 1) LEFT  : 0.7290
	// (0, 1) RIGHT : 0.9000
	// (0, 2) UP    : 0.9000
	// (0, 2) DOWN  : 0.8100
	// (0, 2) LEFT  : 0.8100
	// (0, 2) RIGHT : 1.0000
	// (1, 0) UP    : 0.7290
	// (1, 0) DOWN  : 0.5905
	// (1, 0) LEFT  : 0.6561
	// (1, 0) RIGHT : 0.6561
	// (1, 2) UP    : 0.9000
	// (1, 2) DOWN  : 0.7290
	// (1, 2) LEFT  : 0.8100
	// (1, 2) RIGHT : -0.1012
	// (1, 3) UP    : 1.0000
	// (1, 3) DOWN  : 0.0000
	// (1, 3) LEFT  : 0.0000
	// (1, 3) RIGHT : 0.0000
	// (2, 0) UP    : 0.6561
	// (2, 0) DOWN  : 0.5905
	// (2, 0) LEFT  : 0.5905
	// (2, 0) RIGHT : 0.6561
	// (2, 1) UP    : 0.6311
	// (2, 1) DOWN  : 0.6311
	// (2, 1) LEFT  : 0.5905
	// (2, 1) RIGHT : 0.7290
	// (2, 2) UP    : 0.8100
	// (2, 2) DOWN  : 0.7289
	// (2, 2) LEFT  : 0.6549
	// (2, 2) RIGHT : 0.0000
	// (2, 3) UP    : -0.0998
	// (2, 3) DOWN  : 0.0000
	// (2, 3) LEFT  : 0.0000
	// (2, 3) RIGHT : 0.0000
}
