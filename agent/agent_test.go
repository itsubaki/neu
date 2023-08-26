package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
)

func ExampleAgent() {
	a := agent.Agent{
		Epsilon: 0.5,
		Qs:      []float64{0, 0, 0, 0, 0},
		Ns:      []float64{0, 0, 0, 0, 0},
		Source:  rand.NewSource(1),
	}

	for i := 0; i < 10; i++ {
		action := a.GetAction()
		a.Update(action, 1.0)
		fmt.Printf("%v: %v\n", action, a.Qs)
	}

	// Output:
	// 0: [1 0 0 0 0]
	// 0: [1 0 0 0 0]
	// 0: [1 0 0 0 0]
	// 1: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 4: [1 1 0 0 1]
}
