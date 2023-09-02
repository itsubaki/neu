package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
)

func ExampleDefaultMap() {
	m := agent.DefaultMap[agent.RandomActions]{}

	fmt.Println(agent.Get(m, env.GridState{Height: 1, Width: 1}, agent.RandomActions{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}).Probs())
	for k, v := range m {
		fmt.Println(k, v)
	}

	// Output:
	// [0.25 0.25 0.25 0.25]
	// (1, 1) map[0:0.25 1:0.25 2:0.25 3:0.25]
}
