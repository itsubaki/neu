package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
)

func ExampleDeque() {
	q := agent.NewDeque[agent.Memory](2)
	q.Add(agent.Memory{State: "a", Action: 1, Reward: 1, Done: false})
	q.Add(agent.Memory{State: "b", Action: 2, Reward: 2, Done: false})
	q.Add(agent.Memory{State: "c", Action: 3, Reward: 3, Done: true})

	fmt.Println(q.Get(0))
	fmt.Println(q.Get(1))
	fmt.Println(q.Len())
	fmt.Println(q.Size())

	// Output:
	// {b 2 2 false}
	// {c 3 3 true}
	// 2
	// 2
}
