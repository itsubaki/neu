package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
)

func ExampleReplayBuffer() {
	buf := agent.NewReplyBuffer(10, 3, rand.NewSource(1))
	for i := 0; i < 10; i++ {
		buf.Append("s", i, float64(i), "n", false)
	}
	fmt.Println(buf.Len())

	batch := buf.Batch()
	for _, b := range batch {
		fmt.Println(b)
	}

	// Unordered output:
	// 10
	// {s 7 7 n false}
	// {s 9 9 n false}
	// {s 1 1 n false}
}

func ExampleReplayBuffer_rand() {
	buf := agent.NewReplyBuffer(10, 3)
	for i := 0; i < 10; i++ {
		buf.Append("s", i, float64(i), "n", false)
	}
	fmt.Println(buf.Len())

	// Output:
	// 10
}
