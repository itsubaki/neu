package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
)

func ExampleReplayBuffer() {
	buf := agent.NewReplyBuffer[string](10, 3, rand.NewSource(1))
	for i := 0; i < 10; i++ {
		buf.Append(fmt.Sprintf("%d", i))
	}
	fmt.Println(buf.Len())
	fmt.Println(buf.Batch())

	// Output:
	// 10
	// [1 7 9]
}

func ExampleReplayBuffer_rand() {
	buf := agent.NewReplyBuffer[string](10, 3)
	for i := 0; i < 10; i++ {
		buf.Append(fmt.Sprintf("%d", i))
	}
	fmt.Println(buf.Len())

	// Output:
	// 10
}
