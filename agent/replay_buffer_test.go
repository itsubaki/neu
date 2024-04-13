package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleReplayBuffer() {
	buf := agent.NewReplayBuffer(10, 3, rand.Const(1))
	for i := 0; i < 10; i++ {
		buf.Add([]float64{float64(i), float64(i)}, i, float64(i), []float64{float64(i * 10), float64(i * 10)}, false)
	}
	fmt.Println(buf.Len())

	state, action, reward, next, done := buf.Batch()
	for i := range state {
		fmt.Println(state[i], action[i], reward[i], next[i], done[i])
	}

	// Unordered output:
	// 10
	// [0 0] 0 0 [0 0] false
	// [7 7] 7 7 [70 70] false
	// [5 5] 5 5 [50 50] false
}

func ExampleReplayBuffer_rand() {
	buf := agent.NewReplayBuffer(10, 3)
	for i := 0; i < 10; i++ {
		buf.Add([]float64{float64(i)}, i, float64(i), []float64{float64(i * 10)}, false)
	}
	fmt.Println(buf.Len())

	// Output:
	// 10
}
