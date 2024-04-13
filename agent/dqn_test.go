package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleDQNAgent() {
	e := env.NewGridWorld()
	s := rand.Const(1)
	a := &agent.DQNAgent{
		Gamma:        0.98,
		Epsilon:      0.1,
		ActionSize:   4,
		ReplayBuffer: agent.NewReplayBuffer(10000, 32, s),
		Q: model.NewQNet(&model.QNetConfig{
			InputSize:  12,
			OutputSize: 4,
			HiddenSize: []int{128, 128},
			WeightInit: weight.Xavier,
		}, s),
		QTarget: model.NewQNet(&model.QNetConfig{
			InputSize:  12,
			OutputSize: 4,
			HiddenSize: []int{128, 128},
			WeightInit: weight.Xavier,
		}, s),
		Optimizer: &optimizer.Adam{
			Alpha: 0.0005,
			Beta1: 0.9,
			Beta2: 0.999,
		},
		Source: s,
	}

	episodes, syncInterval := 1, 1
	for i := 0; i < episodes; i++ {
		state := e.OneHot(e.Reset())
		var totalLoss, totalReward float64
		var count int

		for {
			action := a.GetAction(state)
			next, reward, done := e.Step(action)
			nextoh := e.OneHot(next)
			loss := a.Update(state, action, reward, nextoh, done)
			state = nextoh

			totalLoss += loss[0][0]
			totalReward += reward
			count++

			if done {
				break
			}
		}

		if (i+1)%syncInterval == 0 {
			a.Sync()
		}

		fmt.Printf("%d: %.4f, %.4f\n", i, totalLoss/float64(count), totalReward/float64(count))
	}

	for _, s := range e.State {
		if s.Equals(e.GoalState) || s.Equals(e.WallState) {
			continue
		}

		q := a.Q.Predict(matrix.New(e.OneHot(&s)))
		for _, a := range e.Actions() {
			fmt.Printf("%s %-6s: %.4f\n", s, e.ActionMeaning[a], q[0][a])
		}
	}

	// Output:
	// 0: 0.0143, -0.0046
	// (0, 0) UP    : 0.1057
	// (0, 0) DOWN  : 0.0411
	// (0, 0) LEFT  : -0.1044
	// (0, 0) RIGHT : -0.1153
	// (0, 1) UP    : 0.2575
	// (0, 1) DOWN  : 0.0565
	// (0, 1) LEFT  : -0.0090
	// (0, 1) RIGHT : -0.1412
	// (0, 2) UP    : -0.1237
	// (0, 2) DOWN  : 0.3362
	// (0, 2) LEFT  : 0.0124
	// (0, 2) RIGHT : -0.0446
	// (1, 0) UP    : 0.0993
	// (1, 0) DOWN  : 0.0425
	// (1, 0) LEFT  : -0.1653
	// (1, 0) RIGHT : -0.0591
	// (1, 2) UP    : -0.4625
	// (1, 2) DOWN  : 0.0474
	// (1, 2) LEFT  : -0.2263
	// (1, 2) RIGHT : -0.1712
	// (1, 3) UP    : 0.7964
	// (1, 3) DOWN  : 0.0965
	// (1, 3) LEFT  : 0.0643
	// (1, 3) RIGHT : -0.1828
	// (2, 0) UP    : -0.4854
	// (2, 0) DOWN  : 0.2162
	// (2, 0) LEFT  : -0.2302
	// (2, 0) RIGHT : -0.1094
	// (2, 1) UP    : 0.2301
	// (2, 1) DOWN  : 0.0680
	// (2, 1) LEFT  : -0.0531
	// (2, 1) RIGHT : -0.0764
	// (2, 2) UP    : -1.2781
	// (2, 2) DOWN  : 0.2185
	// (2, 2) LEFT  : -0.6493
	// (2, 2) RIGHT : -0.4158
	// (2, 3) UP    : -0.7843
	// (2, 3) DOWN  : 0.1689
	// (2, 3) LEFT  : -0.4043
	// (2, 3) RIGHT : -0.2855
}

func Example_target() {
	fmt.Println(agent.Target(
		[]float64{1, 2, 3},
		[]bool{false, false, true},
		0.98,
		[]float64{1, 2, 3},
	))

	// Output:
	// [[1.98] [3.96] [3]]
}
