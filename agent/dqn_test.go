package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleDQNAgent() {
	e := env.NewGridWorld()
	s := rand.NewSource(1)
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
	// 0: 0.0106, 0.0067
	// (0, 0) UP    : -0.1061
	// (0, 0) DOWN  : -0.0477
	// (0, 0) LEFT  : 0.1598
	// (0, 0) RIGHT : -0.2651
	// (0, 1) UP    : -0.0276
	// (0, 1) DOWN  : -0.0434
	// (0, 1) LEFT  : 0.1282
	// (0, 1) RIGHT : -0.1524
	// (0, 2) UP    : 0.3050
	// (0, 2) DOWN  : 0.1813
	// (0, 2) LEFT  : -0.0105
	// (0, 2) RIGHT : 0.1292
	// (1, 0) UP    : -0.6514
	// (1, 0) DOWN  : 0.1122
	// (1, 0) LEFT  : -0.2203
	// (1, 0) RIGHT : -0.4828
	// (1, 2) UP    : -0.8882
	// (1, 2) DOWN  : -0.0749
	// (1, 2) LEFT  : -0.0114
	// (1, 2) RIGHT : -0.4060
	// (1, 3) UP    : -0.0637
	// (1, 3) DOWN  : -0.0555
	// (1, 3) LEFT  : 0.1383
	// (1, 3) RIGHT : -0.1666
	// (2, 0) UP    : 0.1170
	// (2, 0) DOWN  : 0.1217
	// (2, 0) LEFT  : 0.1104
	// (2, 0) RIGHT : -0.0538
	// (2, 1) UP    : 0.1745
	// (2, 1) DOWN  : 0.0733
	// (2, 1) LEFT  : 0.1112
	// (2, 1) RIGHT : -0.3337
	// (2, 2) UP    : 0.8314
	// (2, 2) DOWN  : -0.1022
	// (2, 2) LEFT  : 0.0660
	// (2, 2) RIGHT : 0.1461
	// (2, 3) UP    : -2.3387
	// (2, 3) DOWN  : 0.1448
	// (2, 3) LEFT  : -0.4197
	// (2, 3) RIGHT : -1.0760
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
