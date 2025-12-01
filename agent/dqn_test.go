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
	for i := range episodes {
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
	// 0: 0.0088, -0.0082
	// (0, 0) UP    : 0.0940
	// (0, 0) DOWN  : -0.0011
	// (0, 0) LEFT  : -0.0951
	// (0, 0) RIGHT : -0.0126
	// (0, 1) UP    : 0.4905
	// (0, 1) DOWN  : 0.1174
	// (0, 1) LEFT  : -0.1345
	// (0, 1) RIGHT : 0.0601
	// (0, 2) UP    : -1.1773
	// (0, 2) DOWN  : 0.0345
	// (0, 2) LEFT  : -0.2830
	// (0, 2) RIGHT : 0.0377
	// (1, 0) UP    : 0.0477
	// (1, 0) DOWN  : 0.0067
	// (1, 0) LEFT  : -0.2127
	// (1, 0) RIGHT : 0.0289
	// (1, 2) UP    : 0.2000
	// (1, 2) DOWN  : 0.1005
	// (1, 2) LEFT  : -0.1068
	// (1, 2) RIGHT : 0.0478
	// (1, 3) UP    : 0.9551
	// (1, 3) DOWN  : 0.1666
	// (1, 3) LEFT  : 0.1643
	// (1, 3) RIGHT : -0.1554
	// (2, 0) UP    : -0.3087
	// (2, 0) DOWN  : 0.2201
	// (2, 0) LEFT  : -0.1539
	// (2, 0) RIGHT : 0.0535
	// (2, 1) UP    : 0.1924
	// (2, 1) DOWN  : -0.0042
	// (2, 1) LEFT  : -0.1180
	// (2, 1) RIGHT : -0.0138
	// (2, 2) UP    : -1.0109
	// (2, 2) DOWN  : 0.0833
	// (2, 2) LEFT  : -0.3053
	// (2, 2) RIGHT : -0.0602
	// (2, 3) UP    : -0.4370
	// (2, 3) DOWN  : 0.0578
	// (2, 3) LEFT  : -0.3560
	// (2, 3) RIGHT : -0.1734
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
