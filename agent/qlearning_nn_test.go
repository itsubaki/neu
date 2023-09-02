package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleQLearningAgentNN() {
	env := env.NewGridWorld()
	a := &agent.QLearningAgentNN{
		Gamma:      0.9,
		Epsilon:    0.1,
		ActionSize: 4,
		Q: model.NewQNet(&model.QNetConfig{
			InputSize:  12,
			OutputSize: 4,
			HiddenSize: 100,
			WeightInit: weight.Xavier,
		}, rand.NewSource(1)),
		Optimizer: &optimizer.SGD{
			LearningRate: 0.01,
		},
		Source: rand.NewSource(1),
	}

	episodes := 10
	for i := 0; i < episodes; i++ {
		state := env.OneHot(env.Reset())
		var totalLoss float64
		var count int

		for {
			action := a.GetAction(state)
			next, reward, done := env.Step(action)
			nextoh := env.OneHot(next)
			loss := a.Update(state, action, reward, nextoh, done)
			totalLoss += loss[0][0]
			count++

			if done {
				break
			}

			state = nextoh
		}

		fmt.Println(i, totalLoss/float64(count))
	}

	for _, s := range env.State {
		if s.Equals(env.GoalState) {
			continue
		}

		q := a.Q.Predict(env.OneHot(&s))
		for _, a := range env.Actions() {
			fmt.Printf("%s %-6s: %.4f\n", s, env.ActionMeaning[a], q[0][a])
		}
	}

	// Output:
	// 0 0.021614082692006962
	// 1 0.020214900789190127
	// 2 0.0157352274399023
	// 3 0.024148847567304278
	// 4 0.06361324855857328
	// 5 0.05365485126764651
	// 6 0.03897906184890752
	// 7 0.041966519004113224
	// 8 0.029798790108796564
	// 9 0.031657969064067815
	// (0, 0) UP    : 0.2969
	// (0, 0) DOWN  : 0.3578
	// (0, 0) LEFT  : 0.7252
	// (0, 0) RIGHT : 0.5834
	// (0, 1) UP    : -0.7730
	// (0, 1) DOWN  : -1.1099
	// (0, 1) LEFT  : -0.9157
	// (0, 1) RIGHT : -1.2035
	// (0, 2) UP    : 0.4119
	// (0, 2) DOWN  : -0.0461
	// (0, 2) LEFT  : -0.2786
	// (0, 2) RIGHT : -0.2180
	// (1, 0) UP    : 0.3308
	// (1, 0) DOWN  : -0.5254
	// (1, 0) LEFT  : -0.4311
	// (1, 0) RIGHT : -0.2963
	// (1, 1) UP    : -0.2138
	// (1, 1) DOWN  : -0.4489
	// (1, 1) LEFT  : 0.1267
	// (1, 1) RIGHT : 0.2966
	// (1, 2) UP    : 0.1548
	// (1, 2) DOWN  : -0.3841
	// (1, 2) LEFT  : -0.2472
	// (1, 2) RIGHT : 0.0078
	// (1, 3) UP    : -0.2564
	// (1, 3) DOWN  : -0.1858
	// (1, 3) LEFT  : -0.4932
	// (1, 3) RIGHT : -0.0310
	// (2, 0) UP    : 0.2096
	// (2, 0) DOWN  : -0.0150
	// (2, 0) LEFT  : 0.6087
	// (2, 0) RIGHT : 0.2333
	// (2, 1) UP    : 0.0714
	// (2, 1) DOWN  : -0.2692
	// (2, 1) LEFT  : -0.1603
	// (2, 1) RIGHT : -0.2023
	// (2, 2) UP    : -0.0546
	// (2, 2) DOWN  : -0.4982
	// (2, 2) LEFT  : -0.2138
	// (2, 2) RIGHT : -0.2034
	// (2, 3) UP    : -0.0442
	// (2, 3) DOWN  : -0.1283
	// (2, 3) LEFT  : -0.1734
	// (2, 3) RIGHT : -0.0173
}
