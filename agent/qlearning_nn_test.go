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

func ExampleQLearningAgentNN() {
	env := env.NewGridWorld()
	a := &agent.QLearningAgentNN{
		Gamma:      0.9,
		Epsilon:    0.1,
		ActionSize: 4,
		Q: model.NewQNet(&model.QNetConfig{
			InputSize:  12,
			OutputSize: 4,
			HiddenSize: []int{100},
			WeightInit: weight.Xavier,
		}, rand.NewSource(1)),
		Optimizer: &optimizer.SGD{
			LearningRate: 0.01,
		},
		Source: rand.NewSource(1),
	}

	episodes := 2
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
		if s.Equals(env.GoalState) || s.Equals(env.WallState) {
			continue
		}

		q := a.Q.Predict(matrix.New(env.OneHot(&s)))
		for _, a := range env.Actions() {
			fmt.Printf("%s %-6s: %.4f\n", s, env.ActionMeaning[a], q[0][a])
		}
	}

	// Output:
}
