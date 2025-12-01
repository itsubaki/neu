package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	var episode, syncInterval, hiddenSize, bufferSize, batchSize int
	var gamma, epsilon, alpha, beta1, beta2 float64
	flag.IntVar(&episode, "episode", 300, "")
	flag.IntVar(&syncInterval, "sync-interval", 100, "")
	flag.IntVar(&hiddenSize, "hidden-size", 128, "")
	flag.IntVar(&bufferSize, "buffer-size", 10000, "")
	flag.IntVar(&batchSize, "batch-size", 2, "")
	flag.Float64Var(&gamma, "gamma", 0.98, "")
	flag.Float64Var(&epsilon, "epsilon", 0.1, "")
	flag.Float64Var(&alpha, "alpha", 0.001, "")
	flag.Float64Var(&beta1, "beta1", 0.9, "")
	flag.Float64Var(&beta2, "beta2", 0.999, "")
	flag.Parse()

	e := env.NewGridWorld()
	a := &agent.DQNAgent{
		Gamma:        gamma,
		Epsilon:      epsilon,
		ActionSize:   len(e.ActionSpace),
		ReplayBuffer: agent.NewReplayBuffer(bufferSize, batchSize),
		Q: model.NewQNet(&model.QNetConfig{
			InputSize:  e.Size(),
			OutputSize: len(e.ActionSpace),
			HiddenSize: []int{hiddenSize, hiddenSize},
			WeightInit: weight.Xavier,
		}),
		QTarget: model.NewQNet(&model.QNetConfig{
			InputSize:  e.Size(),
			OutputSize: len(e.ActionSpace),
			HiddenSize: []int{hiddenSize, hiddenSize},
			WeightInit: weight.Xavier,
		}),
		Optimizer: &optimizer.Adam{
			Alpha: alpha,
			Beta1: beta1,
			Beta2: beta2,
		},
		Source: rand.NewSource(rand.MustRead()),
	}

	for i := range episode {
		state := e.Reset()
		var totalLoss, totalReward float64
		var count int

		for {
			stateoh := e.OneHot(state)
			action := a.GetAction(stateoh)
			next, reward, done := e.Step(action)
			loss := a.Update(stateoh, action, reward, e.OneHot(next), done)
			state = next

			totalLoss += loss[0][0]
			totalReward += reward
			count++

			if done {
				break
			}

		}

		if (i+1)%syncInterval == 0 {
			a.Sync()
			fmt.Printf("%d: sync\n", i)
		}

		if (i+1)%10 == 0 {
			fmt.Printf("%d: %.8f, %.8f\n", i, totalLoss/float64(count), totalReward/float64(count))

			for _, s := range e.State {
				if s.Equals(e.GoalState) || s.Equals(e.WallState) {
					continue
				}

				q := a.Q.Predict(matrix.New(e.OneHot(&s)))
				for _, a := range e.Actions() {
					fmt.Printf("%s %-6s: %.4f\n", s, e.ActionMeaning[a], q[0][a])
				}
			}
		}
	}
}
