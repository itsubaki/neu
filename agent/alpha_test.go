package agent_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/vector"
)

func ExampleAlphaAgent() {
	a := &agent.AlphaAgent{
		Epsilon: 0.5,
		Alpha:   0.8,
		Qs:      []float64{0, 0, 0, 0, 0},
		Source:  rand.Const(1),
	}

	for i := 0; i < 10; i++ {
		action := a.GetAction()
		a.Update(action, 1.0)
		fmt.Printf("%v: %.4f\n", action, a.Qs)
	}

	// Output:
	// 0: [0.8000 0.0000 0.0000 0.0000 0.0000]
	// 0: [0.9600 0.0000 0.0000 0.0000 0.0000]
	// 0: [0.9920 0.0000 0.0000 0.0000 0.0000]
	// 0: [0.9984 0.0000 0.0000 0.0000 0.0000]
	// 2: [0.9984 0.0000 0.8000 0.0000 0.0000]
	// 0: [0.9997 0.0000 0.8000 0.0000 0.0000]
	// 2: [0.9997 0.0000 0.9600 0.0000 0.0000]
	// 4: [0.9997 0.0000 0.9600 0.0000 0.8000]
	// 4: [0.9997 0.0000 0.9600 0.0000 0.9600]
	// 3: [0.9997 0.0000 0.9600 0.8000 0.9600]
}

func ExampleAlphaAgent_bandit() {
	arms, steps, runs := 10, 1000, 200
	eps, alpha := 0.1, 0.8
	s := rand.Const(1)

	all := make([][]float64, runs)
	for r := 0; r < runs; r++ {
		bandit := env.NewNonStatBandit(arms, s)
		agent := &agent.AlphaAgent{Epsilon: eps, Alpha: alpha, Qs: make([]float64, arms), Source: s}

		var total float64
		rates := make([]float64, steps)
		for i := 0; i < steps; i++ {
			action := agent.GetAction()
			reward := bandit.Play(action)
			agent.Update(action, reward)

			total += reward
			rates[i] = total / float64(i+1)
		}

		all[r] = rates
	}

	for _, i := range []int{190, 191, 192, 193, 194, 195, 196, 197, 198, 199} {
		fmt.Printf("step=%3v: mean(rate)=%.4f\n", i, vector.Mean(all[i]))
	}

	// Output:
	// step=190: mean(rate)=0.8010
	// step=191: mean(rate)=0.9436
	// step=192: mean(rate)=0.9249
	// step=193: mean(rate)=0.6519
	// step=194: mean(rate)=0.8373
	// step=195: mean(rate)=0.8367
	// step=196: mean(rate)=0.8759
	// step=197: mean(rate)=0.9196
	// step=198: mean(rate)=0.8844
	// step=199: mean(rate)=0.8871
}
