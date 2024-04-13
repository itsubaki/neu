package agent_test

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/vector"
)

func ExampleAgent() {
	a := &agent.Agent{
		Epsilon: 0.5,
		Qs:      []float64{0, 0, 0, 0, 0},
		Ns:      []float64{0, 0, 0, 0, 0},
		Source:  rand.Const(1),
	}

	for i := 0; i < 10; i++ {
		action := a.GetAction()
		a.Update(action, 1.0)
		fmt.Printf("%v: %v\n", action, a.Qs)
	}

	// Output:
	// 0: [1 0 0 0 0]
	// 0: [1 0 0 0 0]
	// 0: [1 0 0 0 0]
	// 0: [1 0 0 0 0]
	// 2: [1 0 1 0 0]
	// 0: [1 0 1 0 0]
	// 2: [1 0 1 0 0]
	// 4: [1 0 1 0 1]
	// 4: [1 0 1 0 1]
	// 3: [1 0 1 1 1]
}

func ExampleAgent_bandit() {
	arms, steps, runs, eps := 10, 1000, 200, 0.1
	s := rand.Const(1)

	all := make([][]float64, runs)
	for r := 0; r < runs; r++ {
		bandit := env.NewNonStatBandit(arms, s)
		agent := &agent.Agent{Epsilon: eps, Qs: make([]float64, arms), Ns: make([]float64, arms), Source: s}

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
	// step=190: mean(rate)=0.7875
	// step=191: mean(rate)=0.9418
	// step=192: mean(rate)=0.9185
	// step=193: mean(rate)=0.6062
	// step=194: mean(rate)=0.8260
	// step=195: mean(rate)=0.8314
	// step=196: mean(rate)=0.8781
	// step=197: mean(rate)=0.8273
	// step=198: mean(rate)=0.8844
	// step=199: mean(rate)=0.8903
}

func Example_rand() {
	for i := 0; i < 5; i++ {
		r := randv2.New(rand.Const(1))
		fmt.Println(r.Float64())
	}

	s := rand.Const(1)
	for i := 0; i < 5; i++ {
		r := randv2.New(s)
		fmt.Println(r.Float64())
	}

	// Output:
	// 0.23842319087387442
	// 0.23842319087387442
	// 0.23842319087387442
	// 0.23842319087387442
	// 0.23842319087387442
	// 0.23842319087387442
	// 0.50092138792625
	// 0.04999911180706662
	// 0.4894631469238666
	// 0.7500167893718852
}
