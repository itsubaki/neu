package agent_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/vector"
)

func ExampleAgent() {
	a := &agent.Agent{
		Epsilon: 0.5,
		Qs:      []float64{0, 0, 0, 0, 0},
		Ns:      []float64{0, 0, 0, 0, 0},
		Source:  rand.NewSource(1),
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
	// 1: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 0: [1 1 0 0 0]
	// 4: [1 1 0 0 1]
}

func ExampleAgent_bandit() {
	arms, steps, runs, eps := 10, 1000, 200, 0.1
	s := rand.NewSource(1)

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
	// step=190: mean(rate)=0.8534
	// step=191: mean(rate)=0.9438
	// step=192: mean(rate)=0.9169
	// step=193: mean(rate)=0.7310
	// step=194: mean(rate)=0.9367
	// step=195: mean(rate)=0.8822
	// step=196: mean(rate)=0.8339
	// step=197: mean(rate)=0.8462
	// step=198: mean(rate)=0.7172
	// step=199: mean(rate)=0.8401
}

func Example_rand() {
	for i := 0; i < 5; i++ {
		rng := rand.New(rand.NewSource(1))
		fmt.Println(rng.Float64())
	}

	s := rand.NewSource(1)
	for i := 0; i < 5; i++ {
		rng := rand.New(s)
		fmt.Println(rng.Float64())
	}

	// Output:
	// 0.6046602879796196
	// 0.6046602879796196
	// 0.6046602879796196
	// 0.6046602879796196
	// 0.6046602879796196
	// 0.6046602879796196
	// 0.9405090880450124
	// 0.6645600532184904
	// 0.4377141871869802
	// 0.4246374970712657
}
