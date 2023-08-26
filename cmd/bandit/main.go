package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/agent"
	"github.com/itsubaki/neu/math/vector"
)

type Bandit struct {
	Rates  []float64
	Source rand.Source
}

func (b *Bandit) Play(arm int) float64 {
	rng := rand.New(b.Source)
	if b.Rates[arm] > rng.Float64() {
		return 1
	}

	return 0
}

type NonStatBandit struct {
	Arms   int
	Rates  []float64
	Source rand.Source
}

func (b *NonStatBandit) Play(arm int) float64 {
	rate := b.Rates[arm]
	b.Rates = vector.Add(b.Rates, vector.Mul(vector.Randn(b.Arms), -0.1))

	if rate > rand.New(b.Source).Float64() {
		return 1
	}

	return 0
}

func main() {
	var arms, steps, runs int
	var eps, alpha float64
	flag.IntVar(&arms, "arms", 10, "")
	flag.IntVar(&steps, "steps", 1000, "")
	flag.IntVar(&runs, "runs", 200, "")
	flag.Float64Var(&eps, "epsilon", 0.1, "")
	flag.Float64Var(&alpha, "alpha", 0.8, "")
	flag.Parse()

	s := rand.NewSource(time.Now().UnixNano())
	all := make([][]float64, runs)
	for r := 0; r < runs; r++ {
		bandit := NonStatBandit{Arms: arms, Rates: vector.Rand(arms), Source: s}
		agent := agent.AlphaAgent{Epsilon: eps, Alpha: alpha, Qs: make([]float64, arms), Source: s}

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

	for i, r := range all {
		fmt.Printf("step=%3v: mean(rate)=%.4f\n", i, vector.Mean(r))
	}
}
