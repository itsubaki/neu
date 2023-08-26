package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/math/vector"
)

type Bandit struct {
	Rates []float64
}

func (b *Bandit) Play(arm int) float64 {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	if b.Rates[arm] > rng.Float64() {
		return 1
	}

	return 0
}

type Agent struct {
	Epsilon float64
	Qs      []float64
	Ns      []float64
}

func (a *Agent) GetAction() int {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	if a.Epsilon > rng.Float64() {
		return rng.Intn(len(a.Qs))
	}

	return vector.Argmax(a.Qs)
}

func (a *Agent) Update(action int, reward float64) {
	a.Ns[action]++
	a.Qs[action] += (reward - a.Qs[action]) / a.Ns[action]
}

func main() {
	arms, steps := 10, 1000
	bandit := Bandit{Rates: vector.Rand(arms)}
	agent := Agent{Epsilon: 0.1, Qs: make([]float64, arms), Ns: make([]float64, arms)}

	var total float64
	rewards, rates := make([]float64, steps), make([]float64, steps)

	for i := 0; i < steps; i++ {
		action := agent.GetAction()
		reward := bandit.Play(action)
		agent.Update(action, reward)

		total += reward
		rewards[i] = total
		rates[i] = total / float64(i+1)
	}

	fmt.Printf("%4f\n", bandit.Rates)
	fmt.Printf("%4f\n", agent.Qs)
	fmt.Println(rewards[len(rewards)-1])
	fmt.Println(rates[len(rates)-1])
}
