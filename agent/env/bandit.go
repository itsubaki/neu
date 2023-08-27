package env

import (
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type Bandit struct {
	Rates  []float64
	Source rand.Source
}

func NewBandit(arms int, s rand.Source) *Bandit {
	return &Bandit{
		Rates:  vector.Rand(arms, s),
		Source: s,
	}
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

func NewNonStatBandit(arms int, s rand.Source) *NonStatBandit {
	return &NonStatBandit{
		Arms:   arms,
		Rates:  vector.Rand(arms, s),
		Source: s,
	}
}

func (b *NonStatBandit) Play(arm int) float64 {
	rate := b.Rates[arm]
	randn := vector.Randn(b.Arms, b.Source)
	b.Rates = vector.Add(b.Rates, vector.Mul(randn, -0.1))

	if rate > rand.New(b.Source).Float64() {
		return 1
	}

	return 0
}
