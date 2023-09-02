package env

import (
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

type Bandit struct {
	Rates []float64
	RNG   *rand.Rand
}

func NewBandit(arms int, s rand.Source) *Bandit {
	return &Bandit{
		Rates: vector.Rand(arms, s),
		RNG:   rand.New(s),
	}
}

func (b *Bandit) Play(arm int) float64 {
	if b.Rates[arm] > b.RNG.Float64() {
		return 1
	}

	return 0
}

type NonStatBandit struct {
	Arms  int
	Rates []float64
	RNG   *rand.Rand
}

func NewNonStatBandit(arms int, s rand.Source) *NonStatBandit {
	return &NonStatBandit{
		Arms:  arms,
		Rates: vector.Rand(arms, s),
		RNG:   rand.New(s),
	}
}

func (b *NonStatBandit) Play(arm int, s ...rand.Source) float64 {
	rate := b.Rates[arm]
	randn := vector.Randn(b.Arms, s...)
	b.Rates = vector.Add(b.Rates, vector.Mul(randn, -0.1))

	if rate > b.RNG.Float64() {
		return 1
	}

	return 0
}
