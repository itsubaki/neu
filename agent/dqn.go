package agent

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

type DQNAgent struct {
	Gamma        float64
	Epsilon      float64
	ReplayBuffer *ReplayBuffer
	ActionSize   int
	Q            *model.QNet
	QTarget      *model.QNet
	Optimizer    *optimizer.Adam
	Source       randv2.Source
}

func (a *DQNAgent) Sync() {
	a.QTarget.Sync(a.Q)
}

func (a *DQNAgent) GetAction(state []float64) int {
	rng := randv2.New(a.Source)
	if a.Epsilon > rng.Float64() {
		return rng.IntN(a.ActionSize)
	}

	qs := a.Q.Predict(matrix.New(state))
	return vector.Argmax(qs[0])
}

func (a *DQNAgent) Update(state []float64, action int, reward float64, next []float64, done bool) matrix.Matrix {
	a.ReplayBuffer.Add(state, action, reward, next, done)
	if a.ReplayBuffer.Len() < a.ReplayBuffer.BatchSize {
		return matrix.New([]float64{0.0})
	}

	s, ac, r, n, d := a.ReplayBuffer.Batch()
	qs := a.Q.Predict(s)
	q := q(qs, ac)

	nextqs := a.QTarget.Predict(n)
	nextq := nextqs.MaxAxis1()
	target := target(r, d, a.Gamma, nextq)

	loss := a.Q.Loss(q, target)

	a.Q.Backward()
	a.Optimizer.Update(a.Q)
	return loss
}

func q(qs matrix.Matrix, action []int) matrix.Matrix {
	out := matrix.Zero(len(qs), 1)
	for i := 0; i < len(qs); i++ {
		out[i] = []float64{qs[i][action[i]]}
	}

	return out
}

func target(r []float64, done []bool, gamma float64, nextq []float64) matrix.Matrix {
	single := func(r float64, done bool, gamma float64, nextq float64) float64 {
		if done {
			return r
		}

		return r + gamma*nextq
	}

	out := matrix.Zero(len(r), 1)
	for i := 0; i < len(r); i++ {
		out[i] = []float64{single(r[i], done[i], gamma, nextq[i])}
	}

	return out
}
