package agent

import (
	"math/rand"

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
	Source       rand.Source
}

func (a *DQNAgent) Sync() {
	a.QTarget.Sync(a.Q)
}

func (a *DQNAgent) GetAction(state []float64) int {
	rng := rand.New(a.Source)
	if a.Epsilon > rng.Float64() {
		return rng.Intn(a.ActionSize)
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
	q := matrix.Zero(a.ReplayBuffer.BatchSize, 1)
	for i := 0; i < a.ReplayBuffer.BatchSize; i++ {
		q[i] = []float64{qs[i][ac[i]]}
	}

	nextqs := a.QTarget.Predict(n)
	nextq := nextqs.MaxAxis1()

	target := matrix.Zero(a.ReplayBuffer.BatchSize, 1)
	for i := 0; i < a.ReplayBuffer.BatchSize; i++ {
		target[i] = Target(r[i], nextq[i], d[i], a.Gamma)
	}

	loss := a.Q.Loss(q, target)

	a.Q.Backward()
	a.Optimizer.Update(a.Q)
	return loss
}

func Target(r, nextq float64, done bool, gamma float64) []float64 {
	if done {
		return []float64{r}
	}

	return []float64{r + gamma*nextq}
}
