package optimizer

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

type Adam struct {
	LearningRate float64
	Beta1, Beta2 float64
	Hooks        []Hook
	m, v         [][]matrix.Matrix
	iter         int
}

func (o *Adam) Update(m Model) [][]matrix.Matrix {
	params, grads := m.Params(), m.Grads()
	for _, h := range o.Hooks {
		grads = h(params, grads)
	}

	if len(o.m) == 0 {
		o.m, o.v = ZeroLike(params), ZeroLike(params)
	}

	o.iter++
	top := math.Sqrt(1.0 - math.Pow(o.Beta2, float64(o.iter))) // sqrt(1 - beta2^t)
	bottom := 1.0 - math.Pow(o.Beta1, float64(o.iter))         // 1 - beta1^t
	lrt := o.LearningRate * top / bottom                       // lr * sqrt(1 - beta2^t) / (1 - beta1^t)

	updated := ZeroLike(params)
	for i := range params {
		for j := range params[i] {
			o.m[i][j] = o.m[i][j].Add(grads[i][j].Sub(o.m[i][j]).MulC(1.0 - o.Beta1))        // m = m + (1 - beta1) * (grads - m)
			o.v[i][j] = o.v[i][j].Add(grads[i][j].Pow2().Sub(o.v[i][j]).MulC(1.0 - o.Beta2)) // v = v + (1 - beta2) * (grads * grads - v)
			updated[i][j] = matrix.F3(params[i][j], o.m[i][j], o.v[i][j], adam(lrt))         // params = params - lrt * m / (sqrt(v) + 1e-7)
		}
	}

	m.SetParams(updated)
	return updated
}

func adam(learningRate float64) func(p, m, v float64) float64 {
	return func(p, m, v float64) float64 { return p - learningRate*m/(math.Sqrt(v)+1e-7) }
}
