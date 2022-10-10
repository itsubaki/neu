package optimizer

import "github.com/itsubaki/neu/math/matrix"

type Momentum struct {
	LearningRate float64
	Momentum     float64
	v            map[string]matrix.Matrix
}

func (o *Momentum) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	if o.v == nil {
		o.v = make(map[string]matrix.Matrix)
		for k, v := range params {
			p, q := v.Dimension()
			o.v[k] = matrix.Zero(p, q)
		}
	}

	for k := range params {
		m := o.v[k].Func(func(v float64) float64 { return o.Momentum * v })
		g := grads[k].Func(func(v float64) float64 { return -1.0 * o.LearningRate * v })
		o.v[k] = m.Add(g)                 // v[k] = momentum * v[k] - lr *grads[k]
		params[k] = params[k].Add(o.v[k]) // params[k] = params[k] + v[k]
	}

	return params
}
