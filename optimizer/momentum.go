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
		o.v[k] = o.v[k].Mulf64(o.Momentum).Sub(grads[k].Mulf64(o.LearningRate)) // v[k] = momentum * v[k] - lr *grads[k]
		params[k] = params[k].Add(o.v[k])                                       // params[k] = params[k] + v[k]
	}

	return params
}
