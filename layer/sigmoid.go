package layer

import "github.com/itsubaki/neu/activation"

type Sigmoid struct {
	out []float64
}

func (l *Sigmoid) Forward(x, _ []float64) []float64 {
	l.out = activation.Sigmoid(x)
	return l.out
}

func (l *Sigmoid) Backward(dout []float64) ([]float64, []float64) {
	dx := make([]float64, 0)
	for i := range dout {
		dx = append(dx, dout[i]*(1.0-l.out[i])*l.out[i])
	}

	return dx, []float64{}
}
