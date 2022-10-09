package layer

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/loss"
)

type SoftmaxWithLoss struct {
	loss float64
	y    []float64
	t    []float64
}

func (l *SoftmaxWithLoss) Forward(x, t []float64) []float64 {
	l.t = t
	l.y = activation.Softmax(x)
	l.loss = loss.CrossEntropyError(l.y, l.t)

	return []float64{l.loss}
}

func (l *SoftmaxWithLoss) Backward(_ []float64) ([]float64, []float64) {
	size := len(l.t)
	dx := make([]float64, 0)
	for i := range l.y {
		dx = append(dx, (l.y[i]-l.t[i])/float64(size))
	}

	return dx, []float64{}
}
