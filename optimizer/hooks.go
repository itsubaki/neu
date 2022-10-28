package optimizer

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Hook = weight.Decay(0.1)
)

type Hook func(params, grads [][]matrix.Matrix) [][]matrix.Matrix
