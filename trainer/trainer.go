package trainer

import (
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
)

type Model interface {
	Predict(x matrix.Matrix) matrix.Matrix
	Loss(x, t matrix.Matrix) matrix.Matrix
	Gradient(x, t matrix.Matrix) map[string]matrix.Matrix
	Optimize(grads map[string]matrix.Matrix)
}

type Input struct {
	Model        Model
	X, T, XT, TT matrix.Matrix
	Iter         int
	BatchSize    int
	Verbose      bool
	Func         func(i int, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix)
}

func Train(in *Input) {
	for i := 0; i < in.Iter+1; i++ {
		// batch
		mask := Random(len(in.X), in.BatchSize)
		xbatch := matrix.Batch(in.X, mask)
		tbatch := matrix.Batch(in.T, mask)

		// update
		grads := in.Model.Gradient(xbatch, tbatch)
		in.Model.Optimize(grads)

		if !in.Verbose {
			continue
		}

		if i%(in.Iter/in.BatchSize) == 0 {
			// test data
			mask := Random(len(in.XT), in.BatchSize)
			xtbatch := matrix.Batch(in.XT, mask)
			ttbatch := matrix.Batch(in.TT, mask)

			// func
			in.Func(i, xbatch, tbatch, xtbatch, ttbatch)
		}
	}
}

func Random(trainSize, batchSize int) []int {
	tmp := make(map[int]bool)

	for c := 0; c < batchSize; {
		n := rand.Intn(trainSize)
		if _, ok := tmp[n]; !ok {
			tmp[n] = true
			c++
		}
	}

	out := make([]int, 0, len(tmp))
	for k := range tmp {
		out = append(out, k)
	}

	return out
}

func Accuracy(y, t matrix.Matrix) float64 {
	count := func(x, y []int) int {
		var c int
		for i := range x {
			if x[i] == y[i] {
				c++
			}
		}

		return c
	}

	ymax := y.Argmax()
	tmax := t.Argmax()

	c := count(ymax, tmax)
	return float64(c) / float64(len(ymax))
}
