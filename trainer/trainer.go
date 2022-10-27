package trainer

import (
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

var _ Model = (*model.MLP)(nil)

type Model interface {
	Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Loss(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Gradient(x, t matrix.Matrix) [][]matrix.Matrix
	Optimize(grads [][]matrix.Matrix)
}

type Input struct {
	Model             Model
	Train, TrainLabel matrix.Matrix
	Test, TestLabel   matrix.Matrix
	Iter              int
	BatchSize         int
	Verbose           func(i int, m Model, xbatch, tbatch, xtbatch, ttbatch matrix.Matrix)
}

func Fit(in *Input) {
	for i := 0; i < in.Iter+1; i++ {
		// batch
		mask := Random(len(in.Train), in.BatchSize)
		xbatch := matrix.Batch(in.Train, mask)
		tbatch := matrix.Batch(in.TrainLabel, mask)

		// update
		grads := in.Model.Gradient(xbatch, tbatch)
		in.Model.Optimize(grads)

		if in.Verbose == nil {
			continue
		}

		if i%(in.Iter/in.BatchSize) == 0 {
			// test data
			mask := Random(len(in.Test), in.BatchSize)
			xtbatch := matrix.Batch(in.Test, mask)
			ttbatch := matrix.Batch(in.TestLabel, mask)

			// verbose
			in.Verbose(i, in.Model, xbatch, tbatch, xtbatch, ttbatch)
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
