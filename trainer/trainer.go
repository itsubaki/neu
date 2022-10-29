package trainer

import (
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
)

type Model interface {
	Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Forward(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Backward(x, t matrix.Matrix) matrix.Matrix
	Optimize(opt model.Optimizer) [][]matrix.Matrix
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
}

type Input struct {
	Train      matrix.Matrix
	TrainLabel matrix.Matrix
	Epochs     int
	BatchSize  int
	Verbose    func(epoch, j int, m Model)
}

type Trainer struct {
	Model     Model
	Optimizer model.Optimizer
}

func (t *Trainer) Fit(in *Input) {
	for i := 0; i < in.Epochs; i++ {
		// shuffle dataset
		xs, ts := Shuffle(in.Train, in.TrainLabel)
		for j := 0; j < len(in.Train)/in.BatchSize; j++ {
			// batch
			begin, end := Range(j, in.BatchSize)
			xbatch, tbatch := xs[begin:end], ts[begin:end]

			// update
			t.Model.Forward(xbatch, tbatch)
			t.Model.Backward(xbatch, tbatch)
			t.Model.Optimize(t.Optimizer)

			// verbose
			in.Verbose(i, j, t.Model)
		}
	}
}

func Range(i, batchSize int) (int, int) {
	begin := i * batchSize
	end := begin + batchSize
	return begin, end
}

func Shuffle(x, t matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	xs, ts := make(matrix.Matrix, 0), make(matrix.Matrix, 0)
	for i := 0; i < len(x); i++ {
		xs, ts = append(xs, x[i]), append(ts, t[i])
	}

	for i := 0; i < len(x); i++ {
		j := rand.Intn(i + 1)

		// swap
		xs[i], xs[j] = xs[j], xs[i]
		ts[i], ts[j] = ts[j], ts[i]
	}

	return xs, ts
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
