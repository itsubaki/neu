package trainer

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Model     = (*model.Sequential)(nil)
	_ Model     = (*model.MLP)(nil)
	_ Optimizer = (*optimizer.AdaGrad)(nil)
	_ Optimizer = (*optimizer.Momentum)(nil)
	_ Optimizer = (*optimizer.SGD)(nil)
)

type Model interface {
	Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Forward(x, t matrix.Matrix) matrix.Matrix
	Backward() matrix.Matrix
	Layers() []model.Layer
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
}

type Optimizer interface {
	Update(m optimizer.Model) [][]matrix.Matrix
}

type Input struct {
	Train      matrix.Matrix
	TrainLabel matrix.Matrix
	Epochs     int
	BatchSize  int
	Verbose    func(epoch, j int, loss float64, m Model)
}

type Trainer struct {
	Model     Model
	Optimizer Optimizer
}

func New(m Model, o Optimizer) *Trainer {
	return &Trainer{
		Model:     m,
		Optimizer: o,
	}
}

// Fit trains the model using the provided optimizer.
func (t *Trainer) Fit(in *Input, s ...rand.Source) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	for i := 0; i < in.Epochs; i++ {
		// shuffle dataset
		xs, ts := Shuffle(in.Train, in.TrainLabel, s[0])
		for j := 0; j < len(in.Train)/in.BatchSize; j++ {
			// batch
			begin, end := Range(j, in.BatchSize)
			xbatch, tbatch := xs[begin:end], ts[begin:end]

			// update
			loss := t.Model.Forward(xbatch, tbatch)
			t.Model.Backward()
			t.Optimizer.Update(t.Model)

			// verbose
			in.Verbose(i, j, loss[0][0], t.Model)
		}
	}
}

// Range returns begin and end index of batch.
func Range(i, batchSize int) (int, int) {
	begin := i * batchSize
	end := begin + batchSize
	return begin, end
}

// Shuffle shuffles the dataset.
func Shuffle(x, t matrix.Matrix, s ...rand.Source) (matrix.Matrix, matrix.Matrix) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	xs, ts := make(matrix.Matrix, len(x)), make(matrix.Matrix, len(t))
	for i := 0; i < len(x); i++ {
		xs[i], ts[i] = x[i], t[i]
	}

	for i := 0; i < len(x); i++ {
		j := rng.Intn(i + 1)

		// swap
		xs[i], xs[j] = xs[j], xs[i]
		ts[i], ts[j] = ts[j], ts[i]
	}

	return xs, ts
}

// Random returns random index.
func Random(trainSize, batchSize int, s ...rand.Source) []int {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	tmp := make(map[int]bool)
	for c := 0; c < batchSize; {
		n := rng.Intn(trainSize)
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
