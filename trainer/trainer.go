package trainer

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.CBOWNegs)(nil)
)

var (
	_ Optimizer = (*optimizer.AdaGrad)(nil)
	_ Optimizer = (*optimizer.Adam)(nil)
	_ Optimizer = (*optimizer.Momentum)(nil)
	_ Optimizer = (*optimizer.SGD)(nil)
)

type Model interface {
	Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Forward(x, t matrix.Matrix) matrix.Matrix
	Backward() matrix.Matrix
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
	SetParams(p [][]matrix.Matrix)
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
func (tr *Trainer) Fit(in *Input, s ...rand.Source) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	for i := 0; i < in.Epochs; i++ {
		// shuffle dataset
		xs, ts := vector.Shuffle(in.Train, in.TrainLabel, s[0])
		for j := 0; j < len(in.Train)/in.BatchSize; j++ {
			// batch
			begin, end := Range(j, in.BatchSize)
			xbatch, tbatch := xs[begin:end], ts[begin:end]

			// update
			loss := tr.Model.Forward(xbatch, tbatch)
			tr.Model.Backward()
			tr.Optimizer.Update(tr.Model)

			// verbose
			in.Verbose(i, j, loss[0][0], tr.Model)
		}
	}
}

// Range returns begin and end index of batch.
func Range(i, batchSize int) (int, int) {
	begin := i * batchSize
	end := begin + batchSize
	return begin, end
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
	ymax, tmax := y.Argmax(), t.Argmax()
	c := vector.MatchCount(ymax, tmax)
	return float64(c) / float64(len(ymax))
}
