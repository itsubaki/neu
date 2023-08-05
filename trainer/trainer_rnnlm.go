package trainer

import (
	"math"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

var _ RNNLM = (*model.RNNLM)(nil)

type RNNLM interface {
	Predict(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix
	Forward(xs, ts []matrix.Matrix) matrix.Matrix
	Backward() []matrix.Matrix
	Params() [][]matrix.Matrix
	Grads() [][]matrix.Matrix
	SetParams(p [][]matrix.Matrix)
}

type RNNLMInput struct {
	Train      []int
	TrainLabel []int
	Epochs     int
	BatchSize  int
	TimeSize   int
	Verbose    func(epoch int, perplexity float64, m RNNLM)
}

type RNNLMTrainer struct {
	Model     RNNLM
	Optimizer Optimizer
}

func NewRNNLM(m RNNLM, o Optimizer) *RNNLMTrainer {
	return &RNNLMTrainer{
		Model:     m,
		Optimizer: o,
	}
}

func (t *RNNLMTrainer) Fit(in *RNNLMInput) {
	xs, ts := in.Train, in.TrainLabel
	dataSize := len(xs)

	jump := dataSize / in.BatchSize
	offsets := make([]int, in.BatchSize)
	for i := 0; i < in.BatchSize; i++ {
		offsets[i] = i * jump
	}

	maxIter := dataSize / (in.BatchSize * in.TimeSize)
	var timeIdx, lossCount int
	var totalLoss float64

	for epoch := 0; epoch < in.Epochs; epoch++ {
		for iter := 0; iter < maxIter; iter++ {
			// (Time, N, 1)
			xbatch, tbatch := make([]matrix.Matrix, in.TimeSize), make([]matrix.Matrix, in.TimeSize)
			for t := 0; t < in.TimeSize; t++ {
				xv, tv := make(matrix.Matrix, in.BatchSize), make(matrix.Matrix, in.BatchSize)
				for i, offset := range offsets {
					xv[i] = []float64{float64(xs[(offset+timeIdx)%dataSize])}
					tv[i] = []float64{float64(ts[(offset+timeIdx)%dataSize])}
				}

				xbatch[t], tbatch[t] = xv, tv
				timeIdx++
			}

			loss := t.Model.Forward(xbatch, tbatch)
			t.Model.Backward()
			t.Optimizer.Update(t.Model)

			totalLoss += loss[0][0]
			lossCount++
		}

		ppl := Perplexity(totalLoss, lossCount)
		totalLoss, lossCount = 0, 0

		in.Verbose(epoch, ppl, t.Model)
	}
}

func Perplexity(loss float64, count int) float64 {
	return math.Exp(loss / float64(count))
}
