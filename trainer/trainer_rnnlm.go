package trainer

import (
	"math"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

var (
	_ RNNLM = (*model.RNNLM)(nil)
	_ RNNLM = (*model.LSTMLM)(nil)
	_ RNNLM = (*model.RNNLMGen)(nil)
)

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
	Verbose    func(epoch, j int, perplexity float64, m RNNLM)
}

type RNNLMTrainer struct {
	Model     RNNLM
	Optimizer Optimizer
	timeIdx   int
}

func NewRNNLM(m RNNLM, o Optimizer) *RNNLMTrainer {
	return &RNNLMTrainer{
		Model:     m,
		Optimizer: o,
	}
}

func (tr *RNNLMTrainer) Fit(in *RNNLMInput) {
	xs, ts := in.Train, in.TrainLabel
	dataSize := len(xs)

	jump := dataSize / in.BatchSize
	offsets := make([]int, in.BatchSize)
	for i := 0; i < in.BatchSize; i++ {
		offsets[i] = i * jump
	}

	maxIter := dataSize / (in.BatchSize * in.TimeSize)
	var totalLoss float64
	var lossCount int
	for epoch := 0; epoch < in.Epochs; epoch++ {
		for j := 0; j < maxIter; j++ {
			// (Time, N, 1)
			xbatch, tbatch := tr.Batch(xs, ts, offsets, in.TimeSize, in.BatchSize)

			// update
			loss := tr.Model.Forward(xbatch, tbatch)
			tr.Model.Backward()
			tr.Optimizer.Update(tr.Model)

			totalLoss += loss[0][0]
			lossCount++

			// verbose
			ppl := Perplexity(totalLoss, lossCount)
			in.Verbose(epoch, j, ppl, tr.Model)
		}

		totalLoss, lossCount = 0, 0
	}
}

func (tr *RNNLMTrainer) Batch(xs, ts, offsets []int, T, N int) ([]matrix.Matrix, []matrix.Matrix) {
	xbatch, tbatch := make([]matrix.Matrix, T), make([]matrix.Matrix, T)
	for t := 0; t < T; t++ {
		xv, tv := make(matrix.Matrix, N), make(matrix.Matrix, N)
		for i, offset := range offsets {
			xv[i] = []float64{float64(xs[(offset+tr.timeIdx)%len(xs)])}
			tv[i] = []float64{float64(ts[(offset+tr.timeIdx)%len(xs)])}
		}

		xbatch[t], tbatch[t] = xv, tv
		tr.timeIdx++
	}

	return xbatch, tbatch
}

func Perplexity(loss float64, count int) float64 {
	return math.Exp(loss / float64(count))
}
