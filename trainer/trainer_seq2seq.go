package trainer

import (
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
)

type Seq2SeqInput struct {
	Train      [][]int
	TrainLabel [][]int
	Epochs     int
	BatchSize  int
	Verbose    func(epoch, j int, loss float64, m *model.Seq2Seq)
}

type Seq2SeqTrainer struct {
	Model     *model.Seq2Seq
	Optimizer Optimizer
}

func NewSeq2Seq(m *model.Seq2Seq, o Optimizer) *Seq2SeqTrainer {
	return &Seq2SeqTrainer{
		Model:     m,
		Optimizer: o,
	}
}

func (t *Seq2SeqTrainer) Fit(in *Seq2SeqInput) {
	var total float64
	var count int
	for i := 0; i < in.Epochs; i++ {
		xt, tt := vector.Shuffle(in.Train, in.TrainLabel) // (45000, 7), (45000, 5)
		xs, ts := Float64(xt), Float64(tt)                // (45000, 7), (45000, 5)

		for j := 0; j < len(in.Train)/in.BatchSize; j++ {
			begin, end := Range(j, in.BatchSize)
			xbatch := Time(xs[begin:end]) // (128, 7) -> (7, 128, 1)
			tbatch := Time(ts[begin:end]) // (128, 5) -> (5, 128, 1)

			loss := t.Model.Forward(xbatch, tbatch)
			t.Model.Backward()
			t.Optimizer.Update(t.Model)

			total += loss
			count++

			in.Verbose(i, j, total/float64(count), t.Model)
		}

		total, count = 0.0, 0
	}
}

func Time(xs matrix.Matrix, reverse ...bool) []matrix.Matrix {
	T := len(xs[0])                 // (128, 7)
	out := make([]matrix.Matrix, T) // (7, 128, 1)
	for i := 0; i < T; i++ {
		out[i] = matrix.New()
		for j := 0; j < len(xs); j++ {
			out[i] = append(out[i], []float64{xs[j][i]})
		}
	}

	if len(reverse) > 0 && reverse[0] {
		for i := 0; i < T/2; i++ {
			out[i], out[T-1-i] = out[T-1-i], out[i]
		}
	}

	return out
}

func Float64(x [][]int) [][]float64 {
	out := make([][]float64, len(x))
	for i, r := range x {
		out[i] = make([]float64, len(r))
		for j, v := range r {
			out[i][j] = float64(v)
		}
	}

	return out
}

func SeqAccuracy[T comparable](a, b []T) int {
	if len(a) != len(b) {
		return 0
	}

	for i := range a {
		if a[i] != b[i] {
			return 0
		}
	}

	return 1
}
