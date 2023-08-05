package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/dataset/sequence"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir string
	var batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&batchSize, "batch-size", 128, "")
	flag.Parse()

	x, t, v := sequence.Must(sequence.Load(dir, sequence.Addition))
	xt, tt := Float64(x.Train), Float64(t.Train)

	m := model.NewSeq2Seq(&model.Seq2SeqConfig{
		VocabSize:   len(v.RuneToID),
		WordVecSize: 16,
		HiddenSize:  128,
		WeightInit:  weight.Xavier,
	})
	optimizer := optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(5.0),
		},
	}

	xs, ts := trainer.Shuffle(xt, tt) // (45000, 7), (45000, 5)

	for j := 0; j < len(x.Train)/batchSize; j++ {
		begin, end := trainer.Range(j, batchSize)
		xbatch := xs[begin:end] // (128, 7)
		tbatch := ts[begin:end] // (128, 5)

		loss := m.Forward(xbatch, tbatch)
		m.Backward([]matrix.Matrix{matrix.New([]float64{1})})
		optimizer.Update(m)

		fmt.Println(loss)
	}
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
