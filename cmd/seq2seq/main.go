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

	xt, tt := Float64(x.Train), Float64(t.Train)
	xs, ts := trainer.Shuffle(xt, tt) // (45000, 7), (45000, 5)

	for j := 0; j < len(x.Train)/batchSize; j++ {
		begin, end := trainer.Range(j, batchSize)
		xbatch := Time(xs[begin:end]) // (7, 128, 1)
		tbatch := Time(ts[begin:end]) // (5, 128, 1)

		loss := m.Forward(xbatch, tbatch)
		m.Backward([]matrix.Matrix{matrix.New([]float64{1})})
		optimizer.Update(m)

		fmt.Println(loss)
	}

}

func Time(xbatch matrix.Matrix) []matrix.Matrix {
	T := len(xbatch[0])             // 7
	out := make([]matrix.Matrix, T) // (7, 128, 1)
	for i := 0; i < T; i++ {
		m := matrix.New()
		for j := 0; j < len(xbatch); j++ {
			m = append(m, []float64{xbatch[j][i]})
		}
		out[i] = m
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
