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
	var epochs, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 10, "")
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

	var total float64
	var count int
	for i := 0; i < epochs; i++ {
		xs, ts := trainer.Shuffle(xt, tt) // (45000, 7), (45000, 5)
		for j := 0; j < len(x.Train)/batchSize; j++ {
			begin, end := trainer.Range(j, batchSize)
			xbatch := Time(xs[begin:end]) // (7, 128, 1)
			tbatch := Time(ts[begin:end]) // (5, 128, 1)

			loss := m.Forward(xbatch, tbatch)
			m.Backward()
			optimizer.Update(m)

			total += loss
			count++
		}

		var acc float64
		for k := 0; k < len(x.Test); k++ {
			q, c := Float64(x.Test)[k], t.Test[k]
			guess := m.Generate([]matrix.Matrix{matrix.New(q)}, c[0], len(c))
			//	fmt.Printf("%v %v (%v)\n", v.ToString(x.Test[k]), v.ToString(c), v.ToString(guess))
			if fmt.Sprintf("%v", v.ToString(c)) == fmt.Sprintf("%v", v.ToString(guess)) {
				acc++
			}
		}

		fmt.Printf("%2d: loss=%.4v, acc=%f\n", i, total/float64(count), acc/float64(len(x.Test)))
		total, count = 0.0, 0
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

	for i := 0; i < T/2; i++ {
		out[i], out[T-1-i] = out[T-1-i], out[i]
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
