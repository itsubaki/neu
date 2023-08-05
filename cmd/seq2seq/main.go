package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

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
	flag.IntVar(&epochs, "epochs", 30, "")
	flag.IntVar(&batchSize, "batch-size", 128, "")
	flag.Parse()

	x, t, v := sequence.Must(sequence.Load(dir, sequence.Addition))

	m := model.NewSeq2Seq(&model.Seq2SeqConfig{
		VocabSize:   len(v.RuneToID), // 13
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
		xt, tt := Shuffle(x.Train, t.Train) // (45000, 7), (45000, 5)
		xs, ts := Float64(xt), Float64(tt)  // (45000, 7), (45000, 5)

		for j := 0; j < len(x.Train)/batchSize; j++ {
			begin, end := trainer.Range(j, batchSize)
			xbatch := Time(xs[begin:end]) // (7, 128, 1)
			tbatch := Time(ts[begin:end]) // (5, 128, 1)

			loss := m.Forward(xbatch, tbatch)
			m.Backward()
			optimizer.Update(m)

			for k := 0; k < 10; k++ {
				q, ans := Float64(xt)[k], tt[k]
				guess := m.Generate(Time(matrix.New(q)), ans[0], len(ans[1:]))
				fmt.Printf("%v %v (%v)\n", v.ToString(xt[k]), v.ToString(ans), v.ToString(guess))
			}

			total += loss
			count++
		}

		fmt.Printf("%2d: loss=%.4f\n", i, total/float64(count))
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

func Shuffle(x, t [][]int, s ...rand.Source) ([][]int, [][]int) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	xs, ts := make([][]int, len(x)), make([][]int, len(t))
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

func Accuracy(a, b []int) int {
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
