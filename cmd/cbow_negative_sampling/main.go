package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
)

func main() {
	// flags
	var epochs, hiddenSize, windowSize, sampleSize, batchSize int
	var power, alpha, beta1, beta2 float64
	flag.IntVar(&epochs, "epochs", 1000, "")
	flag.IntVar(&hiddenSize, "hidden-size", 5, "")
	flag.IntVar(&windowSize, "window-size", 1, "")
	flag.IntVar(&sampleSize, "sample-size", 2, "")
	flag.IntVar(&batchSize, "batch-size", 1, "")
	flag.Float64Var(&power, "power", 0.75, "")
	flag.Float64Var(&alpha, "alpha", 0.001, "")
	flag.Float64Var(&beta1, "beta1", 0.9, "")
	flag.Float64Var(&beta2, "beta2", 0.999, "")
	flag.Parse()

	text := "You say goodbye and I say hello ."
	corpus, id2w, w2id := ptb.PreProcess(text)
	fmt.Println(text)
	fmt.Println(corpus)
	fmt.Println(id2w)
	fmt.Println(w2id)
	fmt.Println()

	contexts, target := ptb.CreateContextsTarget(corpus, 1)
	for i := range contexts {
		fmt.Printf("%v: %v\n", contexts[i], target[i])
	}
	fmt.Println()

	m := model.NewCBOWNegativeSampling(model.CBOWNegativeSamplingConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  vector.Max(corpus) + 1,
			HiddenSize: hiddenSize,
		},
		Corpus:     corpus,
		WindowSize: windowSize,
		SampleSize: sampleSize,
		Power:      power,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.New(m, &optimizer.Adam{
		Alpha: alpha,
		Beta1: beta1,
		Beta2: beta2,
	})

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      matrix.From(contexts),
		TrainLabel: matrix.From([][]int{target}).T(),
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, loss float64, m trainer.Model) {
			if epoch%200 != 0 {
				return
			}

			fmt.Printf("%3d, %2d: loss=%.04f\n", epoch, j, loss)
		},
	})
	fmt.Printf("elapsed=%v\n", time.Since(now))
	fmt.Println()

	Win := m.Embedding[0].Params()[0]
	for id, word := range id2w {
		fmt.Printf("%v: %.4f\n", word, Win[id])
	}
	fmt.Println()

	w0, w1, w2, w3 := "I", "You", "and", "say"
	fmt.Printf("cos(%q, %q): %v\n", w0, w1, vector.Cos(Win[w2id[w0]], Win[w2id[w1]]))
	fmt.Printf("cos(%q, %q): %v\n", w0, w2, vector.Cos(Win[w2id[w0]], Win[w2id[w2]]))
	fmt.Printf("cos(%q, %q): %v\n", w0, w3, vector.Cos(Win[w2id[w0]], Win[w2id[w3]]))
}
