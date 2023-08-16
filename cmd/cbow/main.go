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
	var epochs int
	flag.IntVar(&epochs, "epochs", 1000, "")

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

	m := model.NewCBOWNegs(model.CBOWNegsConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  vector.Max(corpus) + 1,
			HiddenSize: 5,
		},
		Corpus:     corpus,
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.New(m, &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
	})

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      matrix.From(contexts),
		TrainLabel: matrix.From([][]int{target}).T(),
		Epochs:     epochs,
		BatchSize:  1,
		Verbose: func(epoch, j int, loss float64, m trainer.Model) {
			if epoch%300 != 0 {
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

	fmt.Println("cos('I', 'You'): ", vector.Cos(Win[w2id["I"]], Win[w2id["You"]]))
}
