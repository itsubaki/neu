package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flags
	var dir string
	var epochs, corpusSize, wordvecSize, hiddenSize, batchSize, timeSize int
	var learningRate, dropoutRatio, max float64
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 10, "")
	flag.IntVar(&corpusSize, "corpus-size", -1, "")
	flag.IntVar(&wordvecSize, "wordvec-size", 100, "")
	flag.IntVar(&hiddenSize, "hidden-size", 100, "")
	flag.IntVar(&batchSize, "batch-size", 10, "")
	flag.IntVar(&timeSize, "time-size", 5, "")
	flag.Float64Var(&dropoutRatio, "dropout-ratio", 0.5, "")
	flag.Float64Var(&learningRate, "learning-rate", 0.1, "")
	flag.Float64Var(&max, "grads-cliping-max", 0.25, "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))
	corpus := train.Corpus
	if corpusSize > 0 {
		corpus = train.Corpus[:corpusSize]
	}

	// model
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   vector.Max(corpus) + 1,
			WordVecSize: wordvecSize,
			HiddenSize:  hiddenSize,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: dropoutRatio,
	})

	// summary
	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.NewRNNLM(m, &optimizer.SGD{
		LearningRate: learningRate,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(max),
		},
	})

	now := time.Now()
	tr.Fit(&trainer.RNNLMInput{
		Train:      corpus[:len(corpus)-1],
		TrainLabel: corpus[1:],
		Epochs:     epochs,
		BatchSize:  batchSize,
		TimeSize:   timeSize,
		Verbose: func(epoch, j int, perplexity float64, m trainer.RNNLM) {
			fmt.Printf("%2d, %2d: train_ppl=%.04f\n", epoch, j, perplexity)
		},
	})

	fmt.Printf("elapsed=%v\n", time.Since(now))
	fmt.Println()
}
