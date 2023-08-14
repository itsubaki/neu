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
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flags
	var dir string
	var epochs, corpusSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&corpusSize, "corpus-size", 1000, "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))
	corpus := train.Corpus[:corpusSize]

	// model
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   vector.Max(corpus) + 1,
			WordVecSize: 100,
			HiddenSize:  100,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	})

	// summary
	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.NewRNNLM(m, &optimizer.SGD{
		LearningRate: 20,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(0.25),
		},
	})

	now := time.Now()
	tr.Fit(&trainer.RNNLMInput{
		Train:      corpus[:len(corpus)-1],
		TrainLabel: corpus[1:],
		Epochs:     epochs,
		BatchSize:  20,
		TimeSize:   35,
		Verbose: func(epoch, j int, perplexity float64, m trainer.RNNLM) {
			fmt.Printf("%2d, %2d: train_ppl=%.04f\n", epoch, j, perplexity)
		},
	})
	fmt.Printf("elapsed=%v\n", time.Since(now))
	fmt.Println()

	// test
	test := ptb.Must(ptb.Load(dir, ptb.TestTxt))
	fmt.Printf("test_ppl=%.04f\n", perplexity(m, test.Corpus[:corpusSize], 10, 35))
}

func perplexity(m trainer.RNNLM, corpus []int, batchSize, timeSize int) float64 {
	corpusSize := len(corpus)
	maxIter := (corpusSize - 1) / (batchSize * timeSize)
	jump := (corpusSize - 1) / batchSize

	var total float64
	for j := 0; j < maxIter; j++ {
		xs, ts := make([]matrix.Matrix, timeSize), make([]matrix.Matrix, timeSize)
		timeOffset := j * timeSize
		offsets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			offsets[i] = timeOffset + (i * jump)
		}

		for t := 0; t < timeSize; t++ {
			xs[t], ts[t] = make(matrix.Matrix, batchSize), make(matrix.Matrix, batchSize)
			for i, offset := range offsets {
				xs[t][i] = []float64{float64(corpus[(offset+t)%corpusSize])}
				ts[t][i] = []float64{float64(corpus[(offset+t+1)%corpusSize])}
			}
		}

		loss := m.Forward(xs, ts)
		total += loss[0][0][0]

		fmt.Printf("%2d, %2d: loss=%.04f\n", j, maxIter, loss[0][0])
	}

	return trainer.Perplexity(total, maxIter)
}
