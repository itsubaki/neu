package main

import (
	"flag"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir string
	var length int
	var epochs, wordvecSize, hiddenSize, batchSize, timeSize int
	var learningRate, dropoutRatio, max float64
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&length, "length", 100, "")
	flag.IntVar(&epochs, "epochs", 10, "")
	flag.IntVar(&wordvecSize, "wordvec-size", 650, "")
	flag.IntVar(&hiddenSize, "hidden-size", 650, "")
	flag.IntVar(&batchSize, "batch-size", 20, "")
	flag.IntVar(&timeSize, "time-size", 35, "")
	flag.Float64Var(&dropoutRatio, "dropout-ratio", 0.5, "")
	flag.Float64Var(&learningRate, "learning-rate", 20, "")
	flag.Float64Var(&max, "grads-cliping-max", 0.25, "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))
	valid := ptb.Must(ptb.Load(dir, ptb.ValidTxt))

	// model
	m := model.NewRNNLMGen(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   vector.Max(train.Corpus) + 1,
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

	// params
	filename := fmt.Sprintf("%s/rnnlm_gen.gob", dir)
	if params, ok := model.Load(filename); ok {
		m.SetParams(params)
	}

	// train
	o := &optimizer.SGD{
		LearningRate: learningRate,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(max),
		},
	}
	tr := trainer.NewRNNLM(m, o)

	now, min := time.Now(), math.MaxFloat64
	for i := 0; i < epochs; i++ {
		tr.Fit(&trainer.RNNLMInput{
			Train:      train.Corpus[:len(train.Corpus)-1],
			TrainLabel: train.Corpus[1:],
			Epochs:     1,
			BatchSize:  batchSize,
			TimeSize:   timeSize,
			Verbose: func(_, j int, perplexity float64, m trainer.RNNLM) {
				fmt.Printf("%2d, %2d: train_ppl=%.04f\n", i, j, perplexity)
			},
		})

		ppl := perplexity(m, valid.Corpus, batchSize, timeSize)
		if min > ppl {
			min = ppl
			if err := model.Save(filename, m.Params()); err != nil {
				fmt.Printf("failed to save params: :%v\n", err)
			}
		} else {
			o.LearningRate /= 4
		}

		m.ResetState()
	}
	fmt.Printf("elapsed=%v\n", time.Since(now))
	fmt.Println()

	// generate
	query := []string{"the", "meaning", "of", "life", "is"}
	startID := train.WordToID[query[len(query)-1]]
	skipIDs := []int{
		train.WordToID["N"],
		train.WordToID["<unk>"],
		train.WordToID["$"],
	}

	for _, q := range query[:len(query)-1] {
		m.Predict([]matrix.Matrix{{{float64(train.WordToID[q])}}})
	}

	wordIDs := m.Generate(startID, skipIDs, length)
	words := make([]string, length)
	for i, id := range wordIDs {
		words[i] = train.IDToWord[id]
	}

	txt := strings.ReplaceAll(strings.Join(words, " "), "\n", "")
	fmt.Println(strings.Join(query, " "), txt)
}

func perplexity(m trainer.RNNLM, corpus []int, batchSize, timeSize int) float64 {
	corpusSize := len(corpus)
	maxIter := (corpusSize - 1) / (batchSize * timeSize)
	jump := (corpusSize - 1) / batchSize

	var total float64
	for j := 0; j < maxIter; j++ {
		xs, ts := tensor.Zero(timeSize, batchSize, 1), tensor.Zero(timeSize, batchSize, 1)

		timeOffset := j * timeSize
		offsets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			offsets[i] = timeOffset + (i * jump)
		}

		for t := 0; t < timeSize; t++ {
			for i, offset := range offsets {
				xs[t][i] = []float64{float64(corpus[(offset+t)%corpusSize])}
				ts[t][i] = []float64{float64(corpus[(offset+t+1)%corpusSize])}
			}
		}

		loss := m.Forward(xs, ts)
		total += loss[0][0][0]

		fmt.Printf("%3d/%3d: loss=%.04f\n", j, maxIter, loss)
	}

	return trainer.Perplexity(total, maxIter)
}
