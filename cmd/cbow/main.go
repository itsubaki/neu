package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

func main() {
	// flags
	var epochs, batchSize int
	flag.IntVar(&epochs, "epochs", 1000, "")
	flag.IntVar(&batchSize, "batch-size", 3, "")

	text := "You say goodbye and I say hello ."
	corpus, w2id, id2w := ptb.PreProcess(text)
	fmt.Println(corpus)
	fmt.Println(w2id)
	fmt.Println(id2w)

	contexts := []matrix.Matrix{
		matrix.New(
			// 0, 2
			[]float64{1, 0, 0, 0, 0, 0, 0},
			[]float64{0, 0, 1, 0, 0, 0, 0},
		),
		matrix.New(
			// 1, 3
			[]float64{0, 1, 0, 0, 0, 0, 0},
			[]float64{0, 0, 0, 1, 0, 0, 0},
		),
		matrix.New(
			// 2, 4
			[]float64{0, 0, 1, 0, 0, 0, 0},
			[]float64{0, 0, 0, 0, 1, 0, 0},
		),
		matrix.New(
			// 3, 1
			[]float64{0, 0, 0, 1, 0, 0, 0},
			[]float64{0, 1, 0, 0, 0, 0, 0},
		),
		matrix.New(
			// 4, 5
			[]float64{0, 0, 0, 0, 1, 0, 0},
			[]float64{0, 0, 0, 0, 0, 1, 0},
		),
		matrix.New(
			// 1, 6
			[]float64{0, 1, 0, 0, 0, 0, 0},
			[]float64{0, 0, 0, 0, 0, 0, 1},
		),
	}

	targets := []matrix.Matrix{matrix.New(
		[]float64{0, 1, 0, 0, 0, 0, 0}, // 1
		[]float64{0, 0, 1, 0, 0, 0, 0}, // 2
		[]float64{0, 0, 0, 1, 0, 0, 0}, // 3
		[]float64{0, 0, 0, 0, 1, 0, 0}, // 4
		[]float64{0, 1, 0, 0, 0, 0, 0}, // 1
		[]float64{0, 0, 0, 0, 0, 1, 0}, // 5
	)}

	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  len(w2id),
		HiddenSize: 5,
	})
	o := &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
	}

	for i := 0; i < epochs; i++ {
		loss := m.Forward(contexts, targets)
		m.Backward()
		o.Update(m)

		if i%100 == 0 {
			fmt.Printf("loss=%.4f\n", loss)
		}
	}
	fmt.Println()

	for id, word := range id2w {
		fmt.Printf("%v: %v\n", word, m.Win0.Params()[0][id])
	}
	fmt.Println()

	score := m.Predict([]matrix.Matrix{
		matrix.New(
			// 0, 2
			[]float64{1, 0, 0, 0, 0, 0, 0}, // you
			[]float64{0, 0, 1, 0, 0, 0, 0}, // goodbye
		),
	})

	fmt.Println(activation.Softmax(score[0][0]))
	fmt.Println(model.Argmax(score)) // say
}
