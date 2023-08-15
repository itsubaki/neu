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
	var epochs int
	flag.IntVar(&epochs, "epochs", 1000, "")

	text := "You say goodbye and I say hello ."
	corpus, id2w, w2id := ptb.PreProcess(text)
	fmt.Println(corpus)
	fmt.Println(id2w)
	fmt.Println(w2id)
	fmt.Println()

	c, t := ptb.CreateContextsTarget(corpus, 1)
	contexts, targets := OneHot(c, t, len(w2id))
	for i := range contexts {
		fmt.Printf("%v(%v): %v(%v)\n", c[i], contexts[i], t[i], targets[i])
	}
	fmt.Println()

	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  len(w2id),
		HiddenSize: 5,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	o := &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
	}

	for i := 0; i < epochs; i++ {
		loss := m.Forward(contexts, targets)
		m.Backward()
		o.Update(m)

		if (i+1)%200 == 0 {
			fmt.Printf("%4v: loss=%.4f\n", i+1, loss)
		}
	}
	fmt.Println()

	for id, word := range id2w {
		fmt.Printf("%v: %.4f\n", word, m.Win0.Params()[0][id])
	}
	fmt.Println()

	you := []float64{1, 0, 0, 0, 0, 0, 0}               // you
	goodbye := []float64{0, 0, 1, 0, 0, 0, 0}           // goodbye
	target := []float64{0, 1, 0, 0, 0, 0, 0}            // say
	score := m.Predict([]matrix.Matrix{{you, goodbye}}) //

	fmt.Printf("you:     %.4f\n", you)
	fmt.Printf("goodbye: %.4f\n", goodbye)
	fmt.Printf("target:  %.4f\n", target)
	fmt.Printf("predict: %.4f\n", activation.Softmax(score[0]))
}

func OneHot(c [][]int, t []int, size int) ([]matrix.Matrix, matrix.Matrix) {
	contexts := make([]matrix.Matrix, 0)
	for _, v := range c {
		contexts = append(contexts, matrix.OneHot(v, size))
	}

	return contexts, matrix.OneHot(t, size)
}
