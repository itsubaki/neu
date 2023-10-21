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
	var epochs, hiddenSize int
	var alpha, beta1, beta2 float64
	flag.IntVar(&epochs, "epochs", 1000, "")
	flag.IntVar(&hiddenSize, "hidden-size", 5, "")
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
	c, t := OneHot(contexts, target, len(w2id))
	for i := range contexts {
		fmt.Printf("%v(%v): %v(%v)\n", contexts[i], c[i], target[i], t[i])
	}
	fmt.Println()

	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  len(w2id),
		HiddenSize: hiddenSize,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	o := &optimizer.Adam{
		Alpha: alpha,
		Beta1: beta1,
		Beta2: beta2,
	}

	for i := 0; i < epochs; i++ {
		loss := m.Forward(c, t)
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
	say := []float64{0, 1, 0, 0, 0, 0, 0}               // say
	score := m.Predict([]matrix.Matrix{{you, goodbye}}) //

	fmt.Printf("you:     %.4f\n", you)
	fmt.Printf("goodbye: %.4f\n", goodbye)
	fmt.Printf("target:  %.4f\n", say)
	fmt.Printf("predict: %.4f\n", activation.Softmax(score[0]))
}

func OneHot(contexts [][]int, target []int, size int) ([]matrix.Matrix, matrix.Matrix) {
	c := make([]matrix.Matrix, 0)
	for _, v := range contexts {
		c = append(c, matrix.OneHot(v, size))
	}

	return c, matrix.OneHot(target, size)
}
