package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleSave() {
	s := rand.NewSource(1)
	m := model.NewSeq2Seq(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	if err := model.Save(m.Params(), "../testdata/example_save.gob"); err != nil {
		fmt.Println("failed to save params:", err)
		return
	}

	params, ok := model.Load("../testdata/example_save.gob")
	if !ok {
		fmt.Println("failed to load params")
		return
	}

	if len(m.Params()) != len(params) {
		fmt.Println("invalid length")
		return
	}

	for i, p := range m.Params() {
		if len(p) != len(params[i]) {
			fmt.Println("invalid length")
			return
		}

		for j := range p {
			if p[j].Sub(params[i][j]).Abs().Sum() > 1e-13 {
				fmt.Println("invalid value")
				return
			}
		}
	}

	// Output:

}

func ExampleSave_nosuchdir() {
	if err := model.Save(nil, "../nosuchdir/hoge.gob"); err != nil {
		fmt.Println("failed to save params:", err)
		return
	}

	// Output:
	// failed to save params: failed to create file: open ../nosuchdir/hoge.gob: no such file or directory
}

func ExampleLoad_invaliddir() {
	if _, ok := model.Load("invalid_dir"); !ok {
		fmt.Println("failed to save params")
		return
	}

	// Output:
	// failed to save params
}

func ExampleLoad_invalidfile() {
	if _, ok := model.Load("../testdata/.gitkeep"); !ok {
		fmt.Println("failed to save params")
		return
	}

	// Output:
	// failed to save params
}
