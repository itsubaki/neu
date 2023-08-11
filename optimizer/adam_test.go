package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
)

func ExampleAdam() {
	m := &TestModel{
		params: append(make([][]matrix.Matrix, 0), []matrix.Matrix{{{1, 2, 3}, {4, 5, 6}}}),
		grads:  append(make([][]matrix.Matrix, 0), []matrix.Matrix{{{2, 4, 6}, {8, 10, 12}}}),
	}

	for _, lr := range []float64{0.0, 0.5, 1.0} {
		opt := optimizer.Adam{
			LearningRate: lr,
			Beta1:        0.9,
			Beta2:        0.999,
		}
		fmt.Println(opt.Update(m)[0][0])
	}
	fmt.Println()

	opt := optimizer.Adam{LearningRate: 0.5, Beta1: 0.9, Beta2: 0.999, Hooks: []optimizer.Hook{hook.WeightDecay(0.0)}}
	fmt.Println(opt.Update(m)[0][0])
	fmt.Println(opt.Update(m)[0][0])
	fmt.Println(opt.Update(m)[0][0])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0.500000790568165 1.500000395284395 2.5000002635229994] [3.500000197642276 4.500000158113833 5.500000131761534]]
	// [[1.5811363300866077e-06 1.00000079056879 2.000000527045999] [3.0000003952845513 4.000000316227666 5.000000263523069]]
	//
	// [[0.500000790568165 1.500000395284395 2.5000002635229994] [3.500000197642276 4.500000158113833 5.500000131761534]]
	// [[0.5000005591561794 1.5000002795782477 2.5000001863855346] [3.5000001397891647 4.500000111831339 5.500000093192786]]
	// [[0.5000004566633607 1.5000002283317846 2.500000152221213] [3.5000001141659185 4.500000091332739 5.500000076110618]]

}
