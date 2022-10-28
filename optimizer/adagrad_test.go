package optimizer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleAdaGrad() {
	params := append(make([][]matrix.Matrix, 0), []matrix.Matrix{
		matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6}),
	})
	grads := append(make([][]matrix.Matrix, 0), []matrix.Matrix{
		matrix.New([]float64{2, 4, 6}, []float64{8, 10, 12}),
	})

	for _, lr := range []float64{0.0, 0.5, 1.0} {
		opt := optimizer.AdaGrad{LearningRate: lr, Hooks: []optimizer.Hook{weight.Decay(0.0)}}
		fmt.Println(opt.Update(params, grads)[0][0])
	}
	fmt.Println()

	opt := optimizer.AdaGrad{LearningRate: 0.5}
	fmt.Println(opt.Update(params, grads)[0][0])
	fmt.Println(opt.Update(params, grads)[0][0])

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[0.5000000249999987 1.5000000124999997 2.500000008333333] [3.50000000625 4.5000000049999995 5.500000004166667]]
	// [[4.99999973646581e-08 1.0000000249999994 2.0000000166666663] [3.0000000124999997 4.00000001 5.000000008333333]]
	//
	// [[0.5000000249999987 1.5000000124999997 2.500000008333333] [3.50000000625 4.5000000049999995 5.500000004166667]]
	// [[0.6464466219067257 1.6464466156567261 2.6464466135733926] [3.6464466125317263 4.646446611906726 5.6464466114900596]]

}
