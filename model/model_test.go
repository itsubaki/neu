package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

func ExampleAccuracy() {
	// data
	y0 := matrix.New([]float64{0, 1}, []float64{1, 0}, []float64{1, 0})
	y1 := matrix.New([]float64{0, 1}, []float64{1, 0}, []float64{0, 1})
	y2 := matrix.New([]float64{0, 1}, []float64{0, 1}, []float64{0, 1})
	y3 := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	fmt.Println(model.Accuracy(y0, t))
	fmt.Println(model.Accuracy(y1, t))
	fmt.Println(model.Accuracy(y2, t))
	fmt.Println(model.Accuracy(y3, t))

	// Output:
	// 0
	// 0.3333333333333333
	// 0.6666666666666666
	// 1
}

func ExampleXavier() {
	fmt.Println(model.Xavier(1))
	fmt.Println(model.Xavier(2))
	fmt.Println(model.Xavier(4))

	// Output:
	// 1
	// 0.7071067811865476
	// 0.5

}

func ExampleHe() {
	fmt.Println(model.He(1))
	fmt.Println(model.He(2))
	fmt.Println(model.He(4))

	// Output:
	// 1.4142135623730951
	// 1
	// 0.7071067811865476

}
