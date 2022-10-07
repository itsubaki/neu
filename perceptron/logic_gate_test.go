package perceptron_test

import (
	"fmt"

	"github.com/itsubaki/neu/perceptron"
)

func ExampleAND() {
	fmt.Println(perceptron.AND([]float64{0, 0}))
	fmt.Println(perceptron.AND([]float64{1, 0}))
	fmt.Println(perceptron.AND([]float64{0, 1}))
	fmt.Println(perceptron.AND([]float64{1, 1}))

	// Output:
	// 0
	// 0
	// 0
	// 1
}

func ExampleNAND() {
	fmt.Println(perceptron.NAND([]float64{0, 0}))
	fmt.Println(perceptron.NAND([]float64{1, 0}))
	fmt.Println(perceptron.NAND([]float64{0, 1}))
	fmt.Println(perceptron.NAND([]float64{1, 1}))

	// Output:
	// 1
	// 1
	// 1
	// 0
}

func ExampleOR() {
	fmt.Println(perceptron.OR([]float64{0, 0}))
	fmt.Println(perceptron.OR([]float64{1, 0}))
	fmt.Println(perceptron.OR([]float64{0, 1}))
	fmt.Println(perceptron.OR([]float64{1, 1}))

	// Output:
	// 0
	// 1
	// 1
	// 1
}

func ExampleXOR() {
	fmt.Println(perceptron.XOR([]float64{0, 0}))
	fmt.Println(perceptron.XOR([]float64{1, 0}))
	fmt.Println(perceptron.XOR([]float64{0, 1}))
	fmt.Println(perceptron.XOR([]float64{1, 1}))

	// Output:
	// 0
	// 1
	// 1
	// 0
}
