package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMul() {
	apple := matrix.New([]float64{100.0})
	appleNum := matrix.New([]float64{2.0})
	tax := matrix.New([]float64{1.1})

	appleLayer := layer.Mul{}
	taxLayer := layer.Mul{}

	applePrice := appleLayer.Forward(apple, appleNum)
	price := taxLayer.Forward(applePrice, tax)

	fmt.Println(price)

	dPrice := matrix.New([]float64{1.0})
	dApplePrice, dTax := taxLayer.Backward(dPrice)
	dApple, dAppleNum := appleLayer.Backward(dApplePrice)

	fmt.Printf("%v %v %v\n", dApple, dAppleNum, dTax)

	// Output:
	// [[220.00000000000003]]
	// [[2.2]] [[110.00000000000001]] [[200]]

}

func ExampleMul_Params() {
	mul := layer.Mul{}

	mul.SetParams(make([]matrix.Matrix, 0))
	mul.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(mul.Params())
	fmt.Println(mul.Grads())

	// Output:
	// []
	// []
}
