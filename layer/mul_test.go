package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleMul() {
	apple := []float64{100.0}
	appleNum := []float64{2.0}
	tax := []float64{1.1}

	appleLayer := layer.Mul{}
	taxLayer := layer.Mul{}

	applePrice := appleLayer.Forward(apple, appleNum)
	price := taxLayer.Forward(applePrice, tax)

	fmt.Println(price)

	dPrice := []float64{1.0}
	dApplePrice, dTax := taxLayer.Backward(dPrice)
	dApple, dAppleNum := appleLayer.Backward(dApplePrice)

	fmt.Printf("%v %v %v\n", dApple, dAppleNum, dTax)

	// Output:
	// [220.00000000000003]
	// [2.2] [110.00000000000001] [200]

}
