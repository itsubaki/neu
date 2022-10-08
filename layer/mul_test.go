package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleMul() {
	apple := 100.0
	appleNum := 2.0
	tax := 1.1

	appleLayer := layer.Mul{}
	taxLayer := layer.Mul{}

	applePrice := appleLayer.Forward([]float64{apple}, []float64{appleNum})
	price := taxLayer.Forward(applePrice, []float64{tax})

	fmt.Println(price)

	dPrice := 1.0
	dApplePrice, dTax := taxLayer.Backwward([]float64{dPrice})
	dApple, dAppleNum := appleLayer.Backwward(dApplePrice)

	fmt.Printf("%v %v %v\n", dApple, dAppleNum, dTax)

	// Output:
	// [220.00000000000003]
	// [2.2] [110.00000000000001] [200]

}
