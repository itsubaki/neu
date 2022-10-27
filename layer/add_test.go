package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAdd() {
	apple := matrix.New([]float64{100.0})
	appleNum := matrix.New([]float64{2.0})
	orange := matrix.New([]float64{150.0})
	orangeNum := matrix.New([]float64{3.0})
	tax := matrix.New([]float64{1.1})

	appleLayer := layer.Mul{}
	orangeLayer := layer.Mul{}
	addLayer := layer.Add{}
	taxLayer := layer.Mul{}

	applePrice := appleLayer.Forward(apple, appleNum)
	orangePrice := orangeLayer.Forward(orange, orangeNum)
	allPrice := addLayer.Forward(applePrice, orangePrice)
	price := taxLayer.Forward(allPrice, tax)

	fmt.Println(price)

	dPrice := matrix.New([]float64{1.0})
	dAllPrice, dTax := taxLayer.Backward(dPrice)
	dApplePrice, dOrangePrice := addLayer.Backward(dAllPrice)
	dOrange, dOrangeNum := orangeLayer.Backward(dOrangePrice)
	dApple, dAppleNum := appleLayer.Backward(dApplePrice)

	fmt.Printf("%v %v\n", dApple, dAppleNum)
	fmt.Printf("%v %v\n", dOrange, dOrangeNum)
	fmt.Println(dTax)

	// Output:
	// [[715.0000000000001]]
	// [[2.2]] [[110.00000000000001]]
	// [[3.3000000000000003]] [[165]]
	// [[650]]

}

func ExampleAdd_Params() {
	add := layer.Add{}

	add.SetParams(make([]matrix.Matrix, 0))
	add.SetGrads(make([]matrix.Matrix, 0))

	fmt.Println(add.Params())
	fmt.Println(add.Grads())

	// Output:
	// []
	// []
}
