package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/model"
)

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
