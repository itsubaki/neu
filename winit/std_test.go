package winit_test

import (
	"fmt"

	"github.com/itsubaki/neu/winit"
)

func ExampleStd() {
	fmt.Println(winit.Std(0.1)(0))
	fmt.Println(winit.Std(0.01)(0))
	fmt.Println(winit.Std(0.001)(0))

	// Output:
	// 0.1
	// 0.01
	// 0.001

}
