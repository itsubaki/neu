package winit_test

import (
	"fmt"

	"github.com/itsubaki/neu/winit"
)

func ExampleXavier() {
	fmt.Println(winit.Xavier(1))
	fmt.Println(winit.Xavier(2))
	fmt.Println(winit.Xavier(4))

	// Output:
	// 1
	// 0.7071067811865476
	// 0.5

}
