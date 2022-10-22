package winit_test

import (
	"fmt"

	"github.com/itsubaki/neu/winit"
)

func ExampleHe() {
	fmt.Println(winit.He(1))
	fmt.Println(winit.He(2))
	fmt.Println(winit.He(4))

	// Output:
	// 1.4142135623730951
	// 1
	// 0.7071067811865476

}
