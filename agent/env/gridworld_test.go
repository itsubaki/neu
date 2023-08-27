package env_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent/env"
)

func ExampleGridWorld() {
	e := env.NewGridWorld()

	fmt.Println(e.Height(), e.Width())
	fmt.Println(e.Shape())

	for _, a := range e.Actions() {
		fmt.Println(a, e.ActionMeaning[a])
	}

	for i := 0; i < e.Height()*e.Width(); i++ {
		s := e.State()
		fmt.Println(s, e.Reward(nil, 0, s))
	}

	// Output:
	// 3 4
	// 3 4
	// 0 UP
	// 1 RIGHT
	// 2 DOWN
	// 3 LEFT
	// &{0 0} 0
	// &{0 1} 0
	// &{0 2} 0
	// &{0 3} 1
	// &{1 0} 0
	// &{1 1} 0
	// &{1 2} 0
	// &{1 3} -1
	// &{2 0} 0
	// &{2 1} 0
	// &{2 2} 0
	// &{2 3} 0
}
