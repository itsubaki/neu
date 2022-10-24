package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleOpts() {
	fmt.Println(layer.Opts{Train: true})

	// Output:
	// {true}
}
