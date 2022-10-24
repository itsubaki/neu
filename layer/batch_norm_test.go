package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleBatchNorm() {
	n := &layer.BatchNorm{}
	fmt.Println(n.Forward(nil, nil, layer.Opts{Train: true}))
	fmt.Println(n.Backward(nil))

	// Output:
	// []
	// [] []
}
