package hook_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer/hook"
)

func ExampleGradsClipping() {
	for _, v := range hook.GradsClipping(1.0)(nil, [][]matrix.Matrix{
		{
			{[]float64{1, 2}, []float64{3, 4}},
			{[]float64{5, 6}, []float64{7, 8}},
		},
	}) {
		// rate: 0.07001399929944005
		// clipped
		fmt.Println(v)
	}

	for _, v := range hook.GradsClipping(1.0)(nil, [][]matrix.Matrix{
		{
			{[]float64{0.1, 0.1}, []float64{0.1, 0.1}},
			{[]float64{0.01, 0.01}, []float64{0.01, 0.01}},
		},
	}) {
		// rate: 4.975161198697846
		// not clipping
		fmt.Println(v)
	}

	// Output:
	// [[[0.07001399929944005 0.1400279985988801] [0.21004199789832015 0.2800559971977602]] [[0.3500699964972003 0.4200839957966403] [0.4900979950960803 0.5601119943955204]]]
	// [[[0.1 0.1] [0.1 0.1]] [[0.01 0.01] [0.01 0.01]]]

}
