package sequence_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/neu/dataset/sequence"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleLoad() {
	s := rand.Const(1)
	x, t, v := sequence.Must(sequence.Load("../../testdata", sequence.AdditionTxt, s))

	fmt.Println(len(x.Train), len(x.Train[0]), len(x.Test), len(x.Test[0]))
	fmt.Println(len(t.Train), len(t.Train[0]), len(t.Test), len(t.Test[0]))
	fmt.Println(len(v.IDToRune), len(v.RuneToID))
	fmt.Println(x.Train[0], t.Train[0])
	fmt.Println(v.ToString(x.Train[0]), v.ToString(t.Train[0]))

	// Output:
	// 45000 7 5000 7
	// 45000 5 5000 5
	// 13 13
	// [9 0 3 2 7 5 5] [12 9 0 3 5]
	// [9 1 7 + 0    ] [_ 9 1 7  ]
}

func ExampleLoad_rand() {
	x, t, v := sequence.Must(sequence.Load("../../testdata", sequence.AdditionTxt))

	fmt.Println(len(x.Train), len(x.Train[0]), len(x.Test), len(x.Test[0]))
	fmt.Println(len(t.Train), len(t.Train[0]), len(t.Test), len(t.Test[0]))
	fmt.Println(len(v.IDToRune), len(v.RuneToID))

	// Output:
	// 45000 7 5000 7
	// 45000 5 5000 5
	// 13 13
}

func ExampleLoad_notfound() {
	_, _, _, err := sequence.Load("invalid_dir", "invlid_file")
	fmt.Println(err)

	// Output:
	// open file=invalid_dir/invlid_file: open invalid_dir/invlid_file: no such file or directory
}

func TestMust(t *testing.T) {
	defer func() {
		if rec := recover(); rec != nil {
			err, ok := rec.(error)
			if !ok {
				t.Fail()
			}

			if err.Error() != "something went wrong" {
				t.Fail()
			}
		}
	}()

	sequence.Must(nil, nil, nil, fmt.Errorf("something went wrong"))
	t.Fail()
}
