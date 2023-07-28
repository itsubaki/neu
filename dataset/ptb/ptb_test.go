package ptb_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/neu/dataset/ptb"
)

func ExampleLoadFile() {
	train, err := ptb.LoadFile("../../testdata", ptb.TrainTxt)
	if err != nil {
		panic(err)
	}

	fmt.Println(len(train.Corpus))
	fmt.Println(train.Corpus[:20])

	fmt.Println(train.IDToWord[0])
	fmt.Println(train.IDToWord[1])
	fmt.Println(train.IDToWord[2])

	fmt.Println(train.WordToID["car"])
	fmt.Println(train.WordToID["happy"])
	fmt.Println(train.WordToID["lexus"])

	// Output:
	// 929589
	// [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]
	// aer
	// banknote
	// berlitz
	// 3856
	// 4428
	// 7426
}

func ExampleLoad() {
	train, test, valid := ptb.Must(ptb.Load("../../testdata"))

	fmt.Println(len(train.Corpus))
	fmt.Println(train.Corpus[:20])
	fmt.Println(train.IDToWord[0])
	fmt.Println(train.IDToWord[1])
	fmt.Println(train.IDToWord[2])
	fmt.Println(train.WordToID["car"])
	fmt.Println(train.WordToID["happy"])
	fmt.Println(train.WordToID["lexus"])
	fmt.Println()

	fmt.Println(len(test.Corpus))
	fmt.Println(test.Corpus[:20])
	fmt.Println(test.IDToWord[0])
	fmt.Println(test.IDToWord[1])
	fmt.Println(test.IDToWord[2])
	fmt.Println(test.WordToID["car"])
	fmt.Println(test.WordToID["happy"])
	fmt.Println(test.WordToID["lexus"])
	fmt.Println()

	fmt.Println(len(valid.Corpus))
	fmt.Println(valid.Corpus[:20])
	fmt.Println(valid.IDToWord[0])
	fmt.Println(valid.IDToWord[1])
	fmt.Println(valid.IDToWord[2])
	fmt.Println(valid.WordToID["car"])
	fmt.Println(valid.WordToID["happy"])
	fmt.Println(valid.WordToID["lexus"])

	// Output:
	// 929589
	// [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]
	// aer
	// banknote
	// berlitz
	// 3856
	// 4428
	// 7426
	//
	// 82430
	// [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 3 15 16 17 18]
	// no
	// it
	// was
	// 2611
	// 2981
	// 5098
	//
	// 73760
	// [0 1 2 3 4 5 6 7 8 9 3 10 11 12 13 14 14 15 16 17]
	// consumers
	// may
	// want
	// 1801
	// 2014
	// 1826
}

func ExampleLoad_notfound() {
	_, _, _, err := ptb.Load("invalid_dir")
	fmt.Println(err)

	// Output:
	// load training data: open file=invalid_dir/ptb.train.txt: open invalid_dir/ptb.train.txt: no such file or directory
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

	ptb.Must(nil, nil, nil, fmt.Errorf("something went wrong"))
	t.Fail()
}
