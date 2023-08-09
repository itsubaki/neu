package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/dataset/sequence"
)

func main() {
	// flag
	var dir string
	var epochs, dataSize, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&dataSize, "data-size", 10, "")
	flag.IntVar(&batchSize, "batch-size", 5, "")
	flag.Parse()

	// data
	x, t, v := sequence.Must(sequence.Load(dir, sequence.DateTxt))

	fmt.Println(len(x.Train), len(x.Train[0]), len(x.Test), len(x.Test[0]))
	fmt.Println(len(t.Train), len(t.Train[0]), len(t.Test), len(t.Test[0]))
	fmt.Println(len(v.IDToRune), len(v.RuneToID))
	fmt.Println(x.Train[0], t.Train[0])
	fmt.Println(v.ToString(x.Train[0]), v.ToString(t.Train[0]))

}
