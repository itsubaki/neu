package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path"
	"path/filepath"
)

const (
	TrainImageGZ = "train-images-idx3-ubyte.gz"
	TrainLabelGZ = "train-labels-idx1-ubyte.gz"
	TestImageGZ  = "t10k-images-idx3-ubyte.gz"
	TestLabelGZ  = "t10k-labels-idx1-ubyte.gz"
)

const (
	Width  = 28
	Height = 28
)

type (
	Image [Width * Height]byte
	Label uint8
)

type Dataset struct {
	N     int
	Image []Image
	Label []Label
}

func image(fileName string) ([]Image, error) {
	f, err := os.Open(filepath.Clean(fileName))
	if err != nil {
		return nil, fmt.Errorf("file=%v open: %v", fileName, err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("new gzip reader: %v", err)
	}

	type header struct {
		Magic  int32
		N      int32
		Height int32
		Width  int32
	}

	h := header{}
	err = binary.Read(gzr, binary.BigEndian, &h)
	if err != nil {
		return nil, fmt.Errorf("binary read: %v", err)
	}

	if h.Magic != 0x00000803 || h.Width != Width || h.Height != Height {
		return nil, fmt.Errorf("invalid header=%v: %v", h, err)
	}

	images := make([]Image, h.N)
	for i := int32(0); i < h.N; i++ {
		if err := binary.Read(gzr, binary.BigEndian, &images[i]); err != nil {
			return nil, fmt.Errorf("read image[%v]: %v", i, err)
		}
	}

	return images, nil
}

func label(fileName string) ([]Label, error) {
	f, err := os.Open(filepath.Clean(fileName))
	if err != nil {
		return nil, fmt.Errorf("file=%v open: %v", fileName, err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("new gzip reader: %v", err)
	}

	type header struct {
		Magic int32
		N     int32
	}

	h := header{}
	err = binary.Read(gzr, binary.BigEndian, &h)
	if err != nil {
		return nil, fmt.Errorf("binary read: %v", err)
	}

	if h.Magic != 0x00000801 {
		return nil, fmt.Errorf("invalid header=%v: %v", h, err)
	}

	labels := make([]Label, h.N)
	for i := int32(0); i < h.N; i++ {
		if err := binary.Read(gzr, binary.BigEndian, &labels[i]); err != nil {
			return nil, fmt.Errorf("read label[%v]: %v", i, err)
		}
	}

	return labels, nil
}

func load(imageFileName, labelFileName string) (*Dataset, error) {
	images, err := image(imageFileName)
	if err != nil {
		return nil, fmt.Errorf("load=%v: %v", imageFileName, err)
	}

	labels, err := label(labelFileName)
	if err != nil {
		return nil, fmt.Errorf("load=%v: %v", labelFileName, err)
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("invalid size. image=%v, labels=%v", len(images), len(labels))
	}

	return &Dataset{
		N:     len(images),
		Image: images,
		Label: labels,
	}, nil
}

func Must(train, test *Dataset, err error) (*Dataset, *Dataset) {
	if err != nil {
		panic(err)
	}

	return train, test
}

func Load(dir string) (*Dataset, *Dataset, error) {
	train, err := load(
		path.Join(dir, TrainImageGZ),
		path.Join(dir, TrainLabelGZ),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("load training data: %v", err)
	}

	test, err := load(
		path.Join(dir, TestImageGZ),
		path.Join(dir, TestLabelGZ),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("load test data: %v", err)
	}

	return train, test, nil
}

func OneHot(label []Label) [][]float64 {
	out := make([][]float64, 0, len(label))
	for _, l := range label {
		v := make([]float64, 10)
		v[l] = 1.0
		out = append(out, v)
	}

	return out
}

func Normalize(img []Image) [][]float64 {
	out := make([][]float64, 0)
	for i := range img {
		v := make([]float64, 0)
		for j := range img[i] {
			v = append(v, float64(img[i][j])/float64(math.MaxUint8))
		}

		out = append(out, v)
	}

	return out
}
