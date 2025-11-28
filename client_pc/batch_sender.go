package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"math"
	"io"
)

type Header struct {
	Magic		[4]byte
	BatchIdx	uint32
	DType		uint32
	DataLen		uint64
	LabelLen	uint64
}

const (
	BatchSize = 2
)

func readUint32BE(r io.Reader) uint32 {
	var v uint32
	binary.Read(r, binary.BigEndian, &v)
	return v
}

func loadMNISTImages(path string) [][]byte {
	f, err := os.Open(path)
	if err != nil { panic(err) }
	defer f.Close()

	magic := readUint32BE(f)
	if magic != 2051 {
		panic("Invalid magic number for images")
	}

	num := readUint32BE(f)
	rows := readUint32BE(f)
	cols := readUint32BE(f)
	
	imgSize := int(rows * cols)
	images := make([][]byte, num)

	for i := 0; i < int(num); i++ {
		buf := make([]byte, imgSize)
		io.ReadFull(f, buf)
		images[i] = buf
	}
	return images
}

func loadMNISTLabels(path string) []byte {
	f, err := os.Open(path)
	if err != nil { panic(err) }
	defer f.Close()

	magic := readUint32BE(f)
	if magic != 2049 {
		panic("Invalid magic number for labels")
	}

	num := readUint32BE(f)
	labels := make([]byte, num)
	io.ReadFull(f, labels)
	return labels
}

func main() {
	images := loadMNISTImages("train-images.idx3-ubyte")
	labels := loadMNISTLabels("train-labels.idx1-ubyte")

	conn, err := net.Dial("tcp", "10.1.1.2:9000")
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	fmt.Println("Connected to Jetson!")

	// Receive format:
	// dataBytes	[]byte	// float32 tensor bytes
	// labelBytes	[]byte	// int64 labels

	batchIdx := uint32(0)
	
	for i := 0; i + BatchSize <= len(images); i += BatchSize {
		dataBytes := make([]byte, BatchSize * 1 * 28 * 28 * 4)
		off := 0

		for b:= 0; b < BatchSize; b++ {
			img := images[i + b]
			for _, px := range img {
				f := float32(px) / 255.0
				binary.LittleEndian.PutUint32(dataBytes[off:], math.Float32bits(f))
				off += 4
			}
		}

		labelBytes := labels[i : i + BatchSize]

		header := Header {
			Magic:		[4]byte{'B', 'T', 'S', '0'},
			BatchIdx:	batchIdx,
			DType:		1,
			DataLen:	uint64(len(dataBytes)),
			LabelLen:	uint64(len(labelBytes)),
		}

		buf := new(bytes.Buffer)
		binary.Write(buf, binary.LittleEndian, header)
		buf.Write(dataBytes)
		buf.Write(labelBytes)

		conn.Write(buf.Bytes())

		reply := make([]byte, 256)
		var total string

		for {
			n, err := conn.Read(reply)
			if err != nil {
				fmt.Println("Error reading reply:", err)
				break
			}

			total += string(reply[:n])

			if strings.Contains(total, "OK") {
				fmt.Print(total)
				break
			}
		}

		batchIdx++
	}

	fmt.Println("Finished sending MNIST batches.")
}