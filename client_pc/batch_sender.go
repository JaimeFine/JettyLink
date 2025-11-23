package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"os"
)

type Header struct {
	Magic		[4]byte
	BatchIdx	uint32
	DType		uint32
	DataLen		uint64
	LabelLen	uint64
}

func main() {
	conn, err := net.Dial("tcp", "JETSON_IP:9000")
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	fmt.Println("Connected to Jetson!")

	// Receive format:
	// dataBytes	[]byte	// float32 tensor bytes
	// labelBytes	[]byte	// int64 labels
	
}