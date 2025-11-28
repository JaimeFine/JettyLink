package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"os/exec"
)

func main() {
	ln, err := net.Listen("tcp", ":9000")
	if err != nil { panic(err) }
	fmt.Println("Server listening on :9000")

	conn, err := ln.Accept()
	if err != nil { panic(err) }
	fmt.Println("PC connected.")

	// Control the python code:
	py := exec.Command("python3", "tester.py")
	pyIn, _ := py.StdinPipe()
	pyOut, _ := py.StdoutPipe()

	py.Start()

	/*
	go func() {
		// To PC:
		io.Copy(conn, pyOut)
	} ()
	*/

	// To Receive:
	for {
		header := make([]byte, 32)
		_, err := io.ReadFull(conn, header)
		if err != nil { break }

		magic := string(header[0:4])
		if magic != "BTS0" {
			fmt.Println("Invalid header")
			break
		}

		dataLen := binary.LittleEndian.Uint64(header[16:24])
		labelLen := binary.LittleEndian.Uint64(header[24:32])

		// Read tensor:
		data := make([]byte, dataLen)
		labels := make([]byte, labelLen)
		io.ReadFull(conn, data)
		io.ReadFull(conn, labels)

		// To python:
		pyIn.Write(header)
		pyIn.Write(data)
		pyIn.Write(labels)

		// Read python's response:
		reply := make([]byte, 256)
		n, _ := pyOut.Read(reply)
		fmt.Print(string(reply[:n]))

		conn.Write([]byte("OK"))
	}

	pyIn.Close()
	py.Wait()
}