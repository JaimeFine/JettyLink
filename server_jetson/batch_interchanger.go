package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"os/exec"
)

const (
    MaxDataBytes  = 10 * 1024 * 1024
    MaxLabelBytes = 64 * 1024
)

func main() {
	ln, err := net.Listen("tcp", ":9000")
	if err != nil {
		panic(err)
	}
	fmt.Println("Server listening on :9000")
	defer ln.Close()

	conn, err := ln.Accept()
	if err != nil {
		panic(err)
	}
	fmt.Println("PC connected.")
	defer conn.Close()

	// Control the python code:
	py := exec.Command("python3", "tester.py")
	pyIn, _ := py.StdinPipe()
	pyOut, _ := py.StdoutPipe()
	if err := py.Start(); err != nil {
		panic(err)
	}
	defer py.Wait()

	// To PC:
	go io.Copy(conn, pyOut)

	// To Receive:
	buf := make([]byte, 32)
	for {
		_, err := io.ReadFull(conn, buf)
		if err != nil {
			if err != io.EOF {
                fmt.Printf("Error reading header: %v\n", err)
            }
			break
		}

		if string(buf[0:4] != "BTS0" {
			fmt.Println("Invalid header")
			break
		}

		dataLen := binary.LittleEndian.Uint64(header[16:24])
		labelLen := binary.LittleEndian.Uint64(header[24:32])

		if dataLen > MaxDataBytes || labelLen > MaxLabelBytes || dataLen == 0 {
            fmt.Printf("Invalid lengths: data=%d label=%d\n", dataLen, labelLen)
            conn.Write([]byte("ERR"))
            break
        }

		// Read tensor:
		data := make([]byte, dataLen)
		labels := make([]byte, labelLen)
		if _, err := io.ReadFull(conn, data); err != nil {
            fmt.Printf("Error reading data: %v\n", err)
            break
        }
        if _, err := io.ReadFull(conn, labels); err != nil {
            fmt.Printf("Error reading labels: %v\n", err)
            break
        }

		// To python:
		if _, err := pyIn.Write(header); err != nil {
            fmt.Printf("Error writing header to Python: %v\n", err)
            break
        }
        if _, err := pyIn.Write(data); err != nil {
            fmt.Printf("Error writing data to Python: %v\n", err)
            break
        }
        if _, err := pyIn.Write(labels); err != nil {
            fmt.Printf("Error writing labels to Python: %v\n", err)
            break
        }

		// Read python's response:
		reply := make([]byte, 1024)
		var response string
		for {
			n, err := pyOut.Read(reply)
			if err != nil && err != io.EOF {
				fmt.Printf("Error reading Python response: %v\n", err)
				break
			}
			if n == 0 {
				break
			}
			response += string(reply[:n])
		}
		fmt.Print(response)

		if _, err := conn.Write([]byte("OK")); err != nil {
			fmt.Printf("Error sending OK: %v\n", err)
			break
		}
	}

	pyIn.Close()
}