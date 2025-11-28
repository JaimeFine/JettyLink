package main

import (
    "bufio"
    "encoding/binary"
    "fmt"
    "io"
    "net"
    "os"
    "os/exec"
	"strings"
)

const (
    MaxDataBytes  = 20 * 1024 * 1024
    MaxLabelBytes = 128 * 1024
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
    fmt.Println("PC connected")
    defer conn.Close()

    py := exec.Command("python3", "tester.py")
    pyInRaw, _ := py.StdinPipe()
    pyOutRaw, _ := py.StdoutPipe()
    pyErr, _ := py.StderrPipe()
    if err := py.Start(); err != nil {
        panic(err)
    }
    defer py.Wait()

    go io.Copy(os.Stdout, pyErr)

    pyIn := bufio.NewWriter(pyInRaw)

    pyOut := bufio.NewReader(pyOutRaw)

    headerBuf := make([]byte, 32)

    for {
        fmt.Println("Server: Waiting for header from client")
        _, err := io.ReadFull(conn, headerBuf)
        if err != nil {
            if err != io.EOF {
                fmt.Println("header read error:", err)
            }
            break
        }
        fmt.Println("Server: Header read successfully")

        if string(headerBuf[:4]) != "BTS0" {
            fmt.Println("bad magic")
            break
        }

        batchIdx := binary.LittleEndian.Uint32(headerBuf[4:8])
        dataLen := binary.LittleEndian.Uint64(headerBuf[16:24])
        labelLen := binary.LittleEndian.Uint64(headerBuf[24:32])
        fmt.Printf("Server: Received batch %d, dataLen=%d, labelLen=%d\n", batchIdx, dataLen, labelLen)

        if dataLen > MaxDataBytes || labelLen > MaxLabelBytes || dataLen == 0 {
            fmt.Printf("invalid payload size data=%d label=%d\n", dataLen, labelLen)
            conn.Write([]byte("ERR"))
            break
        }

        data := make([]byte, dataLen)
        label := make([]byte, labelLen)
        fmt.Println("Server: Reading data from client")
        io.ReadFull(conn, data)
        fmt.Println("Server: Data read successfully")
        fmt.Println("Server: Reading labels from client")
        io.ReadFull(conn, label)
        fmt.Println("Server: Labels read successfully")

        // Forward to Python
        fmt.Println("Server: Writing header to Python")
        pyIn.Write(headerBuf)
        fmt.Println("Server: Writing data to Python")
        pyIn.Write(data)
        fmt.Println("Server: Writing labels to Python")
        pyIn.Write(label)
        fmt.Println("Server: Flushing to Python")
        if err := pyIn.Flush(); err != nil {
            fmt.Println("flush error:", err)
        }
        fmt.Println("Server: Sent batch to Python - waiting for output")

        // Read and forward Python output line by line
        for {
            line, err := pyOut.ReadString('\n')
            if err != nil && err != io.EOF {
                fmt.Println("output read error:", err)
                break
            }
            if line == "" {
                fmt.Println("Server: No more output from Python for this batch")
                break
            }

			// To client:
            fmt.Printf("Server: Got line from Python: %s\n", line)
			if _, werr := conn.Write([]byte(line)); werr != nil {
				fmt.Println("error forwarding to client:", werr)
				break
			}
            
			if strings.Contains(line, "OK") {
				fmt.Println("Server: Detected OK from Python")
				break
			}
        }

        fmt.Println("Server: Sending OK to client")
        conn.Write([]byte("OK"))
    }

    pyInRaw.Close()
    fmt.Println("Connection closed")
}