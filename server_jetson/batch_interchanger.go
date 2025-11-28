package main

import (
    "encoding/binary"
    "fmt"
    "io"
    "net"
    "os"
    "os/exec"
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

    py := exec.Command("python3", "trainer.py")
    pyIn, _ := py.StdinPipe()
    pyOut, _ := py.StdoutPipe()
    pyErr, _ := py.StderrPipe()
    if err := py.Start(); err != nil {
        panic(err)
    }
    defer py.Wait()

    go io.Copy(os.Stdout, pyErr)

    headerBuf := make([]byte, 32)
    replyBuf := make([]byte, 2048)

    for {
        _, err := io.ReadFull(conn, headerBuf)
        if err != nil {
            if err != io.EOF {
                fmt.Println("header read error:", err)
            }
            break
        }

        if string(headerBuf[:4]) != "BTS0" {
            fmt.Println("bad magic")
            break
        }

        dataLen := binary.LittleEndian.Uint64(headerBuf[16:24])
        labelLen := binary.LittleEndian.Uint64(headerBuf[24:32])

        if dataLen > MaxDataBytes || labelLen > MaxLabelBytes || dataLen == 0 {
            fmt.Printf("invalid payload size data=%d label=%d\n", dataLen, labelLen)
            conn.Write([]byte("ERR"))
            break
        }

        data := make([]byte, dataLen)
        label := make([]byte, labelLen)
        io.ReadFull(conn, data)
        io.ReadFull(conn, label)

        pyIn.Write(headerBuf)
        pyIn.Write(data)
        pyIn.Write(label)

        for {
            n, err := pyOut.Read(replyBuf)
            if err != nil && err != io.EOF {
                fmt.Println("error reading Python output:", err)
                break
            }
            if n == 0 {
                break
            }
            conn.Write(replyBuf[:n])
        }

        conn.Write([]byte("OK"))
    }

    pyIn.Close()
    fmt.Println("Connection closed")
}