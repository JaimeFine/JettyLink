package main

import (
    "encoding/binary"
    "fmt"
    "io"
    "math"
    "net"
    "os"
    "strings"
)

const BatchSize = 8

func writeHeader(w io.Writer, batchIdx uint32, dataLen, labelLen uint64) error {
    var buf [32]byte
    copy(buf[:4], []byte("BTS0"))
    binary.LittleEndian.PutUint32(buf[4:8], batchIdx)
    binary.LittleEndian.PutUint32(buf[8:12], 1)
    binary.LittleEndian.PutUint64(buf[16:24], dataLen)
    binary.LittleEndian.PutUint64(buf[24:32], labelLen)
    _, err := w.Write(buf[:])
    return err
}

func readUint32BE(r io.Reader) uint32 {
    var v uint32
    binary.Read(r, binary.BigEndian, &v)
    return v
}

func loadMNISTImages(path string) [][]byte {
    f, err := os.Open(path)
    if err != nil {
        panic(err)
    }
    defer f.Close()

    magic := readUint32BE(f)
    if magic != 2051 {
        panic("bad magic in images file")
    }
    num := int(readUint32BE(f))
    readUint32BE(f)
    readUint32BE(f)

    imgSize := 28 * 28
    images := make([][]byte, num)
    for i := 0; i < num; i++ {
        buf := make([]byte, imgSize)
        if _, err := io.ReadFull(f, buf); err != nil {
            panic(err)
        }
        images[i] = buf
    }
    return images
}

func loadMNISTLabels(path string) []byte {
    f, err := os.Open(path)
    if err != nil {
        panic(err)
    }
    defer f.Close()

    if magic := readUint32BE(f); magic != 2049 {
        panic("bad magic in labels file")
    }
    num := int(readUint32BE(f))
    data := make([]byte, num)
    if _, err := io.ReadFull(f, data); err != nil {
        panic(err)
    }
    return data
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

    batchIdx := uint32(0)
    for i := 0; i+BatchSize <= len(images); i += BatchSize {
        dataBytes := make([]byte, BatchSize*1*28*28*4)
        off := 0
        for b := 0; b < BatchSize; b++ {
            img := images[i+b]
            for _, px := range img {
                f := float32(px) / 255.0
                binary.LittleEndian.PutUint32(dataBytes[off:], math.Float32bits(f))
                off += 4
            }
        }
        labelBytes := labels[i : i+BatchSize]

        if err := writeHeader(conn, batchIdx, uint64(len(dataBytes)), uint64(len(labelBytes))); err != nil {
            panic(err)
        }
        if _, err := conn.Write(dataBytes); err != nil {
            panic(err)
        }
        if _, err := conn.Write(labelBytes); err != nil {
            panic(err)
        }

        reply := make([]byte, 512)
        total := ""
        for {
            n, err := conn.Read(reply)
            if err != nil {
                panic(err)
            }
            total += string(reply[:n])
            if strings.Contains(total, "OK") {
                fmt.Print(total)
                break
            }
        }

        batchIdx++
    }
    fmt.Println("Training finished – all batches sent!")
}