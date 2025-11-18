package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {
	json := []byte(`{"data":[1, 2, 3]}`)

	resp, err := http.Post(
		"http://localhost:8080/infer",
		"application/json",
		bytes.NewBuffer(json),
	)
	if err != nil {
		panic(err)
	}

	body, _ := ioutil.ReadAll(resp.Body)
	fmt.Println(string(body))
}