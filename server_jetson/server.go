package main

import (
    "encoding/json"
    "io/ioutil"
    "log"
    "net/http"
    "os/exec"
)

type InputData struct {
    Data []float64 `json:"data"`
}

type OutputData struct {
    Output []float64 `json:"output"`
}

func main() {
    http.HandleFunc("/infer", inferHandler)
    log.Println("Server running on :8080")
    http.ListenAndServe("0.0.0.0:8080", nil)
}

func inferHandler(w http.ResponseWriter, r *http.Request) {
    var input InputData

    body, err := ioutil.ReadAll(r.Body)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }

    if err := json.Unmarshal(body, &input); err != nil {
        http.Error(w, err.Error(), 500)
        return
    }

    inputJSON, _ := json.Marshal(input)

    pythonPath := "/home/ubuntu/myenv/bin/python3"	// A random path
    scriptPath := "/home/ubuntu/projects/gpu_server/infer.py"	// A random path

    cmd := exec.Command(pythonPath, scriptPath)
    stdin, _ := cmd.StdinPipe()
    stdout, _ := cmd.StdoutPipe()
    stderr, _ := cmd.StderrPipe()

    if err := cmd.Start(); err != nil {
        http.Error(w, "Error starting python: "+err.Error(), 500)
        return
    }

    stdin.Write(inputJSON)
    stdin.Close()

    outputBytes, _ := ioutil.ReadAll(stdout)

    errBytes, _ := ioutil.ReadAll(stderr)
    if len(errBytes) > 0 {
        http.Error(w, "Python error: "+string(errBytes), 500)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    w.Write(outputBytes)
}
