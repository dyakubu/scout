// Package mediaworker talks to scout's Python media (image/video)
// embedding worker over stdio, one JSON object per line in each direction.
package mediaworker

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
)

// Job is one unit of work sent to the media worker.
type Job struct {
	ID       string `json:"id"`
	ModelDir string `json:"model_dir"`
	Payload  string `json:"payload"`
}

// Result is the media worker's response to one Job, matched to it by ID.
type Result struct {
	ID        string    `json:"id"`
	Embedding []float32 `json:"embedding"`
	Error     *string   `json:"error"`
}

// Client manages a running media worker subprocess, communicating with it
// over stdin/stdout. The worker is expected to process jobs strictly in
// the order they're written - one line in, one matching line out - so
// SendJob is safe to call concurrently, but calls are serialized rather
// than pipelined. Client implements embedder.MediaEmbedder.
type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner

	modelDir string
	modelID  string

	mu sync.Mutex
}

// Start launches the media worker as a subprocess (pythonPath scriptPath),
// wiring its stdin/stdout for JSON-lines communication. The worker's
// stderr is connected to this process's stderr so its own logging/errors
// surface directly instead of being silently dropped. modelDir is sent
// with every job; modelID is returned by ModelID() and is a caller-chosen
// label for now (e.g. "media-dummy-v1") until a real model's ModelID can
// be derived from its actual weights, the way embedder.LocalEmbedder does.
func Start(pythonPath, scriptPath, modelDir, modelID string) (*Client, error) {
	cmd := exec.Command(pythonPath, scriptPath)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("wiring worker stdin: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("wiring worker stdout: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting media worker: %w", err)
	}

	return &Client{
		cmd:      cmd,
		stdin:    stdin,
		stdout:   bufio.NewScanner(stdout),
		modelDir: modelDir,
		modelID:  modelID,
	}, nil
}

// ModelID returns the label identifying the model this worker is using.
func (c *Client) ModelID() string {
	return c.modelID
}

// EmbedImage embeds a single image by round-tripping a Job through the
// worker subprocess, using path as the job's ID.
func (c *Client) EmbedImage(path string) ([]float32, error) {
	result, err := c.SendJob(Job{ID: path, ModelDir: c.modelDir, Payload: path})
	if err != nil {
		return nil, err
	}

	if result.Error != nil {
		return nil, fmt.Errorf("media worker: %s", *result.Error)
	}

	return result.Embedding, nil
}

// SendJob writes job to the worker's stdin as one JSON line and blocks for
// its matching response on stdout.
func (c *Client) SendJob(job Job) (Result, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	data, err := json.Marshal(job)
	if err != nil {
		return Result{}, fmt.Errorf("encoding job: %w", err)
	}

	if _, err := c.stdin.Write(append(data, '\n')); err != nil {
		return Result{}, fmt.Errorf("writing job to worker: %w", err)
	}

	if !c.stdout.Scan() {
		if err := c.stdout.Err(); err != nil {
			return Result{}, fmt.Errorf("reading worker response: %w", err)
		}
		return Result{}, fmt.Errorf("worker closed stdout without responding")
	}

	var result Result
	if err := json.Unmarshal(c.stdout.Bytes(), &result); err != nil {
		return Result{}, fmt.Errorf("decoding worker response %q: %w", c.stdout.Text(), err)
	}

	if result.ID != job.ID {
		return Result{}, fmt.Errorf("response id %q does not match job id %q", result.ID, job.ID)
	}

	return result, nil
}

// Close closes the worker's stdin, signaling it to exit its read loop, and
// waits for the subprocess to exit.
func (c *Client) Close() error {
	if err := c.stdin.Close(); err != nil {
		return fmt.Errorf("closing worker stdin: %w", err)
	}
	return c.cmd.Wait()
}
