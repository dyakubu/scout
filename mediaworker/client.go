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

// Job is one unit of work sent to the media worker. ModelDir isn't part of
// this - one worker process always serves one fixed model directory,
// passed once as a --model-dir startup argument (see Start), not per job.
type Job struct {
	ID string `json:"id"`

	// Kind selects which CLIP tower embeds Payload: "image" (a file path)
	// or "text" (a search query). Both land in the same 512-dim CLIP
	// space, but "text" is only ever used to embed a query for searching
	// vec_media - indexed text file chunks are embedded entirely
	// separately, by embedder.Embedder, never through this worker.
	Kind    string `json:"kind"`
	Payload string `json:"payload"`
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

	modelID string

	mu sync.Mutex
}

// Start launches the media worker as a subprocess, wiring its stdin/stdout
// for JSON-lines communication and appending "--model-dir modelDir" to
// command so the worker can load its model eagerly at startup rather than
// waiting for the first job. command is the command line to run it with,
// before that flag (e.g. []string{"python3", "media/worker.py"} for the
// dependency-free dummy path, or []string{"uv", "run", "--project",
// "media", "media/worker.py"} for real inference, which needs the media/
// project's own virtualenv - the bare "python3" on PATH won't have
// torch/transformers installed). The worker's stderr is connected to this
// process's stderr so its own logging/errors surface directly instead of
// being silently dropped. modelID is returned by ModelID() and is a
// caller-chosen label for now (e.g. "openai/clip-vit-base-patch32") until
// a real model's ModelID can be derived from its actual weights, the way
// embedder.LocalEmbedder does.
func Start(command []string, modelDir, modelID string) (*Client, error) {
	if len(command) == 0 {
		return nil, fmt.Errorf("command must not be empty")
	}

	args := append(append([]string{}, command[1:]...), "--model-dir", modelDir)
	cmd := exec.Command(command[0], args...)
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
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewScanner(stdout),
		modelID: modelID,
	}, nil
}

// ModelID returns the label identifying the model this worker is using.
func (c *Client) ModelID() string {
	return c.modelID
}

// EmbedImage embeds a single image by round-tripping a Job through the
// worker subprocess, using path as the job's ID.
func (c *Client) EmbedImage(path string) ([]float32, error) {
	result, err := c.SendJob(Job{ID: path, Kind: "image", Payload: path})
	if err != nil {
		return nil, err
	}

	if result.Error != nil {
		return nil, fmt.Errorf("media worker: %s", *result.Error)
	}

	return result.Embedding, nil
}

// EmbedText embeds a search query into the same CLIP space EmbedImage's
// results live in, so it's directly comparable to them - this is only
// ever used to query vec_media, never to embed text file chunks.
func (c *Client) EmbedText(query string) ([]float32, error) {
	result, err := c.SendJob(Job{ID: query, Kind: "text", Payload: query})
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
