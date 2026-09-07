// Package embeddertest provides fake embedder.Embedder and
// embedder.MediaEmbedder implementations for tests that need to exercise
// code depending on those interfaces (e.g. search.Searcher) without a
// real ONNX model, cgo, or a live Python worker.
//
// Both fakes require an exact vector for every input they're asked to
// embed, rather than synthesizing one - vec_chunks and vec_media are
// fixed-width tables (384 and 512 dimensions respectively), so a
// synthesized default risks silently producing the wrong width for
// whichever table a test is exercising. Forcing every test to be
// explicit about its vectors avoids that entirely.
package embeddertest

import "fmt"

// Embedder is a fake embedder.Embedder.
type Embedder struct {
	// Vectors maps each text Embed will be called with to the exact
	// vector to return for it.
	Vectors map[string][]float32
	// Model is returned by ModelID; defaults to "fake-model" if empty.
	Model string
	// Err, if set, is returned by Embed instead of a result.
	Err error
}

func (f *Embedder) Embed(texts []string) ([][]float32, error) {
	if f.Err != nil {
		return nil, f.Err
	}

	out := make([][]float32, len(texts))
	for i, text := range texts {
		v, ok := f.Vectors[text]
		if !ok {
			return nil, fmt.Errorf("embeddertest.Embedder: no vector configured for %q", text)
		}
		out[i] = v
	}

	return out, nil
}

func (f *Embedder) ModelID() string {
	if f.Model == "" {
		return "fake-model"
	}
	return f.Model
}

// MediaEmbedder is a fake embedder.MediaEmbedder.
type MediaEmbedder struct {
	// ImageVectors maps each path EmbedImage will be called with to the
	// exact vector to return for it.
	ImageVectors map[string][]float32
	// TextVectors maps each query EmbedText will be called with to the
	// exact vector to return for it.
	TextVectors map[string][]float32
	// Model is returned by ModelID; defaults to "fake-media-model" if empty.
	Model string
	// Err, if set, is returned by EmbedImage and EmbedText instead of a result.
	Err error
}

func (f *MediaEmbedder) EmbedImage(path string) ([]float32, error) {
	if f.Err != nil {
		return nil, f.Err
	}
	v, ok := f.ImageVectors[path]
	if !ok {
		return nil, fmt.Errorf("embeddertest.MediaEmbedder: no vector configured for image %q", path)
	}
	return v, nil
}

func (f *MediaEmbedder) EmbedText(query string) ([]float32, error) {
	if f.Err != nil {
		return nil, f.Err
	}
	v, ok := f.TextVectors[query]
	if !ok {
		return nil, fmt.Errorf("embeddertest.MediaEmbedder: no vector configured for query %q", query)
	}
	return v, nil
}

func (f *MediaEmbedder) ModelID() string {
	if f.Model == "" {
		return "fake-media-model"
	}
	return f.Model
}

// UnitVector returns a dim-length vector that's 1.0 at index i and 0
// elsewhere - already L2-normalized (norm 1), so distances between two
// UnitVectors are easy to reason about in tests: the same index gives
// distance 0 (identical), different indices give distance sqrt(2)
// (orthogonal, the maximum possible distance between two unit vectors).
func UnitVector(dim, i int) []float32 {
	v := make([]float32, dim)
	v[i] = 1
	return v
}
