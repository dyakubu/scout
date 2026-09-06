package embedder

// MediaEmbedder embeds a single image, mirroring Embedder's shape for
// text. One call per file rather than a batch, since (unlike text) an
// image isn't chunked - see mediaworker.Client for the implementation
// backing this, which talks to a Python worker process.
type MediaEmbedder interface {
	EmbedImage(path string) ([]float32, error)

	// ModelID identifies the exact model producing embeddings, so stored
	// vectors can be detected as stale if the model ever changes.
	ModelID() string
}
