package embedder

// MediaEmbedder embeds images (for indexing) and search queries (for
// searching those embeddings), both into the same joint space - a search
// query is only ever comparable to a stored image embedding if it's been
// mapped into that same space, which is what EmbedText is for. See
// mediaworker.Client for the implementation backing this, which talks to
// a Python worker process.
type MediaEmbedder interface {
	// EmbedImage embeds a single image file. One call per file rather
	// than a batch, since (unlike text) an image isn't chunked.
	EmbedImage(path string) ([]float32, error)

	// EmbedText embeds a search query into the same space EmbedImage's
	// results live in. This is only ever used to query stored image
	// embeddings - indexed text file chunks are embedded entirely
	// separately, by Embedder, never through this.
	EmbedText(query string) ([]float32, error)

	// ModelID identifies the exact model producing embeddings, so stored
	// vectors can be detected as stale if the model ever changes.
	ModelID() string
}
