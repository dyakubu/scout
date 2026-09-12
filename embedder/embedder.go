package embedder

// Embedder embeds a batch of texts in one call, returning one embedding per
// input text in the same order.
type Embedder interface {
	Embed(texts []string) ([][]float32, error)

	// ModelID identifies the exact model producing embeddings, so stored
	// vectors can be detected as stale if the model ever changes.
	ModelID() string

	// TokenBudget is the most tokens the model will look at in one text.
	// Anything beyond it is dropped during encoding, silently, so callers
	// deciding how much text to hand over need to size against this
	// rather than against a character count.
	TokenBudget() int

	// CountTokens reports how many tokens text occupies, so a caller can
	// check it against TokenBudget before embedding.
	CountTokens(text string) int
}
