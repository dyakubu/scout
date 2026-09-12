package indexer

import "errors"

type Chunk struct {
	content   string
	index     int
	startLine int
	endLine   int
}

// ChunkLimits bounds how much text one chunk may hold.
//
// MaxRunes is the ceiling. MaxTokens is the embedding model's real limit,
// which is what actually matters: a model that reads 126 tokens drops
// everything past them, and how many tokens a run of text produces varies
// enormously with content. 500 runes of English prose is about 131 tokens;
// the same 500 runes of JSON is about 299. Sizing on characters alone
// silently truncated most chunks of anything denser than prose.
type ChunkLimits struct {
	MaxRunes int

	// MaxTokens is the token budget, and CountTokens measures a candidate
	// chunk against it. Both are needed for token-aware sizing; with
	// either unset, chunks are bounded by MaxRunes alone.
	MaxTokens   int
	CountTokens func(text string) int
}

func (l ChunkLimits) tokenAware() bool {
	return l.MaxTokens > 0 && l.CountTokens != nil
}

// TODO v0 Chunking strategy is simple fixed length chunking
func ChunkText(text string, limits ChunkLimits) ([]Chunk, error) {
	if limits.MaxRunes <= 0 {
		return nil, errors.New("chunk size must be greater than zero")
	}

	// Chunk by rune, not by byte, so a chunk boundary never lands in the
	// middle of a multi-byte UTF-8 codepoint.
	runes := []rune(text)

	var idx int
	var chunks []Chunk
	line := 1 // 1-indexed, matches editors/grep

	for i := 0; i < len(runes); {
		end := i + limits.fit(runes[i:min(i+limits.MaxRunes, len(runes))])
		startLine := line

		for _, r := range runes[i:end] {
			if r == '\n' {
				line++
			}
		}

		chunks = append(chunks, Chunk{
			index:     idx,
			content:   string(runes[i:end]),
			startLine: startLine,
			endLine:   line,
		})
		idx += 1
		i = end
	}

	return chunks, nil
}

// fit returns how many of candidate's runes fit within the token budget,
// which is all of them unless the content is dense enough to tokenize past
// it.
//
// Each attempt that comes in over budget is scaled down by the ratio it
// missed by, which converges in a couple of passes: tokens per rune is
// near-constant within one stretch of text, so the first estimate is
// usually close. Shrinking by at least one rune each time guarantees it
// terminates even on content that breaks that assumption.
func (l ChunkLimits) fit(candidate []rune) int {
	if !l.tokenAware() {
		return len(candidate)
	}

	n := len(candidate)
	for n > 1 {
		tokens := l.CountTokens(string(candidate[:n]))
		if tokens <= l.MaxTokens {
			break
		}

		scaled := n * l.MaxTokens / tokens
		if scaled >= n {
			scaled = n - 1
		}
		n = scaled
	}

	// A single rune worth more than the whole budget can't be split any
	// further; embedding a truncated version of it beats looping forever.
	if n < 1 {
		n = 1
	}

	return n
}

func ChunkFile(path string, limits ChunkLimits) ([]Chunk, error) {

	content, err := ReadFile(path)

	if err != nil {
		return nil, err
	}

	res, err := ChunkText(content, limits)

	if err != nil {
		return nil, err
	}

	return res, nil
}
