// Package tokenizer defines the Tokenizer interface scout's embedders
// encode text through, plus BertTokenizer, a pure-Go WordPiece
// implementation for BERT-family models that replaces
// github.com/daulet/tokenizers (a cgo binding to HuggingFace's Rust
// tokenizers crate). BertTokenizer reads the same tokenizer.json the Rust
// library does - nothing here requires a different asset.
//
// BertTokenizer's scope is deliberately narrow: it implements exactly the
// pipeline the model's tokenizer.json declares (BertNormalizer,
// BertPreTokenizer, WordPiece, single-sequence [CLS]/[SEP] wrapping,
// fixed-length padding/truncation) rather than the full generality of
// HuggingFace's tokenizers library. It is not a general-purpose
// tokenizers.json reader, and not every model family's tokenizer -
// swapping to a non-BERT model would mean a different Tokenizer
// implementation, not a change to this one.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// tokenizerFile is the subset of tokenizer.json's schema this package
// reads. Fields outside this (decoder, post_processor's full template,
// added_tokens' per-token flags, etc.) aren't needed: post-processing here
// is hardcoded to single-sequence [CLS]/[SEP] wrapping (scout never embeds
// sequence pairs), and CLS/SEP/PAD/UNK ids are read directly out of vocab
// rather than parsed from the post-processor's template.
type tokenizerFile struct {
	Normalizer struct {
		Lowercase          bool  `json:"lowercase"`
		CleanText          bool  `json:"clean_text"`
		HandleChineseChars bool  `json:"handle_chinese_chars"`
		StripAccents       *bool `json:"strip_accents"`
	} `json:"normalizer"`

	Model struct {
		UnkToken                string           `json:"unk_token"`
		ContinuingSubwordPrefix string           `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int              `json:"max_input_chars_per_word"`
		Vocab                   map[string]int64 `json:"vocab"`
	} `json:"model"`

	Truncation struct {
		MaxLength int `json:"max_length"`
	} `json:"truncation"`
}

// Encoding is one text's tokenized form, always exactly MaxLen long in
// every field (padded or truncated to fit) - the fixed-length shape
// embedder.LocalEmbedder's batching relies on.
type Encoding struct {
	IDs           []int64
	AttentionMask []int64
	TypeIDs       []int64
}

// Tokenizer encodes text into the fixed-length id/mask/type-id form a
// transformer model expects. BertTokenizer is the only implementation
// today; the interface exists so embedder.LocalEmbedder doesn't need to
// change if scout ever supports a model family with a different
// tokenization scheme.
type Tokenizer interface {
	Encode(text string) Encoding

	// CountContentTokens returns how many tokens text produces, before
	// [CLS]/[SEP] are added and before any truncation - what Encode would
	// have to drop, if anything.
	CountContentTokens(text string) int

	// ContentBudget returns how many content tokens Encode can keep. Text
	// producing more than this loses the excess silently, so callers
	// sizing input for the model need to check against it.
	ContentBudget() int
}

// BertTokenizer implements WordPiece tokenization for one loaded
// tokenizer.json.
type BertTokenizer struct {
	vocab                   map[string]int64
	unkToken                string
	continuingSubwordPrefix string
	maxInputCharsPerWord    int

	lowercase          bool
	cleanText          bool
	handleChineseChars bool
	stripAccents       bool

	clsID int64
	sepID int64
	padID int64
	unkID int64

	// maxLen is the fixed total length of every Encoding, taken from
	// truncation.max_length. Content is truncated to maxLen-2 to leave
	// room for [CLS] and [SEP].
	maxLen int
}

// NewBertTokenizer loads a tokenizer.json file.
func NewBertTokenizer(path string) (*BertTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading tokenizer file: %w", err)
	}

	var tf tokenizerFile
	if err := json.Unmarshal(data, &tf); err != nil {
		return nil, fmt.Errorf("parsing tokenizer file: %w", err)
	}

	if tf.Truncation.MaxLength <= 0 {
		return nil, fmt.Errorf("tokenizer file has no truncation.max_length")
	}

	vocab := tf.Model.Vocab

	clsID, ok := vocab["[CLS]"]
	if !ok {
		return nil, fmt.Errorf("vocab has no [CLS] token")
	}
	sepID, ok := vocab["[SEP]"]
	if !ok {
		return nil, fmt.Errorf("vocab has no [SEP] token")
	}
	padID, ok := vocab["[PAD]"]
	if !ok {
		return nil, fmt.Errorf("vocab has no [PAD] token")
	}
	unkID, ok := vocab[tf.Model.UnkToken]
	if !ok {
		return nil, fmt.Errorf("vocab has no %q (unk_token) entry", tf.Model.UnkToken)
	}

	// strip_accents: null in the file means "inherit from lowercase", per
	// HuggingFace's BertNormalizer convention - not "don't strip".
	stripAccents := tf.Normalizer.Lowercase
	if tf.Normalizer.StripAccents != nil {
		stripAccents = *tf.Normalizer.StripAccents
	}

	return &BertTokenizer{
		vocab:                   vocab,
		unkToken:                tf.Model.UnkToken,
		continuingSubwordPrefix: tf.Model.ContinuingSubwordPrefix,
		maxInputCharsPerWord:    tf.Model.MaxInputCharsPerWord,
		lowercase:               tf.Normalizer.Lowercase,
		cleanText:               tf.Normalizer.CleanText,
		handleChineseChars:      tf.Normalizer.HandleChineseChars,
		stripAccents:            stripAccents,
		clsID:                   clsID,
		sepID:                   sepID,
		padID:                   padID,
		unkID:                   unkID,
		maxLen:                  tf.Truncation.MaxLength,
	}, nil
}

// Encode tokenizes text into a fixed-length Encoding: [CLS], the text's
// WordPiece tokens (truncated if needed), [SEP], then right-padded with
// [PAD] up to MaxLen. TypeIDs is always all zero - scout only ever embeds
// a single sequence, never a sentence pair.
// CountContentTokens tokenizes text without truncating or padding, so the
// result can be compared against ContentBudget to tell whether Encode
// would drop part of it.
func (t *BertTokenizer) CountContentTokens(text string) int {
	var n int
	for _, word := range t.preTokenize(t.normalize(text)) {
		n += len(t.wordpiece(word))
	}
	return n
}

// ContentBudget is the sequence length from tokenizer.json's truncation
// config, less the two positions [CLS] and [SEP] occupy.
func (t *BertTokenizer) ContentBudget() int {
	return t.maxLen - 2
}

func (t *BertTokenizer) Encode(text string) Encoding {
	words := t.preTokenize(t.normalize(text))

	var ids []int64
	for _, word := range words {
		ids = append(ids, t.wordpiece(word)...)
	}

	// Reserve room for [CLS] and [SEP].
	if maxContent := t.maxLen - 2; len(ids) > maxContent {
		ids = ids[:maxContent]
	}

	ids = append([]int64{t.clsID}, ids...)
	ids = append(ids, t.sepID)

	enc := Encoding{
		IDs:           make([]int64, t.maxLen),
		AttentionMask: make([]int64, t.maxLen),
		TypeIDs:       make([]int64, t.maxLen),
	}
	for i := range enc.IDs {
		enc.IDs[i] = t.padID
	}
	for i, id := range ids {
		enc.IDs[i] = id
		enc.AttentionMask[i] = 1
	}

	return enc
}

// normalize applies BertNormalizer: optional CJK spacing, control-char/
// whitespace cleanup, lowercasing, and accent stripping, in that order -
// matching the reference implementation's pipeline order exactly, since
// e.g. lowercasing before vs. after accent stripping can change output for
// certain scripts.
func (t *BertTokenizer) normalize(s string) string {
	if t.handleChineseChars {
		s = spaceOutCJK(s)
	}
	if t.cleanText {
		s = cleanText(s)
	}
	if t.lowercase {
		s = strings.ToLower(s)
	}
	if t.stripAccents {
		s = stripAccents(s)
	}
	return s
}

// spaceOutCJK inserts spaces around CJK codepoints so BertPreTokenizer's
// later whitespace split gives each CJK character its own token.
func spaceOutCJK(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if isCJK(r) {
			b.WriteRune(' ')
			b.WriteRune(r)
			b.WriteRune(' ')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// isCJK reports whether r falls in one of the CJK unicode blocks BERT's
// reference tokenizer treats specially, per its _is_chinese_char.
func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF,
		r >= 0x3400 && r <= 0x4DBF,
		r >= 0x20000 && r <= 0x2A6DF,
		r >= 0x2A700 && r <= 0x2B73F,
		r >= 0x2B740 && r <= 0x2B81F,
		r >= 0x2B820 && r <= 0x2CEAF,
		r >= 0xF900 && r <= 0xFAFF,
		r >= 0x2F800 && r <= 0x2FA1F:
		return true
	default:
		return false
	}
}

// cleanText drops null/replacement/control characters (except the
// whitespace ones handled below) and collapses any whitespace run to a
// single space, per BERT's _clean_text.
func cleanText(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r == 0 || r == 0xFFFD || isControl(r):
			continue
		case isWhitespace(r):
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// isWhitespace matches BERT's _is_whitespace: the ASCII whitespace chars
// plus anything Unicode itself categorizes as a separator.
func isWhitespace(r rune) bool {
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

// isControl matches BERT's _is_control: \t, \n, \r are explicitly excluded
// (treated as whitespace instead, see isWhitespace), everything else in a
// Unicode "C*" category counts.
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.IsControl(r)
}

// stripAccents NFD-decomposes s and drops any resulting combining mark,
// per BERT's _run_strip_accents.
func stripAccents(s string) string {
	decomposed := norm.NFD.String(s)
	var b strings.Builder
	b.Grow(len(decomposed))
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// preTokenize implements BertPreTokenizer: split on whitespace, then split
// punctuation off each resulting word as its own token.
func (t *BertTokenizer) preTokenize(s string) []string {
	var words []string
	for _, field := range strings.Fields(s) {
		words = append(words, splitPunctuation(field)...)
	}
	return words
}

// splitPunctuation matches BERT's _run_split_on_punc.
func splitPunctuation(word string) []string {
	var out []string
	var cur strings.Builder

	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}

	for _, r := range word {
		if isPunctuation(r) {
			flush()
			out = append(out, string(r))
			continue
		}
		cur.WriteRune(r)
	}
	flush()

	return out
}

// isPunctuation matches BERT's _is_punctuation: explicit ASCII ranges
// first (these cover symbols like '$' or '+' that Unicode itself
// categorizes as Sc/Sm, not P*, but BERT treats as punctuation regardless),
// falling back to Unicode's own punctuation category for everything else.
func isPunctuation(r rune) bool {
	switch {
	case r >= 33 && r <= 47,
		r >= 58 && r <= 64,
		r >= 91 && r <= 96,
		r >= 123 && r <= 126:
		return true
	default:
		return unicode.IsPunct(r)
	}
}

// wordpiece splits one pre-tokenized word into vocab ids via greedy
// longest-match-first, per BERT's WordPieceTokenizer.tokenize. A word
// longer than maxInputCharsPerWord, or one with any unmatched remainder,
// becomes a single [UNK] rather than a partial split.
func (t *BertTokenizer) wordpiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > t.maxInputCharsPerWord {
		return []int64{t.unkID}
	}

	var ids []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var matchID int64
		matched := false

		for end > start {
			candidate := string(runes[start:end])
			if start > 0 {
				candidate = t.continuingSubwordPrefix + candidate
			}
			if id, ok := t.vocab[candidate]; ok {
				matchID = id
				matched = true
				break
			}
			end--
		}

		if !matched {
			return []int64{t.unkID}
		}

		ids = append(ids, matchID)
		start = end
	}

	return ids
}
