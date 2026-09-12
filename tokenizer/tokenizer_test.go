package tokenizer

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// modelTokenizerPath resolves models/tokenizer.json relative to this test
// file's own location (not the working directory), so it works regardless
// of how `go test` is invoked.
func modelTokenizerPath(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "models", "tokenizer.json")
}

func loadTestTokenizer(t *testing.T) *BertTokenizer {
	t.Helper()
	tok, err := NewBertTokenizer(modelTokenizerPath(t))
	if err != nil {
		t.Fatalf("NewBertTokenizer: %v", err)
	}
	return tok
}

func TestNewBertTokenizer_LoadsRealVocab(t *testing.T) {
	tok := loadTestTokenizer(t)

	if got := len(tok.vocab); got != 30522 {
		t.Errorf("vocab size = %d, want 30522", got)
	}
	if tok.clsID != 101 {
		t.Errorf("clsID = %d, want 101", tok.clsID)
	}
	if tok.sepID != 102 {
		t.Errorf("sepID = %d, want 102", tok.sepID)
	}
	if tok.padID != 0 {
		t.Errorf("padID = %d, want 0", tok.padID)
	}
	if tok.unkID != 100 {
		t.Errorf("unkID = %d, want 100", tok.unkID)
	}
	if tok.maxLen != 128 {
		t.Errorf("maxLen = %d, want 128", tok.maxLen)
	}
	// strip_accents is null in the file; lowercase is true, so this must
	// resolve to true, not false - the one HF convention that's easy to
	// get backwards.
	if !tok.stripAccents {
		t.Error("stripAccents = false, want true (null inherits from lowercase=true)")
	}
}

func TestEncode_KnownSentence(t *testing.T) {
	tok := loadTestTokenizer(t)

	// Verified independently against HuggingFace's reference `tokenizers`
	// Python library against this exact tokenizer.json:
	// ids: [101, 7464, 5950, 2229, 6764, 2005, 3945, 1012, 102], then padded.
	enc := tok.Encode("Scout indexes files for search.")

	wantIDs := []int64{101, 7464, 5950, 2229, 6764, 2005, 3945, 1012, 102}

	if len(enc.IDs) != 128 {
		t.Fatalf("len(IDs) = %d, want 128", len(enc.IDs))
	}
	for i, want := range wantIDs {
		if enc.IDs[i] != want {
			t.Errorf("IDs[%d] = %d, want %d", i, enc.IDs[i], want)
		}
	}
	for i := len(wantIDs); i < len(enc.IDs); i++ {
		if enc.IDs[i] != 0 {
			t.Errorf("IDs[%d] = %d, want 0 (pad)", i, enc.IDs[i])
		}
	}

	for i := range enc.AttentionMask {
		want := int64(0)
		if i < len(wantIDs) {
			want = 1
		}
		if enc.AttentionMask[i] != want {
			t.Errorf("AttentionMask[%d] = %d, want %d", i, enc.AttentionMask[i], want)
		}
	}

	for i, id := range enc.TypeIDs {
		if id != 0 {
			t.Errorf("TypeIDs[%d] = %d, want 0", i, id)
		}
	}
}

func TestEncode_SubwordSplit(t *testing.T) {
	tok := loadTestTokenizer(t)

	// "indexes" isn't whole-word in vocab; it must split into "index" +
	// "##es" - the concrete case verified against the reference tokenizer.
	enc := tok.Encode("indexes")

	indexID, ok := tok.vocab["index"]
	if !ok {
		t.Fatal(`vocab has no "index" entry`)
	}
	esID, ok := tok.vocab["##es"]
	if !ok {
		t.Fatal(`vocab has no "##es" entry`)
	}

	want := []int64{tok.clsID, indexID, esID, tok.sepID}
	for i, id := range want {
		if enc.IDs[i] != id {
			t.Errorf("IDs[%d] = %d, want %d", i, enc.IDs[i], id)
		}
	}
}

func TestEncode_Truncation(t *testing.T) {
	tok := loadTestTokenizer(t)

	long := strings.Repeat("scout indexes files for semantic search ", 30)
	enc := tok.Encode(long)

	if len(enc.IDs) != 128 {
		t.Fatalf("len(IDs) = %d, want 128", len(enc.IDs))
	}
	if enc.IDs[0] != tok.clsID {
		t.Errorf("IDs[0] = %d, want [CLS] (%d)", enc.IDs[0], tok.clsID)
	}
	if enc.IDs[127] != tok.sepID {
		t.Errorf("IDs[127] = %d, want [SEP] (%d) - truncation must reserve room for it", enc.IDs[127], tok.sepID)
	}
	for i, mask := range enc.AttentionMask {
		if mask != 1 {
			t.Errorf("AttentionMask[%d] = %d, want 1 (a full 128-token sequence has no padding)", i, mask)
		}
	}
}

func TestEncode_UnknownWord(t *testing.T) {
	tok := loadTestTokenizer(t)

	// A run of characters with no vocab match at all (not even a
	// single-character fallback) must become exactly one [UNK].
	enc := tok.Encode("\U0001F600\U0001F601\U0001F602")

	if enc.IDs[1] != tok.unkID {
		t.Errorf("IDs[1] = %d, want [UNK] (%d)", enc.IDs[1], tok.unkID)
	}
	if enc.IDs[2] != tok.sepID {
		t.Errorf("IDs[2] = %d, want [SEP] immediately after the single [UNK] (%d)", enc.IDs[2], tok.sepID)
	}
}

func TestEncode_EmptyString(t *testing.T) {
	tok := loadTestTokenizer(t)

	enc := tok.Encode("")

	if enc.IDs[0] != tok.clsID || enc.IDs[1] != tok.sepID {
		t.Errorf("IDs[:2] = %v, want [CLS, SEP]", enc.IDs[:2])
	}
	for i := 2; i < len(enc.IDs); i++ {
		if enc.IDs[i] != tok.padID {
			t.Errorf("IDs[%d] = %d, want [PAD]", i, enc.IDs[i])
		}
	}
	if enc.AttentionMask[0] != 1 || enc.AttentionMask[1] != 1 || enc.AttentionMask[2] != 0 {
		t.Errorf("AttentionMask[:3] = %v, want [1, 1, 0]", enc.AttentionMask[:3])
	}
}

func TestEncode_LowercaseAndAccents(t *testing.T) {
	tok := loadTestTokenizer(t)

	// "Café" should normalize to "cafe" (lowercased, accent stripped)
	// before WordPiece even runs.
	enc := tok.Encode("Café")

	cafeID, ok := tok.vocab["cafe"]
	if !ok {
		t.Skip(`vocab has no "cafe" entry - can't assert exact id, but normalization is exercised regardless`)
	}
	if enc.IDs[1] != cafeID {
		t.Errorf("IDs[1] = %d, want %d (\"cafe\")", enc.IDs[1], cafeID)
	}
}

func TestSplitPunctuation(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"search.", []string{"search", "."}},
		{"hello", []string{"hello"}},
		{"$100", []string{"$", "100"}},
		{"", nil},
	}

	for _, tt := range tests {
		got := splitPunctuation(tt.in)
		if !equalStrings(got, tt.want) {
			t.Errorf("splitPunctuation(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestCountContentTokens_MatchesEncode ties the counter to what Encode
// actually keeps: below the budget they agree exactly, and above it the
// count keeps climbing while Encode silently stops at the budget. That gap
// is what chunking has to size against.
func TestCountContentTokens_MatchesEncode(t *testing.T) {
	tok := loadTestTokenizer(t)

	budget := tok.ContentBudget()
	if budget <= 0 {
		t.Fatalf("ContentBudget() = %d, want a positive budget", budget)
	}

	short := "exponential backoff with jitter avoids thundering-herd retries"
	count := tok.CountContentTokens(short)
	if count > budget {
		t.Fatalf("test string is longer than the budget (%d > %d)", count, budget)
	}

	// Encode pads to a fixed width, so the content length is the number of
	// non-padding positions, less [CLS] and [SEP].
	enc := tok.Encode(short)
	var attended int
	for _, m := range enc.AttentionMask {
		attended += int(m)
	}
	if got := attended - 2; got != count {
		t.Errorf("CountContentTokens = %d, but Encode kept %d content tokens", count, got)
	}

	long := strings.Repeat("tokenization density varies by content type. ", 40)
	if tok.CountContentTokens(long) <= budget {
		t.Fatal("test string should exceed the budget, adjust it")
	}

	enc = tok.Encode(long)
	attended = 0
	for _, m := range enc.AttentionMask {
		attended += int(m)
	}
	if got := attended - 2; got != budget {
		t.Errorf("Encode kept %d content tokens for over-budget input, want it capped at %d", got, budget)
	}
}
