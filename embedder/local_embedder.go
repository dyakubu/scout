package embedder

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"

	"github.com/dyakubu/scout/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	inputIDsName      = "input_ids"
	attentionMaskName = "attention_mask"
	tokenTypeIDsName  = "token_type_ids"
	outputName        = "last_hidden_state"
)

// EmbedderConfig configures a LocalEmbedder: where its model and tokenizer
// live on disk, and how it should run.
type EmbedderConfig struct {
	ModelPath      string
	TokenizerPath  string
	OrtLibraryPath string
	BatchSize      int
}

type LocalEmbedder struct {
	tokenizer tokenizer.Tokenizer
	session   *ort.DynamicAdvancedSession
	batchSize int
	hiddenDim int64
	modelID   string
}

func NewLocalEmbedder(cfg EmbedderConfig) (*LocalEmbedder, error) {
	modelID, err := hashFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("hashing model file: %w", err)
	}

	tok, err := tokenizer.NewBertTokenizer(cfg.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	ort.SetSharedLibraryPath(cfg.OrtLibraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initializing onnxruntime environment: %w", err)
	}

	_, outputs, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("reading model input/output info: %w", err)
	}

	hiddenDim, err := hiddenDimOf(outputs)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, err
	}

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{inputIDsName, attentionMaskName, tokenTypeIDsName},
		[]string{outputName},
		nil,
	)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("creating onnxruntime session: %w", err)
	}

	return &LocalEmbedder{
		tokenizer: tok,
		session:   session,
		batchSize: cfg.BatchSize,
		hiddenDim: hiddenDim,
		modelID:   modelID,
	}, nil
}

func hashFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// ModelID returns a hash of the model file's bytes. It changes if and only
// if the model file itself changes, so it's used to detect embeddings that
// were produced by a different (or since-updated) model.
func (e *LocalEmbedder) ModelID() string {
	return e.modelID
}

// TokenBudget and CountTokens expose the tokenizer's limits, since the
// model's real input size is measured in tokens while the text handed to
// it is chunked by runes.
func (e *LocalEmbedder) TokenBudget() int {
	return e.tokenizer.ContentBudget()
}

func (e *LocalEmbedder) CountTokens(text string) int {
	return e.tokenizer.CountContentTokens(text)
}

func hiddenDimOf(outputs []ort.InputOutputInfo) (int64, error) {
	for _, out := range outputs {
		if out.Name != outputName {
			continue
		}

		if len(out.Dimensions) == 0 {
			return 0, fmt.Errorf("model output %q has no dimensions", outputName)
		}

		hiddenDim := out.Dimensions[len(out.Dimensions)-1]
		if hiddenDim <= 0 {
			return 0, fmt.Errorf("model output %q has a non-fixed hidden dimension", outputName)
		}

		return hiddenDim, nil
	}

	return 0, fmt.Errorf("model has no output named %q", outputName)
}

// Close releases the session and onnxruntime resources held by e. The
// tokenizer needs no explicit cleanup - it's plain Go memory, not a cgo
// resource.
func (e *LocalEmbedder) Close() error {
	defer ort.DestroyEnvironment()
	defer e.session.Destroy()
	return nil
}

// Embed tokenizes and embeds texts in one or more batches of size
// batchSize, returning one embedding per input text in the same order.
func (e *LocalEmbedder) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	batchSize := e.batchSize
	if batchSize <= 0 {
		batchSize = len(texts)
	}

	embeddings := make([][]float32, 0, len(texts))

	for start := 0; start < len(texts); start += batchSize {
		end := min(start+batchSize, len(texts))

		batchEmbeddings, err := e.embedBatch(texts[start:end])
		if err != nil {
			return nil, err
		}

		embeddings = append(embeddings, batchEmbeddings...)
	}

	return embeddings, nil
}

// embedBatch runs one ONNX Runtime inference call over a batch of texts.
// It relies on the tokenizer's own fixed-padding configuration (see
// tokenizer.json) to guarantee every encoding in the batch has the same
// sequence length, so no manual padding is needed here.
func (e *LocalEmbedder) embedBatch(texts []string) ([][]float32, error) {
	batchLen := len(texts)

	encodings := make([]tokenizer.Encoding, batchLen)
	for i, text := range texts {
		encodings[i] = e.tokenizer.Encode(text)
	}

	seqLen := len(encodings[0].IDs)

	inputIDs := make([]int64, batchLen*seqLen)
	attentionMask := make([]int64, batchLen*seqLen)
	tokenTypeIDs := make([]int64, batchLen*seqLen)

	for i, enc := range encodings {
		for j := range seqLen {
			offset := i*seqLen + j
			inputIDs[offset] = enc.IDs[j]
			attentionMask[offset] = enc.AttentionMask[j]
			tokenTypeIDs[offset] = enc.TypeIDs[j]
		}
	}

	shape := ort.NewShape(int64(batchLen), int64(seqLen))

	inputIDsTensor, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("creating input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(shape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("creating attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	tokenTypeIDsTensor, err := ort.NewTensor(shape, tokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("creating token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDsTensor.Destroy()

	outputShape := ort.NewShape(int64(batchLen), int64(seqLen), e.hiddenDim)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = e.session.Run(
		[]ort.Value{inputIDsTensor, attentionMaskTensor, tokenTypeIDsTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("running onnxruntime session: %w", err)
	}

	return meanPool(outputTensor.GetData(), attentionMask, batchLen, seqLen, int(e.hiddenDim)), nil
}

// meanPool reduces per-token hidden states to one embedding per sequence via
// attention-mask-weighted mean pooling, then L2-normalizes each result -
// the standard sentence-transformers pooling strategy for this model
// family, and what lets cosine similarity be computed as a plain dot
// product downstream.
func meanPool(hidden []float32, attentionMask []int64, batchLen, seqLen, hiddenDim int) [][]float32 {
	embeddings := make([][]float32, batchLen)

	for b := range batchLen {
		sum := make([]float32, hiddenDim)
		var count float32

		for t := range seqLen {
			if attentionMask[b*seqLen+t] == 0 {
				continue
			}
			count++

			base := (b*seqLen + t) * hiddenDim
			for d := range hiddenDim {
				sum[d] += hidden[base+d]
			}
		}

		if count == 0 {
			count = 1
		}

		var norm float32
		for d := range hiddenDim {
			sum[d] /= count
			norm += sum[d] * sum[d]
		}

		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for d := range hiddenDim {
				sum[d] /= norm
			}
		}

		embeddings[b] = sum
	}

	return embeddings
}
