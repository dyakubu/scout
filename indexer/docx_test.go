package indexer

import (
	"archive/zip"
	"bytes"
	"strings"
	"testing"
)

// buildDOCX assembles a minimal .docx around the given document.xml body.
// A real one carries styles, settings and relationship parts too, none of
// which extraction reads, so a fixture doesn't need them - which keeps
// this out of the repo as a binary blob.
func buildDOCX(t *testing.T, body string) []byte {
	t.Helper()

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)

	w, err := zw.Create(documentPart)
	if err != nil {
		t.Fatalf("creating %s: %v", documentPart, err)
	}
	doc := `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>` + body + `</w:body></w:document>`
	if _, err := w.Write([]byte(doc)); err != nil {
		t.Fatalf("writing document.xml: %v", err)
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("closing zip: %v", err)
	}
	return buf.Bytes()
}

func para(runs ...string) string {
	var b strings.Builder
	b.WriteString("<w:p>")
	for _, r := range runs {
		b.WriteString("<w:r><w:t>" + r + "</w:t></w:r>")
	}
	b.WriteString("</w:p>")
	return b.String()
}

func TestExtractDOCXText(t *testing.T) {
	raw := buildDOCX(t, para("Queue design")+
		// Word splits a sentence across runs wherever formatting changes;
		// they have to join without a space appearing between them.
		para("Exponential backoff with", " jitter")+
		para("Col A")+
		"<w:p><w:r><w:t>Left</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>Right</w:t></w:r></w:p>"+
		"<w:p/>"+
		para("After a blank paragraph"))

	got, err := extractDOCXText(raw)
	if err != nil {
		t.Fatalf("extractDOCXText: %v", err)
	}

	for _, want := range []string{
		"Queue design",
		"Exponential backoff with jitter",
		"Left\tRight",
		"After a blank paragraph",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("extracted text missing %q\ngot:\n%s", want, got)
		}
	}

	// Paragraphs are the unit line numbers count, so they have to be
	// separated rather than run together.
	if strings.Contains(got, "Queue designExponential") {
		t.Error("paragraphs ran together - no newline at </w:p>")
	}
}

// Formatting elements sit between the text runs. Their character data
// would otherwise land in the output as stray whitespace.
func TestExtractDOCXText_IgnoresNonTextElements(t *testing.T) {
	raw := buildDOCX(t, `<w:p>
		<w:pPr><w:spacing w:after="200"/><w:rPr><w:b/></w:rPr></w:pPr>
		<w:r><w:rPr><w:b/></w:rPr><w:t>Only this</w:t></w:r>
	</w:p>`)

	got, err := extractDOCXText(raw)
	if err != nil {
		t.Fatalf("extractDOCXText: %v", err)
	}

	if strings.TrimSpace(got) != "Only this" {
		t.Errorf("extracted %q, want just the run text", got)
	}
}

// Table cells hold ordinary <w:t> runs, so they need no special handling -
// worth pinning, since losing table text would be silent.
func TestExtractDOCXText_ReadsTables(t *testing.T) {
	raw := buildDOCX(t, `<w:tbl><w:tr>
		<w:tc>`+para("retry budget")+`</w:tc>
		<w:tc>`+para("three attempts")+`</w:tc>
	</w:tr></w:tbl>`)

	got, err := extractDOCXText(raw)
	if err != nil {
		t.Fatalf("extractDOCXText: %v", err)
	}

	for _, want := range []string{"retry budget", "three attempts"} {
		if !strings.Contains(got, want) {
			t.Errorf("table text missing %q, got %q", want, got)
		}
	}
}

func TestExtractDOCXText_Errors(t *testing.T) {
	tests := []struct {
		name string
		raw  []byte
		want string
	}{
		{"not a zip", []byte("this is plain text"), "opening docx"},
		{"zip without a document part", func() []byte {
			var buf bytes.Buffer
			zw := zip.NewWriter(&buf)
			w, _ := zw.Create("word/styles.xml")
			w.Write([]byte("<styles/>"))
			zw.Close()
			return buf.Bytes()
		}(), "no word/document.xml"},
		{"malformed xml", func() []byte {
			var buf bytes.Buffer
			zw := zip.NewWriter(&buf)
			w, _ := zw.Create(documentPart)
			w.Write([]byte("<w:body><w:p>unclosed"))
			zw.Close()
			return buf.Bytes()
		}(), "parsing word/document.xml"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := extractDOCXText(tt.raw)
			if err == nil {
				t.Fatal("want an error, got none")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

// extractText has to route .docx to the docx reader, case-insensitively,
// and leave everything else as raw bytes.
func TestExtractText_RoutesDOCX(t *testing.T) {
	raw := buildDOCX(t, para("routed correctly"))

	for _, path := range []string{"/notes/report.docx", "/notes/REPORT.DOCX"} {
		got, err := extractText(path, raw)
		if err != nil {
			t.Fatalf("extractText(%s): %v", path, err)
		}
		if !strings.Contains(got, "routed correctly") {
			t.Errorf("extractText(%s) = %q, want the unpacked text", path, got)
		}
	}

	if got, _ := extractText("/notes/plain.txt", []byte("raw bytes")); got != "raw bytes" {
		t.Errorf("extractText(.txt) = %q, want the bytes unchanged", got)
	}
}
