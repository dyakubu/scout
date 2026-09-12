package indexer

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"strings"
)

// documentPart is the entry in a .docx's zip that holds the body text.
// Headers, footers and footnotes live in their own parts and aren't read.
const documentPart = "word/document.xml"

// extractDOCXText reads a .docx and returns its text, one paragraph per
// line, so chunk line numbers land on paragraph boundaries the way they
// land on page markers for PDFs.
//
// A .docx is a zip of XML. The text sits in <w:t> elements, which a
// paragraph can split into several of - Word starts a new one wherever
// formatting changes mid-sentence - so the runs are concatenated and the
// paragraph ends at </w:p>. Table cell text is <w:t> too, so it comes
// through without special handling.
func extractDOCXText(raw []byte) (string, error) {
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return "", fmt.Errorf("opening docx: %w", err)
	}

	var part *zip.File
	for _, f := range zr.File {
		if f.Name == documentPart {
			part = f
			break
		}
	}
	if part == nil {
		return "", fmt.Errorf("no %s inside the archive", documentPart)
	}

	rc, err := part.Open()
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", documentPart, err)
	}
	defer rc.Close()

	return parseDocumentXML(rc)
}

func parseDocumentXML(r io.Reader) (string, error) {
	var text strings.Builder
	decoder := xml.NewDecoder(r)
	inText := false

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("parsing %s: %w", documentPart, err)
		}

		switch t := token.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t":
				inText = true
			case "tab":
				text.WriteString("\t")
			case "br", "cr":
				text.WriteString("\n")
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				text.WriteString("\n")
			}
		case xml.CharData:
			// Only inside <w:t>. Everything else at this level is
			// formatting and layout, which would otherwise arrive as
			// stray whitespace between runs.
			if inText {
				text.Write(t)
			}
		}
	}

	return cleanExtractedText(text.String()), nil
}
