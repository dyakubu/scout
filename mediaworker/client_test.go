package mediaworker

import (
	"os/exec"
	"testing"
)

// scriptPath is relative to this package's directory, which is also `go
// test`'s working directory by default.
const scriptPath = "../media/worker.py"

func TestSendJob_DummyRoundTrip(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found on PATH")
	}
	// Real inference needs network access and heavy dependencies neither
	// tests nor CI should require - see media/worker.py's SCOUT_MEDIA_DUMMY.
	t.Setenv("SCOUT_MEDIA_DUMMY", "1")

	client, err := Start([]string{"python3", scriptPath}, "/fake/model/dir", "media-dummy-v1")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Close()

	if got := client.ModelID(); got != "media-dummy-v1" {
		t.Errorf("ModelID() = %q, want %q", got, "media-dummy-v1")
	}

	result, err := client.SendJob(Job{
		ID:      "job-1",
		Payload: "dummy",
	})
	if err != nil {
		t.Fatalf("SendJob: %v", err)
	}

	if result.ID != "job-1" {
		t.Errorf("id = %q, want %q", result.ID, "job-1")
	}
	if result.Error != nil {
		t.Errorf("unexpected error: %v", *result.Error)
	}
	if len(result.Embedding) != 512 {
		t.Errorf("embedding length = %d, want 512", len(result.Embedding))
	}
}

func TestSendJob_MissingModelDir(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found on PATH")
	}
	// Real inference needs network access and heavy dependencies neither
	// tests nor CI should require - see media/worker.py's SCOUT_MEDIA_DUMMY.
	t.Setenv("SCOUT_MEDIA_DUMMY", "1")

	client, err := Start([]string{"python3", scriptPath}, "", "media-dummy-v1")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Close()

	result, err := client.SendJob(Job{ID: "job-2", Payload: "dummy"})
	if err != nil {
		t.Fatalf("SendJob: %v", err)
	}

	if result.Error == nil {
		t.Fatal("expected an error for a missing model_dir, got none")
	}
}

func TestEmbedImage(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not found on PATH")
	}
	// Real inference needs network access and heavy dependencies neither
	// tests nor CI should require - see media/worker.py's SCOUT_MEDIA_DUMMY.
	t.Setenv("SCOUT_MEDIA_DUMMY", "1")

	client, err := Start([]string{"python3", scriptPath}, "/fake/model/dir", "media-dummy-v1")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer client.Close()

	embedding, err := client.EmbedImage("/fake/image.jpg")
	if err != nil {
		t.Fatalf("EmbedImage: %v", err)
	}

	if len(embedding) != 512 {
		t.Errorf("embedding length = %d, want 512", len(embedding))
	}
}
