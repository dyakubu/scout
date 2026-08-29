package commands

import (
	"fmt"
	"os"

	"github.com/dyakubu/scout/config"
)

// Clean removes scout's generated artifacts - the search index and trace
// log - so the next index starts completely fresh. It does not touch
// scoutconfig.toml, since that holds user customization, not generated
// state; deleting it (if that's ever wanted) already works on its own,
// since it's recreated from the embedded default on next run.
func Clean(cfg *config.Config) error {
	logPath, err := config.LogPath()
	if err != nil {
		return err
	}

	removed := 0

	for _, path := range []string{cfg.DB.Path, logPath} {
		if err := os.Remove(path); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return fmt.Errorf("removing %s: %w", path, err)
		}

		fmt.Printf("removed %s\n", path)
		removed++
	}

	if removed == 0 {
		fmt.Println("nothing to clean")
	}

	return nil
}
