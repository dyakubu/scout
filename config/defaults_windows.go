package config

import _ "embed"

// See defaults_darwin.go for why this is split per-OS.
//
//go:embed scoutconfig_windows.toml
var defaultConfig []byte
