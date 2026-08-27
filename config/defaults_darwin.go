package config

import _ "embed"

// defaultConfig is the config scout writes on first run. It's split per-OS
// (see defaults_linux.go) because ort_library_path needs the platform's
// shared library extension - a Linux user handed a ".dylib" default would
// have a broken config from the very first run.
//
//go:embed scoutconfig_darwin.toml
var defaultConfig []byte
