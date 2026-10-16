#!/usr/bin/env bash
# Compatibility entry point; defaults now target Amazon -> Yahoo.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run.sh" collect
