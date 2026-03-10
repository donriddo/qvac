'use strict'

// Wrapper around the auto-generated all.js that keeps the bare event loop
// alive during native model loading. Without this, bare exits prematurely
// when C++ model loading runs in background threads while JS has no pending work.
setInterval(() => {}, 30_000)

require('./all.js')
