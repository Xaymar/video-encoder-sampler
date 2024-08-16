// --------------------------------------------------------------------------------
// Development options (for user options, see config.json)
global.debug = false;
global.version = [0, 2, 0, 0];

// --------------------------------------------------------------------------------
// Actual Code

// Node.js libraries
const PATH = require('path');
const PROCESS = require('process');
const FS = require('fs');
const UTIL = require('util');

// Our libraries
const APP = require('./app.js');
const LOG = require('./log.js');

// Constants
const E_OK = 0;
const E_FAIL = 1;

// Encoders
const ENCODERS = [
	"x264", "x265", "qsv", "nvenc", "amf", "libsvtav1"
];

class MyApp extends APP {
	constructor() {
		super();

		// Load any required libraries.
		this._libs = {
			// Third-Party
			argparse: require('../third-party/argparse/argparse.js'),

			// First-Party
			ffmpeg: require('./ffmpeg.js'),
			poolqueue: require('./poolqueue.js'),
		};

		// Paths
		this._paths = {
			config: `${PATH.join(PROCESS.cwd(), 'config.json')}`,
			ffmpeg: `${PATH.join(PROCESS.cwd(), 'ffmpeg')}`,
			input: `${PATH.join(PROCESS.cwd(), 'input')}`,
			cache: `${PATH.join(PROCESS.cwd(), 'cache')}`,
			output: `${PATH.join(PROCESS.cwd(), 'output')}`,
		}

		// VMAF
		this._vmaf = {
			model: "vmaf_4k_rb_v0.6.2.pkl",
			threads: 4,
		}

		// Color
		this._color = new Map();

		// Tests
		this._encoders = new Map();
		this._codecs = new Set();
		this._resolutions = new Set();
		this._bitrates = new Set();
		this._kfintervals = new Set();

		// Load any defined encoders.
		for (let encoder of ENCODERS) {
			try {
				this._encoders.set(encoder, new (require(`./encoder_${encoder}.js`)));
			} catch (ex) {
				this.logger.log(LOG.ELEVEL_ERROR, "Failed to compile encoder '%s' with error: %s", encoder, ex.toString());
			}
		}

		{ // Argument Parsing
			class HelpFormatter extends this._libs.argparse.RawDescriptionHelpFormatter {
				// executes parent _split_lines for each line of the help, then flattens the result
				_split_lines(text, width) {
					return [].concat(...text.split('\n').map(line => super._split_lines(line, width)));
				}
			}

			// Set up our argparse instance.
			this._ap = new this._libs.argparse.ArgumentParser({
				description: 'Video Encoding Samples Tool',
				add_help: true,
				formatter_class: HelpFormatter,
			});

			{ // ArgParse: Core
				this._ap.add_argument('-v', '--version', {
					action: 'version',
					version: `VES Tool v${global.version[0]}.${global.version[1]}.${global.version[2]}.${global.version[3]}`,
					help: 'Show the version of the tool.',
				});
				this._ap.add_argument('--copyright', {
					help: 'Hide the copyright header when running the tool. This does not change the copyright itself, only hides it.',
					action: 'store_false'
				});
				this._ap.add_argument('--license', {
					help: 'Hide the license header when running the tool. This does not change the license itself, only hides it.',
					action: 'store_false'
				});
				this._ap.add_argument('--loglevel', {
					help: 'Change which log level will be shown in the output. From most spammy to least spammy.',
					type: 'str',
					action: 'store',
					choices: ['debug', 'verbose', 'info', 'warning', 'error', 'critical'],
					default: 'info',
				});
			}

			{ // ArgParse: Paths
				this._ap_paths = this._ap.add_argument_group('Paths');
				this._ap_paths.add_argument('-cfg', '--config', {
					help: `The path to the configuration file, with relative paths being relative to the current working directory.\n(Default: %(default)s)`,
					type: 'str',
					action: 'store',
					default: this._paths.config,
				});
				this._ap_paths.add_argument('-ff', '--ffmpeg', {
					help: `The path to where the FFmpeg binaries are located, with relative paths being relative to the current working directory.\n(Default: ${this._paths.ffmpeg})`,
					type: 'str',
					action: 'store',
				});
				this._ap_paths.add_argument('-i', '--input', {
					help: `The path to the input videos, with relative paths being relative to the current working directory.\n(Default: ${this._paths.input})`,
					type: 'str',
					action: 'store'
				});
				this._ap_paths.add_argument('-c', '--cache', {
					help: `The path for the intermediate cache storage, with relative paths being relative to the current working directory.\nThis should ideally point at a very high speed storage, such as SSDs.\n(Default: ${this._paths.cache})`,
					type: 'str',
					action: 'store'
				});
				this._ap_paths.add_argument('-o', '--output', {
					help: `The path at which to store output files, with relative paths being relative to the current working directory.\nThis should ideally point a a very high speed storage, such as SSDs.\n(Default: ${this._paths.output})`,
					type: 'str',
					action: 'store'
				});
			}

			{// ArgParse: Colors
				this._ap_color = this._ap.add_argument_group('Colors');
				this._ap_color.add_argument('-cf', '--color-format', {
					help: `Override the detected color format used for transcoding.`,
					type: 'str',
					action: 'store',
				});
				this._ap_color.add_argument('-cm', '--color-matrix', {
					help: `Override the detected color matrix used for transcoding.`,
					type: 'str',
					action: 'store',
				});
				this._ap_color.add_argument('-ct', '--color-tc', '--color-transfer-characteristics', {
					help: `Override the detected color transfer characteristics used for transcoding.`,
					type: 'str',
					action: 'store',
				});
				this._ap_color.add_argument('-cp', '--color-primaries', {
					help: `Override the detected color primaries used for transcoding.`,
					type: 'str',
					action: 'store',
				});
				this._ap_color.add_argument('-cr', '--color-range', {
					help: `Override the detected color range used for transcoding.`,
					type: 'str',
					action: 'store',
				});
			}

			{// ArgParse: VMAF
				this._ap_vmaf = this._ap.add_argument_group('VMAF');
				this._ap_vmaf.add_argument('--vmaf-model', {
					help: "What model should be used for VMAF processing? Note that different models are not compatible with each other.\n(Default: %(default)s)",
					type: 'str',
					action: 'store',
					default: this._vmaf.model,
				});
				this._ap_vmaf.add_argument('--vmaf-threads', {
					help: "How many threads should be used by each VMAF instance?\n(Default: %(default)i)",
					type: 'int',
					action: 'store',
					default: this._vmaf.threads,
				});
			}

			{ // ArgParse: Tests
				this._ap_tests = this._ap.add_argument_group('Tests');
				this._ap_tests.add_argument('-enc', '--encoder', {
					help: "Enable an encoder by name.\n(Default: %(default)s)",
					type: 'str',
					action: 'append',
					default: ['x264', 'nvenc', 'amf', 'qsv']
				});
				this._ap_tests.add_argument('-cd', '--codec', {
					help: "Enable a codec by name.\n(Default: %(default)s)",
					type: 'str',
					action: 'append',
					default: ['h264']
				});
				this._ap_tests.add_argument('-res', '--resolution', {
					help: "Add or remove a resolution (WxH) to the tests.\nPrefix with - to remove an entry (-1920x1080 instead of 1920x1080).",
					type: 'str',
					action: 'append',
				});
				this._ap_tests.add_argument('-br', '--bitrate', {
					help: "Add or remove a bitrate to the tests.\nPrefix with - to remove an entry (-3000 instead of 3000).",
					type: 'int',
					action: 'append',
				});
				this._ap_tests.add_argument('-kf', '--kfinterval', {
					help: "Add or remove a key frame interval (in seconds) to list of tests.\nPrefix with - to remove an entry (-1.0 instead of 1.0).",
					type: 'float',
					action: 'append',
				});
			}

			// Parse arguments
			this._args = this._ap.parse_args();

			// Update log level
			this.logger.level = this._args.loglevel;
		}
	}

	// ---------------------------------------------------------------------------------------------- //
	// Copyright and License
	// ---------------------------------------------------------------------------------------------- //
	_copyright() {
		// Log copyright information.
		if (this._args.copyright) {
			this.logger.log(LOG.ELEVEL_INFO, "Copyright 2017-2021 Michael Fabian 'Xaymar' Dirks");
		}
	}

	_license() {
		// Log license.
		if (this._args.license) {
			for (let line of [
				"Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:",
				"1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.",
				"2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.",
				"3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.",
				"THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
			]) {
				this.logger.log(LOG.ELEVEL_INFO, line);
			}
		}
	}

	// ---------------------------------------------------------------------------------------------- //
	// Configuration Parsing
	// ---------------------------------------------------------------------------------------------- //
	_parse_resolution(str) {
		if (((typeof str) == "string") || (str instanceof String)) {
			let parts = str.split('x');
			if (parts.length != 2) {
				throw SyntaxError(`Resolution '${str}' must be a string with the format WxH.`);
			}
			return str;
		} else {
			throw TypeError(`Resolution must be a string or String, is ${typeof str}.`);
		}
	}

	_parse_bitrate(str) {
		if ((typeof str) == "number") {
			let ival = parseInt(str);
			if (ival != str) {
				throw SyntaxError(`Non-integer bitrate '${str}' is not supported.`);
			}
			return ival;
		} else {
			throw TypeError(`Bitrate must be a number, is ${typeof str}.`);
		}
	}

	_parse_kfinterval(str) {
		if ((typeof str) == "number") {
			return Math.round(parseFloat(str) * 1000) / 1000;
		} else {
			throw TypeError(`Interval must be a number, is ${typeof str}.`);
		}
	}

	_load_configuration() {
		// This takes care of general parsing of command line and configuration.

		// Try and load the on-disk configuration if specified and existing.
		let config = undefined;
		this._paths.config = PATH.resolve(this._args.config);
		if (FS.existsSync(this._paths.config)) {
			try {
				config = JSON.parse(FS.readFileSync(this._paths.config));
				this.logger.log(LOG.ELEVEL_VERBOSE, "Loaded configuration file '%s'.", this._paths.config);
			} catch (ex) {
				this.logger.log(LOG.ELEVEL_ERROR, "Configuration file '%s' is invalid: %s", this._paths.config, ex.stack);
			}
		} else {
			this.logger.log(LOG.ELEVEL_WARNING, "Configuration file '%s' is missing.", this._paths.config);
		}

		// Decipher the paths for common things.
		{
			for (let k of ["ffmpeg", "input", "cache", "output"]) {
				let path = "";
				if (config && config.paths && config.paths[k] && ((typeof config.paths[k]) == "string")) {
					let file = config.paths[k];
					if (file.substr(0, 1) == '.') {
						file = PATH.join(PATH.dirname(this._paths.config), file);
					}
					path = PATH.resolve(file);
				}
				if (this._args[k] && ((typeof this._args[k]) == "string")) {
					path = PATH.resolve(this._args[k]);
				}
				if (path.length > 0) {
					this._paths[k] = path;
				}
				this.logger.log(LOG.ELEVEL_DEBUG, "'%s' is located at: '%s'.", k, this._paths[k]);
			}
		}

		// Allow overriding the detected color format/matrix/trc/primaries/range
		for (let k of ["format", "matrix", "tc", "primaries", "range"]) {
			let value = "";
			if ((config) && (config.color) && (config.color[k]) && ((typeof config.color[k]) == "string")) {
				value = config.color[k];
			}
			if (this._args[`color_${k}`] && ((typeof this._args[`color_${k}`]) == "string")) {
				value = this._args[`color_${k}`];
			}
			if (value.length > 0) {
				this._color.set(k, value);
				this.logger.log(LOG.ELEVEL_INFO, "Overriding color %s with '%s' during transcoding.", k, value);
			}
		}

		{ // VMAF
			this._vmaf.model = this._args.vmaf_model;
			this._vmaf.threads = this._args.vmaf_threads;
			this.logger.log(LOG.ELEVEL_DEBUG, "VMAF model is '%s' with %d threads.", this._vmaf.model, this._vmaf.threads);
		}

		{ // Tests
			{ // Resolutions
				if (config && config.resolutions) {
					if (Array.isArray(config.resolutions)) {
						for (let v of config.resolutions) {
							try {
								this._resolutions.add(this._parse_resolution(v));
							} catch (ex) {
								console.log(`Skipping resolution '${v}' due to exception:`, ex, ex.stack);
							}
						}
					} else {
						console.log(`Configuration '${this._paths.config}' has invalid 'resolutions' entry, expected 'Array', is '${typeof config.resolutions}'.`);
					}
				}
				if (this._args.resolution && Array.isArray(this._args.resolution)) {
					for (let v of this._args.resolution) {
						try {
							if (v.substr(0, 1) == '-') {
								// Remove
								let v2 = this._parse_resolution(v.substr(1));
								this._resolutions.delete(v2);
							} else {
								// Insert
								let v2 = this._parse_resolution(v);
								this._resolutions.add(v2);
							}
						} catch (ex) {
							console.log(`Skipping resolution '${v}' due to exception:`, ex, ex.stack);
						}
					}
				}
			}

			{ // Bitrates
				if (config && config.bitrates) {
					if (Array.isArray(config.bitrates)) {
						for (let v of config.bitrates) {
							try {
								this._bitrates.add(this._parse_bitrate(v));
							} catch (ex) {
								console.log(`Skipping bitrate '${v}' due to exception:`, ex);
							}
						}
					} else {
						console.log(`Configuration '${this._paths.config}' has invalid 'bitrates' entry, expected 'Array', is '${typeof config.bitrates}'.`);
					}
				}
				if (this._args.bitrate && Array.isArray(this._args.bitrate)) {
					for (let v of this._args.bitrate) {
						try {
							if (v < 0) {
								// Remove
								let v2 = this._parse_bitrate(-v);
								this._bitrates.delete(v2);
							} else {
								// Insert
								let v2 = this._parse_bitrate(v);
								this._bitrates.add(v2);
							}
						} catch (ex) {
							console.log(`Skipping bitrate '${v}' due to exception:`, ex);
						}
					}
				}
			}

			{ // Intervals
				if (config && config.kfintervals) {
					if (Array.isArray(config.kfintervals)) {
						for (let res of config.kfintervals) {
							try {
								this._kfintervals.add(this._parse_kfinterval(res));
							} catch (ex) {
								console.log(`Skipping key frame interval '${res}' due to exception:`, ex);
							}
						}
					} else {
						console.log(`Configuration '${this._paths.config}' has invalid 'kfintervals' entry, expected 'Array', is '${typeof config.bitrates}'.`);
					}
				}
				if (this._args.kfinterval && Array.isArray(this._args.kfinterval)) {
					for (let v of this._args.kfinterval) {
						try {
							if (v < 0) {
								// Remove
								let v2 = this._parse_kfinterval(-v);
								this._kfintervals.delete(v2);
							} else {
								// Insert
								let v2 = this._parse_kfinterval(v);
								this._kfintervals.add(v2);
							}
						} catch (ex) {
							console.log(`Skipping key frame interval '${v}' due to exception:`, ex);
						}
					}
				}
			}

			{ // Encoders
				for (let name of this._args.encoder) {
					try {
						if ((typeof name) != "string") {
							throw TypeError(`Encoder must be a 'string', is '${typeof name}'.`);
						}

						let target = true;
						if (name.substr(0, 1) == '-') {
							target = false;
							name = name.substr(1);
						}

						if (this._encoders.has(name)) {
							this._encoders.get(name).enabled = target;
						} else {
							this.logger.log(LOG.ELEVEL_ERROR, "Specified encoder '%s' does not exist.", name);
						}
					} catch (ex) {
						console.log(`Unexpected error parsing encoder '${name}':`, ex);
					}
				}
			}

			{ // Codecs
				for (let v of this._args.codec) {
					try {
						if ((typeof v) != "string") {
							throw TypeError(`Codec must be a 'string', is '${typeof v}'.`);
						}
						if (v.substr(0, 1) == '-') {
							// Remove
							this._codecs.delete(v.substr(1));
						} else {
							// Insert
							this._codecs.add(v);
						}
					} catch (ex) {
						console.log(`Unexpected error parsing codec '${kv}':`, ex);
					}
				}
			}
		}
	}

	// ---------------------------------------------------------------------------------------------- //
	// Logic
	// ---------------------------------------------------------------------------------------------- //
	loop() {
		{ // Initialization
			// Encoder arguments
			for (let kv of this._encoders.entries()) {
				try {
					kv[1].parse(this._libs.argparse, this._ap);
				} catch (ex) {
					this.logger.log(LOG.ELEVEL_DEBUG, "Encoder '%s' failed argument parsing with exception: %s", kv[0], ex.toString());
				}
			}

			// Parse arguments
			this._args = this._ap.parse_args();

			// Copyright
			this._copyright();

			// License
			this._license();

			// Load configuration.
			this._load_configuration();

			// Load FFmpeg.
			try {
				this._ffmpeg = new this._libs.ffmpeg(this.paths.ffmpeg);
			} catch (ex) {
				console.error(ex, ex.stack);
				return E_FAIL;
			}
		}

		{ // Work
			// Load enabled encoders.
			for (let kv of this._encoders.entries()) {
				try {
					if (kv[1].enabled) {
						kv[1].load();
					}
				} catch (ex) {
					this.logger.log(LOG.ELEVEL_ERROR, "Failed to load encoder '%s': %s", kv[0], ex.toString());
					kv[1].enabled = false;
				}
			}

			// Create any missing paths.
			for (let v of ['input', 'cache', 'output']) {
				if (!FS.existsSync(v)) {
					FS.mkdirSync(v, { recursive: true });
				}
			}



			// TODO: What's next?
			// - Encode/Compare
			//   1. Iterate over each input video
			//   2. Generate the cache if it is missing for the current video, resolution and framerate.
			//   3. Launch N encodes (up to the maximum for each pool, limit by vmaf)
			//     - VMAF shares CPU pool, not GPU pool.

		}

		return E_OK;
	}

	// ---------------------------------------------------------------------------------------------- //
	// Accessors
	// ---------------------------------------------------------------------------------------------- //
	get paths() {
		return this._paths;
	}

	get argparse() {
		return this._ap;
	}

	get args() {
		return this._args;
	}

	get ffmpeg() {
		return this._ffmpeg;
	}

	get codecs() {
		return this._codecs;
	}
};

{ // Main
	let main = new MyApp();
	main.run();
}

/*


// Import modules
const ffmpeg = require('./ffmpeg.js');
const poolqueue = require('./poolqueue.js');

const fs = require('fs');
const path = require('path');
const process = require('process');
const util = require('util');
const { config } = require('process');

// Define some Helpers
function float_eq(a, b, edge) { return (Math.abs(a - b) <= edge); }
function float_lt(a, b, edge) { return ((a + edge) < b); }
function float_le(a, b, edge) { return float_lt(a, b, edge) || float_eq(a, b, edge); }
function float_gt(a, b, edge) { return ((a - edge) > b); }
function float_ge(a, b, edge) { return float_gt(a, b, edge) || float_eq(a, b, edge); }
Object.size = function (obj) { var size = 0, key; for (key in obj) { if (obj.hasOwnProperty(key)) { size++; } } return size; };

// Actual Code
async function load_config() { // Load Configuration
}

async function load_encoders(config, ff) { // Load Encoders
	console.group("Loading encoders...");
	console.time("Total");

	let encoders = new Map();
	for (let name in config.encoders) {
		let encoder = config.encoders[name];

		// Is the encoder disabled, or invalid?
		if (!encoder || !encoder.enabled) {
			if (global.debug) console.debug(`${name} is disabled or invalid.`);
			continue;
		}

		// Does the encoder exist?
		if (!fs.existsSync(path.join(".", `encoder_${name}.js`))) {
			console.error(`${name} does not exist.`);
			continue;
		}

		// Attempt to load the encoder, if possible.
		console.group(name);
		console.time("Subtotal");
		try {
			let instance = new (require(`./encoder_${name}.js`))(ff, config, encoder);
			instance.load();
			encoders.set(name, instance);
		} catch (ex) {
			console.log(ex);
		}
		console.timeEnd("Subtotal");
		console.groupEnd();
	}

	console.timeEnd("Total");
	console.groupEnd();
	return encoders;
}

async function load_videos(config, ff) { // Load Videos
	console.group("Loading videos...");
	console.time("Total");

	let videos = new Map();
	let promises = []; // Asynchronous loading.
	for (let name in config.videos) {
		let video = config.videos[name];

		// Is the video disabled, or invalid?
		if (!video || !video.enabled) {
			if (global.debug) console.debug(`${name} is disabled or invalid.`);
			continue;
		}

		// Does the video exist?
		if (!fs.existsSync(path.join(config.paths.videos, `${name}.mkv`))) {
			console.error(`${name} does not exist.`);
			continue;
		}

		// Load video information asynchronously.
		promises.push(new Promise((resolve, reject) => {
			console.time(name);

			let file_name = path.join(config.paths.videos, `${name}.mkv`);
			let probeprom = ff.probe(file_name);
			probeprom.then((json) => {
				let data = new Object();

				data.name = name;
				data.info = json;
				data.caches = {};

				// File Information
				data.file_name = `${name}.mkv`;
				data.file_ext = path.extname(data.file_name);
				data.file_base = path.basename(data.file_name, data.file_ext);
				data.file = path.join(config.paths.videos, data.file_name);

				// Video Information
				data.resolution = { width: data.info.streams[0].width, height: data.info.streams[0].height };
				data.framerate = eval(data.info.streams[0].r_frame_rate);
				data.duration = data.info.streams[0].duration;

				// Color Information
				data.color = {};
				data.color.range = data.info.streams[0].color_range ? data.info.streams[0].color_range : 'tv';
				data.color.trc = data.info.streams[0].color_transfer ? data.info.streams[0].color_transfer : 'bt709';
				data.color.primaries = data.info.streams[0].color_primaries ? data.info.streams[0].color_primaries : 'bt709';
				data.color.matrix = data.info.streams[0].color_space ? data.info.streams[0].color_space : 'bt709';

				videos.set(data.name, data);
				console.timeEnd(name);
				resolve(data);
			});
		}));
	}

	await Promise.allSettled(promises);

	// Sort the videos list again. (We don't care about what order the user wants!)
	videos = new Map([...videos].sort((a, b) => a[0] > b[0] ? 1 : -1));

	console.timeEnd("Total");
	console.groupEnd();
	return videos;
}

async function create_caches(config, ff, videos, encoders) { // Create Caches
	console.group("Creating caches...");
	console.time("Total");

	for (let name of videos.keys()) {
		let video = videos.get(name);

		console.group(name);
		console.time("Subtotal");

		video.caches = new Map();
		for (let resolution of config.options.resolutions) {
			let aspect_ratio = (video.resolution.height / video.resolution.width);
			for (let framerate_scaling of config.options.framerate_scalings) {
				let height = (Math.round(resolution[0] * aspect_ratio * 0.5) * 2);
				// Skip resolutions that are too large horizontally.
				if (resolution[0] > video.resolution.width) {
					if (global.debug) console.debug(`${resolution[0]}x${height}x${(video.framerate * framerate_scaling)} skipped.`);
					continue;
				}

				// Create Cache Information
				let cache = {
					framerate: (video.framerate * framerate_scaling),
					width: resolution[0],
					height: height
				};
				cache.framerate_s = cache.framerate.toFixed(2);
				cache.duration = Math.floor(video.duration * video.framerate * framerate_scaling) / cache.framerate;
				let key = `${cache.width}x${cache.height}x${cache.framerate_s}`;
				cache.file = path.join(config.paths.cache, `${video.file_base}-${key}.mkv`);
				cache.queues = new Map();

				video.caches.set(`${key}`, cache);
			}
		}

		for (let key of video.caches.keys()) {
			let cache = video.caches.get(key);

			// Check if cache file already exists and is correct.
			if (fs.existsSync(cache.file)) {
				let info = ff.probeSync(cache.file);
				if ((info.streams)
					&& (info.streams.length > 0)
					&& (info.streams[0].width == cache.width)
					&& (info.streams[0].height == cache.height)
					&& float_eq(eval(info.streams[0].r_frame_rate), cache.framerate, 0.01)
					&& float_eq(info.streams[0].duration, video.duration, 0.1)) {
					if (global.debug) console.debug(`${key} already exists.`);
					continue;
				}
			}

			console.time(`${key} created`);
			let command = [
				"-y",
				"-hide_banner",
				"-v", "error",
				"-i", video.file,
				"-filter_complex", `fps=fps=${cache.framerate_s},scale=flags=bicubic+full_chroma_inp+full_chroma_int:w=${cache.width}:h=${cache.height},colorspace=space=${config.options.transcode.color.matrix}:trc=${config.options.transcode.color.trc}:primaries=${config.options.transcode.color.primaries}:range=${config.options.transcode.color.range},format=pix_fmts=${config.options.transcode.format}`,
				"-an",
			];
			if (encoders.has("h264_nvenc")) {
				command.push(
					"-c:v", "h264_nvenc",
					"-profile:v", "high",
					"-preset", "p1",
					"-tune", "lossless",
					"-rc", "constqp",
					"-rc-lookahead", "0",
					"-multipass", "0",
					"-b:v", "0",
					"-minrate", "0",
					"-maxrate", "0",
					"-bufsize", "0",
					"-qp", "0",
					"-init_qpI", "0",
					"-init_qpP", "0",
					"-init_qpB", "0",
					"-bf", "0",
					"-g", `15`,
				);
			} else {
				command.push(
					"-c:v", "libx264",
					"-preset", "veryfast",
					"-crf", "0",
					"-b:v", "0",
					"-minrate", "0",
					"-maxrate", "0",
					"-bufsize", "0",
					"-g", `15`,
				);
			}
			command.push(cache.file);
			let res = ff.ffmpegSync(command);
			if (res.status != 0) {
				console.log(res.stderr.toString());
			}
			if (global.debug) console.log(res.stdout.toString(), res.stderr.toString());
			console.timeEnd(`${key} created`);
		}

		console.timeEnd("Subtotal");
		console.groupEnd(name);
	}

	console.timeEnd("Total");
	console.groupEnd();
	return videos;
}

async function queue(config, ff, videos, encoders) {
	console.group("Queueing...");
	console.time("Total");

	let promises = [];
	for (let video_key of videos.keys()) {
		promises.push(new Promise(async (resolve, reject) => {
			let video = videos.get(video_key);
			let video_promises = [];

			console.time(video_key);
			for (let cache_key of video.caches.keys()) {
				video_promises.push(new Promise(async (resolve1) => {
					let cache = video.caches.get(cache_key);

					let queue_commands = new poolqueue();
					let queue_files = new poolqueue();
					let queue_promises = new Array();

					for (let encoder_key of encoders.keys()) {
						let encoder = encoders.get(encoder_key);
						let encoder_pool = encoder.pool();
						let encoder_extra = encoder.extra();
						for (let idx = 0; idx < encoder.count(); idx++) {
							let command = encoder.get(idx, cache.width, cache.height, cache.framerate);
							for (let bitrate of config.options.bitrates) {
								for (let kfinterval of config.options.keyframeinterval) {
									queue_promises.push(new Promise((resolve2) => {
										let file = path.join(
											config.paths.output,
											video.name,
											cache_key,
											encoder_key,
											bitrate.toFixed(0),
											(cache.framerate * kfinterval).toFixed(0),
											`${command.hash}.mkv`
										);
										let file_json = path.join(
											config.paths.output,
											video.name,
											cache_key,
											encoder_key,
											bitrate.toFixed(0),
											(cache.framerate * kfinterval).toFixed(0),
											`${command.hash}.json`
										);

										// Check if this file already exists at the target location.
										if (fs.existsSync(file_json)) {
											// Don't need to check for a video here, since we only care about results.
											if (global.debug) console.debug(`${file} already completed.`);

											resolve2(false);
											return;
										}

										let line = [
											"-map", "0:v:0",
											"-an",
											"-c:v", encoder_key,
											"-g", (cache.framerate * kfinterval).toFixed(0),
											"-b:v", `${bitrate}k`,
											"-minrate", "0",
											"-maxrate", "0",
											"-bufsize", `${2 * bitrate}k`,
										].concat(command.options).concat(encoder_extra).concat([file]);

										queue_commands.push(encoder_pool, line, command.cost);
										queue_files.push(encoder_pool, [[file, file_json]], command.cost);

										resolve2(true);
									}));
								}
							}
						}
					}

					await Promise.allSettled(queue_promises);

					let data = {
						commands: queue_commands.finalize(),
						files: queue_files.finalize(),
					};
					cache.queues = data;

					resolve1(true);
				}));
			}

			await Promise.allSettled(video_promises);
			console.timeEnd(video_key);

			resolve();
		}));
	}
	await Promise.allSettled(promises);
	console.timeEnd("Total");
	console.groupEnd();
}

async function work(config, ff, videos, encoders) {
	// Process from here on out.
	// LOOP
	// 1. Pull out front of the command and file queue.
	// 2. Encode using the given command(s).
	// 3. Compare resulting files with real input (libvmaf).
	// 4. Delete encoded files.
	// 5. Repeat until queues empty, no more caches for video, and no more videos.

	let vmaf_model = ff.consolify(path.resolve(path.join(config.paths.ffmpeg, "vmaf", config.options.vmaf.model)));

	console.group("Processing...")
	console.time("Total");
	for (let video_key of videos.keys()) {
		console.group(video_key);
		console.time(video_key);
		let video = videos.get(video_key);
		for (let cache_key of video.caches.keys()) {
			console.group(cache_key);
			console.time(cache_key);
			let cache = video.caches.get(cache_key);

			let length = cache.queues.commands.length;
			console.log(`0.0% (0 / ${length}): 0.000s`);
			while (cache.queues.commands.length > 0) {
				let commands = cache.queues.commands.shift();
				let files = cache.queues.files.shift();

				let prc1 = length - cache.queues.commands.length;
				let prc2 = prc1 / length * 100.0;
				let LABEL = `${prc2.toFixed(1)}% (${prc1} / ${length})`
				console.time(LABEL)
				console.group();


				for (let file of files) { // Create directories.
					fs.mkdirSync(path.dirname(file[0]), { recursive: true });
				}


				{ // Encode
					console.time("Encoding");
					let opts = [
						"-y",
						"-hide_banner",
						"-v", "error",
						"-hwaccel", "auto",
						"-i", cache.file
					].concat(commands);
					let res = ff.ffmpegSync(opts);
					if (res.status != 0) {
						console.log(res.stdout.toString(), res.stderr.toString());
						continue;
					}
					console.timeEnd("Encoding");
				}

				{ // Process
					console.time("Processing");
					let opts = [
						"-hide_banner",
						"-v", "warning",
						"-hwaccel", "auto",
						"-i", video.file,
					];
					let filter = "";
					let references = [];

					// Build Filter Graph
					if (files.length > 1) {
						filter = `[0:v:0]split=${files.length}`
						for (let idx = 0; idx < files.length; idx++) {
							filter = `${filter}[ref:${idx}]`;
							references[idx] = `[ref:${idx}]`;
						}
						filter = `${filter}`
					} else {
						references[0] = "[0:v:0]";
					}

					// Rescale and Resample all inputs and compare them.
					for (let idx = 0; idx < files.length; idx++) {
						let file = files[idx];
						opts.push("-i", file[0]);

						if (filter != "")
							filter = `${filter};` // [temp:${idx}];[temp:${idx}]
						filter += `[${idx+1}:v:0]scale=flags=bicubic+full_chroma_inp+full_chroma_int:w=${video.resolution.width.toFixed(0)}:h=${video.resolution.height.toFixed(0)}`
							+ `,colorspace=ispace=${config.options.transcode.color.matrix}:itrc=${config.options.transcode.color.trc}:iprimaries=${config.options.transcode.color.primaries}:irange=${config.options.transcode.color.range}:space=${video.color.matrix}:trc=${video.color.trc}:primaries=${video.color.primaries}:range=${video.color.range}`
							+ `,format=pix_fmts=yuv444p,fps=fps=${video.framerate.toFixed(2)}[scaled:${idx}]`
							+ `;[scaled:${idx}][${references[idx]}libvmaf=model_path=${vmaf_model}:log_fmt=json:log_path=${ff.consolify(file[1])}:enable_conf_interval=1:psnr=1:ssim=1:n_threads=${(config.options.vmaf.threads / files.length).toFixed(0)}[main:${idx}]`
					}

					opts.push("-filter_complex", filter);
					for (let idx = 0; idx < files.length; idx++) {
						opts.push(
							"-map", `[main:${idx}]`,
							"-f", "null",
							"-"
						)
					}

					let res = ff.ffmpegSync(opts);
					if (res.status != 0) {
						console.log(res.stdout.toString(), res.stderr.toString());
						continue;
					}
					console.timeEnd("Processing");
				}

				for (let file of files) { // Clean up
					fs.unlinkSync(file[0]);
				}

				console.groupEnd();
				console.timeEnd(LABEL);
			}
			console.groupEnd(cache_key);
			console.timeEnd(cache_key);
		}
		console.timeEnd(video_key);
		console.groupEnd();
	}
	console.timeEnd("Total");
	console.groupEnd();
}

async function main() {
	let config;
	let ff;
	let encoders;
	let videos;

	await load_config().then((p) => { config = p; });
	ff = new ffmpeg(config.paths.ffmpeg);
	await load_encoders(config, ff).then((p) => { encoders = p; });
	await load_videos(config, ff).then((p) => { videos = p; });
	await create_caches(config, ff, videos, encoders);
	await queue(config, ff, videos, encoders);
	await work(config, ff, videos, encoders);
}

main();
*/
