const ENCODER = require('./encoder.js');
const LOG = require('./log.js');
const CRYPTO = require('crypto');
const PATH = require('path');
const FS = require('fs');

let version = [1, 0, 0, 0];

class x264Encoder extends ENCODER {
	constructor(parent) {
		super(parent, "x264");

		this._args = new Map();

		{ // Presets
			let values = new Set();
			values.add("ultrafast");
			values.add("superfast");
			values.add("veryfast");
			values.add("faster");
			values.add("fast");
			values.add("medium");
			values.add("slow");
			values.add("slower");
			values.add("placebo");
			this._args.set('preset', values);
		}

		{ // Tunes
			let values = new Set();
			values.add(null);
			values.add("film");
			values.add("animation");
			values.add("grain");
			values.add("psnr");
			values.add("ssim");
			values.add("touhou"); // Undocumented tune option,
			this._args.set('tune', values);
		}

		{ // Scenecut
			let values = new Set();
			values.add(null);
			values.add(0);
			this._args.set('scenecut', values);
		}

		this._index = new Map();
	}

	// ---------------------------------------------------------------------------------------------- //
	// Accessors
	// ---------------------------------------------------------------------------------------------- //

	// ---------------------------------------------------------------------------------------------- //
	// Functions
	// ---------------------------------------------------------------------------------------------- //

	/** Parse arguments with ArgParse
	 *
	 * @param ArgParse The ArgParse library itself.
	 * @param Parser The Argument Parser instance.
	 */
	parse(ArgParse, Parser) {

	}

	/** Load the encoder and any configurations.
	 *
	 */
	load() {
		{ // Check if the encoder is actually available on this system.
			let res = this.ffmpeg.ffmpegSync(
				"-hide_banner", "-v", "error",
				"-f", "lavfi",
				"-i", "color=size=256x256:duration=1:rate=30:color=black",
				"-c:v", "libx264",
				"-f", "null",
				"-"
			);

			if (res.status != 0) {
				this.enabled = false;
				this.logger.log(LOG.ELEVEL_DEBUG, res.stdout.toString());
				this.logger.log(LOG.ELEVEL_DEBUG, res.stderr.toString());
				throw new Error("x264 (via FFmpeg) is not available.");
			}
		}

		{ // Build an index of all option combinations.
			for (let preset of this._args.get("preset")) {
				for (let tune of this._args.get("tune")) {
					for (let scenecut of this._args.get("scenecut")) {
						let opts = [
							"-profile:v", "high",
							"-x264-params", "nal-hrd=cbr:force-cfr=1",
							"-qmin", "0", "-qmax", "1024",
							"-psnr", "0", "-ssim", "0",
							"-preset", preset,
						];
						if (tune !== null) {
							opts.push("-tune", tune);
						};
						if (scenecut !== null) {
							opts.push("-sc_threshold", scenecut.toFixed(0));
						};

						let hash = CRYPTO.createHash("sha256").update(JSON.stringify(opts)).digest("hex");
						hash = `preset=${preset}_tune=${tune}_scenecut=${scenecut}_${hash}`;
						this._index.set(hash, opts);
					}
				}
			}

			// Store on disk for later identification.
			let data = {};
			for (let kv of this._index.entries()) {
				data[kv[0]] = kv[1];
			}
			FS.writeFileSync(
				PATH.join(this.parent.paths.output, 'x264.json'),
				JSON.stringify({
					version: version,
					data: data,
				}, null, '\t'),
				{ encoding: "utf8" }
			);
		}
	}

}

module.exports = x264Encoder;
