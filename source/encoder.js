class Encoder extends EventTarget {
	constructor(parent, identifier) {
		super();

		this._parent = parent;
		this._id = identifier;
		this._enabled = false;
		this._config = {};
	}

	// ---------------------------------------------------------------------------------------------- //
	// Accessors
	// ---------------------------------------------------------------------------------------------- //
	get parent() {
		return this._parent;
	}

	get ffmpeg() {
		return this.parent.ffmpeg;
	}

	get logger() {
		return this.parent.logger;
	}

	get enabled() {
		return this._enabled;
	}
	set enabled(v) {
		this._enabled = v ? true : false;
	}

	get config() {
		return this._config;
	}
	set config(v) {
		this._config = v;
	}

	// ---------------------------------------------------------------------------------------------- //
	// Functions
	// ---------------------------------------------------------------------------------------------- //

	/** Parse arguments with ArgParse
	 *
	 * @param ArgParse The ArgParse library itself.
	 * @param Parser The Argument Parser instance.
	 */
	parse(ArgParse, Parser) {
		throw new Error("Not yet implemented.");
	}

	/** Load the encoder and any configurations.
	 *
	 */
	load() {
		throw new Error("Not yet implemented.");
	}



}

module.exports = Encoder;
