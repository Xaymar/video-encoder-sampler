const ELEVEL_CRITICAL = 0;
const ELEVEL_ERROR = -8;
const ELEVEL_WARNING = -16;
const ELEVEL_INFO = -24;
const ELEVEL_VERBOSE = -32;
const ELEVEL_DEBUG = -48;

class Logger {
	constructor() {
		this._level = ELEVEL_DEBUG;
	}

	set level(level) {
		if (typeof (level) == "number") {
			this._level = parseInt(level, 10);
		} else if (typeof (level) == "string") {
			switch (level.toLowerCase()) {
				case "debug":
					this._level = ELEVEL_DEBUG;
					break;
				case "verbose":
					this._level = ELEVEL_VERBOSE;
					break;
				case "info":
					this._level = ELEVEL_INFO;
					break;
				case "warning":
					this._level = ELEVEL_WARNING;
					break;
				case "error":
					this._level = ELEVEL_ERROR;
					break;
				case "critical":
					this._level = ELEVEL_CRITICAL;
					break;
			}
		}
	}

	get level() {
		return this._level;
	}

	log(level, message, ...extra) {
		if (level < this._level) {
			return;
		}

		let time = (new Date()).toISOString();
		switch (level) {
			case ELEVEL_CRITICAL:
				level = "CRITICAL";
				break;
			case ELEVEL_ERROR:
				level = "ERROR";
				break;
			case ELEVEL_WARNING:
				level = "WARNING";
				break;
			case ELEVEL_INFO:
				level = "INFO";
				break;
			case ELEVEL_VERBOSE:
				level = "VERBOSE";
				break;
			case ELEVEL_DEBUG:
			default:
				level = "DEBUG";
				break;
		}

		let str = `${time} [${level}] ${message}`;
		if (level >= ELEVEL_ERROR) {
			console.error(str, ...extra);
		}
		console.log(str, ...extra);
	}
}

module.exports = exports = {
	Logger,
	ELEVEL_CRITICAL: ELEVEL_CRITICAL,
	ELEVEL_ERROR: ELEVEL_ERROR,
	ELEVEL_WARNING: ELEVEL_WARNING,
	ELEVEL_INFO: ELEVEL_INFO,
	ELEVEL_VERBOSE: ELEVEL_VERBOSE,
	ELEVEL_DEBUG: ELEVEL_DEBUG,
};
