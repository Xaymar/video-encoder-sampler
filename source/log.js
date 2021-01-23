const ELEVEL_CRITICAL = 0;
const ELEVEL_ERROR = -8;
const ELEVEL_WARNING = -16;
const ELEVEL_INFO = -24;
const ELEVEL_VERBOSE = -32;
const ELEVEL_DEBUG = -48;

class Logger {
	constructor() {
	}

	log(level, message, ...extra) {
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
