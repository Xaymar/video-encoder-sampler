const process = require('process');
const log = require('./log.js');

class app {
	constructor() {
		this._log = new log.Logger();
	}

	get logger() {
		return this._log;
	}

	run() {
		try {
			process.exit(this.loop());
		} catch(ex) {
			console.log(ex);
			process.exit(1);
		}
	}
}

module.exports = app;
