"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = exports.SuspendAllSession = exports.ResumeAllSession = exports.MemoryServer = void 0;
/*********************************************************************
 * Copyright (c) 2018 QNX Software Systems and others
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
const vscode_1 = require("vscode");
const MemoryServer_1 = require("./memory/server/MemoryServer");
var MemoryServer_2 = require("./memory/server/MemoryServer");
Object.defineProperty(exports, "MemoryServer", { enumerable: true, get: function () { return MemoryServer_2.MemoryServer; } });
const ResumeAllSession_1 = require("./ResumeAllSession");
var ResumeAllSession_2 = require("./ResumeAllSession");
Object.defineProperty(exports, "ResumeAllSession", { enumerable: true, get: function () { return ResumeAllSession_2.ResumeAllSession; } });
const SuspendAllSession_1 = require("./SuspendAllSession");
var SuspendAllSession_2 = require("./SuspendAllSession");
Object.defineProperty(exports, "SuspendAllSession", { enumerable: true, get: function () { return SuspendAllSession_2.SuspendAllSession; } });
function activate(context) {
    new MemoryServer_1.MemoryServer(context);
    new ResumeAllSession_1.ResumeAllSession(context);
    new SuspendAllSession_1.SuspendAllSession(context);
    context.subscriptions.push(vscode_1.commands.registerCommand('cdt.debug.askProgramPath', (_config) => {
        return vscode_1.window.showInputBox({
            placeHolder: 'Please enter the path to the program',
        });
    }));
    context.subscriptions.push(vscode_1.commands.registerCommand('cdt.debug.askProcessId', (_config) => {
        return vscode_1.window.showInputBox({
            placeHolder: 'Please enter ID of process to attach to',
        });
    }));
}
exports.activate = activate;
function deactivate() {
    // empty, nothing to do on deactivating extension
}
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map