"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MemoryServer = void 0;
/*********************************************************************
 * Copyright (c) 2019 QNX Software Systems and others
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
const vscode = require("vscode");
const path = require("path");
class MemoryServer {
    constructor(context) {
        context.subscriptions.push(vscode.commands.registerCommand('cdt.gdb.memory.open', () => this.openPanel(context)));
    }
    openPanel(context) {
        const newPanel = vscode.window.createWebviewPanel('cdtMemoryBrowser', 'Memory Browser', vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.file(path.resolve(context.extensionPath, 'out')),
            ],
            retainContextWhenHidden: true,
        });
        context.subscriptions.push(newPanel);
        newPanel.webview.onDidReceiveMessage((message) => {
            this.onDidReceiveMessage(message, newPanel);
        });
        newPanel.webview.html = `
             <html>
                 <head>
                     <meta charset="utf-8">
                     <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                 </head>
                 <body>
                     <div id="app"></div>
                     ${this.loadScript(context, 'out/packages.js', newPanel)}
                     ${this.loadScript(context, 'out/MemoryBrowser.js', newPanel)}
                 </body>
             </html>
         `;
    }
    loadScript(context, path, panel) {
        const uri = vscode.Uri.file(context.asAbsolutePath(path));
        const fp = panel.webview.asWebviewUri(uri);
        return `<script src="${fp.toString()}"></script>`;
    }
    onDidReceiveMessage(request, panel) {
        switch (request.command) {
            case 'ReadMemory':
                this.handleReadMemory(request, panel);
                break;
            case 'getChildDapNames':
                this.handleGetChildDapNames(request, panel);
                break;
        }
    }
    sendResponse(panel, request, response) {
        if (panel) {
            response.token = request.token;
            response.command = request.command;
            panel.webview.postMessage(response);
        }
    }
    handleReadMemory(request, panel) {
        return __awaiter(this, void 0, void 0, function* () {
            const session = vscode.debug.activeDebugSession;
            if (session) {
                try {
                    const result = yield session.customRequest(request.args.child === undefined
                        ? 'cdt-gdb-adapter/Memory'
                        : 'cdt-amalgamator/Memory', request.args);
                    this.sendResponse(panel, request, {
                        result,
                    });
                }
                catch (err) {
                    this.sendResponse(panel, request, {
                        err: err + '',
                    });
                }
            }
            else {
                this.sendResponse(panel, request, {
                    err: 'No Debug Session',
                });
            }
        });
    }
    handleGetChildDapNames(request, panel) {
        return __awaiter(this, void 0, void 0, function* () {
            const session = vscode.debug.activeDebugSession;
            if (session) {
                try {
                    const result = yield session.customRequest('cdt-amalgamator/getChildDapNames');
                    this.sendResponse(panel, request, {
                        result,
                    });
                }
                catch (err) {
                    this.sendResponse(panel, request, {
                        err: err + '',
                    });
                }
            }
            else {
                this.sendResponse(panel, request, {
                    err: 'No Debug Session',
                });
            }
        });
    }
}
exports.MemoryServer = MemoryServer;
//# sourceMappingURL=MemoryServer.js.map