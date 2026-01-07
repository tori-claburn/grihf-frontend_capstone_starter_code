"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = void 0;
const child_process_1 = require("child_process");
const execa_1 = __importDefault(require("execa"));
const net = __importStar(require("net"));
const path = __importStar(require("path"));
const semver = __importStar(require("semver"));
const url = __importStar(require("url"));
const vscode = __importStar(require("vscode"));
const node_1 = require("vscode-languageclient/node");
const composerJson = require('../composer.json');
function activate(context) {
    return __awaiter(this, void 0, void 0, function* () {
        const conf = vscode.workspace.getConfiguration('php');
        const executablePath = conf.get('executablePath') ||
            conf.get('validate.executablePath') ||
            (process.platform === 'win32' ? 'php.exe' : 'php');
        const memoryLimit = conf.get('memoryLimit') || '4095M';
        if (memoryLimit !== '-1' && !/^\d+[KMG]?$/.exec(memoryLimit)) {
            const selected = yield vscode.window.showErrorMessage('The memory limit you\'d provided is not numeric, nor "-1" nor valid php shorthand notation!', 'Open settings');
            if (selected === 'Open settings') {
                yield vscode.commands.executeCommand('workbench.action.openGlobalSettings');
            }
            return;
        }
        // Check path (if PHP is available and version is ^7.0.0)
        let stdout;
        try {
            stdout = (yield (0, execa_1.default)(executablePath, ['--version'], { preferLocal: false })).stdout;
        }
        catch (err) {
            if (err.code === 'ENOENT') {
                const selected = yield vscode.window.showErrorMessage(`PHP executable not found. Install PHP ${composerJson.config.platform.php} or higher and add it to your PATH or set the php.executablePath setting`, 'Open settings');
                if (selected === 'Open settings') {
                    yield vscode.commands.executeCommand('workbench.action.openGlobalSettings');
                }
            }
            else {
                vscode.window.showErrorMessage('Error spawning PHP: ' + err.message);
                console.error(err);
            }
            return;
        }
        // Parse version and discard OS info like 7.0.8--0ubuntu0.16.04.2
        const match = stdout.match(/^PHP ([^\s]+)/m);
        if (!match) {
            vscode.window.showErrorMessage('Error parsing PHP version. Please check the output of php --version');
            return;
        }
        let version = match[1].split('-')[0];
        // Convert PHP prerelease format like 7.0.0rc1 to 7.0.0-rc1
        if (!/^\d+.\d+.\d+$/.test(version)) {
            version = version.replace(/(\d+.\d+.\d+)/, '$1-');
        }
        if (semver.lt(version, composerJson.config.platform.php)) {
            vscode.window.showErrorMessage(`The language server needs at least PHP ${composerJson.config.platform.php} installed. Version found: ${version}`);
            return;
        }
        let client;
        const serverOptions = () => new Promise((resolve, reject) => {
            // Use a TCP socket because of problems with blocking STDIO
            const server = net.createServer((socket) => {
                // 'connection' listener
                console.log('PHP process connected');
                socket.on('end', () => {
                    console.log('PHP process disconnected');
                });
                server.close();
                resolve({ reader: socket, writer: socket });
            });
            // Listen on random port
            server.listen(0, '127.0.0.1', () => {
                // The server is implemented in PHP
                const childProcess = (0, child_process_1.spawn)(executablePath, [
                    context.asAbsolutePath(path.join('vendor', 'felixfbecker', 'language-server', 'bin', 'php-language-server.php')),
                    '--tcp=127.0.0.1:' + server.address().port,
                    '--memory-limit=' + memoryLimit,
                ]);
                childProcess.stderr.on('data', (chunk) => {
                    const str = chunk.toString();
                    console.log('PHP Language Server:', str);
                    client.outputChannel.appendLine(str);
                });
                // childProcess.stdout.on('data', (chunk: Buffer) => {
                //     console.log('PHP Language Server:', chunk + '');
                // });
                childProcess.on('exit', (code, signal) => {
                    client.outputChannel.appendLine(`Language server exited ` + (signal ? `from signal ${signal}` : `with exit code ${code}`));
                    if (code !== 0) {
                        client.outputChannel.show();
                    }
                });
                return childProcess;
            });
        });
        // Options to control the language client
        const clientOptions = {
            // Register the server for php documents
            documentSelector: [
                { scheme: 'file', language: 'php' },
                { scheme: 'untitled', language: 'php' },
            ],
            revealOutputChannelOn: node_1.RevealOutputChannelOn.Never,
            uriConverters: {
                // VS Code by default %-encodes even the colon after the drive letter
                // NodeJS handles it much better
                code2Protocol: (uri) => url.format(url.parse(uri.toString(true))),
                protocol2Code: (str) => vscode.Uri.parse(str),
            },
            synchronize: {
                // Synchronize the setting section 'php' to the server
                configurationSection: 'php',
                // Notify the server about changes to PHP files in the workspace
                fileEvents: vscode.workspace.createFileSystemWatcher('**/*.php'),
            },
        };
        // Create the language client and start the client.
        client = new node_1.LanguageClient('php-intellisense', 'PHP Language Server', serverOptions, clientOptions);
        const disposable = client.start();
        // Push the disposable to the context's subscriptions so that the
        // client can be deactivated on extension deactivation
        context.subscriptions.push(disposable);
    });
}
exports.activate = activate;
//# sourceMappingURL=extension.js.map