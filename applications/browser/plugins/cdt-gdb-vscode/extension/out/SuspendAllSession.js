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
exports.SuspendAllSession = void 0;
const vscode = require("vscode");
class SuspendAllSession {
    constructor(context) {
        this.registerCommands = (context) => {
            context.subscriptions.push(vscode.commands.registerCommand('cdt.debug.suspendAllSession', () => __awaiter(this, void 0, void 0, function* () {
                const session = vscode.debug.activeDebugSession;
                if (session) {
                    yield session.customRequest('cdt-amalgamator/suspendAll');
                }
            })));
        };
        this.registerCommands(context);
    }
}
exports.SuspendAllSession = SuspendAllSession;
//# sourceMappingURL=SuspendAllSession.js.map