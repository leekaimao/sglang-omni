import Foundation

public struct CorrectionResult {
    public enum Status: String { case ok, noChange = "no_change" }
    public let status: Status
    public let correctedText: String
    public init(status: Status, correctedText: String) {
        self.status = status
        self.correctedText = correctedText
    }
}

@MainActor
public protocol TextCorrecting: AnyObject {
    func correct(original: String, instruction: String, personalBackground: String) async throws -> CorrectionResult
}

/// Explicit user edits have a different contract from automatic light polishing.
@MainActor
public final class OllamaCorrector: TextCorrecting {
    private let configuration: LocalModelConfiguration
    private let transport: LocalHTTPTransport

    public init(configuration: LocalModelConfiguration = .ollama, transport: LocalHTTPTransport) {
        self.configuration = configuration
        self.transport = transport
    }

    public func correct(original: String, instruction: String, personalBackground: String) async throws -> CorrectionResult {
        let request = try Self.request(original: original, instruction: instruction,
                                       personalBackground: personalBackground, configuration: configuration)
        let bytes = try await transport.send(request, stage: "Ollama 更正")
        let result = try Self.decode(bytes, original: original, instruction: instruction)
        if let explicit = SpokenSpelling.explicitRevision(original: original, instruction: instruction, terminology: personalBackground) {
            return CorrectionResult(status: .ok, correctedText: explicit.corrected)
        }
        guard result.status == .ok else { return result }
        // A valid JSON envelope does not make copied editing instructions valid prose.
        // Ignore ASR/model punctuation and spacing when checking a full instruction echo.
        let compact = { (text: String) in
            text.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
        }
        let command = compact(instruction)
        if !command.isEmpty, compact(result.correctedText).contains(command), !compact(original).contains(command) {
            throw DictationError("模型把修改意见写进了正文，已拦截；请明确说明要替换的词和正确写法。")
        }
        return CorrectionResult(status: .ok,
                                correctedText: SpokenSpelling.normalize(result.correctedText, original: original,
                                                                        instruction: instruction, terminology: personalBackground))
    }

    public static func request(original: String, instruction: String, personalBackground: String = "",
                               configuration: LocalModelConfiguration = .ollama) throws -> URLRequest {
        guard !original.isEmpty, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DictationError("没有上一段文字或没有听清修改意见；原文保持不变。")
        }
        let system = """
        你是文字编辑器。执行修改意见，只修改指定的词，保留原文中其他文字。修改意见是编辑指令，不是正文；不要把它追加、复述或解释在结果里。明确拼写优先于参考背景，逐个字母拼出的单词不要加点或空格。只输出 JSON，text 只能是修改后的完整原文，不包含修改意见或参考背景。不能确定修改位置时保留原文。
        """
        let normalizedInstruction = SpokenSpelling.normalizeInstruction(instruction, terminology: personalBackground)
        // Plain labels are easier for small local models than a classification task
        // over nested JSON. The current editing instruction remains last.
        let input = "参考背景：" + personalBackground + "\n原文：" + original + "\n修改意见：" + normalizedInstruction
        let messages = [["role": "system", "content": system], ["role": "user", "content": input]]
        guard try JSONSerialization.data(withJSONObject: messages).count <= 5000 else {
            throw DictationError("上一段、修改意见或术语背景过长，请精简后重试。原文保持不变。")
        }
        // Ask the small model for the edit itself. Derive unchanged status locally;
        // branch-heavy classification plus editing produced false-success copies.
        let schema: [String: Any] = ["type": "object", "additionalProperties": false,
                                    "required": ["text"], "properties": ["text": ["type": "string"]]]
        let payload: [String: Any] = [
            "model": configuration.model, "messages": messages, "format": schema,
            "think": false, "stream": false, "keep_alive": "10m",
            "options": ["temperature": 0, "top_p": 1, "num_ctx": 8192, "num_predict": 2048],
        ]
        var request = URLRequest(url: configuration.endpoint("api/chat"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        return request
    }

    public static func decode(_ data: Data, original: String, instruction: String) throws -> CorrectionResult {
        struct Response: Decodable {
            struct Message: Decodable { let content: String }
            let message: Message?
            let done: Bool?
            let done_reason: String?
            let error: String?
        }
        do {
            guard data.count <= 128_000, !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DictationError("更正响应过长或修改意见为空。")
            }
            let response = try JSONDecoder().decode(Response.self, from: data)
            guard response.error == nil, response.done == true, response.done_reason == "stop",
                  let content = response.message?.content else { throw DictationError("Ollama 更正未完整结束。") }
            let bytes = Data(content.utf8)
            guard let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                  Set(object.keys) == Set(["text"]), let text = object["text"] as? String else {
                throw DictationError("Ollama 更正字段异常。")
            }
            guard text.count <= 8000 else { throw DictationError("Ollama 更正输出过长。") }
            if text.isEmpty, !explicitlyDeletesAll(instruction) {
                throw DictationError("没有明确的整段删除指令，已拒绝空更正结果。")
            }
            let result = CorrectionResult(status: text.utf16.elementsEqual(original.utf16) ? .noChange : .ok,
                                          correctedText: text)
            return result
        } catch let error as DictationError { throw error }
        catch { throw DictationError("Ollama 更正返回格式异常，原文保持不变。") }
    }

    private static func explicitlyDeletesAll(_ instruction: String) -> Bool {
        let text = instruction.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
        // Match the whole command: "delete all commas" must not authorize deleting
        // the entire dictation. Ambiguous/compound instructions fail conservatively.
        let patterns = [
            "^(请|请帮我|帮我)?((删除|删掉|清空)(全部|整段|上一段|这一段)|(全部|整段|上一段|这一段)(删除|删掉|清空)|把(整段|上一段|这一段)(全部|都)?(删除|删掉|清空))$",
            "^(please)?(delete|clear|remove)(all|everything|alltext|allthetext|the(entire|whole)(text|paragraph)|the(last|previous)paragraph)$",
        ]
        return patterns.contains { text.range(of: $0, options: .regularExpression) != nil }
    }
}
