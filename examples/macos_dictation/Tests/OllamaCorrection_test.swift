import Foundation

private func response(_ content: [String: Any], done: Bool = true, reason: String = "stop") throws -> Data {
    let inner = try JSONSerialization.data(withJSONObject: content)
    return try JSONSerialization.data(withJSONObject: ["done": done, "done_reason": reason,
        "message": ["content": String(decoding: inner, as: UTF8.self)]])
}

private func rejected(_ action: () throws -> Void) {
    do { try action(); preconditionFailure("Invalid correction must be rejected") }
    catch { }
}

@main
private enum OllamaCorrectionTests {
    @MainActor
    static func main() async throws {
        let original = "明天下午两点见。"
        let valid: [String: Any] = ["text": "明天下午三点见。"]
        let reply = try response(valid)
        let result = try OllamaCorrector.decode(reply, original: original, instruction: "两点改三点")
        precondition(result.correctedText == "明天下午三点见。")
        for reason in ["length", "", "error"] {
            rejected { _ = try OllamaCorrector.decode(response(valid, reason: reason), original: original, instruction: "修改") }
        }
        rejected { _ = try OllamaCorrector.decode(response(valid, done: false), original: original, instruction: "修改") }
        rejected { _ = try OllamaCorrector.decode(reply, original: original, instruction: " \n") }
        rejected { _ = try OllamaCorrector.decode(Data("not JSON".utf8), original: original, instruction: "修改") }
        for bad in [
            ["decision": "unknown", "text": original, "question": ""],
            ["decision": "no_change", "text": "另一段", "question": ""],
            ["decision": "needs_clarification", "text": original, "question": ""],
            ["decision": "needs_clarification", "text": "已经改过的文本", "question": "请说明要改哪里"],
            ["decision": "ok", "text": "新版", "question": "我不知道"],
            ["decision": "ok", "text": "新版", "question": "", "extra": "unexpected"],
        ] {
            rejected { _ = try OllamaCorrector.decode(response(bad), original: original, instruction: "修改") }
        }
        let unchanged = try OllamaCorrector.decode(response(["text": original]),
                                                   original: original, instruction: "不改了")
        precondition(unchanged.status == .noChange)
        let deletion = try response(["text": ""])
        for instruction in ["删除全部", "把上一段全部删掉", "delete all"] {
            let result = try OllamaCorrector.decode(deletion, original: original, instruction: instruction)
            precondition(result.correctedText.isEmpty)
        }
        for instruction in ["改一下人名", "不要删除全部", "don't delete all", "删除全部标点", "delete all commas", "remove small error"] {
            rejected { _ = try OllamaCorrector.decode(deletion, original: original, instruction: instruction) }
        }
        rejected { _ = try OllamaCorrector.request(original: "", instruction: "改一下") }
        rejected { _ = try OllamaCorrector.request(original: String(repeating: "长", count: 10_000), instruction: "改一下") }

        let transport = LocalHTTPTransport(protocolClasses: [HTTPStub.self])
        let model = try LocalModelConfiguration(baseURL: "http://127.0.0.1:11434", model: "correction-model")
        let corrector = OllamaCorrector(configuration: model, transport: transport)
        HTTPStub.reset(["/api/chat": .http(200, String(decoding: reply, as: UTF8.self))])
        let corrected = try await corrector.correct(original: original, instruction: "两点改三点", personalBackground: "术语")
        precondition(corrected.correctedText == result.correctedText)
        let body = try JSONSerialization.jsonObject(with: Data(HTTPStub.requests[0].body.utf8)) as! [String: Any]
        precondition(body["model"] as? String == "correction-model" && body["format"] is [String: Any])
        let messages = body["messages"] as! [[String: String]]
        let input = messages.last!["content"]!.components(separatedBy: "\n")
        precondition(input == ["参考背景：术语", "原文：明天下午两点见。", "修改意见：两点改三点"])
        HTTPStub.reset(["/api/chat": .held])
        let task = Task { try await corrector.correct(original: original, instruction: "修改", personalBackground: "") }
        while HTTPStub.heldCount == 0 { try await Task.sleep(nanoseconds: 1_000_000) }
        task.cancel()
        do { _ = try await task.value; preconditionFailure("Cancellation must propagate") }
        catch is CancellationError { }
        print("PASS: correction schema, completion, explicit deletion guard, input limits, local request and cancellation")
    }
}
