import Combine
import Foundation

/// A completed round dismisses existing results; explicitly reopening them keeps them readable.
@MainActor
public final class ResultWindowFeedback: ObservableObject {
    @Published public private(set) var opacity: Double = 1
    private let session: DictationSession
    private let insertion: DictationInsertion
    private let feedback: DictationFeedback
    private var automaticallyDismisses = false
    private var phaseObserver: AnyCancellable?
    private var opacityObserver: AnyCancellable?

    public init(session: DictationSession, insertion: DictationInsertion,
                holdDuration: Double = 3, fadeDuration: Double = 0.3,
                attentionHoldDuration: Double = 5,
                reduceMotion: @escaping () -> Bool = { false }) {
        self.session = session
        self.insertion = insertion
        // Separate timing from the floating bar: hovering that bar must not pin this window.
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: holdDuration, fadeDuration: fadeDuration,
                                     attentionHoldDuration: attentionHoldDuration, reduceMotion: reduceMotion)
        phaseObserver = session.$phase.removeDuplicates().sink { [weak self] phase in
            guard let self else { return }
            // Use the emitted phase because @Published sends before storing the new value.
            if [.authorizing, .recording, .recognizing, .polishing, .failed].contains(phase) {
                self.automaticallyDismisses = true
                self.opacity = 1
            }
        }
        opacityObserver = feedback.$opacity.removeDuplicates().sink { [weak self] opacity in
            guard let self, self.automaticallyDismisses else { return }
            self.opacity = opacity
        }
    }

    /// Called by an explicit "show results" action, never by a completion notification.
    public func show() {
        automaticallyDismisses = session.isBusy || insertion.isDelivering
        opacity = 1
    }
}
