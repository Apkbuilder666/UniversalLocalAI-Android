#pragma once

#include "localai/backend.hpp"
#include "localai/benchmark.hpp"
#include "localai/model.hpp"
#include "localai/scheduler.hpp"
#include "localai/settings.hpp"

#include <QAudioFormat>
#include <QByteArray>
#include <QImage>
#include <QMainWindow>

#include <future>
#include <memory>
#include <optional>

class QAudioSink;
class QAudioSource;
class QBuffer;
class QComboBox;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDropEvent;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QStackedWidget;
class QTableWidget;
class QTextBrowser;
class QTimer;

namespace localai::ui {

class MainWindow final : public QMainWindow {
public:
    MainWindow();
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

private:
    QWidget* createChatPage();
    QWidget* createVisionPage();
    QWidget* createImagePage();
    QWidget* createAudioPage();
    QWidget* createModelsPage();
    QWidget* createHardwarePage();
    QWidget* createBenchmarksPage();
    QWidget* createSettingsPage();
    QWidget* createLogsPage();
    void refreshModels();
    void refreshHardware();
    void importModel();
    void loadSelectedModel();
    void unloadModel();
    const ModelDescriptor* selectedModel() const;
    void startInference(InferenceRequest request,
                        std::function<void(const Result<InferenceOutput>&)> completion);
    void pollInference();
    void pollLoad();
    void startRecording();
    void stopRecording();
    void transcribeAudio();
    void synthesizeSpeech();
    void chooseVisionImage();
    void useClipboardImage();
    void runVision();
    void playAudio(const InferenceOutput& output);
    void setBusy(bool busy, const QString& status = {});

    ModelCatalog catalog_;
    Settings settings_;
    BenchmarkStore benchmarks_;
    InferenceScheduler scheduler_{1};
    std::unique_ptr<IModelBackend> backend_;
    std::string loadedModelId_;
    std::optional<ScheduledTask> activeTask_;
    std::function<void(const Result<InferenceOutput>&)> completion_;
    std::future<Result<BackendInfo>> loadFuture_;
    std::unique_ptr<IModelBackend> loadingBackend_;
    std::string loadingModelId_;

    QStackedWidget* pages_{};
    QListWidget* navigation_{};
    QLabel* status_{};
    QTableWidget* modelsTable_{};
    QComboBox* backendChoice_{};
    QPushButton* loadButton_{};
    QPushButton* unloadButton_{};
    QTextBrowser* chatTranscript_{};
    QListWidget* conversations_{};
    QPlainTextEdit* chatInput_{};
    QLineEdit* systemPrompt_{};
    QLineEdit* stopSequences_{};
    QPushButton* chatSend_{};
    QPushButton* stopButton_{};
    QLabel* visionPreview_{};
    QLabel* visionPathLabel_{};
    QPlainTextEdit* visionPrompt_{};
    QTextBrowser* visionResponse_{};
    QPlainTextEdit* imagePrompt_{};
    QLineEdit* negativePrompt_{};
    QSpinBox* imageWidth_{};
    QSpinBox* imageHeight_{};
    QSpinBox* imageSteps_{};
    QDoubleSpinBox* imageCfg_{};
    QSpinBox* imageSeed_{};
    QComboBox* imageSampler_{};
    QComboBox* imageScheduler_{};
    QDoubleSpinBox* imageStrength_{};
    QProgressBar* imageProgress_{};
    QLabel* imagePreview_{};
    QLineEdit* audioText_{};
    QLabel* audioFileLabel_{};
    QPlainTextEdit* transcription_{};
    QTextBrowser* hardwareText_{};
    QTextBrowser* benchmarkText_{};
    QPlainTextEdit* logs_{};
    QSpinBox* settingThreads_{};
    QSpinBox* settingContext_{};
    QSpinBox* settingBatch_{};
    QSpinBox* settingGpuLayers_{};
    QComboBox* settingDevice_{};
    QComboBox* settingMemoryPolicy_{};
    QComboBox* settingKvCache_{};
    QTimer* taskTimer_{};
    QTimer* loadTimer_{};

    std::filesystem::path audioFile_;
    QImage visionImage_;
    QImage imageInitial_;
    QImage imageMask_;
    std::vector<float> recordedSamples_;
    std::unique_ptr<QAudioSource> audioSource_;
    std::unique_ptr<QBuffer> microphoneBuffer_;
    QByteArray microphoneBytes_;
    QAudioFormat microphoneFormat_;
    std::unique_ptr<QAudioSink> audioSink_;
    std::unique_ptr<QBuffer> playbackBuffer_;
    QByteArray playbackBytes_;
    std::vector<std::string> conversationDocuments_;
    int currentConversation_{};
    std::string lastChatPrompt_;
};

} // namespace localai::ui
