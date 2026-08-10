#include "main_window.hpp"

#include "localai/audio.hpp"
#include "localai/hardware.hpp"
#include "localai/log.hpp"
#include "localai/settings.hpp"

#include <QAudioDevice>
#include <QAudioSink>
#include <QAudioSource>
#include <QApplication>
#include <QBuffer>
#include <QBoxLayout>
#include <QCloseEvent>
#include <QClipboard>
#include <QComboBox>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QListView>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMediaDevices>
#include <QMessageBox>
#include <QMimeData>
#include <QtGlobal>
#ifdef Q_OS_ANDROID
#include <QJniObject>
#include <QNativeInterface>
#endif
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
#include <QMicrophonePermission>
#endif
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QPixmap>
#include <QRegularExpression>
#include <QSaveFile>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTextBrowser>
#include <QTextCursor>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <thread>

namespace localai::ui {
namespace {

QString bytesText(std::uint64_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit{};
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    return QString::number(value, 'f', unit ? 2 : 0) + ' ' + units[unit];
}

QWidget* messagePage(const QString& title, const QString& text) {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* heading = new QLabel(title);
    heading->setStyleSheet("font-size: 24px; font-weight: 600");
    auto* body = new QLabel(text);
    body->setWordWrap(true);
    body->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(heading);
    layout->addWidget(body);
    layout->addStretch();
    return page;
}

QString persistentLocalPath(QWidget* parent, const QString& selected, QString& failure) {
#ifdef Q_OS_ANDROID
    const QUrl url(selected);
    if (url.scheme().compare("content", Qt::CaseInsensitive) != 0) return selected;
    QFile source(selected);
    if (!source.open(QIODevice::ReadOnly)) {
        failure = "Android document provider could not open the selected file: " + source.errorString();
        return {};
    }
    QString fileName = QFileInfo(url.path()).fileName();
    if (fileName.isEmpty()) fileName = "imported-model.bin";
    fileName.replace(QRegularExpression("[^A-Za-z0-9._-]"), "_");
    const auto digest = QCryptographicHash::hash(selected.toUtf8(), QCryptographicHash::Sha256).toHex().left(16);
    QDir directory(QString::fromStdString((applicationDataDirectory() / "imported-files").string()));
    if (!directory.mkpath(".")) {
        failure = "Could not create the private Android model directory";
        return {};
    }
    const QString destination = directory.filePath(QString::fromLatin1(digest) + "-" + fileName);
    if (QFileInfo existing(destination); existing.isFile() && existing.size() == source.size()) return destination;
    QSaveFile output(destination);
    if (!output.open(QIODevice::WriteOnly)) {
        failure = "Could not create a persistent private copy: " + output.errorString();
        return {};
    }
    const qint64 totalMiB = source.size() > 0 ? source.size() / (1024 * 1024) : 0;
    QProgressDialog progress("Copying selected file into private app storage…", "Cancel", 0,
                             static_cast<int>(std::min<qint64>(totalMiB, std::numeric_limits<int>::max())), parent);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(250);
    QByteArray buffer(4 * 1024 * 1024, Qt::Uninitialized);
    qint64 copied{};
    while (!source.atEnd()) {
        const qint64 read = source.read(buffer.data(), buffer.size());
        if (read <= 0 || output.write(buffer.constData(), read) != read) {
            output.cancelWriting();
            failure = "Android document copy failed while reading or writing model bytes";
            return {};
        }
        copied += read;
        progress.setValue(static_cast<int>(std::min<qint64>(copied / (1024 * 1024),
                                                            std::numeric_limits<int>::max())));
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        if (progress.wasCanceled()) {
            output.cancelWriting();
            failure = "Import cancelled";
            return {};
        }
    }
    if (!output.commit()) {
        failure = "Could not atomically finish the private file copy: " + output.errorString();
        return {};
    }
    return destination;
#else
    static_cast<void>(parent);
    static_cast<void>(failure);
    return selected;
#endif
}

QString persistentBundlePath(QWidget* parent, const QString& selected, QString& failure) {
#ifdef Q_OS_ANDROID
    const QUrl url(selected);
    if (url.scheme().compare("content", Qt::CaseInsensitive) != 0) return selected;
    const auto digest = QCryptographicHash::hash(selected.toUtf8(), QCryptographicHash::Sha256).toHex().left(16);
    QDir root(QString::fromStdString((applicationDataDirectory() / "imported-bundles").string()));
    if (!root.mkpath(".")) {
        failure = "Could not create the private Android bundle directory";
        return {};
    }
    const QString destination = root.filePath(QString::fromLatin1(digest));
    const QJniObject context = QNativeInterface::QAndroidApplication::context();
    auto copy = std::async(std::launch::async, [context, selected, destination] {
        const QJniObject treeUri = QJniObject::fromString(selected);
        const QJniObject destinationPath = QJniObject::fromString(destination);
        return QJniObject::callStaticObjectMethod(
            "ai/local/universal/DocumentTreeCopier", "copyTree",
            "(Landroid/content/Context;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
            context.object(), treeUri.object<jstring>(), destinationPath.object<jstring>());
    });
    QProgressDialog progress("Copying the selected model bundle into private app storage…", {}, 0, 0, parent);
    progress.setWindowModality(Qt::WindowModal);
    progress.setCancelButton(nullptr);
    progress.show();
    while (copy.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const QJniObject error = copy.get();
    if (!error.isValid()) {
        failure = "Android document-tree import helper did not return a result";
        return {};
    }
    failure = error.toString();
    return failure.isEmpty() ? destination : QString{};
#else
    static_cast<void>(parent);
    static_cast<void>(failure);
    return selected;
#endif
}

}

MainWindow::MainWindow()
    : catalog_(applicationDataDirectory() / "models.db"),
      settings_(applicationDataDirectory() / "settings.db"),
      benchmarks_(applicationDataDirectory() / "benchmarks.db") {
    registerCompiledBackends();
    std::filesystem::create_directories(applicationDataDirectory());
    Logger::instance().setFile(applicationDataDirectory() / "localai.log.jsonl");
    static_cast<void>(catalog_.load());
    static_cast<void>(settings_.load());
    static_cast<void>(benchmarks_.load());

    auto* central = new QWidget;
    auto* root = new QVBoxLayout(central);
    QBoxLayout* content{};
#ifdef Q_OS_ANDROID
    content = new QVBoxLayout;
#else
    content = new QHBoxLayout;
#endif
    navigation_ = new QListWidget;
#ifdef Q_OS_ANDROID
    navigation_->setFlow(QListView::LeftToRight);
    navigation_->setWrapping(false);
    navigation_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    navigation_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    navigation_->setFixedHeight(58);
#else
    navigation_->setFixedWidth(180);
#endif
    navigation_->addItems({"Chat", "Vision", "Image", "Audio", "Models", "Hardware", "Benchmarks", "Settings", "Logs"});
    pages_ = new QStackedWidget;
    pages_->addWidget(createChatPage());
    pages_->addWidget(createVisionPage());
    pages_->addWidget(createImagePage());
    pages_->addWidget(createAudioPage());
    pages_->addWidget(createModelsPage());
    pages_->addWidget(createHardwarePage());
    pages_->addWidget(createBenchmarksPage());
    pages_->addWidget(createSettingsPage());
    pages_->addWidget(createLogsPage());
#ifdef Q_OS_ANDROID
    content->addWidget(navigation_);
#else
    content->addWidget(navigation_);
#endif
    content->addWidget(pages_, 1);
    status_ = new QLabel("System ready");
    status_->setStyleSheet("padding: 7px; border-top: 1px solid palette(mid)");
    root->addLayout(content, 1);
    root->addWidget(status_);
    setCentralWidget(central);
    resize(1180, 760);
    setWindowTitle("Universal Local AI");
    setAcceptDrops(true);
#ifdef Q_OS_ANDROID
    qApp->setStyleSheet(
        "QWidget{font-size:15px;} QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,QLineEdit{min-height:42px;}"
        "QPushButton{padding:6px 12px;} QListWidget::item{padding:10px;}"
        "QPlainTextEdit,QTextBrowser,QTableWidget{font-size:14px;}");
#endif
    auto* sendShortcut = new QShortcut(QKeySequence("Ctrl+Return"), this);
    connect(sendShortcut, &QShortcut::activated, chatSend_, &QPushButton::click);
    auto* stopShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    connect(stopShortcut, &QShortcut::activated, stopButton_, &QPushButton::click);

    connect(navigation_, &QListWidget::currentRowChanged, pages_, &QStackedWidget::setCurrentIndex);
    navigation_->setCurrentRow(0);
    taskTimer_ = new QTimer(this);
    taskTimer_->setInterval(40);
    connect(taskTimer_, &QTimer::timeout, this, [this] { pollInference(); });
    loadTimer_ = new QTimer(this);
    loadTimer_->setInterval(75);
    connect(loadTimer_, &QTimer::timeout, this, [this] { pollLoad(); });
    Logger::instance().setListener([this](const LogEntry& entry) {
        QMetaObject::invokeMethod(this, [this, entry] {
            if (logs_) logs_->appendPlainText(QString::fromStdString(toString(entry.category) + " | " + entry.message));
        });
    });
    refreshModels();
    refreshHardware();
}

MainWindow::~MainWindow() {
    Logger::instance().setListener({});
    if (activeTask_) activeTask_->cancel();
    if (backend_) backend_->cancel();
    scheduler_.shutdown();
    if (loadFuture_.valid()) {
        loadFuture_.wait();
        static_cast<void>(loadFuture_.get());
    }
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (activeTask_) activeTask_->cancel();
    if (backend_) backend_->cancel();
    event->accept();
}

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* event) {
    QStringList failures;
    for (const auto& url : event->mimeData()->urls()) {
        QString copyFailure;
        const auto path = persistentLocalPath(this, url.isLocalFile() ? url.toLocalFile() : url.toString(), copyFailure);
        if (path.isEmpty()) {
            if (!copyFailure.isEmpty()) failures << copyFailure;
            continue;
        }
        const auto extension = QFileInfo(path).suffix().toLower();
        const bool stillImage = extension == "png" || extension == "jpg" || extension == "jpeg" ||
                                extension == "bmp" || extension == "webp";
        if (pages_->currentIndex() == 1 && stillImage) {
            QImage image(path);
            if (image.isNull()) failures << path + ": image decoder rejected the file";
            else {
                visionImage_ = image.convertToFormat(QImage::Format_RGB888);
                visionPathLabel_->setText(path);
                visionPreview_->setPixmap(QPixmap::fromImage(visionImage_).scaled(
                    visionPreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }
            continue;
        }
        const auto imported = catalog_.importModel(path.toStdString());
        if (!imported) failures << path + ": " + QString::fromStdString(imported.error().message);
    }
    refreshModels();
    if (!failures.isEmpty()) QMessageBox::warning(this, "Some dropped files were rejected", failures.join("\n"));
    event->acceptProposedAction();
}

QWidget* MainWindow::createChatPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Local Chat");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    systemPrompt_ = new QLineEdit;
    stopSequences_ = new QLineEdit;
    chatTranscript_ = new QTextBrowser;
    chatTranscript_->setOpenExternalLinks(false);
    conversations_ = new QListWidget;
    conversations_->addItem("Conversation 1");
    conversationDocuments_.emplace_back();
    auto* newConversation = new QPushButton("New");
    auto* deleteConversation = new QPushButton("Delete");
    auto* conversationButtons = new QHBoxLayout;
    conversationButtons->addWidget(newConversation);
    conversationButtons->addWidget(deleteConversation);
    auto* conversationPanel = new QWidget;
    auto* conversationLayout = new QVBoxLayout(conversationPanel);
    conversationLayout->setContentsMargins(0, 0, 0, 0);
    conversationLayout->addWidget(conversations_);
    conversationLayout->addLayout(conversationButtons);
    auto* transcriptSplit = new QSplitter;
    transcriptSplit->addWidget(conversationPanel);
    transcriptSplit->addWidget(chatTranscript_);
    transcriptSplit->setStretchFactor(1, 1);
    transcriptSplit->setSizes({180, 800});
    chatInput_ = new QPlainTextEdit;
    chatInput_->setMaximumHeight(120);
    chatSend_ = new QPushButton("Generate");
    auto* editLast = new QPushButton("Edit last prompt");
    auto* regenerate = new QPushButton("Regenerate");
    stopButton_ = new QPushButton("Stop");
    stopButton_->setEnabled(false);
    auto* buttons = new QHBoxLayout;
    buttons->addStretch();
    buttons->addWidget(editLast);
    buttons->addWidget(regenerate);
    buttons->addWidget(stopButton_);
    buttons->addWidget(chatSend_);
    layout->addWidget(title);
    layout->addWidget(new QLabel("System prompt (optional)"));
    layout->addWidget(systemPrompt_);
    layout->addWidget(new QLabel("Stop sequences separated by | (optional)"));
    layout->addWidget(stopSequences_);
    layout->addWidget(transcriptSplit, 1);
    layout->addWidget(chatInput_);
    layout->addLayout(buttons);
    connect(chatSend_, &QPushButton::clicked, this, [this] {
        if (!backend_) {
            QMessageBox::information(this, "Model required", "Load a text-generation model from Models first.");
            return;
        }
        const auto prompt = chatInput_->toPlainText();
        if (prompt.trimmed().isEmpty()) return;
        lastChatPrompt_ = prompt.toStdString();
        chatTranscript_->append("<p><b>You</b></p><pre>" + prompt.toHtmlEscaped() +
                                "</pre><p><b>Assistant</b></p>");
        chatInput_->clear();
        InferenceRequest request;
        request.kind = TaskKind::Text;
        request.prompt = prompt.toStdString();
        request.text["system"] = systemPrompt_->text().toStdString();
        request.text["stop_sequences"] = stopSequences_->text().toStdString();
        request.numeric = {{"max_tokens", 512}, {"temperature", 0.8}, {"top_k", 40}, {"top_p", 0.95}, {"min_p", 0.05}};
        request.onText = [this](std::string_view piece) {
            const auto text = QString::fromUtf8(piece.data(), static_cast<qsizetype>(piece.size()));
            QMetaObject::invokeMethod(this, [this, text] { chatTranscript_->moveCursor(QTextCursor::End); chatTranscript_->insertPlainText(text); });
        };
        startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
            if (!result) chatTranscript_->append("<p><b>Error:</b> " + QString::fromStdString(result.error().message).toHtmlEscaped() + "</p>");
            else {
                const auto& performance = result.value().performance;
                chatTranscript_->append(QString("<hr><small>Input: %1 tokens · Output: %2 tokens · First token: %3 ms · %4 tokens/s</small>")
                    .arg(performance.inputUnits).arg(performance.outputUnits)
                    .arg(performance.firstOutputMilliseconds, 0, 'f', 1)
                    .arg(performance.outputUnitsPerSecond, 0, 'f', 2));
                status_->setText(QString("Completed at %1 tokens/s").arg(performance.outputUnitsPerSecond, 0, 'f', 2));
            }
        });
    });
    connect(stopButton_, &QPushButton::clicked, this, [this] {
        if (activeTask_) activeTask_->cancel();
        if (backend_) backend_->cancel();
    });
    connect(editLast, &QPushButton::clicked, this, [this] {
        chatInput_->setPlainText(QString::fromStdString(lastChatPrompt_));
        chatInput_->setFocus();
    });
    connect(regenerate, &QPushButton::clicked, this, [this] {
        if (lastChatPrompt_.empty() || activeTask_) return;
        chatInput_->setPlainText(QString::fromStdString(lastChatPrompt_));
        chatSend_->click();
    });
    connect(newConversation, &QPushButton::clicked, this, [this] {
        if (activeTask_) return;
        if (currentConversation_ >= 0 && currentConversation_ < static_cast<int>(conversationDocuments_.size()))
            conversationDocuments_[static_cast<std::size_t>(currentConversation_)] = chatTranscript_->toHtml().toStdString();
        conversationDocuments_.emplace_back();
        conversations_->addItem(QString("Conversation %1").arg(conversationDocuments_.size()));
        conversations_->setCurrentRow(static_cast<int>(conversationDocuments_.size()) - 1);
    });
    connect(deleteConversation, &QPushButton::clicked, this, [this] {
        if (activeTask_ || conversationDocuments_.size() <= 1) return;
        const int row = conversations_->currentRow();
        if (row < 0) return;
        currentConversation_ = -1;
        conversationDocuments_.erase(conversationDocuments_.begin() + row);
        delete conversations_->takeItem(row);
        for (int index = 0; index < conversations_->count(); ++index)
            conversations_->item(index)->setText(QString("Conversation %1").arg(index + 1));
        conversations_->setCurrentRow(std::min(row, static_cast<int>(conversationDocuments_.size()) - 1));
    });
    connect(conversations_, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0 || row >= static_cast<int>(conversationDocuments_.size())) return;
        if (currentConversation_ >= 0 && currentConversation_ < static_cast<int>(conversationDocuments_.size()))
            conversationDocuments_[static_cast<std::size_t>(currentConversation_)] = chatTranscript_->toHtml().toStdString();
        currentConversation_ = row;
        chatTranscript_->setHtml(QString::fromStdString(conversationDocuments_[static_cast<std::size_t>(row)]));
    });
    conversations_->setCurrentRow(0);
    return page;
}

QWidget* MainWindow::createVisionPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Still-image Vision");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    visionPathLabel_ = new QLabel("No still image selected");
    visionPreview_ = new QLabel;
    visionPreview_->setMinimumHeight(280);
    visionPreview_->setAlignment(Qt::AlignCenter);
    visionPreview_->setStyleSheet("border: 1px solid palette(mid); background: palette(base)");
    auto* open = new QPushButton("Open image");
    auto* clipboard = new QPushButton("Paste image from clipboard");
    auto* buttons = new QHBoxLayout;
    buttons->addWidget(open);
    buttons->addWidget(clipboard);
    buttons->addStretch();
    visionPrompt_ = new QPlainTextEdit;
    visionPrompt_->setMaximumHeight(90);
    visionPrompt_->setPlainText("Describe this image accurately.");
    auto* analyze = new QPushButton("Analyze still image");
    visionResponse_ = new QTextBrowser;
    layout->addWidget(title);
    layout->addWidget(visionPathLabel_);
    layout->addLayout(buttons);
    layout->addWidget(visionPreview_);
    layout->addWidget(visionPrompt_);
    layout->addWidget(analyze);
    layout->addWidget(visionResponse_, 1);
    connect(open, &QPushButton::clicked, this, [this] { chooseVisionImage(); });
    connect(clipboard, &QPushButton::clicked, this, [this] { useClipboardImage(); });
    connect(analyze, &QPushButton::clicked, this, [this] { runVision(); });
    return page;
}

QWidget* MainWindow::createImagePage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Still-image Generation");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    imagePrompt_ = new QPlainTextEdit;
    imagePrompt_->setMaximumHeight(90);
    negativePrompt_ = new QLineEdit;
    imageWidth_ = new QSpinBox;
    imageWidth_->setRange(64, 4096); imageWidth_->setSingleStep(64);
#ifdef Q_OS_ANDROID
    imageWidth_->setValue(384);
#else
    imageWidth_->setValue(512);
#endif
    imageHeight_ = new QSpinBox;
    imageHeight_->setRange(64, 4096); imageHeight_->setSingleStep(64);
#ifdef Q_OS_ANDROID
    imageHeight_->setValue(384);
#else
    imageHeight_->setValue(512);
#endif
    imageSteps_ = new QSpinBox;
    imageSteps_->setRange(1, 200);
#ifdef Q_OS_ANDROID
    imageSteps_->setValue(12);
#else
    imageSteps_->setValue(20);
#endif
    imageCfg_ = new QDoubleSpinBox;
    imageCfg_->setRange(0.0, 50.0); imageCfg_->setDecimals(2); imageCfg_->setValue(7.0);
    imageSeed_ = new QSpinBox;
    imageSeed_->setRange(-1, std::numeric_limits<int>::max()); imageSeed_->setValue(42);
    imageSampler_ = new QComboBox;
    imageSampler_->addItems({"euler_a", "euler", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm"});
    imageScheduler_ = new QComboBox;
    imageScheduler_->addItems({"discrete", "karras", "exponential", "ays", "gits"});
    imageStrength_ = new QDoubleSpinBox;
    imageStrength_->setRange(0.0, 1.0); imageStrength_->setDecimals(2); imageStrength_->setSingleStep(0.05); imageStrength_->setValue(0.75);
    auto* initialLabel = new QLabel("No initial image: text-to-image mode");
    auto* maskLabel = new QLabel("No inpainting mask");
    auto* openInitial = new QPushButton("Open initial image");
    auto* clearInitial = new QPushButton("Clear initial image");
    auto* openMask = new QPushButton("Open mask");
    auto* clearMask = new QPushButton("Clear mask");
    auto* sourceControls = new QHBoxLayout;
    sourceControls->addWidget(openInitial); sourceControls->addWidget(clearInitial);
    sourceControls->addWidget(openMask); sourceControls->addWidget(clearMask); sourceControls->addStretch();
    auto* form = new QFormLayout;
    form->addRow("Width", imageWidth_);
    form->addRow("Height", imageHeight_);
    form->addRow("Steps", imageSteps_);
    form->addRow("Guidance (CFG)", imageCfg_);
    form->addRow("Seed (-1 = random)", imageSeed_);
    form->addRow("Sampler", imageSampler_);
    form->addRow("Scheduler", imageScheduler_);
    form->addRow("Transformation strength", imageStrength_);
    auto* generate = new QPushButton("Generate image");
    imageProgress_ = new QProgressBar;
    imageProgress_->setRange(0, 100);
    imageProgress_->setValue(0);
    imagePreview_ = new QLabel("Generated image appears here");
    imagePreview_->setAlignment(Qt::AlignCenter);
    imagePreview_->setMinimumHeight(320);
    imagePreview_->setStyleSheet("border: 1px solid palette(mid); background: palette(base)");
    layout->addWidget(title);
    layout->addWidget(imagePrompt_);
    layout->addWidget(negativePrompt_);
    layout->addWidget(initialLabel);
    layout->addWidget(maskLabel);
    layout->addLayout(sourceControls);
    layout->addLayout(form);
    layout->addWidget(generate);
    layout->addWidget(imageProgress_);
    layout->addWidget(imagePreview_, 1);
    connect(openInitial, &QPushButton::clicked, this, [this, initialLabel] {
        const auto path = QFileDialog::getOpenFileName(this, "Open initial still image", {},
            "Still images (*.png *.jpg *.jpeg *.bmp *.webp)");
        if (path.isEmpty()) return;
        QImage image(path);
        if (image.isNull()) QMessageBox::warning(this, "Image rejected", "Qt could not decode the selected still image.");
        else {
            imageInitial_ = image.convertToFormat(QImage::Format_RGB888);
            initialLabel->setText(path);
        }
    });
    connect(clearInitial, &QPushButton::clicked, this, [this, initialLabel, maskLabel] {
        imageInitial_ = {};
        imageMask_ = {};
        initialLabel->setText("No initial image: text-to-image mode");
        maskLabel->setText("No inpainting mask");
    });
    connect(openMask, &QPushButton::clicked, this, [this, maskLabel] {
        if (imageInitial_.isNull()) {
            QMessageBox::information(this, "Initial image required", "Choose an initial image before adding an inpainting mask.");
            return;
        }
        const auto path = QFileDialog::getOpenFileName(this, "Open inpainting mask", {},
            "Still images (*.png *.jpg *.jpeg *.bmp *.webp)");
        if (path.isEmpty()) return;
        QImage image(path);
        if (image.isNull()) QMessageBox::warning(this, "Mask rejected", "Qt could not decode the selected mask image.");
        else {
            imageMask_ = image.convertToFormat(QImage::Format_Grayscale8);
            maskLabel->setText(path);
        }
    });
    connect(clearMask, &QPushButton::clicked, this, [this, maskLabel] {
        imageMask_ = {};
        maskLabel->setText("No inpainting mask");
    });
    connect(generate, &QPushButton::clicked, this, [this] {
        if (!backend_ || backend_->backendInfo().id != "stable-diffusion.cpp") {
            QMessageBox::information(this, "Model required", "Load a compatible still-image generation model first.");
            return;
        }
        InferenceRequest request;
        request.kind = TaskKind::ImageGeneration;
        request.prompt = imagePrompt_->toPlainText().toStdString();
        request.negativePrompt = negativePrompt_->text().toStdString();
        request.numeric = {{"width", static_cast<double>(imageWidth_->value())},
                           {"height", static_cast<double>(imageHeight_->value())},
                           {"steps", static_cast<double>(imageSteps_->value())},
                           {"cfg", imageCfg_->value()}, {"seed", static_cast<double>(imageSeed_->value())},
                           {"strength", imageStrength_->value()}};
        request.text["sampler"] = imageSampler_->currentText().toStdString();
        request.text["scheduler"] = imageScheduler_->currentText().toStdString();
        if (!imageInitial_.isNull()) {
            request.imageWidth = imageInitial_.width();
            request.imageHeight = imageInitial_.height();
            request.imageChannels = 3;
            request.imagePixels.resize(static_cast<std::size_t>(request.imageWidth * request.imageHeight * 3));
            for (int y = 0; y < request.imageHeight; ++y)
                std::memcpy(request.imagePixels.data() + static_cast<std::size_t>(y * request.imageWidth * 3),
                            imageInitial_.constScanLine(y), static_cast<std::size_t>(request.imageWidth * 3));
        }
        if (!imageMask_.isNull()) {
            request.maskWidth = imageMask_.width();
            request.maskHeight = imageMask_.height();
            request.maskChannels = 1;
            request.maskPixels.resize(static_cast<std::size_t>(request.maskWidth * request.maskHeight));
            for (int y = 0; y < request.maskHeight; ++y)
                std::memcpy(request.maskPixels.data() + static_cast<std::size_t>(y * request.maskWidth),
                            imageMask_.constScanLine(y), static_cast<std::size_t>(request.maskWidth));
        }
        imageProgress_->setValue(0);
        request.onProgress = [this](double value) {
            QMetaObject::invokeMethod(this, [this, value] {
                imageProgress_->setValue(static_cast<int>(std::clamp(value, 0.0, 1.0) * 100.0));
            });
        };
        startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
            if (!result) {
                QMessageBox::critical(this, "Generation failed", QString::fromStdString(result.error().message));
                return;
            }
            const auto& value = result.value();
            QImage image(value.imagePixels.data(), value.imageWidth, value.imageHeight,
                         value.imageWidth * value.imageChannels,
                         value.imageChannels == 4 ? QImage::Format_RGBA8888 : QImage::Format_RGB888);
            const auto owned = image.copy();
            imageProgress_->setValue(100);
            imagePreview_->setPixmap(QPixmap::fromImage(owned).scaled(imagePreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            const auto destination = QFileDialog::getSaveFileName(this, "Save generated image", {}, "PNG image (*.png);;JPEG image (*.jpg)");
            if (!destination.isEmpty() && !owned.save(destination)) QMessageBox::warning(this, "Save failed", "Qt could not encode the selected image format.");
        });
    });
    return page;
}

QWidget* MainWindow::createAudioPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Offline Audio");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    audioFileLabel_ = new QLabel("No audio selected");
    auto* open = new QPushButton("Open WAVE file");
    auto* record = new QPushButton("Start microphone");
    auto* transcribe = new QPushButton("Transcribe");
    transcription_ = new QPlainTextEdit;
    transcription_->setReadOnly(true);
    audioText_ = new QLineEdit;
    auto* synthesize = new QPushButton("Generate and play speech");
    auto* row = new QHBoxLayout;
    row->addWidget(open);
    row->addWidget(record);
    row->addWidget(transcribe);
    layout->addWidget(title);
    layout->addWidget(audioFileLabel_);
    layout->addLayout(row);
    layout->addWidget(transcription_, 1);
    layout->addWidget(audioText_);
    layout->addWidget(synthesize);
    connect(open, &QPushButton::clicked, this, [this] {
        const auto path = QFileDialog::getOpenFileName(this, "Open audio", {}, "WAVE audio (*.wav)");
        if (!path.isEmpty()) {
            QString failure;
            const auto localPath = persistentLocalPath(this, path, failure);
            if (localPath.isEmpty()) {
                QMessageBox::warning(this, "Audio import failed", failure);
                return;
            }
            audioFile_ = localPath.toStdString();
            recordedSamples_.clear();
            audioFileLabel_->setText(path);
        }
    });
    connect(record, &QPushButton::clicked, this, [this, record] {
        if (audioSource_) {
            stopRecording();
            record->setText("Start microphone");
        } else {
            startRecording();
            if (audioSource_) record->setText("Stop microphone");
        }
    });
    connect(transcribe, &QPushButton::clicked, this, [this] { transcribeAudio(); });
    connect(synthesize, &QPushButton::clicked, this, [this] { synthesizeSpeech(); });
    return page;
}

QWidget* MainWindow::createModelsPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Model Library");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    modelsTable_ = new QTableWidget(0, 5);
    modelsTable_->setHorizontalHeaderLabels({"Name", "Format", "Size", "Capabilities", "Location"});
    modelsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    modelsTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    modelsTable_->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    modelsTable_->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    backendChoice_ = new QComboBox;
    auto* import = new QPushButton("Import model or bundle");
    import->setShortcut(QKeySequence("Ctrl+I"));
    auto* remove = new QPushButton("Remove from library");
    loadButton_ = new QPushButton("Load");
    unloadButton_ = new QPushButton("Unload");
    unloadButton_->setEnabled(false);
    auto* controls = new QHBoxLayout;
    controls->addWidget(import);
    controls->addWidget(remove);
    controls->addStretch();
    controls->addWidget(new QLabel("Backend"));
    controls->addWidget(backendChoice_);
    controls->addWidget(unloadButton_);
    controls->addWidget(loadButton_);
    layout->addWidget(title);
    layout->addWidget(modelsTable_, 1);
    layout->addLayout(controls);
    connect(import, &QPushButton::clicked, this, [this] { importModel(); });
    connect(remove, &QPushButton::clicked, this, [this] {
        if (const auto* model = selectedModel()) {
            if (model->id == loadedModelId_) unloadModel();
            catalog_.remove(model->id);
            refreshModels();
        }
    });
    connect(loadButton_, &QPushButton::clicked, this, [this] { loadSelectedModel(); });
    connect(unloadButton_, &QPushButton::clicked, this, [this] { unloadModel(); });
    connect(modelsTable_, &QTableWidget::itemSelectionChanged, this, [this] {
        backendChoice_->clear();
        if (const auto* model = selectedModel()) {
            const auto compatible = BackendRegistry::instance().compatible(*model);
            if (!compatible.empty()) backendChoice_->addItem("Auto");
            for (const auto& id : compatible) backendChoice_->addItem(QString::fromStdString(id));
        }
        loadButton_->setEnabled(backendChoice_->count() > 0 && !activeTask_);
    });
    return page;
}

QWidget* MainWindow::createHardwarePage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Hardware and Runtime Verification");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    auto* refresh = new QPushButton("Refresh");
    hardwareText_ = new QTextBrowser;
    layout->addWidget(title);
    layout->addWidget(refresh);
    layout->addWidget(hardwareText_, 1);
    connect(refresh, &QPushButton::clicked, this, [this] { refreshHardware(); });
    return page;
}

QWidget* MainWindow::createBenchmarksPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Real Inference Benchmarks");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    auto* run = new QPushButton("Benchmark loaded text model");
    benchmarkText_ = new QTextBrowser;
    layout->addWidget(title);
    layout->addWidget(run);
    layout->addWidget(benchmarkText_, 1);
    connect(run, &QPushButton::clicked, this, [this] {
        if (!backend_ || backend_->backendInfo().capabilities != std::vector<Capability>{Capability::TextGeneration}) {
            QMessageBox::information(this, "Text model required", "Load a text-generation backend first.");
            return;
        }
        InferenceRequest request;
        request.prompt = "Explain in two sentences why local inference protects privacy.";
        request.numeric = {{"max_tokens", 96}, {"temperature", 0.0}};
        startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
            if (!result) benchmarkText_->setText(QString::fromStdString(result.error().message));
            else {
                const auto& stats = result.value().performance;
                const auto info = backend_->backendInfo();
                benchmarks_.record({loadedModelId_, info.id, info.provider, info.device, stats});
                const auto saved = benchmarks_.save();
                const auto storage = saved ? QString("\nResult stored for future Auto selection.")
                                           : "\n" + QString::fromStdString(saved.error().message);
                benchmarkText_->setText(QString("Load: %1 ms\nFirst output: %2 ms\nTotal: %3 ms\nGeneration: %4 tokens/s")
                    .arg(stats.loadMilliseconds, 0, 'f', 2).arg(stats.firstOutputMilliseconds, 0, 'f', 2)
                    .arg(stats.totalMilliseconds, 0, 'f', 2).arg(stats.outputUnitsPerSecond, 0, 'f', 2) + storage);
            }
        });
    });
    return page;
}

QWidget* MainWindow::createSettingsPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Settings");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    auto* theme = new QComboBox;
    theme->addItems({"System", "Dark", "Light"});
    theme->setCurrentText(QString::fromStdString(settings_.get("theme", "System")));
    settingThreads_ = new QSpinBox;
    settingThreads_->setRange(1, 1024);
#ifdef Q_OS_ANDROID
    const int defaultThreads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1U, 4U));
#else
    const int defaultThreads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
#endif
    settingThreads_->setValue(settings_.getInt("threads", defaultThreads));
    settingContext_ = new QSpinBox;
    settingContext_->setRange(512, 1048576);
    settingContext_->setSingleStep(512);
    settingContext_->setValue(settings_.getInt("context",
#ifdef Q_OS_ANDROID
        2048
#else
        4096
#endif
    ));
    settingBatch_ = new QSpinBox;
    settingBatch_->setRange(32, 8192);
    settingBatch_->setSingleStep(32);
    settingBatch_->setValue(settings_.getInt("batch",
#ifdef Q_OS_ANDROID
        128
#else
        512
#endif
    ));
    settingGpuLayers_ = new QSpinBox;
    settingGpuLayers_->setRange(0, 999);
    settingGpuLayers_->setValue(settings_.getInt("gpu_layers", 999));
    settingDevice_ = new QComboBox;
    settingDevice_->addItems({"auto", "cpu"});
    const auto discovered = HardwareDiscovery::discover();
    const bool usableGpu = std::ranges::any_of(discovered.devices, [](const ExecutionDevice& device) {
        return device.type == AcceleratorType::Gpu && device.available &&
               device.runtime != "Linux DRM hardware discovery";
    });
    const bool usableNpu = std::ranges::any_of(discovered.devices, [](const ExecutionDevice& device) {
        return device.type == AcceleratorType::Npu && device.available;
    });
    if (usableGpu) settingDevice_->addItem("gpu");
    if (usableNpu) settingDevice_->addItem("npu");
    settingDevice_->setCurrentText(QString::fromStdString(settings_.get("device", "auto")));
    settingMemoryPolicy_ = new QComboBox;
    settingMemoryPolicy_->addItems({"Balanced", "Performance", "Low Memory"});
    settingMemoryPolicy_->setCurrentText(QString::fromStdString(settings_.get("memory_policy", "Balanced")));
    settingKvCache_ = new QComboBox;
    settingKvCache_->addItems({"f16", "f32", "q8_0", "q4_0"});
    settingKvCache_->setCurrentText(QString::fromStdString(settings_.get("kv_cache", "f16")));
    auto* save = new QPushButton("Save settings");
    auto* form = new QFormLayout;
    form->addRow("Theme", theme);
    form->addRow("CPU threads", settingThreads_);
    form->addRow("Context tokens", settingContext_);
    form->addRow("Batch tokens", settingBatch_);
    form->addRow("Maximum GPU layers", settingGpuLayers_);
    form->addRow("Execution device", settingDevice_);
    form->addRow("Memory policy", settingMemoryPolicy_);
    form->addRow("KV cache type", settingKvCache_);
    layout->addWidget(title);
    layout->addLayout(form);
    layout->addWidget(save);
    layout->addStretch();
    auto applyTheme = [](const QString& value) {
        if (value == "Dark") qApp->setStyleSheet("QWidget { background:#1e1f22; color:#e8e8e8; } QLineEdit,QPlainTextEdit,QTextBrowser,QTableWidget,QListWidget,QComboBox,QSpinBox { background:#292b30; } QPushButton { padding:6px 12px; background:#353841; border:1px solid #565a66; border-radius:4px; }");
        else if (value == "Light") qApp->setStyleSheet("QWidget { background:#f7f7f8; color:#1f2328; } QLineEdit,QPlainTextEdit,QTextBrowser,QTableWidget,QListWidget,QComboBox,QSpinBox { background:white; } QPushButton { padding:6px 12px; }");
        else qApp->setStyleSheet(QString{});
    };
    applyTheme(theme->currentText());
    connect(theme, &QComboBox::currentTextChanged, this, applyTheme);
    connect(save, &QPushButton::clicked, this, [this, theme] {
        settings_.set("theme", theme->currentText().toStdString());
        settings_.set("threads", std::to_string(settingThreads_->value()));
        settings_.set("context", std::to_string(settingContext_->value()));
        settings_.set("batch", std::to_string(settingBatch_->value()));
        settings_.set("gpu_layers", std::to_string(settingGpuLayers_->value()));
        settings_.set("device", settingDevice_->currentText().toStdString());
        settings_.set("memory_policy", settingMemoryPolicy_->currentText().toStdString());
        settings_.set("kv_cache", settingKvCache_->currentText().toStdString());
        const auto saved = settings_.save();
        status_->setText(saved ? "Settings saved" : QString::fromStdString(saved.error().message));
    });
    return page;
}

QWidget* MainWindow::createLogsPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    auto* title = new QLabel("Structured Logs");
    title->setStyleSheet("font-size: 24px; font-weight: 600");
    logs_ = new QPlainTextEdit;
    logs_->setReadOnly(true);
    layout->addWidget(title);
    layout->addWidget(logs_, 1);
    return page;
}

void MainWindow::refreshModels() {
    if (!modelsTable_) return;
    modelsTable_->setRowCount(static_cast<int>(catalog_.models().size()));
    for (int row = 0; row < static_cast<int>(catalog_.models().size()); ++row) {
        const auto& model = catalog_.models()[static_cast<std::size_t>(row)];
        QStringList capabilities;
        for (const auto capability : model.capabilities) capabilities << QString::fromStdString(toString(capability));
        const QString values[] = {QString::fromStdString(model.displayName), QString::fromStdString(toString(model.format)),
                                  bytesText(model.fileBytes), capabilities.isEmpty() ? "Adapter-specific" : capabilities.join(", "),
                                  QString::fromStdString(model.path.string())};
        for (int column = 0; column < 5; ++column) {
            auto* item = new QTableWidgetItem(values[column]);
            item->setData(Qt::UserRole, QString::fromStdString(model.id));
            modelsTable_->setItem(row, column, item);
        }
    }
}

void MainWindow::refreshHardware() {
    if (!hardwareText_) return;
    const auto snapshot = HardwareDiscovery::discover();
    QString text = QString("<h2>CPU</h2><p>%1<br>Architecture: %2<br>Logical cores: %3</p><h2>Memory</h2><p>Total: %4<br>Available: %5</p><h2>Execution devices and runtimes</h2><table cellspacing='8'>")
        .arg(QString::fromStdString(snapshot.cpu.model), QString::fromStdString(snapshot.cpu.architecture))
        .arg(snapshot.cpu.logicalCores).arg(bytesText(snapshot.memory.totalBytes), bytesText(snapshot.memory.availableBytes));
    for (const auto& device : snapshot.devices) {
        text += QString("<tr><td><b>%1</b></td><td>%2</td><td>%3</td><td>%4</td></tr>")
            .arg(QString::fromStdString(toString(device.type)), QString::fromStdString(device.name),
                 QString::fromStdString(device.runtime), QString::fromStdString(device.status));
    }
    hardwareText_->setHtml(text + "</table>");
}

void MainWindow::importModel() {
    const auto paths = QFileDialog::getOpenFileNames(this, "Import models", {},
        "Supported models (*.gguf *.safetensors *.onnx *.litert *.tflite *.pt *.bin);;All files (*)");
    if (!paths.isEmpty()) {
        QStringList failures;
        for (const auto& path : paths) {
            QString copyFailure;
            const auto localPath = persistentLocalPath(this, path, copyFailure);
            if (localPath.isEmpty()) {
                failures << path + ": " + copyFailure;
                continue;
            }
            auto result = catalog_.importModel(localPath.toStdString());
            if (!result) failures << path + ": " + QString::fromStdString(result.error().message);
            else Logger::instance().write(LogLevel::Info, LogCategory::Model, "Imported " + result.value().displayName);
        }
        refreshModels();
        if (!failures.isEmpty()) QMessageBox::critical(this, "Some imports failed", failures.join("\n"));
        return;
    }
    const auto directory = QFileDialog::getExistingDirectory(this, "Import offline audio model bundle");
    if (directory.isEmpty()) return;
    QString copyFailure;
    const auto localDirectory = persistentBundlePath(this, directory, copyFailure);
    if (localDirectory.isEmpty()) {
        QMessageBox::critical(this, "Bundle import failed", copyFailure);
        return;
    }
    auto result = catalog_.importModel(localDirectory.toStdString());
    if (!result) QMessageBox::critical(this, "Import failed", QString::fromStdString(result.error().message));
    else refreshModels();
}

void MainWindow::chooseVisionImage() {
    const auto path = QFileDialog::getOpenFileName(this, "Open still image", {},
        "Still images (*.png *.jpg *.jpeg *.bmp *.webp)");
    if (path.isEmpty()) return;
    QImage image(path);
    if (image.isNull()) {
        QMessageBox::critical(this, "Image rejected", "Qt could not decode this file as a still image.");
        return;
    }
    visionImage_ = image.convertToFormat(QImage::Format_RGB888);
    visionPathLabel_->setText(path);
    visionPreview_->setPixmap(QPixmap::fromImage(visionImage_).scaled(visionPreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::useClipboardImage() {
    const auto image = QApplication::clipboard()->image();
    if (image.isNull()) {
        QMessageBox::information(this, "Clipboard", "The clipboard does not contain a still image.");
        return;
    }
    visionImage_ = image.convertToFormat(QImage::Format_RGB888);
    visionPathLabel_->setText("Image pasted from clipboard");
    visionPreview_->setPixmap(QPixmap::fromImage(visionImage_).scaled(visionPreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::runVision() {
    if (!backend_) {
        QMessageBox::information(this, "Model required", "Load a compatible VLM or image-classification model first.");
        return;
    }
    if (visionImage_.isNull()) {
        QMessageBox::information(this, "Image required", "Open or paste a still image first.");
        return;
    }
    const auto info = backend_->backendInfo();
    const bool understanding = std::ranges::find(info.capabilities, Capability::ImageUnderstanding) != info.capabilities.end();
    const bool classification = std::ranges::find(info.capabilities, Capability::ImageClassification) != info.capabilities.end();
    if (!understanding && !classification) {
        QMessageBox::information(this, "Incompatible model", "The loaded model does not expose a verified still-image capability.");
        return;
    }
    InferenceRequest request;
    request.kind = understanding ? TaskKind::ImageUnderstanding : TaskKind::ImageClassification;
    request.prompt = visionPrompt_->toPlainText().toStdString();
    request.imageWidth = visionImage_.width();
    request.imageHeight = visionImage_.height();
    request.imageChannels = 3;
    request.imagePixels.resize(static_cast<std::size_t>(visionImage_.width() * visionImage_.height() * 3));
    for (int y = 0; y < visionImage_.height(); ++y) {
        std::memcpy(request.imagePixels.data() + static_cast<std::size_t>(y * visionImage_.width() * 3),
                    visionImage_.constScanLine(y), static_cast<std::size_t>(visionImage_.width() * 3));
    }
    visionResponse_->clear();
    request.numeric = {{"max_tokens", 256}, {"temperature", 0.2}, {"top_k", 40}, {"top_p", 0.95}};
    request.onText = [this](std::string_view value) {
        const auto text = QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
        QMetaObject::invokeMethod(this, [this, text] {
            visionResponse_->moveCursor(QTextCursor::End);
            visionResponse_->insertPlainText(text);
        });
    };
    startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
        if (!result) QMessageBox::critical(this, "Vision inference failed", QString::fromStdString(result.error().message));
        else if (visionResponse_->toPlainText().isEmpty()) visionResponse_->setPlainText(QString::fromStdString(result.value().text));
    });
}

const ModelDescriptor* MainWindow::selectedModel() const {
    if (!modelsTable_ || modelsTable_->currentRow() < 0) return nullptr;
    const auto id = modelsTable_->item(modelsTable_->currentRow(), 0)->data(Qt::UserRole).toString().toStdString();
    const auto found = std::ranges::find(catalog_.models(), id, &ModelDescriptor::id);
    return found == catalog_.models().end() ? nullptr : &*found;
}

void MainWindow::loadSelectedModel() {
    const auto* model = selectedModel();
    if (!model || backendChoice_->currentText().isEmpty() || loadFuture_.valid()) return;
    const auto memory = HardwareDiscovery::discover().memory;
    if (model->estimatedRamBytes && memory.availableBytes > 0 && *model->estimatedRamBytes > memory.availableBytes) {
        QMessageBox::warning(this, "Memory estimate exceeds availability",
            QString("Estimated model RAM is %1, but only %2 is currently available. Choose a smaller model or free memory before loading.")
                .arg(bytesText(*model->estimatedRamBytes), bytesText(memory.availableBytes)));
        return;
    }
    const auto compatible = BackendRegistry::instance().compatible(*model);
    std::string selectedBackend = backendChoice_->currentText().toStdString();
    if (selectedBackend == "Auto") {
        selectedBackend = benchmarks_.fastestBackend(model->id, compatible).value_or(compatible.front());
    }
    loadingBackend_ = BackendRegistry::instance().create(selectedBackend);
    if (!loadingBackend_) return;
    const auto descriptor = *model;
    LoadOptions options;
    options.threads = settingThreads_ ? settingThreads_->value() : static_cast<int>(std::thread::hardware_concurrency());
    options.contextSize = settingContext_ ? settingContext_->value() : 4096;
    options.batchSize = settingBatch_ ? settingBatch_->value() : 512;
    options.gpuLayers = settingGpuLayers_ ? settingGpuLayers_->value() : 999;
    options.deviceId = settingDevice_ ? settingDevice_->currentText().toStdString() : "auto";
    options.kvCacheType = settingKvCache_ ? settingKvCache_->currentText().toStdString() : "f16";
    const auto memoryPolicy = settingMemoryPolicy_ ? settingMemoryPolicy_->currentText() : QString("Balanced");
    options.mmap = true;
    options.mlock = memoryPolicy == "Performance";
    if (memoryPolicy == "Low Memory") {
        options.contextSize = std::min(options.contextSize, 2048);
        options.batchSize = std::min(options.batchSize, 128);
    }
#ifdef Q_OS_ANDROID
    constexpr std::uint64_t sixGiB = 6ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t threeGiB = 3ULL * 1024ULL * 1024ULL * 1024ULL;
    if ((memory.totalBytes > 0 && memory.totalBytes < sixGiB) ||
        (memory.availableBytes > 0 && memory.availableBytes < threeGiB)) {
        options.contextSize = std::min(options.contextSize, 2048);
        options.batchSize = std::min(options.batchSize, 128);
        if (options.gpuLayers > 24) options.gpuLayers = 24;
        Logger::instance().write(LogLevel::Info, LogCategory::Memory,
            "Android low-memory guard reduced context, batch, and initial GPU offload");
    }
#endif
    loadingModelId_ = model->id;
    auto* pointer = loadingBackend_.get();
    loadFuture_ = std::async(std::launch::async, [pointer, descriptor, options] { return pointer->load(descriptor, options); });
    setBusy(true, "Loading model…");
    loadTimer_->start();
}

void MainWindow::pollLoad() {
    if (!loadFuture_.valid() || loadFuture_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    auto result = loadFuture_.get();
    loadTimer_->stop();
    if (!result) {
        QMessageBox::critical(this, "Model load failed", QString::fromStdString(result.error().message));
        loadingBackend_.reset();
        loadingModelId_.clear();
        setBusy(false, "Model load failed");
        return;
    }
    if (backend_) backend_->unload();
    backend_ = std::move(loadingBackend_);
    loadedModelId_ = std::move(loadingModelId_);
    unloadButton_->setEnabled(true);
    setBusy(false, QString("Loaded with %1 on %2").arg(QString::fromStdString(result.value().runtime),
                                                        QString::fromStdString(result.value().device)));
}

void MainWindow::unloadModel() {
    if (activeTask_) return;
    if (backend_) backend_->unload();
    backend_.reset();
    loadedModelId_.clear();
    unloadButton_->setEnabled(false);
    setBusy(false, "Model unloaded");
}

void MainWindow::startInference(InferenceRequest request,
                                std::function<void(const Result<InferenceOutput>&)> completion) {
    if (!backend_ || activeTask_) return;
    auto* pointer = backend_.get();
    activeTask_.emplace(scheduler_.submit(
        [pointer, request = std::move(request)](const CancellationFlag& cancellation) mutable {
            return pointer->infer(request, cancellation);
        }));
    completion_ = std::move(completion);
    setBusy(true, "Inference running…");
    taskTimer_->start();
}

void MainWindow::pollInference() {
    if (!activeTask_ || activeTask_->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    auto result = activeTask_->future.get();
    activeTask_.reset();
    taskTimer_->stop();
    setBusy(false, result ? "Inference completed" : QString::fromStdString(result.error().message));
    if (completion_) completion_(result);
    completion_ = {};
}

void MainWindow::setBusy(bool busy, const QString& status) {
    chatSend_->setEnabled(!busy);
    stopButton_->setEnabled(busy && activeTask_.has_value());
    loadButton_->setEnabled(!busy && backendChoice_ && backendChoice_->count() > 0);
    if (conversations_) conversations_->setEnabled(!busy);
    if (!status.isEmpty()) status_->setText(status);
}

void MainWindow::startRecording() {
#if defined(Q_OS_ANDROID) && QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    QMicrophonePermission permission;
    const auto status = qApp->checkPermission(permission);
    if (status == Qt::PermissionStatus::Undetermined) {
        qApp->requestPermission(permission, this, [this](const QPermission& result) {
            if (result.status() == Qt::PermissionStatus::Granted) startRecording();
            else QMessageBox::warning(this, "Microphone permission", "Microphone access was denied by Android.");
        });
        return;
    }
    if (status != Qt::PermissionStatus::Granted) {
        QMessageBox::warning(this, "Microphone permission", "Enable microphone permission in Android settings to record audio.");
        return;
    }
#endif
    const auto device = QMediaDevices::defaultAudioInput();
    microphoneFormat_.setSampleRate(16000);
    microphoneFormat_.setChannelCount(1);
    microphoneFormat_.setSampleFormat(QAudioFormat::Int16);
    if (!device.isFormatSupported(microphoneFormat_)) {
        QMessageBox::warning(this, "Microphone format", "The default input device does not support 16 kHz mono PCM recording.");
        return;
    }
    microphoneBytes_.clear();
    microphoneBuffer_ = std::make_unique<QBuffer>(&microphoneBytes_);
    microphoneBuffer_->open(QIODevice::WriteOnly);
    audioSource_ = std::make_unique<QAudioSource>(device, microphoneFormat_);
    audioSource_->start(microphoneBuffer_.get());
    audioFileLabel_->setText("Recording microphone at 16 kHz mono…");
}

void MainWindow::stopRecording() {
    if (!audioSource_) return;
    audioSource_->stop();
    microphoneBuffer_->close();
    audioSource_.reset();
    microphoneBuffer_.reset();
    const auto count = microphoneBytes_.size() / static_cast<qsizetype>(sizeof(std::int16_t));
    recordedSamples_.resize(static_cast<std::size_t>(count));
    for (qsizetype index = 0; index < count; ++index) {
        std::int16_t value{};
        std::memcpy(&value, microphoneBytes_.constData() + index * 2, 2);
        recordedSamples_[static_cast<std::size_t>(index)] = value / 32768.0F;
    }
    audioFile_.clear();
    audioFileLabel_->setText(QString("Recorded %1 seconds").arg(recordedSamples_.size() / 16000.0, 0, 'f', 1));
}

void MainWindow::transcribeAudio() {
    if (!backend_ || backend_->backendInfo().id != "whisper.cpp") {
        QMessageBox::information(this, "Model required", "Load a whisper.cpp speech-to-text model first.");
        return;
    }
    InferenceRequest request;
    request.kind = TaskKind::SpeechToText;
    request.inputPath = audioFile_;
    request.tensor = recordedSamples_;
    request.onText = [this](std::string_view value) {
        const auto text = QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
        QMetaObject::invokeMethod(this, [this, text] { transcription_->insertPlainText(text); });
    };
    transcription_->clear();
    startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
        if (!result) QMessageBox::critical(this, "Transcription failed", QString::fromStdString(result.error().message));
    });
}

void MainWindow::synthesizeSpeech() {
    if (!backend_ || backend_->backendInfo().id != "sherpa-onnx-tts") {
        QMessageBox::information(this, "Model required", "Load a supported sherpa-onnx TTS bundle first.");
        return;
    }
    InferenceRequest request;
    request.kind = TaskKind::TextToSpeech;
    request.prompt = audioText_->text().toStdString();
    request.numeric = {{"speaker", 0}, {"speed", 1.0}, {"silence", 0.2}};
    startInference(std::move(request), [this](const Result<InferenceOutput>& result) {
        if (!result) QMessageBox::critical(this, "Speech synthesis failed", QString::fromStdString(result.error().message));
        else {
            playAudio(result.value());
            const auto destination = QFileDialog::getSaveFileName(this, "Save generated speech", {}, "WAVE audio (*.wav)");
            if (!destination.isEmpty()) {
                const auto saved = writeWaveFile(destination.toStdString(), result.value().audioSamples,
                                                 result.value().audioSampleRate, 1);
                if (!saved) QMessageBox::warning(this, "Save failed", QString::fromStdString(saved.error().message));
            }
        }
    });
}

void MainWindow::playAudio(const InferenceOutput& output) {
    QAudioFormat format;
    format.setSampleRate(output.audioSampleRate);
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);
    const auto device = QMediaDevices::defaultAudioOutput();
    if (!device.isFormatSupported(format)) {
        QMessageBox::warning(this, "Audio output", "Default output device does not support the generated sample format.");
        return;
    }
    playbackBytes_.resize(static_cast<qsizetype>(output.audioSamples.size() * 2));
    for (std::size_t index = 0; index < output.audioSamples.size(); ++index) {
        const auto value = static_cast<std::int16_t>(std::clamp(output.audioSamples[index], -1.0F, 1.0F) * 32767.0F);
        std::memcpy(playbackBytes_.data() + static_cast<qsizetype>(index * 2), &value, 2);
    }
    playbackBuffer_ = std::make_unique<QBuffer>(&playbackBytes_);
    playbackBuffer_->open(QIODevice::ReadOnly);
    audioSink_ = std::make_unique<QAudioSink>(device, format);
    audioSink_->start(playbackBuffer_.get());
}

} // namespace localai::ui
