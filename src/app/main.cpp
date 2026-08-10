#include "../ui/main_window.hpp"

#include <QApplication>
#include <QCoreApplication>

int main(int argc, char** argv) {
    QApplication application(argc, argv);
    QCoreApplication::setApplicationName("Universal Local AI");
    QCoreApplication::setOrganizationName("Universal Local AI");
    QCoreApplication::setApplicationVersion("0.3.0");
    localai::ui::MainWindow window;
    window.show();
    return application.exec();
}
