#include <QApplication>
#include <seatui/launcher/login_window.hpp>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    LoginWindow w;
    w.show();
    return app.exec();
}
