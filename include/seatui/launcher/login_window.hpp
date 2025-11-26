#pragma once
#include <QMainWindow>

class QLineEdit;
class QPushButton;
class QLabel;
class QStackedWidget;

class LoginWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit LoginWindow(QWidget* parent = nullptr);

private slots:
    void onLoginClicked();

private:
    QWidget* buildLoginPage();
    QWidget* buildRolePage();

private:
    QStackedWidget* stacked_;
    // 登录页控件
    QLineEdit* user_;
    QLineEdit* pass_;
    QLabel*    msg_;
};



