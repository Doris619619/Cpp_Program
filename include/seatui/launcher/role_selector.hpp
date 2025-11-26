#pragma once
#include <QWidget>

class QPushButton;

class RoleSelector : public QWidget {
    Q_OBJECT
public:
    explicit RoleSelector(QWidget* parent = nullptr);

signals:
    void openStudent();
    void openAdmin();
};




