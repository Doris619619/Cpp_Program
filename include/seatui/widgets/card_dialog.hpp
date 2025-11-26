#pragma once
#include <QDialog>
#include <QPoint>

class CardDialog : public QDialog {
    Q_OBJECT
public:
    explicit CardDialog(const QString& title,
                        const QString& text,
                        QWidget* parent = nullptr);

protected:
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;

private:
    QPoint dragOffset_;
};
