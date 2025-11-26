#include <seatui/widgets/card_dialog.hpp>

#include <QFrame>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QGraphicsDropShadowEffect>
#include <QMouseEvent>

CardDialog::CardDialog(const QString& title,
                       const QString& text,
                       QWidget* parent)
    : QDialog(parent)
{
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    setModal(true);

    auto outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);

    auto card = new QFrame(this);
    card->setObjectName("card");
    auto sh = new QGraphicsDropShadowEffect(card);
    sh->setBlurRadius(24);
    sh->setOffset(0, 6);
    sh->setColor(QColor(0, 0, 0, 40));
    card->setGraphicsEffect(sh);

    auto v = new QVBoxLayout(card);
    v->setContentsMargins(24, 22, 24, 18);
    v->setSpacing(12);

    auto head = new QHBoxLayout();
    head->setSpacing(12);

    auto icon = new QLabel(u8"⚠️", card);
    icon->setFixedWidth(28);
    icon->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    auto textCol = new QVBoxLayout();
    auto t = new QLabel(title, card);
    t->setStyleSheet("font-size: 14pt; font-weight: 600;");
    auto m = new QLabel(text, card);
    m->setWordWrap(true);

    textCol->addWidget(t);
    textCol->addWidget(m);

    head->addWidget(icon);
    head->addLayout(textCol, 1);

    auto btns = new QHBoxLayout();
    btns->addStretch();
    auto ok = new QPushButton(u8"知道了", card);
    ok->setProperty("type", "primary");
    ok->setMinimumHeight(32);
    btns->addWidget(ok);

    v->addLayout(head);
    v->addSpacing(6);
    v->addLayout(btns);

    outer->addWidget(card);

    connect(ok, &QPushButton::clicked, this, &QDialog::accept);
    resize(420, sizeHint().height());
}

void CardDialog::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton)
        dragOffset_ = e->globalPosition().toPoint() - frameGeometry().topLeft();
    QDialog::mousePressEvent(e);
}

void CardDialog::mouseMoveEvent(QMouseEvent* e) {
    if (e->buttons() & Qt::LeftButton)
        move(e->globalPosition().toPoint() - dragOffset_);
    QDialog::mouseMoveEvent(e);
}
