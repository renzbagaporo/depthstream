/********************************************************************************
** Form generated from reading UI file 'dsdemo.ui'
**
** Created by: Qt User Interface Compiler version 5.5.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DSDEMO_H
#define UI_DSDEMO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_dsdemoClass
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QLabel *depthstream_label;
    QPushButton *depthmap_demo_button;
    QPushButton *tracking_demo_button;
    QPushButton *pointcloud_demo_button;

    void setupUi(QMainWindow *dsdemoClass)
    {
        if (dsdemoClass->objectName().isEmpty())
            dsdemoClass->setObjectName(QStringLiteral("dsdemoClass"));
        dsdemoClass->resize(387, 230);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(dsdemoClass->sizePolicy().hasHeightForWidth());
        dsdemoClass->setSizePolicy(sizePolicy);
        QIcon icon;
        icon.addFile(QStringLiteral("D:/Scratch/stereo-camera_318-100805.jpg"), QSize(), QIcon::Normal, QIcon::Off);
        dsdemoClass->setWindowIcon(icon);
        dsdemoClass->setAutoFillBackground(false);
        centralWidget = new QWidget(dsdemoClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        depthstream_label = new QLabel(centralWidget);
        depthstream_label->setObjectName(QStringLiteral("depthstream_label"));
        depthstream_label->setPixmap(QPixmap(QString::fromUtf8("D:/Dropbox/Documents/Thesis/label.png")));
        depthstream_label->setScaledContents(true);

        verticalLayout->addWidget(depthstream_label);

        depthmap_demo_button = new QPushButton(centralWidget);
        depthmap_demo_button->setObjectName(QStringLiteral("depthmap_demo_button"));
        QFont font;
        font.setFamily(QStringLiteral("Segoe UI"));
        font.setPointSize(11);
        depthmap_demo_button->setFont(font);

        verticalLayout->addWidget(depthmap_demo_button);

        tracking_demo_button = new QPushButton(centralWidget);
        tracking_demo_button->setObjectName(QStringLiteral("tracking_demo_button"));
        QFont font1;
        font1.setFamily(QStringLiteral("Segoe UI"));
        font1.setPointSize(12);
        tracking_demo_button->setFont(font1);

        verticalLayout->addWidget(tracking_demo_button);

        pointcloud_demo_button = new QPushButton(centralWidget);
        pointcloud_demo_button->setObjectName(QStringLiteral("pointcloud_demo_button"));
        pointcloud_demo_button->setFont(font);

        verticalLayout->addWidget(pointcloud_demo_button);

        dsdemoClass->setCentralWidget(centralWidget);

        retranslateUi(dsdemoClass);

        QMetaObject::connectSlotsByName(dsdemoClass);
    } // setupUi

    void retranslateUi(QMainWindow *dsdemoClass)
    {
        dsdemoClass->setWindowTitle(QApplication::translate("dsdemoClass", "Depthstream Demo", 0));
        depthstream_label->setText(QString());
        depthmap_demo_button->setText(QApplication::translate("dsdemoClass", "Depth Map Demo", 0));
        tracking_demo_button->setText(QApplication::translate("dsdemoClass", "Tracking Demo", 0));
        pointcloud_demo_button->setText(QApplication::translate("dsdemoClass", "PointCloud Demo", 0));
    } // retranslateUi

};

namespace Ui {
    class dsdemoClass: public Ui_dsdemoClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DSDEMO_H
