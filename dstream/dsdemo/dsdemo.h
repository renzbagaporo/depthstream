#ifndef DSDEMO_H
#define DSDEMO_H

#include <qmainwindow.h>
#include <qmessagebox.h>
#include <qprocess.h>
#include "ui_dsdemo.h"

class dsdemo : public QMainWindow
{
	Q_OBJECT

public:
	dsdemo(QWidget *parent = 0);
	~dsdemo();

private:
	Ui::dsdemoClass ui;

	void disable_buttons();
	void enable_buttons();

private Q_SLOTS:
	void on_depthmap_demo_button_clicked();
	void on_pointcloud_demo_button_clicked();
	void on_tracking_demo_button_clicked();

};

#endif // DSDEMO_H
