#include "dsdemo.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	dsdemo w;
	w.show();
	return a.exec();
}
