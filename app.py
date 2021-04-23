from PyQt5 import uic, QtWidgets, QtCore, QtGui
from lib.modules import evolution, test_generation
from time import time
import numpy
from PyQt5.QtChart import QChart, QLineSeries, QScatterSeries
from lib.models import Test
import csv
import random

Form, Window = uic.loadUiType("gui.ui")
app = QtWidgets.QApplication([])
window = Window()
form = Form()
form.setupUi(window)
chart = QChart()
chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
form.widget.setChart(chart)
form.widget_test.setChart(chart)
form.tabWidget.setTabText(0, "Algorytm")
form.tabWidget.setTabText(1, "Testy")
window.show()

def run_evolution():
    range_a = float(str(form.input_a.text()))
    range_b = float(str(form.input_b.text()))
    precision = int(str(form.input_d.text()))
    generations_number = int(str(form.input_t.text()))

    app.setOverrideCursor(QtCore.Qt.WaitCursor)

    best_reals, best_binary, best_fxs, local_fxs, _ = evolution(range_a, range_b, precision, generations_number)

    form.best_table.item(1,0).setText(str(best_reals[generations_number-1]))
    form.best_table.item(1,1).setText(''.join(map(str, best_binary[generations_number-1])))
    form.best_table.item(1,2).setText(str(best_fxs[generations_number-1]))
    
    chart = QChart()
    bests = QLineSeries() 

    pen_best = bests.pen()
    pen_best.setWidth(1)
    pen_best.setBrush(QtGui.QColor("red"))
    bests.setPen(pen_best)

    for i in range(0, generations_number):
        if len(local_fxs[i]) - 1 == 0:
            fxs = QScatterSeries()
            fxs.append(i + 0.99, local_fxs[i][0])
            pen = fxs.pen()
            color = QtGui.QColor(random.randint(50,255), random.randint(50,255), random.randint(50,255))
            fxs.setColor(color)
            pen.setColor(color)
            fxs.setPen(pen)
            fxs.setMarkerSize(5)

        else:
            fxs = QLineSeries()
            tick = 1 / (len(local_fxs[i]) - 1)
            for j in range(len(local_fxs[i])):
                fxs.append(i + j * tick, local_fxs[i][j])
            pen = fxs.pen()
            pen.setWidth(1)
            pen.setBrush(QtGui.QColor(random.randint(50,255), random.randint(50,255), random.randint(50,255)))
            fxs.setPen(pen)
            
        bests.append(i+1, best_fxs[i])
        chart.addSeries(fxs)

    chart.addSeries(bests)

    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisX().setTickCount(11)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisX().setGridLineColor(QtGui.QColor("grey"))
    chart.axisX().setLabelFormat("%i")
    chart.axisY().setRange(-2,2)
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    chart.axisY().setGridLineColor(QtGui.QColor("grey"))
    form.widget.setChart(chart)

    app.restoreOverrideCursor()

    '''
    
    with open('best_history.csv', 'w', newline='', encoding='utf8') as history_csvfile:
        history_writer = csv.writer(
            history_csvfile, delimiter=';', dialect=csv.excel)
        history_writer.writerow(['Parametry'])
        history_writer.writerow(['Precyzja: 10^-%d' % precision])
        history_writer.writerow(['Tau: %d' % tau])
        history_writer.writerow(['Pokolenia: %d' % generations_number])
        history_writer.writerow(['', 'vbest', 'vbin', 'f(vbest)'])
        index = 1
        for generation in numpy.arange(generations_number):
            history_writer.writerow([index,  best_real[generation], best_binary[generation], best_fx[generation]])
            index += 1

    app.restoreOverrideCursor()
    '''

def test_generations():
    range_a = float(str(form.input_a_test.text()))
    range_b = float(str(form.input_b_test.text()))
    precision = int(str(form.input_d_test.text()))
    generations = int(str(form.input_generations_test.text()))

    app.setOverrideCursor(QtCore.Qt.WaitCursor)
    start = time()
    result  = test_generation(range_a, range_b, precision, generations)
    app.restoreOverrideCursor()

    chart = QChart()
    series = QLineSeries()

    form.test_table.setRowCount(0)

    form.test_table.insertRow(0)
    item = QtWidgets.QTableWidgetItem("iteracje")
    item.setTextAlignment(QtCore.Qt.AlignCenter)
    form.test_table.setItem(0, 0, item)

    item = QtWidgets.QTableWidgetItem("wystąpienia")
    item.setTextAlignment(QtCore.Qt.AlignCenter)
    form.test_table.setItem(0, 1, item)

    item = QtWidgets.QTableWidgetItem("%")
    item.setTextAlignment(QtCore.Qt.AlignCenter)
    form.test_table.setItem(0, 2, item)

    for i in range(0, generations):
        percent = sum(result[:i+1])/10*100
        series.append(i+1, percent)

        form.test_table.insertRow(i+1)
        form.test_table.setItem(i+1, 0, QtWidgets.QTableWidgetItem(str(i+1)))
        form.test_table.setItem(i+1, 1, QtWidgets.QTableWidgetItem(str(result[i])))
        form.test_table.setItem(i+1, 2, QtWidgets.QTableWidgetItem(str(round(percent, 2))))
 
    chart.addSeries(series)

    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisX().setTickCount(10)
    chart.axisY().setRange(0, 100)
    chart.axisY().setTickCount(11)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget_test.setChart(chart)


form.button_start.clicked.connect(run_evolution)
form.button_test_generations.clicked.connect(test_generations)
app.exec()