from PyQt5 import uic, QtWidgets, QtCore, QtGui
from lib.modules import evolution
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

    #best_binary, best_real, fxs, best_fx = evolution(range_a, range_b, precision, tau, generations_number)
    best_reals, best_binary, best_fxs, local_fxs = evolution(range_a, range_b, precision, generations_number)
    app.restoreOverrideCursor()
    
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
            fxs.append(i + 0.5, local_fxs[i][0])
            pen = fxs.pen()
            fxs.setMarkerSize(3)
            pen.setBrush(QtGui.QColor(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            fxs.setPen(pen)

        else:
            fxs = QLineSeries()
            tick = 1 / (len(local_fxs[i]) - 1)
            for j in range(len(local_fxs[i])):
                fxs.append(i + j * tick, local_fxs[i][j])
            pen = fxs.pen()
            pen.setWidth(1)
            pen.setBrush(QtGui.QColor(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            fxs.setPen(pen)
            
            
        
        bests.append(i+1, best_fxs[i])
        chart.addSeries(fxs)

    chart.addSeries(bests)

    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisY().setRange(-2,2)
    chart.axisX().setTickCount(11)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisX().setLabelFormat("%i")
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget.setChart(chart)

    '''
    
    series = QLineSeries()
    
    points = QScatterSeries()

    pen = series.pen()
    pen.setWidth(1)
    pen.setBrush(QtGui.QColor(114, 137, 218))
    series.setPen(pen)

    

    pen_points = points.pen()
    pen_points.setBrush(QtGui.QColor("white"))
    points.setMarkerSize(8)
    points.setColor(QtGui.QColor("red"))
    points.setPen(pen_points)

    for i in range(1, generations_number+1):
        if fxs[i-1] == best_fx[generations_number-1]:
            points.append(i, fxs[i-1])
        series.append(i, fxs[i-1])
        bests.append(i, best_fx[i-1])
    
    chart.addSeries(series)
    chart.addSeries(bests)
    chart.addSeries(points)
    
    

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

def test_generations():
    range_a = float(str(form.input_a_test.text()))
    range_b = float(str(form.input_b_test.text()))
    precision = int(str(form.input_d_test.text()))
    tau = float(str(form.input_tau_test.text()))

    app.setOverrideCursor(QtCore.Qt.WaitCursor)
    start = time()
    result  = test_generation(range_a, range_b, precision, tau)

    chart = QChart()
    series_bests = QLineSeries()

    form.test_table.setRowCount(0)

    for i in range(0, 40):
        series_bests.append(result[i,0], result[i,1])

    chart.addSeries(series_bests)
    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisX().setTickCount(9)
    chart.axisY().setRange(-2, 2)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget_test.setChart(chart)

    result_sort = result[numpy.argsort(-result[:, 1])] 

    for i in range(0, 40):
        form.test_table.insertRow(i)
        form.test_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(round(result_sort[i,0],2))))
        form.test_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(tau)))
        form.test_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(result_sort[i,1])))

    app.restoreOverrideCursor()

def test_taus():
    range_a = float(str(form.input_a_test.text()))
    range_b = float(str(form.input_b_test.text()))
    precision = int(str(form.input_d_test.text()))
    generations_number = int(str(form.input_t_test.text()))

    app.setOverrideCursor(QtCore.Qt.WaitCursor)
    start = time()
    result  = test_tau(range_a, range_b, precision, generations_number)
    
    chart = QChart()
    series_bests = QLineSeries()

    form.test_table.setRowCount(0)

    for i in range(0, 50):
        series_bests.append(result[i,0], result[i,1])

    chart.addSeries(series_bests)
    chart.setBackgroundBrush(QtGui.QColor(41, 43, 47))
    chart.createDefaultAxes()
    chart.legend().hide()
    chart.setContentsMargins(-10, -10, -10, -10)
    chart.layout().setContentsMargins(0, 0, 0, 0)
    chart.axisX().setTickCount(8)
    chart.axisY().setRange(-2, 2)
    chart.axisX().setLabelsColor(QtGui.QColor("white"))
    chart.axisY().setLabelsColor(QtGui.QColor("white"))
    form.widget_test.setChart(chart)

    result_sort = result[numpy.argsort(-result[:, 1])] 

    for i in range(0, 50):
        form.test_table.insertRow(i)
        form.test_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(round(result_sort[i,0],2))))
        form.test_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(generations_number)))
        form.test_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(result_sort[i,1])))
    
    app.restoreOverrideCursor()

'''
form.button_start.clicked.connect(run_evolution)
#form.button_test_generations.clicked.connect(test_generations)
#form.button_test_tau.clicked.connect(test_taus)
app.exec()