# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Izana Alcalde (izana.alcalde@alumnos.upm.es)
# *
# * Universidad Politécnica de Madrid
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
from pwchem.wizards import VariableWizard
from reinvent.protocols import ReinventStagedLearning
import ast


class AddStageWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol

        if getattr(protocol, inputParam[0]).get() != '':
            index = int(getattr(protocol, inputParam[0]).get())

        msjDic = {
            'termination': "simple",
            'max_score': protocol.maxScore.get(),
            'min_steps': protocol.minSteps.get(),
            'max_steps': protocol.maxSteps.get(),
            'scoring': {'type': protocol.getEnumText('scoreFunct'),
                        'component': []}
        }

        currentText = getattr(protocol, outputParam[0]).get() or ''
        workSteps = [line for line in currentText.strip().split('\n') if line.strip()]

        if index > len(workSteps) or index <= 0:
            newText = currentText.strip() + '\n' + str(msjDic) + '\n'
        else:
            workSteps.insert(index - 1, str(msjDic))
            newText = '\n'.join(workSteps) + '\n'

        form.setVar(outputParam[0], newText.lstrip())

        newSummary = protocol._updateSummary()
        form.setVar('summarySteps', newSummary)


class AddScoringComponentWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol

        rawIndex = getattr(protocol, inputParam[0]).get()
        targetIdx = int(rawIndex) - 1 if rawIndex and str(rawIndex).strip() != '' else 0
        compName = protocol.getEnumText('compType')
        endpointData = {
            'name': compName,
            'weight': protocol.weight.get()
        }

        if protocol.trans.get() is True:
            transTypeIndex = protocol.transFunc.get()
            endpointData['transform.type'] = protocol.getEnumText('transFunc')

            if transTypeIndex in [0, 1, 2, 4, 5]:
                endpointData['transform.low'] = protocol.low.get()

            if transTypeIndex in [0, 1, 2, 3, 5]:
                endpointData['transform.high'] = protocol.up.get()

            if transTypeIndex == 2:
                endpointData['transform.coef_div'] = 100.00
                endpointData['transform.coef_si'] = 10.00
                endpointData['transform.coef_se'] = 10.00

            if transTypeIndex == 6:
                endpointData['transform.mapping'] = {
                    compName: protocol.scoreMatch.get(),
                    "No %s" % compName: protocol.scoreNoMatch.get()
                }

        compDict = {
            compName: {
                'endpoint': [endpointData]
            }
        }
        currentText = getattr(protocol, outputParam[0]).get() or ''
        workSteps = [line for line in currentText.strip().split('\n') if line.strip()]

        if 0 <= targetIdx < len(workSteps):
            stageData = ast.literal_eval(workSteps[targetIdx])

            if 'component' not in stageData['scoring']:
                stageData['scoring']['component'] = []

            stageData['scoring']['component'].append(compDict)

            workSteps[targetIdx] = str(stageData)
            newText = '\n'.join(workSteps) + '\n'

            protocol.workFlowSteps.set(newText)
            form.setVar(outputParam[0], newText.lstrip())

            newSummary = protocol._updateSummary()
            form.setVar('summarySteps', newSummary)


class DeleteStageWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        inputParam, outputParam = self.getInputOutput(form)
        protocol = form.protocol
        try:
            index = int(getattr(protocol, inputParam[0]).get())
            currentText = getattr(protocol, outputParam[0]).get() or ''
            workSteps = [line for line in currentText.strip().split('\n') if line.strip()]
            if 1 <= index <= len(workSteps):
                del workSteps[index - 1]
                if workSteps:
                    newText = '\n'.join(workSteps) + '\n'
                else:
                    newText = ''
                form.setVar(outputParam[0], newText)
                newSum = protocol._updateSummary()
                form.setVar('summarySteps', newSum)
        except:
            print('Incorrect index')

class DeleteComponentWizard(VariableWizard):
    _targets, _inputs, _outputs = [], {}, {}

    def show(self, form, *params):
        protocol = form.protocol
        try:
            sIdx = int(protocol.delStage.get())
            cIdx = int(protocol.delComponent.get())

            currentText = protocol.workFlowSteps.get() or ''
            workSteps = [l for l in currentText.strip().split('\n') if l.strip()]

            if 1 <= sIdx <= len(workSteps):
                stageData = ast.literal_eval(workSteps[sIdx - 1])

                components = stageData.get('scoring', {}).get('component', [])

                if 1 <= cIdx <= len(components):
                    del components[cIdx - 1]

                    workSteps[sIdx - 1] = str(stageData)
                    newText = '\n'.join(workSteps) + '\n'

                    protocol.workFlowSteps.set(newText)
                    form.setVar('workFlowSteps', newText)

                    if hasattr(protocol, '_updateSummary'):
                        newSum = protocol._updateSummary()
                        form.setVar('summarySteps', newSum)

        except:
            print('Incorrect index')

AddStageWizard().addTarget(protocol=ReinventStagedLearning,
                           targets=['insertStep'],
                           inputs=['insertStep'],
                           outputs=['workFlowSteps', 'summarySteps'])

AddScoringComponentWizard().addTarget(protocol=ReinventStagedLearning,
                           targets=['insertComponent'],
                           inputs=['insertComponent'],
                           outputs=['workFlowSteps', 'summarySteps'])

DeleteStageWizard().addTarget(protocol=ReinventStagedLearning,
                              targets=['delStage'],
                              inputs=['delStage'],
                              outputs=['workFlowSteps', 'summarySteps'])

DeleteComponentWizard().addTarget(protocol=ReinventStagedLearning,
                                  targets=['delComponent'],
                                  inputs=['delComponent', 'delStage'],
                                  outputs=['workFlowSteps', 'summarySteps'])
