# Many-to-many data preprocessing in TimeseriesGenerator

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Marat Kopytjuk (kopytjuk@gmail.com)                  |
| **Sponsor**   |                   |
| **Updated**   | 2020-04-10                                         |

## Objective

Currently the `TimeseriesGenerator` class only supports many-to-one model architectures (scenario 1).

![architectures](http://karpathy.github.io/assets/rnn/diags.jpeg)

Source: [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

The goal of this RFC to extend the existing `TimeseriesGenerator` to the 4. and 5. usecase shown above.

## Motivation

A common use-case comes from engineering domain (control theory) where the user models nonlinear dynamical systems based on input and output timesieres.

Those models usually modeled as state-space systems where the next system state **s** is modelled as a function of current state and the current input vector **u**:

![ss](./assets/20190502-preprocessing-layers-ss.png)

The recurrent part `f_theta` calculates the next state (green blocks) given inputs (red), the stateless part `g_omega` calculates the output (blue). 

In order to train those systems the model has to be fed like in many-to-many scenario above (5). Users have to write custom code to prepare their datasets which can be done with TimeseriesGenerator in a more generic and "common" way.

## User Benefit

Support for data-preprocessing for dynamical system modelling (recurrent many-to-many architectures).

Users don't have to maintain own codebase for data preparation and can use "off-the-shelf" `keras` helper utilites.

## Design Proposal



## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.