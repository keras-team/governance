# Keras API proposal "Request For Comment" (RFC) docs

This folder contains approved API proposals. To propose a new API to be considered for review, you can open a Pull Request in this repository to add a new RFC `.md` doc.

## Process

The process for writing and submitting design proposals is same as the [TensorFlow RFC process](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md).

- Start from [this template](https://github.com/keras-team/governance/blob/master/rfcs/yyyymmdd-rfc-template.md).
- Fill in the content. Note that you will need to insert code examples.
    - Provide enough context information for anyone to undertsand what's going on.
    - Provide a solid argument as for why the feature is neeed.
    - Include a code example of the **end-to-end workflow** you have in mind.
- Open a Pull Request in the [Keras API proposals folder in this repository](https://github.com/keras-team/governance/rfcs/).
- Send the Pull Request link to `keras-users@googlegroups.com` with a subject that starts with `[API DESIGN REVIEW]` (all caps) so that we notice it.
- Wait for comments, and answer them as they come. Edit the proposal as necessary.
- The proposal will finally be approved or rejected during a meeting of the Keras SIG chairs. Once approved, you can send out Pull Requests to implement the API changes or ask others to write Pull Requests (targeting `tf.keras` and `keras-team/keras`).

Note that:
- Anyone is free to send out API proposals.
- Anyone is free to comment on API proposals or ask questions.
- Anyone is free to attend design review meetings as an observer.
- Participation in design review meetings is restricted to Keras SIG chairs.
- Design review meeting notes will be posted publicly after each meeting.

## Template

Use [this template](https://github.com/keras-team/governance/blob/master/rfcs/yyyymmdd-rfc-template.md) to draft an RFC.
