# Keras API Special Interest Group (SIG)

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)


## Scope of the Keras SIG

The purpose of the Keras SIG is to provide a formal structure to supervise the evolution of the Keras API,
while giving a voice to all keys stakeholders. The SIG consists of a process that anyone can follow to make API proposals,
public design review meetings, and a comittee that reviews the proposals and makes final decisions about what should be in the Keras API.

The SIG covers all aspects of the Keras API, in particular:

- Design of the core Keras API specification (including its implementations in `tf.keras` and `keras-team/keras`).
- Design of TensorFlow-specific Keras features to be included in `tf.keras`.
- Design of the Keras model file format.
- Design of the API of the subprojects Keras Applications and Keras Preprocessing, which are shared across `tf.keras` and `keras-team/keras`.
- Design of the API of all Keras-brand projects, such as AutoKeras and Keras-Tuner.

## Leadership

### BDFL

Role: final call in decisions related to the Keras API. Must approve all changes to the Keras API.

- Francois Chollet (fchollet@google.com)

### Committee chairs

Role: represent the interests of different stakeholders during design discussions.
Selection process: From time to time, the chairs may revise the group's membership to ensure the project's interests are well represented.

Key stakeholders:

- Karmel Allison, Google (karmel@google.com)
- Martin Wicke, Google (wicke@google.com)
- Stan Bileschi, Google (bileschi@google.com)
- Paige Bailey, Google (webpaige@google.com)
- Adam Gibson, Skymind (adam@skymind.io)
- Sandeep, Amazon & Keras-MXNet (sandeep.krishna98@gmail.com)
- JJ Allaire, RStudio (jj@rstudio.com)

Core Keras community contributors:

- Fariz Rahman (farizrahman4u@gmail.com)
- Frederic Branchaud-Charron (Frederic.Branchaud-Charron@usherbrooke.ca)
- Taehoon Lee (me@taehoonlee.com)


## Design review process

The Keras SIG meets once a month (first Friday of the month at 11am PT) to discuss current development directions.
This is a public meeting open to anyone who is interested: the [agenda and notes are public](http://bit.ly/keras-meeting-notes). To get notifications and a calendar invitation to the public meeting, please join the [keras-meetings Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/keras-meetings).

Other communications are expected to happen primarily asynchronously via:

- The Pull Requests used for API proposals.
- [The Keras mailing list](https://groups.google.com/forum/#!forum/keras-users).

The process for writing and submitting design proposals is same as the [TensorFlow RFC process](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md).

- Start from [this template](https://github.com/keras-team/governance/blob/master/rfcs/yyyymmdd-rfc-template.md).
- Fill in the content. Note that you will need to insert code examples.
    - Provide enough context information for anyone to undertsand what's going on.
    - Provide a solid argument as for why the feature is neeed.
    - Include a code example of the **end-to-end workflow** you have in mind.
- Open a Pull Request in the [Keras API proposals folder in this repository](https://github.com/keras-team/governance/tree/master/rfcs).
- Send the Pull Request link to `keras-users@googlegroups.com` with a subject that starts with `[API DESIGN REVIEW]` (all caps) so that we notice it.
- Wait for comments, and answer them as they come. Edit the proposal as necessary.
- The proposal will finally be approved or rejected during a meeting of the Keras SIG chairs. Once approved, you can send out Pull Requests to implement the API changes or ask others to write Pull Requests (targeting `tf.keras` and `keras-team/keras`).

Note that:
- Anyone is free to send out API proposals.
- Anyone is free to comment on API proposals or ask questions.
- Anyone is free to attend design review meetings as an observer.
- Participation in design review meetings is restricted to Keras SIG chairs.
- Design review meeting notes will be posted publicly after each meeting.


## Our mission

The purpose of our work is to democratize access to machine learning through dependable standards and usable, productive APIs.
We seek to empower as many people as possible, from a wide diversity of backgrounds, to take ownership of ML technology and to use it to build their own solutions to their own problems.

Existing machine learning technology has the potential to solve a huge amount of problems in the world today, across every industry, and to help a tremendous amount of people.
The potential is sky-high. We've barely even started. So how do we fully realize this potential?

We believe that we will only fully realize the potential of machine learning if it becomes a tool in everyone's hands -- not just a technology developed behind closed doors by an "AI industry" in Silicon Valley, that you could only deploy by waiting for a turnkey cloud API to become available commercially, or by contracting an expensive consulting firm. We can't wait for experts to solve every problem -- experts in Silicon Valley don't even have visibility into 1% of the problems that can be solved. The end users should solve their own problems. And our mission is to empower them to do just that.

Our mission is to make these capabilities available to anyone with basic computer literacy, for free. This is how we maximize the realized potential of these technologies, and how we maximize our positive impact on the world.


## Our milestones

As of Q1 2019, 320,000 developers worldwide are using the Keras API to build intelligent applications or conduct research (about 1 developer in 60 globally). Meanwhile, about 1,000,000 developers using machine learning in some form (about 1 developer in 20 globally).

The size of the Keras user base should rise to 500,000 by Q1 2020, and to 1,000,000 by Q1 2022. In the long term (2025 and beyond), about one developer in ten will be using Keras.

In everything we do, we should keep in mind the next million developers and the next million problems we are empowering them to solve.

## Code of conduct

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone. All activity connected with this SIG will abide by the [Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).

## Our values and our strengths

We will be able to reach our milestones because we yield superpowers of a kind that is quite uncommon among developers of ML tools:

- We have empathy for our users.
- We value and practice good design.
- We embrace openness and we are dedicated to foster our developer community.
- We value and practice good communication. 
- We have a unique brand that users love.

**1) We have empathy for our users.** We know that every decision we make should be made with the user in mind, whether a design decision or a strategy decision. "What will be the total impact of our choices on the people who rely on our software?" is the question behind everything we do.

Having empathy for our users means:
- Being users ourselves -- actively using our product in similar scenarios as what our users face.
- Understand our users by being closely in touch with them and having clear visibility into the developer experience: actively seeking community input in everything we do, listening to feedback, talking with users at external talks and developer events.
- Putting ourselves in our users' shoes: always going above and beyond to be helpful to our users and to improve the user experience of our products.

**2) We value good design.** We are fully aware that a delightful UX is what has made us successful so far and what will keep making us successful in the future. We know that things should be as simple as possible (but no simpler). We prefer elegance and minimalism over technical prowess. We follow formal principles for good design.

**3) We embrace openness and we are dedicated to foster our developer community.** We know that long-term community building is critical to the success of our project, and we know that developer communities don't get built behind closed doors at Google. We understand the necessity of doing our work in the open and the importance of involving open-source contributors at all stages of the development process.

**4) We value and practice good communication.** We understand that great documentation, great code examples, and transparency of governance are essential to Keras adoption and contribute hugely to the Keras UX. We value external communication, documentation, and developers relations as much or more than we value technical contributions.

**5) We value what makes us different and unique, and we value our brand that users love, Keras.** The Keras brand is an essential tool in reaching our goals: it stands for user-friendliness, accessibility, and good design, in a way that doesn't ring true for our allies and our foes. We are proud of our banner and we will carry it forward.
