# Contributing to Our Project
Please first refer to these links to understand how to contribute to open source projects.

- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Building Welcoming Communities](https://opensource.guide/building-community/)

## Get involved

There are many ways to contribute to our project, and many of them do not involve writing any code. Here's a few ideas to get started:

- Simply start using the site. Does everything work as expected? If not, we're always looking for improvements. Let us know by [opening an issue](#reporting-new-issues).
- Look through the [open issues](https://github.com/sveltejs/svelte/issues). A good starting point would be issues tagged [good first issue](https://github.com/sveltejs/svelte/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Provide workarounds, ask for clarification, or suggest labels. Help [triage issues](#triaging-issues-and-pull-requests).
- If you find an issue you would like to fix, [open a pull request](#pull-requests).
- Read through our [tutorials](https://learn.svelte.dev/). If you find anything that is confusing or can be improved, you can make edits by clicking "Edit this page" at the bottom left of the tutorial page.
- Take a look at the [features requested](https://github.com/sveltejs/svelte/labels/feature%20request) by others in the community and consider opening a pull request if you see something you want to work on.

Contributions are very welcome. If you think you need help planning your contribution, please ping us on Discord at [svelte.dev/chat](https://svelte.dev/chat) and let us know you are looking for a bit of help.

### Triaging issues and pull requests

One great way you can contribute to the project without writing any code is to help triage issues and pull requests as they come in.

- Ask for more information if you believe the issue does not provide all the details required to solve it.
- Flag issues that are stale or that should be closed.
- Ask for test plans and review code.

## Our process

## Bugs

If you would like to report a problem, take a look around and see if someone already opened an issue about it. If you are certain this is a new unreported bug, you can submit a [bug report](#reporting-new-issues).

If you have questions about using the project, contact us on Discord and we will do our best to answer your questions.

If you see anything you'd like to be implemented, make sure to refer our github repo before doing so.

### Reporting new issues

When [opening a new issue](https://github.com/sveltejs/svelte/issues/new/choose), always make sure to fill out the issue template. **This step is very important!** Not doing so may result in your issue not being managed in a timely fashion. Don't take this personally if this happens, and feel free to open a new issue once you've gathered all the information required by the template.

- **One issue, one bug:** Please report a single bug per issue.
- **Provide reproduction steps:** List all the steps necessary to reproduce the issue. The person reading your bug report should be able to follow these steps to reproduce your issue with minimal effort. If possible, use the [REPL](https://svelte.dev/repl) to create your reproduction.

### Proposing a change

If you would like to request a new feature or enhancement but are not yet thinking about opening a pull request, you can also file an issue with [feature template](https://github.com/sveltejs/svelte/issues/new?template=feature_request.yml).

If you're only fixing a bug, it's fine to submit a pull request right away, but we still recommend that you file an issue detailing what you're fixing. This is helpful in case we don't accept that specific fix but want to keep track of the issue.

Small pull requests are much easier to review and more likely to get merged.

### Creating a branch

Fork [the repository](https://github.com/sveltejs/svelte) and create your branch from `master`. If you've never sent a GitHub pull request before, you can learn how from [this free video series](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

### Testing

A good test plan has the exact commands you ran and their output, provides screenshots or videos if the pull request changes UI.

- If you've changed APIs, update the documentation.

#### Writing tests

All tests are located in `/test` folder.

Test samples are kept in `/test/xxx/samples` folder.

### Style guide

[Eslint](https://eslint.org) will catch most styling issues that may exist in your code. You can check the status of your code styling by simply running `pnpm lint`.

### Sending your pull request

Please make sure the following is done when submitting a pull request:

1. Describe your **test plan** in your pull request description. Make sure to test your changes.
1. Make sure your code lints (`pnpm lint`).
1. Make sure your tests pass (`pnpm test`).

All pull requests should be opened against the `master` branch. Make sure the PR does only one thing, otherwise please split it.

#### Breaking changes

When adding a new breaking change, follow this template in your pull request:

```md
### New breaking change here

- **Who does this affect**:
- **How to migrate**:
- **Why make this breaking change**:
- **Severity (number of people affected x effort)**:
```

## License

By contributing to our project, you agree that your contributions will be licensed under its [MIT license]

## Questions

Feel free to ask in [#contributing](https://discord.com/channels/457912077277855764/750401468569354431) on [Discord](https://svelte.dev/chat) if you have questions about our process, how to proceed, etc.
