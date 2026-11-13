// What can go wrong in the editor, with no Node and no HTTP: post.ts and suggestion.ts throw these, the API turns them
// into statuses (api.ts), and the page reads them back (remote.ts).

// Why an editor operation failed, in the editor's own words. The API turns each kind into an HTTP status (api.ts),
// so this module never needs to know about HTTP.
export type Failure =
  | "invalid" // the input breaks a rule: a bad folder name, frontmatter the site can't build
  | "missing" // no such post or image
  | "exists" // a new post would take an existing folder
  | "conflict" // the post changed since the editor read it
  | "unsupported" // not an image type the site serves
  | "too-large"; // an upload over the limit

export class EditorError extends Error {
  readonly failure: Failure;
  // Shaped like Error's own constructor: the message, then options, here with the failure besides the cause.
  constructor(message: string, { failure, ...options }: ErrorOptions & { failure: Failure }) {
    super(message, options);
    this.failure = failure;
  }
}
