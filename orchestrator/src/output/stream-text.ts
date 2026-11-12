/**
 * Turns a turn's streamed text fragments into terminal writes.
 *
 * The Anthropic provider yields many small `text` events per reply. Printing
 * each with its own line break shreds the reply, so this writes fragments
 * as they come, indents each new line once, and closes the line only when the
 * caller says the run of text is over.
 */

export interface TextStream {
  /** The string to write for one fragment. */
  push(fragment: string): string;
  /** The string that closes the current line, or "" if none is open. */
  end(): string;
}

export function createTextStream(indent: string): TextStream {
  let atLineStart = true;
  return {
    push(fragment) {
      let out = "";
      const lines = fragment.split("\n");
      lines.forEach((line, i) => {
        if (i > 0) {
          out += "\n";
          atLineStart = true;
        }
        if (line !== "") {
          if (atLineStart) out += indent;
          out += line;
          atLineStart = false;
        }
      });
      return out;
    },
    end() {
      if (atLineStart) return "";
      atLineStart = true;
      return "\n";
    },
  };
}
