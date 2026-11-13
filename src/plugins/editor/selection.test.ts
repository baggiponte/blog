import assert from "node:assert/strict";
import { test } from "node:test";

import { rows } from "./selection.ts";

test("a row's text boxes join into one box, and each row of text keeps its own", () => {
  const boxes = [
    // first row: plain text, then a bold span a little taller
    { left: 10, top: 100, right: 60, bottom: 120 },
    { left: 60, top: 99, right: 90, bottom: 121 },
    // second row of the same wrapped line
    { left: 0, top: 130, right: 40, bottom: 150 },
    // an empty box, as browsers give for a line break
    { left: 40, top: 130, right: 40, bottom: 150 },
  ];
  assert.deepEqual(rows(boxes), [
    { left: 10, top: 99, right: 90, bottom: 121 },
    { left: 0, top: 130, right: 40, bottom: 150 },
  ]);
});

test("boxes with their fields on getters, like the browser's DOMRect, keep their size", () => {
  class Rect {
    get left(): number {
      return 5;
    }
    get top(): number {
      return 10;
    }
    get right(): number {
      return 25;
    }
    get bottom(): number {
      return 30;
    }
  }
  assert.deepEqual(rows([new Rect()]), [{ left: 5, top: 10, right: 25, bottom: 30 }]);
});
