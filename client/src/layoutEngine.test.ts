import { expect, test } from "vitest";
import type { PdfChar } from "./types";
import { layout, copyChars } from "./layoutEngine";

test("pad right and bottom single char", () => {
  const width = 100;
  const height = 200;
  const pdfChars: PdfChar[] = [["a", { x0: 0, x1: 10, y0: 0, y1: 10 }]];

  expect(layout(width, height, copyChars(pdfChars))).toEqual([
    [
      "a",
      { x0: 0, x1: 10, y0: 0, y1: 10, lpad: 0, tpad: 0, rpad: 90, bpad: 190 },
    ],
  ]);
});

test("pad every direction single char", () => {
  const width = 100;
  const height = 200;
  const pdfChars: PdfChar[] = [["a", { x0: 50, x1: 60, y0: 40, y1: 60 }]];

  expect(layout(width, height, copyChars(pdfChars))).toEqual([
    [
      "a",
      {
        x0: 50,
        x1: 60,
        y0: 40,
        y1: 60,
        lpad: 50,
        tpad: 40,
        rpad: 40,
        bpad: 140,
      },
    ],
  ]);
});

test("pad with two chars", () => {
  const width = 100;
  const height = 100;
  const pdfChars: PdfChar[] = [
    ["a", { x0: 20, x1: 30, y0: 10, y1: 20 }],
    ["b", { x0: 50, x1: 60, y0: 15, y1: 25 }],
  ];

  expect(layout(width, height, copyChars(pdfChars))).toEqual([
    [
      "a",
      {
        x0: 20,
        x1: 30,
        y0: 10,
        y1: 25,
        lpad: 20,
        tpad: 10,
        rpad: 20,
        bpad: 75,
      },
    ],
    [
      "b",
      {
        x0: 50,
        x1: 60,
        y0: 10,
        y1: 25,
        lpad: 0,
        tpad: 10,
        rpad: 40,
        bpad: 75,
      },
    ],
  ]);
});

test("pad with three chars", () => {
  const width = 100;
  const height = 100;
  const pdfChars: PdfChar[] = [
    ["a", { x0: 20, x1: 30, y0: 10, y1: 20 }],
    ["b", { x0: 50, x1: 60, y0: 15, y1: 25 }],
    ["c", { x0: 80, x1: 85, y0: 5, y1: 16 }],
  ];

  expect(layout(width, height, copyChars(pdfChars))).toEqual([
    [
      "a",
      {
        x0: 20,
        x1: 30,
        y0: 5,
        y1: 25,
        lpad: 20,
        tpad: 5,
        rpad: 20,
        bpad: 75,
      },
    ],
    [
      "b",
      {
        x0: 50,
        x1: 60,
        y0: 5,
        y1: 25,
        lpad: 0,
        tpad: 5,
        rpad: 20,
        bpad: 75,
      },
    ],
    [
      "c",
      {
        x0: 80,
        x1: 85,
        y0: 5,
        y1: 25,
        lpad: 0,
        tpad: 5,
        rpad: 15,
        bpad: 75,
      },
    ],
  ]);
});
