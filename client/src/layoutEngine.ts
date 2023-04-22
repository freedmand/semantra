import type { PdfChar } from "./types";
import RBush from "rbush";

interface RBushItem {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  char: PdfChar;
  i: number;
}

type Direction = "left" | "right" | "up" | "down";

function createTree(chars: PdfChar[]): RBush {
  const tree = new RBush();
  for (let i = 0; i < chars.length; i++) {
    const char = chars[i];
    const x0 = char[1].x0;
    const y0 = char[1].y0;
    const x1 = char[1].x1;
    const y1 = char[1].y1;
    tree.insert({
      minX: x0,
      minY: y0,
      maxX: x1,
      maxY: y1,
      char,
      i,
    });
  }
  return tree;
}

function getAccessor(direction: Direction) {
  switch (direction) {
    case "left":
      return (c) => accessWithPad(c, "x0");
    case "right":
      return (c) => accessWithPad(c, "x1");
    case "up":
      return (c) => accessWithPad(c, "y0");
    case "down":
      return (c) => accessWithPad(c, "y1");
  }
}

function switchDirection(direction: Direction) {
  switch (direction) {
    case "left":
      return "right";
    case "right":
      return "left";
    case "up":
      return "down";
    case "down":
      return "up";
  }
}

function normalize(value: number) {
  return Math.max(0, value);
}

function getPad(
  width: number,
  height: number,
  char: PdfChar,
  i: number,
  searchResults: RBushItem[],
  direction: Direction
): [number, PdfChar | null] {
  const reverse = switchDirection(direction);
  const reverseAccessor = getAccessor(direction);
  const accessor = getAccessor(reverse);
  let most: [number, PdfChar | null] =
    reverse === "right" || reverse === "down"
      ? [0, null]
      : reverse === "left"
      ? [width, null]
      : [height, null];
  const seek = most[0] === 0 ? "max" : "min";
  for (const result of searchResults) {
    if (result.i === i) {
      continue;
    }
    if (
      seek === "max" &&
      accessor(result.char) > reverseAccessor(char) &&
      reverseAccessor(result.char) < accessor(char)
    ) {
      continue;
    }
    if (
      seek === "min" &&
      accessor(result.char) < reverseAccessor(char) &&
      reverseAccessor(result.char) > accessor(char)
    ) {
      continue;
    }
    if (seek === "max" && accessor(result.char) > most[0]) {
      most = [accessor(result.char), result.char];
    }
    if (seek === "min" && accessor(result.char) < most[0]) {
      most = [accessor(result.char), result.char];
    }
  }
  return [
    normalize(
      seek === "max"
        ? reverseAccessor(char) - most[0]
        : most[0] - reverseAccessor(char)
    ),
    most[1],
  ];
}

export function accessWithPad(char: PdfChar, side: "x0" | "x1" | "y0" | "y1") {
  if (side === "x0") {
    return char[1].x0 - (char[1].lpad || 0);
  }
  if (side === "x1") {
    return char[1].x1 + (char[1].rpad || 0);
  }
  if (side === "y0") {
    return char[1].y0 - (char[1].tpad || 0);
  }
  if (side === "y1") {
    return char[1].y1 + (char[1].bpad || 0);
  }
}

function consolidateHeight(c1: PdfChar, c2: PdfChar) {
  const y0 = Math.min(accessWithPad(c1, "y0"), accessWithPad(c2, "y0"));
  const y1 = Math.max(accessWithPad(c1, "y1"), accessWithPad(c2, "y1"));
  c1[1].y0 = y0;
  c1[1].y1 = y1;
  c2[1].y0 = y0;
  c2[1].y1 = y1;
}

function consolidateWidth(c1: PdfChar, c2: PdfChar) {
  const x0 = Math.min(accessWithPad(c1, "x0"), accessWithPad(c2, "x0"));
  const x1 = Math.max(accessWithPad(c1, "x1"), accessWithPad(c2, "x1"));
  c1[1].x0 = x0;
  c1[1].x1 = x1;
  c2[1].x0 = x0;
  c2[1].x1 = x1;
}

function project(
  width: number,
  height: number,
  tree: RBush,
  char: PdfChar,
  i: number,
  direction: Direction
) {
  if (direction === "left") {
    const x1 = accessWithPad(char, "x0");
    const y0 = accessWithPad(char, "y0");
    const y1 = accessWithPad(char, "y1");
    const x0 = 0;
    const [lpad, most] = getPad(
      width,
      height,
      char,
      i,
      tree.search({ minX: x0, minY: y0, maxX: x1, maxY: y1 }),
      direction
    );
    char[1].lpad = lpad;
    if (most) {
      consolidateHeight(char, most);
    }
  } else if (direction === "right") {
    const x0 = accessWithPad(char, "x1");
    const y0 = accessWithPad(char, "y0");
    const y1 = accessWithPad(char, "y1");
    const x1 = width;
    const [rpad, most] = getPad(
      width,
      height,
      char,
      i,
      tree.search({ minX: x0, minY: y0, maxX: x1, maxY: y1 }),
      direction
    );
    char[1].rpad = rpad;
    if (most) {
      consolidateHeight(char, most);
    }
  } else if (direction === "up") {
    const x0 = accessWithPad(char, "x0");
    const x1 = accessWithPad(char, "x1");
    const y1 = accessWithPad(char, "y0");
    const y0 = 0;
    char[1].tpad = getPad(
      width,
      height,
      char,
      i,
      tree.search({ minX: x0, minY: y0, maxX: x1, maxY: y1 }),
      direction
    )[0];
  } else if (direction === "down") {
    const x0 = accessWithPad(char, "x0");
    const x1 = accessWithPad(char, "x1");
    const y0 = accessWithPad(char, "y1");
    const y1 = height;
    char[1].bpad = getPad(
      width,
      height,
      char,
      i,
      tree.search({ minX: x0, minY: y0, maxX: x1, maxY: y1 }),
      direction
    )[0];
  }
}

function projectAll(
  width: number,
  height: number,
  chars: PdfChar[],
  tree: RBush,
  direction: Direction
) {
  for (let i = 0; i < chars.length; i++) {
    project(width, height, tree, chars[i], i, direction);
  }
}

export function layout(
  width: number,
  height: number,
  chars: PdfChar[]
): PdfChar[] {
  const tree = createTree(chars);

  projectAll(width, height, chars, tree, "right");
  projectAll(width, height, chars, tree, "left");
  projectAll(width, height, chars, tree, "up");
  projectAll(width, height, chars, tree, "down");

  return chars;
}

export function copyChars(chars: PdfChar[]): PdfChar[] {
  return chars.map((char) => [char[0], { ...char[1] }]);
}
