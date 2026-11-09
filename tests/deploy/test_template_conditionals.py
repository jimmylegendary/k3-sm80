from __future__ import annotations

import unittest

from deploy.tooling.render_lanes import _apply_conditionals, render_template


def resolve(text: str, **context: object) -> str:
    return _apply_conditionals("inline", text, dict(context))


class TemplateConditionalsTest(unittest.TestCase):
    def test_true_block_keeps_body_and_drops_control_lines_with_their_newlines(self) -> None:
        text = "a\n@@IF X@@\nb\n@@ENDIF@@\nc\n"

        self.assertEqual(resolve(text, X=1), "a\nb\nc\n")

    def test_false_block_vanishes_without_leaving_a_blank_line(self) -> None:
        text = "a\n@@IF X@@\nb\n\nc\n@@ENDIF@@\nd\n"

        self.assertEqual(resolve(text, X=0), "a\nd\n")

    def test_template_without_blocks_is_returned_byte_for_byte(self) -> None:
        text = "x\r\ny\n\n\nz"  # CRLF, blank lines, and no trailing newline

        self.assertEqual(resolve(text), text)

    def test_else_branch_is_exclusive(self) -> None:
        text = "@@IF X@@\nyes\n@@ELSE@@\nno\n@@ENDIF@@\n"

        self.assertEqual(resolve(text, X=1), "yes\n")
        self.assertEqual(resolve(text, X=0), "no\n")

    def test_truthiness_rules(self) -> None:
        text = "@@IF X@@\nyes\n@@ENDIF@@\n"

        for value in (1, True, "1", "text"):
            with self.subTest(true=value):
                self.assertEqual(resolve(text, X=value), "yes\n")
        for value in (0, False, "0", "", None):
            with self.subTest(false=value):
                self.assertEqual(resolve(text, X=value), "")

    def test_nested_blocks_respect_the_outer_condition(self) -> None:
        text = (
            "@@IF OUTER@@\n"
            "o1\n"
            "@@IF INNER@@\n"
            "i\n"
            "@@ELSE@@\n"
            "not-i\n"
            "@@ENDIF@@\n"
            "o2\n"
            "@@ELSE@@\n"
            "not-o\n"
            "@@IF INNER@@\n"
            "must-not-appear\n"
            "@@ENDIF@@\n"
            "@@ENDIF@@\n"
        )

        self.assertEqual(resolve(text, OUTER=1, INNER=1), "o1\ni\no2\n")
        self.assertEqual(resolve(text, OUTER=1, INNER=0), "o1\nnot-i\no2\n")
        self.assertEqual(resolve(text, OUTER=0, INNER=1), "not-o\nmust-not-appear\n")
        self.assertEqual(resolve(text, OUTER=0, INNER=0), "not-o\n")

    def test_value_placeholders_inside_a_dropped_block_are_never_required(self) -> None:
        text = "@@IF X@@\n@@ONLY_WHEN_X@@\n@@ENDIF@@\nkeep\n"

        self.assertEqual(resolve(text, X=0), "keep\n")

    def test_malformed_blocks_are_rejected(self) -> None:
        cases = (
            ("unknown condition", "@@IF NOPE@@\nx\n@@ENDIF@@\n"),
            ("unterminated", "@@IF X@@\nx\n"),
            ("@@ENDIF@@ without an open", "x\n@@ENDIF@@\n"),
            ("@@ELSE@@ without an open", "x\n@@ELSE@@\n"),
            ("@@ELSE@@ without an open", "@@IF X@@\n@@ELSE@@\n@@ELSE@@\n@@ENDIF@@\n"),
            ("malformed control line", "@@IF@@\nx\n@@ENDIF@@\n"),
            ("malformed control line", "@@ELSE X@@\n"),
        )
        for message, text in cases:
            with self.subTest(message=message, text=text):
                with self.assertRaisesRegex(ValueError, message):
                    resolve(text, X=1)

    def test_control_syntax_must_occupy_the_whole_line(self) -> None:
        text = "prefix @@IF X@@ suffix\n  @@IF X@@\n"

        self.assertEqual(resolve(text, X=0), text)

    def test_render_template_still_reports_unresolved_value_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unresolved template keys"):
            render_template("compose.yaml.tmpl", {"LANE_NUMBER": "01"})


if __name__ == "__main__":
    unittest.main()
