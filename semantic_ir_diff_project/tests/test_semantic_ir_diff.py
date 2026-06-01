import os
import tempfile
import unittest

from semantic_ir_diff.classify import classify_function_diff
from semantic_ir_diff.compiler import ClangNotFound, compile_to_ir
from semantic_ir_diff.diff import FunctionDiff, diff_functions
from semantic_ir_diff.normalize import normalize_ir_text
from semantic_ir_diff.parser import IRFunction, parse_functions


class TestSemanticIRDiff(unittest.TestCase):
    def test_normalize_ir_text_strips_metadata_and_registers(self):
        raw = """
        ; ModuleID = 'test'
        define i32 @foo() #0 {
        entry:
          %1 = add i32 %0, 1 ; increment
          br label %exit
        exit:
          ret i32 %1
        }
        !0 = metadata
        """
        normalized = normalize_ir_text(raw)
        self.assertNotIn("!0", normalized)
        self.assertNotIn("; increment", normalized)
        self.assertIn("%r0 = add i32 %r1, 1", normalized)

    def test_parse_functions_extracts_function_and_blocks(self):
        ir = """
        define void @foo() {
        entry:
          br label %bb1
        bb1:
          ret void
        }
        """
        funcs = parse_functions(normalize_ir_text(ir))
        self.assertIn("foo", funcs)
        foo = funcs["foo"]
        self.assertEqual(set(foo.blocks), {"bb0", "bb1"})
        self.assertEqual(foo.terminators["bb0"], ["br label %bb1"])

    def test_diff_functions_detects_added_removed_calls(self):
        old_ir = """
        define void @foo() {
        entry:
          call void @bar()
          ret void
        }
        define void @bar() {
        entry:
          ret void
        }
        """
        new_ir = """
        define void @foo() {
        entry:
          ret void
        }
        """
        old_funcs = parse_functions(normalize_ir_text(old_ir))
        new_funcs = parse_functions(normalize_ir_text(new_ir))
        diff = diff_functions(old_funcs, new_funcs)
        self.assertEqual(diff.added_functions, [])
        self.assertEqual(diff.removed_functions, ["bar"])
        self.assertEqual(len(diff.changed_functions), 1)
        self.assertEqual(diff.changed_functions[0].name, "foo")
        self.assertIn("removed calls", diff.changed_functions[0].facts[0])

    def test_classify_function_diff_detects_inlining_change(self):
        old_func = IRFunction(name="foo", calls={"bar"})
        new_func = IRFunction(name="foo", calls=set())
        diff = FunctionDiff(
            name="foo",
            changed=True,
            added_blocks=0,
            removed_blocks=0,
            instruction_delta=0,
            added_calls=[],
            removed_calls=[],
            branch_changed=False,
            facts=[],
        )
        categories = classify_function_diff(old_func, new_func, diff)
        self.assertIn("calls removed; function may have been inlined", categories)

    def test_compile_to_ir_raises_when_clang_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "dummy.c")
            out = os.path.join(temp_dir, "dummy.ll")
            with open(src, "w", encoding="utf-8") as f:
                f.write("int main() { return 0; }\n")
            with self.assertRaises(ClangNotFound):
                compile_to_ir(src, out, clang_path="nonexistent-clang")


if __name__ == "__main__":
    unittest.main()
