import Lake
open Lake DSL

package LeanData where
  -- Add any package-wide settings here

lean_lib NuminaMath where
  -- This will include all .lean files under the NuminaMath directory
  srcDir := "NuminaMath"

@[default_target]
lean_exe leandata where
  root := `Main
  -- Assuming you'll have a Main.lean file in the root directory