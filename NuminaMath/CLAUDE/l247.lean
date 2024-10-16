import Mathlib

namespace NUMINAMATH_CALUDE_inspector_ratio_l247_24744

/-- Represents the daily production of a workshop relative to the first workshop -/
structure WorkshopProduction where
  relative_production : ℚ

/-- Represents an inspector group -/
structure InspectorGroup where
  num_inspectors : ℕ

/-- Represents the factory setup and inspection process -/
structure Factory where
  workshops : Fin 6 → WorkshopProduction
  initial_products : ℚ
  inspector_speed : ℚ
  group_a : InspectorGroup
  group_b : InspectorGroup

/-- The theorem stating the ratio of inspectors in group A to group B -/
theorem inspector_ratio (f : Factory) : 
  f.workshops 0 = ⟨1⟩ ∧ 
  f.workshops 1 = ⟨1⟩ ∧ 
  f.workshops 2 = ⟨1⟩ ∧ 
  f.workshops 3 = ⟨1⟩ ∧ 
  f.workshops 4 = ⟨3/4⟩ ∧ 
  f.workshops 5 = ⟨8/3⟩ ∧
  (6 * (f.workshops 0).relative_production + 
   6 * (f.workshops 1).relative_production + 
   6 * (f.workshops 2).relative_production + 
   3 * f.initial_products = 6 * f.inspector_speed * f.group_a.num_inspectors) ∧
  (2 * (f.workshops 3).relative_production + 
   2 * (f.workshops 4).relative_production + 
   2 * f.initial_products = 2 * f.inspector_speed * f.group_b.num_inspectors) ∧
  (6 * (f.workshops 5).relative_production + 
   f.initial_products = 4 * f.inspector_speed * f.group_b.num_inspectors) →
  f.group_a.num_inspectors * 19 = f.group_b.num_inspectors * 18 :=
by sorry

end NUMINAMATH_CALUDE_inspector_ratio_l247_24744


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l247_24761

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/16384 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l247_24761


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l247_24724

/-- Calculates the total surface area of a cube with holes -/
def cubeWithHolesSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (numHoles : ℕ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdge^2
  let holeArea := numHoles * holeEdge^2
  let newExposedArea := numHoles * 4 * cubeEdge * holeEdge
  originalSurfaceArea - holeArea + newExposedArea

/-- Theorem stating the surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  cubeWithHolesSurfaceArea 5 2 3 = 258 := by
  sorry

#eval cubeWithHolesSurfaceArea 5 2 3

end NUMINAMATH_CALUDE_specific_cube_surface_area_l247_24724


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_l247_24705

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = 56 / 147 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_l247_24705


namespace NUMINAMATH_CALUDE_linear_combination_proof_l247_24754

theorem linear_combination_proof (A B : Matrix (Fin 3) (Fin 3) ℤ) :
  A = ![![2, -4, 0], ![-1, 5, 1], ![0, 3, -7]] →
  B = ![![4, -1, -2], ![0, -3, 5], ![2, 0, -4]] →
  3 • A - 2 • B = ![![-2, -10, 4], ![-3, 21, -7], ![-4, 9, -13]] := by
  sorry

end NUMINAMATH_CALUDE_linear_combination_proof_l247_24754


namespace NUMINAMATH_CALUDE_inequality_constraint_l247_24737

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x > 1 → (a + 1) / x + Real.log x > a) → a ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_l247_24737


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l247_24794

theorem no_integer_roots_for_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coefficients_l247_24794


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l247_24783

theorem quadratic_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l247_24783


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l247_24750

theorem square_perimeters_sum (x y : ℕ) (h : x^2 - y^2 = 19) : 4*x + 4*y = 76 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l247_24750


namespace NUMINAMATH_CALUDE_hex_sum_equals_451A5_l247_24746

/-- Represents a hexadecimal digit --/
def HexDigit : Type := Fin 16

/-- Represents a hexadecimal number as a list of digits --/
def HexNumber := List HexDigit

/-- Convert a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : HexNumber := sorry

/-- Convert a hexadecimal number to its natural number representation --/
def fromHex (h : HexNumber) : ℕ := sorry

/-- Addition of hexadecimal numbers --/
def hexAdd (a b : HexNumber) : HexNumber := sorry

theorem hex_sum_equals_451A5 :
  let a := toHex 25  -- 19₁₆
  let b := toHex 12  -- C₁₆
  let c := toHex 432 -- 1B0₁₆
  let d := toHex 929 -- 3A1₁₆
  let e := toHex 47  -- 2F₁₆
  hexAdd a (hexAdd b (hexAdd c (hexAdd d e))) = toHex 283045 -- 451A5₁₆
  := by sorry

end NUMINAMATH_CALUDE_hex_sum_equals_451A5_l247_24746


namespace NUMINAMATH_CALUDE_parallel_vectors_theorem_l247_24728

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def Parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem parallel_vectors_theorem (e₁ e₂ a b : V) (m : ℝ) 
  (h_non_collinear : ¬ Parallel e₁ e₂)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : b = m • e₁ + 3 • e₂)
  (h_parallel : Parallel a b) :
  m = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_theorem_l247_24728


namespace NUMINAMATH_CALUDE_permutation_square_diff_l247_24792

theorem permutation_square_diff (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, n = 2 * k + 1) :
  (∃ a : Fin (n / 2 + 1) → Fin (n / 2 + 1),
    Function.Bijective a ∧
    ∀ i : Fin (n / 2), ∃ d : ℕ, ∀ j : Fin (n / 2),
      (a (j + 1))^2 - (a j)^2 ≡ d [ZMOD n]) →
  n = 3 ∨ n = 5 := by
sorry

end NUMINAMATH_CALUDE_permutation_square_diff_l247_24792


namespace NUMINAMATH_CALUDE_div_by_eleven_iff_alternating_sum_div_by_eleven_l247_24775

/-- Calculates the alternating sum of digits of a natural number -/
def alternatingDigitSum (n : ℕ) : ℤ :=
  sorry

/-- Proves the equivalence of divisibility by 11 and divisibility of alternating digit sum by 11 -/
theorem div_by_eleven_iff_alternating_sum_div_by_eleven (n : ℕ) :
  11 ∣ n ↔ 11 ∣ (alternatingDigitSum n) :=
sorry

end NUMINAMATH_CALUDE_div_by_eleven_iff_alternating_sum_div_by_eleven_l247_24775


namespace NUMINAMATH_CALUDE_final_pen_count_l247_24718

def pen_collection (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count : pen_collection 20 22 19 = 65 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l247_24718


namespace NUMINAMATH_CALUDE_square_property_iff_4_or_100_l247_24727

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The condition for n to satisfy the square property -/
def is_square_property (n : ℕ+) : Prop :=
  ∃ k : ℕ, (n ^ (d n + 1) * (n + 21) ^ (d n) : ℕ) = k ^ 2

/-- The main theorem -/
theorem square_property_iff_4_or_100 :
  ∀ n : ℕ+, is_square_property n ↔ n = 4 ∨ n = 100 := by sorry

end NUMINAMATH_CALUDE_square_property_iff_4_or_100_l247_24727


namespace NUMINAMATH_CALUDE_tower_combinations_l247_24743

theorem tower_combinations (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) : 
  (Nat.choose n k) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l247_24743


namespace NUMINAMATH_CALUDE_tetrahedron_triangles_l247_24741

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles :
  distinct_triangles = 4 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_triangles_l247_24741


namespace NUMINAMATH_CALUDE_constant_distance_to_line_l247_24753

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the related circle E
def related_circle_E (x y : ℝ) : Prop := x^2 + y^2 = 2/3

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Theorem statement
theorem constant_distance_to_line
  (k m x1 y1 x2 y2 : ℝ)
  (h1 : ellipse_C x1 y1)
  (h2 : ellipse_C x2 y2)
  (h3 : line_l k m x1 y1)
  (h4 : line_l k m x2 y2)
  (h5 : ∃ (x y : ℝ), related_circle_E x y ∧ line_l k m x y) :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), line_l k m x y → (x^2 + y^2 = d^2)) :=
sorry

end NUMINAMATH_CALUDE_constant_distance_to_line_l247_24753


namespace NUMINAMATH_CALUDE_exists_good_not_next_good_l247_24721

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The function f(n) = n - S(n) where S(n) is the digit sum of n -/
def f (n : ℕ) : ℕ := n - digitSum n

/-- f^k is f applied k times iteratively -/
def fIterate (k : ℕ) : ℕ → ℕ :=
  match k with
  | 0 => id
  | k+1 => f ∘ fIterate k

/-- A number x is k-good if there exists a y such that f^k(y) = x -/
def isGood (k : ℕ) (x : ℕ) : Prop :=
  ∃ y, fIterate k y = x

/-- The main theorem: for all n, there exists an x that is n-good but not (n+1)-good -/
theorem exists_good_not_next_good :
  ∀ n : ℕ, ∃ x : ℕ, isGood n x ∧ ¬isGood (n + 1) x := sorry

end NUMINAMATH_CALUDE_exists_good_not_next_good_l247_24721


namespace NUMINAMATH_CALUDE_long_jump_records_correct_l247_24752

/-- Represents a long jump record -/
structure LongJumpRecord where
  height : Real
  record : Real

/-- Checks if a long jump record is correctly calculated and recorded -/
def is_correct_record (standard : Real) (jump : LongJumpRecord) : Prop :=
  jump.record = jump.height - standard

/-- The problem statement -/
theorem long_jump_records_correct (standard : Real) (xiao_ming : LongJumpRecord) (xiao_liang : LongJumpRecord)
  (h1 : standard = 1.5)
  (h2 : xiao_ming.height = 1.95)
  (h3 : xiao_ming.record = 0.45)
  (h4 : xiao_liang.height = 1.23)
  (h5 : xiao_liang.record = -0.23) :
  ¬(is_correct_record standard xiao_ming ∧ is_correct_record standard xiao_liang) :=
sorry

end NUMINAMATH_CALUDE_long_jump_records_correct_l247_24752


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l247_24720

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l247_24720


namespace NUMINAMATH_CALUDE_min_l_pieces_in_8x8_l247_24762

/-- Represents an 8x8 square board --/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a three-cell L-shaped piece --/
structure LPiece where
  x : Fin 8
  y : Fin 8
  orientation : Fin 4

/-- Checks if an L-piece can be placed on the board --/
def canPlace (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Places an L-piece on the board --/
def placePiece (board : Board) (piece : LPiece) : Board :=
  sorry

/-- Checks if any more L-pieces can be placed on the board --/
def canPlaceMore (board : Board) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_pieces_in_8x8 :
  ∃ (pieces : List LPiece),
    pieces.length = 11 ∧
    (∃ (board : Board),
      (∀ p ∈ pieces, canPlace board p) ∧
      (∀ p ∈ pieces, board = placePiece board p) ∧
      ¬canPlaceMore board) ∧
    (∀ (pieces' : List LPiece),
      pieces'.length < 11 →
      ∀ (board : Board),
        (∀ p ∈ pieces', canPlace board p) →
        (∀ p ∈ pieces', board = placePiece board p) →
        canPlaceMore board) :=
  sorry

end NUMINAMATH_CALUDE_min_l_pieces_in_8x8_l247_24762


namespace NUMINAMATH_CALUDE_highest_point_parabola_l247_24769

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 28 * x + 418

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 7

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := parabola vertex_x

theorem highest_point_parabola :
  ∀ x : ℝ, parabola x ≤ vertex_y :=
by sorry

end NUMINAMATH_CALUDE_highest_point_parabola_l247_24769


namespace NUMINAMATH_CALUDE_parallel_vectors_xy_value_l247_24786

/-- Given two parallel vectors a and b in R³, prove that xy = -1/4 --/
theorem parallel_vectors_xy_value (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2*x, 1, 3)
  let b : ℝ × ℝ × ℝ := (1, -2*y, 9)
  (∃ (k : ℝ), a = k • b) → x * y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_xy_value_l247_24786


namespace NUMINAMATH_CALUDE_largest_N_equals_n_l247_24778

theorem largest_N_equals_n (n : ℕ) (hn : n ≥ 2) :
  ∃ N : ℕ, N > 0 ∧
  (∀ M : ℕ, M > N →
    ¬∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ M - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  (∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ N - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  N = n :=
sorry

end NUMINAMATH_CALUDE_largest_N_equals_n_l247_24778


namespace NUMINAMATH_CALUDE_min_value_fraction_l247_24751

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → (x + y) / (x * y) ≥ (a + b) / (a * b)) ∧
  (a + b) / (a * b) = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l247_24751


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l247_24776

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 432 →
  total_cost = 54 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l247_24776


namespace NUMINAMATH_CALUDE_initial_volumes_l247_24706

/-- Represents the initial state and operations on three cubic containers --/
structure ContainerSystem where
  -- Capacities of the containers
  c₁ : ℝ
  c₂ : ℝ
  c₃ : ℝ
  -- Initial volumes of liquid
  v₁ : ℝ
  v₂ : ℝ
  v₃ : ℝ
  -- Ratio of capacities
  hCapRatio : c₂ = 8 * c₁ ∧ c₃ = 27 * c₁
  -- Ratio of initial volumes
  hVolRatio : v₂ = 2 * v₁ ∧ v₃ = 3 * v₁
  -- Total volume remains constant
  hTotalVol : ℝ
  hTotalVolDef : hTotalVol = v₁ + v₂ + v₃
  -- Volume transferred in final operation
  transferVol : ℝ
  hTransferVol : transferVol = 128 + 4/7
  -- Final state properties
  hFinalState : ∃ (h₁ h₂ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧
    h₁ * c₁ + transferVol = v₁ - 100 ∧
    h₂ * c₂ - transferVol = v₂ ∧
    h₁ = 2 * h₂

/-- Theorem stating the initial volumes of liquid in the three containers --/
theorem initial_volumes (s : ContainerSystem) : 
  s.v₁ = 350 ∧ s.v₂ = 700 ∧ s.v₃ = 1050 := by
  sorry


end NUMINAMATH_CALUDE_initial_volumes_l247_24706


namespace NUMINAMATH_CALUDE_expression_evaluation_l247_24758

theorem expression_evaluation (y : ℝ) (h : y = -3) : 
  (5 + y * (4 + y) - 4^2) / (y - 2 + y^2) = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l247_24758


namespace NUMINAMATH_CALUDE_paper_shredder_capacity_l247_24793

theorem paper_shredder_capacity (total_contracts : ℕ) (shred_operations : ℕ) : 
  total_contracts = 2132 → shred_operations = 44 → 
  (total_contracts / shred_operations : ℕ) = 48 := by
  sorry

end NUMINAMATH_CALUDE_paper_shredder_capacity_l247_24793


namespace NUMINAMATH_CALUDE_max_value_of_f_l247_24730

theorem max_value_of_f (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  (∃ (a b : ℝ), f 0 + f 2017 ≤ a ∧ f 0 + f 2017 ≥ b) ∧ 
  (∀ (y : ℝ), f 0 + f 2017 ≤ y → y ≤ 1 + Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l247_24730


namespace NUMINAMATH_CALUDE_f_properties_l247_24781

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) * Real.exp x - (Real.exp 1 / 3) * x^3

noncomputable def g (x : ℝ) : ℝ := f x - 2

theorem f_properties :
  (∀ M : ℝ, ∃ x : ℝ, f x > M) ∧
  (∃ x₀ : ℝ, x₀ = 1 ∧ f x₀ = (2/3) * Real.exp 1 ∧ ∀ x : ℝ, f x ≥ f x₀) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ ∀ x ∈ Set.Ioo x₁ x₂, g x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l247_24781


namespace NUMINAMATH_CALUDE_other_diagonal_length_l247_24797

-- Define the rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  triangle_area : ℝ

-- Define the properties of the rhombus
def rhombus_properties (r : Rhombus) : Prop :=
  r.diagonal1 = 20 ∧ r.triangle_area = 75

-- Theorem statement
theorem other_diagonal_length (r : Rhombus) 
  (h : rhombus_properties r) : r.diagonal2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l247_24797


namespace NUMINAMATH_CALUDE_weeks_to_cover_all_combinations_l247_24782

/-- Represents a lottery ticket grid -/
structure LotteryGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (row_constraint : rows ≥ 5)
  (col_constraint : cols ≥ 14)

/-- Represents the marking strategy -/
structure MarkingStrategy :=
  (square_size : ℕ)
  (extra_number : ℕ)
  (square_constraint : square_size = 2)
  (extra_constraint : extra_number = 1)

/-- Represents the weekly ticket filling strategy -/
def weekly_tickets : ℕ := 4

/-- Theorem stating the time required to cover all combinations -/
theorem weeks_to_cover_all_combinations 
  (grid : LotteryGrid) 
  (strategy : MarkingStrategy) : 
  (((grid.rows - 2) * (grid.cols - 2)) + weekly_tickets - 1) / weekly_tickets = 52 :=
sorry

end NUMINAMATH_CALUDE_weeks_to_cover_all_combinations_l247_24782


namespace NUMINAMATH_CALUDE_probability_less_than_20_l247_24755

theorem probability_less_than_20 (total : ℕ) (more_than_30 : ℕ) (h1 : total = 160) (h2 : more_than_30 = 90) :
  let less_than_20 := total - more_than_30
  (less_than_20 : ℚ) / total = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l247_24755


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l247_24739

/-- Given an ellipse and a hyperbola with coinciding foci, prove that the parameter d of the ellipse satisfies d² = 667/36 -/
theorem ellipse_hyperbola_coinciding_foci :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / d^2 = 1
  let hyperbola := fun (x y : ℝ) => x^2 / 169 - y^2 / 64 = 1 / 36
  ∀ d : ℝ, (∃ c : ℝ, c^2 = 25 - d^2 ∧ c^2 = 169 / 36 + 64 / 36) →
    d^2 = 667 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l247_24739


namespace NUMINAMATH_CALUDE_alpha_minus_beta_range_l247_24710

theorem alpha_minus_beta_range (α β : Real) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∃ (x : Real), x = α - β ∧ -3*π/2 ≤ x ∧ x ≤ 0 ∧
  ∀ (y : Real), (-3*π/2 ≤ y ∧ y ≤ 0) → ∃ (α' β' : Real), 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ y = α' - β' :=
by
  sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_range_l247_24710


namespace NUMINAMATH_CALUDE_reeya_average_is_67_l247_24771

def reeya_scores : List ℝ := [55, 67, 76, 82, 55]

theorem reeya_average_is_67 : 
  (reeya_scores.sum / reeya_scores.length : ℝ) = 67 := by
  sorry

end NUMINAMATH_CALUDE_reeya_average_is_67_l247_24771


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l247_24788

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l247_24788


namespace NUMINAMATH_CALUDE_problem_1_l247_24703

theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l247_24703


namespace NUMINAMATH_CALUDE_abc_fraction_theorem_l247_24757

theorem abc_fraction_theorem (a b c : ℕ+) :
  ∃ (n : ℕ), n > 0 ∧ n = (a * b * c + a * b + a) / (a * b * c + b * c + c) → n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_theorem_l247_24757


namespace NUMINAMATH_CALUDE_max_value_cos_theta_l247_24777

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x

theorem max_value_cos_theta :
  ∀ θ : ℝ, (∀ x : ℝ, f x ≤ f θ) → Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_theta_l247_24777


namespace NUMINAMATH_CALUDE_valid_fraction_l247_24780

theorem valid_fraction (x : ℝ) (h : x ≠ 3) : ∃ (f : ℝ → ℝ), f x = 1 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_valid_fraction_l247_24780


namespace NUMINAMATH_CALUDE_average_age_increase_l247_24770

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 23 →
  student_avg_age = 22 →
  teacher_age = 46 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end NUMINAMATH_CALUDE_average_age_increase_l247_24770


namespace NUMINAMATH_CALUDE_divisors_of_6440_l247_24748

theorem divisors_of_6440 : 
  let n : ℕ := 6440
  let prime_factorization : List (ℕ × ℕ) := [(2, 3), (5, 1), (7, 1), (23, 1)]
  ∀ (is_valid_factorization : n = (List.foldl (λ acc (p, e) => acc * p^e) 1 prime_factorization)),
  (List.foldl (λ acc (_, e) => acc * (e + 1)) 1 prime_factorization) = 32 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_6440_l247_24748


namespace NUMINAMATH_CALUDE_difference_of_expressions_l247_24798

theorem difference_of_expressions : 
  (Real.sqrt (0.9 * 40) - (4/5 * (2/3 * 25))) = -22/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_expressions_l247_24798


namespace NUMINAMATH_CALUDE_smallest_number_proof_l247_24733

def number_set : Set ℝ := {0.8, 1/2, 0.5}

theorem smallest_number_proof :
  (∀ x ∈ number_set, x ≥ 0.1) →
  (∃ m ∈ number_set, ∀ x ∈ number_set, m ≤ x) →
  (0.5 ∈ number_set ∧ ∀ x ∈ number_set, 0.5 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l247_24733


namespace NUMINAMATH_CALUDE_range_of_a_l247_24759

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l247_24759


namespace NUMINAMATH_CALUDE_area_quadrilateral_OBEC_l247_24787

/-- A line with slope -3 passing through (3,6) -/
def line1 (x y : ℝ) : Prop := y - 6 = -3 * (x - 3)

/-- The x-coordinate of point A where line1 intersects the x-axis -/
def point_A : ℝ := 5

/-- The y-coordinate of point B where line1 intersects the y-axis -/
def point_B : ℝ := 15

/-- A line passing through points (6,0) and (3,6) -/
def line2 (x y : ℝ) : Prop := y = 2 * x - 12

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 72

theorem area_quadrilateral_OBEC :
  line1 3 6 →
  line1 point_A 0 →
  line1 0 point_B →
  line2 3 6 →
  line2 6 0 →
  area_OBEC = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_quadrilateral_OBEC_l247_24787


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_member_l247_24799

def systematic_sample (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) : Prop :=
  ∃ (start : ℕ) (k : ℕ),
    k = total / sample_size ∧
    ∀ (i : ℕ), i < sample_size →
      (start + i * k) % total + 1 ∈ known_members ∪ {(start + (sample_size - 1) * k) % total + 1}

theorem systematic_sample_fourth_member 
  (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) 
  (h_total : total = 52)
  (h_sample_size : sample_size = 4)
  (h_known_members : known_members = [6, 32, 45]) :
  systematic_sample total sample_size known_members →
  (19 : ℕ) ∈ known_members ∪ {19} :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_member_l247_24799


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l247_24713

theorem complex_product_real_imag_parts 
  (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3) 
  (h2 : Complex.abs d = 5) 
  (h3 : c * d = x + 6 * Complex.I) :
  x = 3 * Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l247_24713


namespace NUMINAMATH_CALUDE_triangle_inequality_arithmetic_sequence_l247_24714

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem triangle_inequality_arithmetic_sequence 
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_arithmetic_sequence_l247_24714


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_l247_24749

def qin_jiushao_algorithm (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  let f := λ acc coeff => acc * x + coeff
  List.scanl f 0 coeffs.reverse

def polynomial : List ℤ := [64, -192, 240, -160, 60, -12, 1]

theorem qin_jiushao_v3 :
  (qin_jiushao_algorithm polynomial 2).get! 3 = -80 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_l247_24749


namespace NUMINAMATH_CALUDE_stability_comparison_l247_24717

/-- Represents a student's scores in the competition -/
structure StudentScores where
  scores : List ℝ
  mean : ℝ
  variance : ℝ

/-- The competition has 5 rounds -/
def num_rounds : ℕ := 5

/-- Stability comparison of two students' scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : StudentScores) 
  (h1 : a.scores.length = num_rounds)
  (h2 : b.scores.length = num_rounds)
  (h3 : a.mean = 90)
  (h4 : b.mean = 90)
  (h5 : a.variance = 15)
  (h6 : b.variance = 3) :
  more_stable b a :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l247_24717


namespace NUMINAMATH_CALUDE_real_part_of_z_l247_24735

theorem real_part_of_z (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l247_24735


namespace NUMINAMATH_CALUDE_expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l247_24722

-- 1. Prove that (a+1)(a+2) = a^2 + 3a + 2
theorem expand_a_plus_one_a_plus_two (a : ℝ) : 
  (a + 1) * (a + 2) = a^2 + 3*a + 2 := by sorry

-- 2. Prove that (3a+b)(3a-b) = 9a^2 - b^2
theorem expand_three_a_plus_b_three_a_minus_b (a b : ℝ) : 
  (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by sorry

-- 3. Prove that 101^2 = 10201
theorem square_of_101 : 
  (101 : ℕ)^2 = 10201 := by sorry

-- 4. Prove that (y+2)(y-2)-(y-1)(y+5) = -4y + 1
theorem expand_and_simplify (y : ℝ) : 
  (y + 2) * (y - 2) - (y - 1) * (y + 5) = -4*y + 1 := by sorry

end NUMINAMATH_CALUDE_expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l247_24722


namespace NUMINAMATH_CALUDE_min_value_of_function_l247_24734

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (4 / x + 1 / (1 - x)) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l247_24734


namespace NUMINAMATH_CALUDE_nina_running_distance_l247_24701

theorem nina_running_distance (total : ℝ) (first : ℝ) (second_known : ℝ) 
  (h1 : total = 0.83)
  (h2 : first = 0.08)
  (h3 : second_known = 0.08) :
  total - (first + second_known) = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l247_24701


namespace NUMINAMATH_CALUDE_carson_saw_five_octopuses_l247_24790

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := 40

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := total_legs / legs_per_octopus

theorem carson_saw_five_octopuses : num_octopuses = 5 := by
  sorry

end NUMINAMATH_CALUDE_carson_saw_five_octopuses_l247_24790


namespace NUMINAMATH_CALUDE_only_101_prime_l247_24763

/-- A number in the form 101010...101 with 2n+1 digits -/
def A (n : ℕ) : ℕ := (10^(2*n+2) - 1) / 99

/-- Predicate to check if a number is in the form 101010...101 -/
def is_alternating_101 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = A n

/-- Main theorem: 101 is the only prime number with alternating 1s and 0s -/
theorem only_101_prime :
  ∀ p : ℕ, Prime p ∧ is_alternating_101 p ↔ p = 101 :=
sorry

end NUMINAMATH_CALUDE_only_101_prime_l247_24763


namespace NUMINAMATH_CALUDE_exam_mean_score_l247_24732

/-- Given an exam score distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean score is 76. -/
theorem exam_mean_score (mean std_dev : ℝ)
  (h1 : mean - 2 * std_dev = 60)
  (h2 : mean + 3 * std_dev = 100) :
  mean = 76 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l247_24732


namespace NUMINAMATH_CALUDE_regina_farm_earnings_l247_24789

def farm_earnings (num_cows : ℕ) (pig_cow_ratio : ℕ) (price_pig : ℕ) (price_cow : ℕ) : ℕ :=
  let num_pigs := pig_cow_ratio * num_cows
  (num_cows * price_cow) + (num_pigs * price_pig)

theorem regina_farm_earnings :
  farm_earnings 20 4 400 800 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_regina_farm_earnings_l247_24789


namespace NUMINAMATH_CALUDE_stretch_cosine_curve_l247_24764

/-- Given a curve y = cos x and a stretch transformation x' = 2x and y' = 3y,
    prove that the new equation of the curve is y' = 3 cos (x' / 2) -/
theorem stretch_cosine_curve (x x' y y' : ℝ) :
  y = Real.cos x →
  x' = 2 * x →
  y' = 3 * y →
  y' = 3 * Real.cos (x' / 2) := by
  sorry

end NUMINAMATH_CALUDE_stretch_cosine_curve_l247_24764


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l247_24702

theorem intersection_point_k_value (k : ℝ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 + k * p.2 = 0) ∧ 
    (2 * p.1 + 3 * p.2 + 8 = 0) ∧ 
    (p.1 - p.2 - 1 = 0)) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l247_24702


namespace NUMINAMATH_CALUDE_min_elements_for_sum_equality_l247_24740

theorem min_elements_for_sum_equality (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m = 2 * n + 2 ∧
  (∀ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) → S.card ≥ m →
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a = b + c + d) ∧
  (∀ m' : ℕ, m' < m →
    ∃ S : Finset ℕ, S ⊆ Finset.range (3 * n + 1) ∧ S.card = m' ∧
    ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a ≠ b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_min_elements_for_sum_equality_l247_24740


namespace NUMINAMATH_CALUDE_polynomial_root_product_l247_24723

theorem polynomial_root_product (y₁ y₂ y₃ : ℂ) : 
  (y₁^3 - 3*y₁ + 1 = 0) → 
  (y₂^3 - 3*y₂ + 1 = 0) → 
  (y₃^3 - 3*y₃ + 1 = 0) → 
  (y₁^3 + 2) * (y₂^3 + 2) * (y₃^3 + 2) = -26 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l247_24723


namespace NUMINAMATH_CALUDE_pentagon_area_error_percentage_l247_24712

def actual_side_A : ℝ := 10
def actual_side_B : ℝ := 20
def error_A : ℝ := 0.02
def error_B : ℝ := 0.03

def erroneous_side_A : ℝ := actual_side_A * (1 + error_A)
def erroneous_side_B : ℝ := actual_side_B * (1 - error_B)

def actual_area_factor : ℝ := actual_side_A * actual_side_B
def erroneous_area_factor : ℝ := erroneous_side_A * erroneous_side_B

theorem pentagon_area_error_percentage :
  (erroneous_area_factor - actual_area_factor) / actual_area_factor * 100 = -1.06 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_error_percentage_l247_24712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l247_24756

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_diff : a 7 - 2 * a 4 = 6) 
  (h_third : a 3 = 2) : 
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l247_24756


namespace NUMINAMATH_CALUDE_roots_equation_problem_l247_24729

theorem roots_equation_problem (x₁ x₂ m : ℝ) :
  (2 * x₁^2 - 3 * x₁ + m = 0) →
  (2 * x₂^2 - 3 * x₂ + m = 0) →
  (8 * x₁ - 2 * x₂ = 7) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_problem_l247_24729


namespace NUMINAMATH_CALUDE_roots_negatives_of_each_other_l247_24785

theorem roots_negatives_of_each_other (b : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 3 * x^2 + 2 * b * x + 9 = 0 ∧ 3 * y^2 + 2 * b * y + 9 = 0 ∧ x = -y) → 
  b = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_negatives_of_each_other_l247_24785


namespace NUMINAMATH_CALUDE_direct_proportion_through_3_6_l247_24795

/-- A direct proportion function passing through (3,6) -/
def f (x : ℝ) : ℝ := 2 * x

theorem direct_proportion_through_3_6 :
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x) ∧ 
  f 3 = 6 ∧
  ∀ x : ℝ, f x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_3_6_l247_24795


namespace NUMINAMATH_CALUDE_angle_symmetry_l247_24765

theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = π/3 + 2*k*π) →  -- Condition 1 (symmetry implies α = π/3 + 2kπ)
  α ∈ Set.Ioo (-4*π) (-2*π) →   -- Condition 2
  (α = -11*π/3 ∨ α = -5*π/3) :=
by sorry

end NUMINAMATH_CALUDE_angle_symmetry_l247_24765


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l247_24774

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem stating that 4 * 3 = 37 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 37 := by sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l247_24774


namespace NUMINAMATH_CALUDE_total_attendees_l247_24731

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 1.5

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total amount collected in dollars -/
def total_collected : ℚ := 5050

/-- The number of children who attended -/
def num_children : ℕ := 700

/-- The number of adults who attended -/
def num_adults : ℕ := 1500

/-- Theorem: The total number of people who entered the fair is 2200 -/
theorem total_attendees : num_children + num_adults = 2200 := by
  sorry

end NUMINAMATH_CALUDE_total_attendees_l247_24731


namespace NUMINAMATH_CALUDE_boat_speed_l247_24738

/-- The speed of a boat in still water, given its speeds with and against the stream -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) 
  (h1 : along_stream = 16) 
  (h2 : against_stream = 6) : 
  (along_stream + against_stream) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l247_24738


namespace NUMINAMATH_CALUDE_only_two_is_sum_of_squares_in_22_form_l247_24779

def is_22_form (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * (10^k - 1) / 9

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

theorem only_two_is_sum_of_squares_in_22_form :
  ∀ n : ℕ, is_22_form n ∧ is_sum_of_two_squares n → n = 2 :=
sorry

end NUMINAMATH_CALUDE_only_two_is_sum_of_squares_in_22_form_l247_24779


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l247_24784

/-- An arithmetic sequence with its sum function and common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  d : ℝ       -- Common difference
  sum_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2
  seq_def : ∀ n, a n = a 1 + (n - 1) * d

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l247_24784


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l247_24715

-- Define the quadratic polynomial f
def f (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem quadratic_polynomial_property 
  (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : a ≠ c)
  (hf : ∃ (p q r : ℝ), 
    f p q r a = b * c ∧ 
    f p q r b = c * a ∧ 
    f p q r c = a * b) : 
  ∃ (p q r : ℝ), f p q r (a + b + c) = a * b + b * c + a * c := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l247_24715


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l247_24709

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -2 < m ∧ m < -1

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m → m_range m :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l247_24709


namespace NUMINAMATH_CALUDE_square_field_perimeter_l247_24726

/-- Given a square field enclosed by posts, calculate the outer perimeter of the fence. -/
theorem square_field_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℝ)
  (gap_between_posts_feet : ℝ)
  (h_num_posts : num_posts = 36)
  (h_post_width : post_width_inches = 6)
  (h_gap_between : gap_between_posts_feet = 6) :
  let post_width_feet : ℝ := post_width_inches / 12
  let side_length : ℝ := (num_posts / 4 - 1) * gap_between_posts_feet + num_posts / 4 * post_width_feet
  let perimeter : ℝ := 4 * side_length
  perimeter = 236 := by
  sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l247_24726


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l247_24711

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 360 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l247_24711


namespace NUMINAMATH_CALUDE_wall_painting_contribution_l247_24745

/-- Calculates the individual contribution for a wall painting project --/
theorem wall_painting_contribution
  (total_area : ℝ)
  (coverage_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (num_coats : ℕ)
  (h_total_area : total_area = 1600)
  (h_coverage : coverage_per_gallon = 400)
  (h_cost : cost_per_gallon = 45)
  (h_coats : num_coats = 2) :
  (total_area / coverage_per_gallon * cost_per_gallon * num_coats) / 2 = 180 := by
  sorry

#check wall_painting_contribution

end NUMINAMATH_CALUDE_wall_painting_contribution_l247_24745


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l247_24736

/-- A six-digit number represented as a list of its digits -/
def SixDigitNumber := List Nat

/-- Checks if a list represents a valid six-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ ∀ d ∈ n.toFinset, 0 ≤ d ∧ d ≤ 9

/-- Converts a six-digit number to its integer value -/
def toInt (n : SixDigitNumber) : ℕ :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Left-shifts the digits of a six-digit number -/
def leftShift (n : SixDigitNumber) : SixDigitNumber :=
  match n with
  | [a, b, c, d, e, f] => [f, a, b, c, d, e]
  | _ => []

/-- The condition that needs to be satisfied -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  isValidSixDigitNumber n ∧
  toInt (leftShift n) = n.head! * toInt n

theorem six_digit_number_theorem :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (n = [1, 1, 1, 1, 1, 1] ∨ n = [1, 0, 2, 5, 6, 4]) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l247_24736


namespace NUMINAMATH_CALUDE_fraction_addition_l247_24773

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l247_24773


namespace NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l247_24768

/-- Represents a cubic structure composed of unit cubes -/
structure CubicStructure where
  size : ℕ
  deriving Repr

/-- Calculates the maximum number of visible unit cubes from a single point outside the cube -/
def maxVisibleUnitCubes (c : CubicStructure) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  maxVisibleUnitCubes ⟨12⟩ = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l247_24768


namespace NUMINAMATH_CALUDE_hall_length_l247_24707

/-- Given a rectangular hall with width 15 meters and total floor area 950 square meters,
    prove that the length of the hall is approximately 63.33 meters. -/
theorem hall_length (width : ℝ) (total_area : ℝ) (length : ℝ) : 
  width = 15 →
  total_area = 950 →
  length = total_area / width →
  ∃ ε > 0, abs (length - 63.33) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_hall_length_l247_24707


namespace NUMINAMATH_CALUDE_set_operations_l247_24704

def A : Set ℝ := {x | 1 < 2*x - 1 ∧ 2*x - 1 < 7}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∪ B) = {x | x ≤ -1 ∨ x ≥ 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l247_24704


namespace NUMINAMATH_CALUDE_equation_rewrite_and_product_l247_24742

theorem equation_rewrite_and_product (a b x y : ℝ) (m n p : ℤ) :
  a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1) →
  ∃ m n p : ℤ, (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5 ∧ m*n*p = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_rewrite_and_product_l247_24742


namespace NUMINAMATH_CALUDE_min_value_f_neg_reals_l247_24760

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x - 8/x + 4/x^2 + 5

theorem min_value_f_neg_reals :
  ∃ (x_min : ℝ), x_min < 0 ∧
  ∀ (x : ℝ), x < 0 → f x ≥ f x_min ∧ f x_min = 9 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_neg_reals_l247_24760


namespace NUMINAMATH_CALUDE_sum_in_base_5_l247_24766

/-- Converts a natural number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list representing a number in a given base to base 10 -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Adds two numbers in a given base -/
def addInBase (a b : List ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem sum_in_base_5 :
  let n1 := 29
  let n2 := 45
  let base4 := toBase n1 4
  let base5 := toBase n2 5
  let sum := addInBase base4 base5 5
  sum = [2, 4, 4] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_5_l247_24766


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l247_24708

def initial_budget : ℝ := 1000
def first_discount : ℝ := 100
def total_discount : ℝ := 280

def price_after_first_discount : ℝ := initial_budget - first_discount
def final_price : ℝ := initial_budget - total_discount
def additional_discount : ℝ := price_after_first_discount - final_price

theorem additional_discount_percentage : 
  (additional_discount / price_after_first_discount) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l247_24708


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l247_24725

-- Define the lines
def line1 (x y c : ℝ) : Prop := 3 * x - 4 * y = c
def line2 (x y c d : ℝ) : Prop := 8 * x + d * y = -c

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem intersection_of_perpendicular_lines (c d : ℝ) :
  -- Lines are perpendicular
  perpendicular (3/4) (-8/d) →
  -- Lines intersect at (2, -3)
  line1 2 (-3) c →
  line2 2 (-3) c d →
  -- Then c = 18
  c = 18 := by sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l247_24725


namespace NUMINAMATH_CALUDE_bryce_raisins_l247_24796

theorem bryce_raisins (bryce carter : ℚ) 
  (h1 : bryce = carter + 10)
  (h2 : carter = (1 / 4) * bryce) : 
  bryce = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l247_24796


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_sum_and_11_l247_24747

/-- Represents a three-digit integer -/
structure ThreeDigitInteger where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem largest_three_digit_divisible_by_sum_and_11 :
  ∃ (n : ThreeDigitInteger),
    (n.value % sum_of_digits n.value = 0) ∧
    (sum_of_digits n.value % 11 = 0) ∧
    (∀ (m : ThreeDigitInteger),
      (m.value % sum_of_digits m.value = 0) ∧
      (sum_of_digits m.value % 11 = 0) →
      m.value ≤ n.value) ∧
    n.value = 990 :=
  sorry


end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_sum_and_11_l247_24747


namespace NUMINAMATH_CALUDE_minimum_containers_l247_24791

theorem minimum_containers (medium_capacity small_capacity : ℚ) 
  (h1 : medium_capacity = 450)
  (h2 : small_capacity = 28) : 
  ⌈medium_capacity / small_capacity⌉ = 17 := by
  sorry

end NUMINAMATH_CALUDE_minimum_containers_l247_24791


namespace NUMINAMATH_CALUDE_absolute_value_problem_l247_24719

theorem absolute_value_problem (x y : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l247_24719


namespace NUMINAMATH_CALUDE_even_function_property_l247_24700

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x > 0, f x = 10^x) →  -- f(x) = 10^x for x > 0
  (∀ x < 0, f x = 10^(-x)) := by  -- f(x) = 10^(-x) for x < 0
sorry

end NUMINAMATH_CALUDE_even_function_property_l247_24700


namespace NUMINAMATH_CALUDE_mod_seven_equivalence_l247_24767

theorem mod_seven_equivalence : 47^1357 - 23^1357 ≡ 3 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_mod_seven_equivalence_l247_24767


namespace NUMINAMATH_CALUDE_square_side_length_l247_24772

theorem square_side_length (square_area rectangle_area : ℝ) 
  (rectangle_width rectangle_length : ℝ) (h1 : rectangle_width = 4) 
  (h2 : rectangle_length = 4) (h3 : square_area = rectangle_area) 
  (h4 : rectangle_area = rectangle_width * rectangle_length) : 
  ∃ (side_length : ℝ), side_length * side_length = square_area ∧ side_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l247_24772


namespace NUMINAMATH_CALUDE_fraction_of_percentages_l247_24716

theorem fraction_of_percentages (P R M N : ℝ) 
  (hM : M = 0.4 * R)
  (hR : R = 0.25 * P)
  (hN : N = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_percentages_l247_24716
