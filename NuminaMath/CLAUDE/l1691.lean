import Mathlib

namespace NUMINAMATH_CALUDE_tea_box_theorem_l1691_169149

/-- The amount of tea leaves in a box, given daily consumption and duration -/
def tea_box_amount (daily_consumption : ℚ) (weeks : ℕ) : ℚ :=
  daily_consumption * 7 * weeks

/-- Theorem: A box of tea leaves containing 28 ounces lasts 20 weeks with 1/5 ounce daily consumption -/
theorem tea_box_theorem :
  tea_box_amount (1/5) 20 = 28 := by
  sorry

#eval tea_box_amount (1/5) 20

end NUMINAMATH_CALUDE_tea_box_theorem_l1691_169149


namespace NUMINAMATH_CALUDE_function_periodicity_l1691_169170

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_cond : satisfies_condition f) : 
  periodic f 20 := by sorry

end NUMINAMATH_CALUDE_function_periodicity_l1691_169170


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1691_169100

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of three consecutive terms in an arithmetic sequence -/
def sum_three_consecutive (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  sum_three_consecutive a 4 = 36 →
  a 1 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1691_169100


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l1691_169163

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → (4*x + 3*y ≤ 42) ∧ ∃ x y, x^2 + y^2 = 16*x + 8*y + 10 ∧ 4*x + 3*y = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l1691_169163


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1691_169189

/-- The cost ratio of a muffin to a banana given Susie and Calvin's purchases -/
theorem muffin_banana_cost_ratio :
  ∀ (m b c : ℚ),
  (5 * m + 2 * b + 3 * c = 1) →  -- Normalize Susie's purchase to 1
  (4 * m + 18 * b + c = 3) →     -- Calvin's purchase is 3 times Susie's
  (c = 2 * b) →                  -- A cookie costs twice as much as a banana
  (m / b = 4 / 11) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1691_169189


namespace NUMINAMATH_CALUDE_bd_length_l1691_169135

/-- Given four points A, B, C, and D on a line in that order, prove that BD = 6 -/
theorem bd_length 
  (A B C D : ℝ) -- Points represented as real numbers on a line
  (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D) -- Order of points on the line
  (h_AB : B - A = 2) -- Length of AB
  (h_AC : C - A = 5) -- Length of AC
  (h_CD : D - C = 3) -- Length of CD
  : D - B = 6 := by
  sorry

end NUMINAMATH_CALUDE_bd_length_l1691_169135


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l1691_169192

theorem complex_symmetry_product : 
  ∀ (z₁ z₂ : ℂ), 
  z₁ = 3 + 2*I → 
  (z₂.re = z₁.im ∧ z₂.im = z₁.re) → 
  z₁ * z₂ = 13*I := by
sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l1691_169192


namespace NUMINAMATH_CALUDE_ana_number_puzzle_l1691_169191

theorem ana_number_puzzle (x : ℝ) : (((x + 3) * 3 - 4) / 2) = 10 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ana_number_puzzle_l1691_169191


namespace NUMINAMATH_CALUDE_line_point_k_value_l1691_169113

/-- Given a line containing points (2, 9), (10, k), and (25, 4), prove that k = 167/23 -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 9 = m * 2 + b ∧ k = m * 10 + b ∧ 4 = m * 25 + b) → 
  k = 167 / 23 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l1691_169113


namespace NUMINAMATH_CALUDE_monotonic_f_range_l1691_169123

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 3 else (a + 2) * Real.exp (a * x)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_f_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_range_l1691_169123


namespace NUMINAMATH_CALUDE_M_mod_51_l1691_169152

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 15 := by sorry

end NUMINAMATH_CALUDE_M_mod_51_l1691_169152


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l1691_169147

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.B = 75 ∧
  d t.A t.C = 45 ∧
  d t.B t.C = 90 ∧
  -- X is on the angle bisector of angle ACB
  (t.X.1 - t.A.1) / (t.C.1 - t.A.1) = (t.X.2 - t.A.2) / (t.C.2 - t.A.2) ∧
  (t.X.1 - t.B.1) / (t.C.1 - t.B.1) = (t.X.2 - t.B.2) / (t.C.2 - t.B.2)

-- Theorem statement
theorem angle_bisector_theorem (t : Triangle) (h : is_valid_triangle t) :
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.X = 25 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l1691_169147


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l1691_169188

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l1691_169188


namespace NUMINAMATH_CALUDE_johns_spending_l1691_169136

theorem johns_spending (initial_amount : ℚ) (snack_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : snack_fraction = 1/5)
  (h3 : final_amount = 4) :
  let remaining_after_snacks := initial_amount - snack_fraction * initial_amount
  (remaining_after_snacks - final_amount) / remaining_after_snacks = 3/4 := by
sorry

end NUMINAMATH_CALUDE_johns_spending_l1691_169136


namespace NUMINAMATH_CALUDE_portrait_problem_l1691_169179

theorem portrait_problem (total_students : ℕ) (before_lunch : ℕ) (after_lunch : ℕ) 
  (h1 : total_students = 24)
  (h2 : before_lunch = total_students / 3)
  (h3 : after_lunch = 10) :
  total_students - (before_lunch + after_lunch) = 6 := by
  sorry

end NUMINAMATH_CALUDE_portrait_problem_l1691_169179


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1691_169110

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The 9th term is the arithmetic mean of 1 and 3 -/
def ninth_term_is_mean (b : ℕ → ℝ) : Prop :=
  b 9 = (1 + 3) / 2

theorem geometric_sequence_product (b : ℕ → ℝ) 
  (h1 : geometric_sequence b) 
  (h2 : ninth_term_is_mean b) : 
  b 2 * b 16 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1691_169110


namespace NUMINAMATH_CALUDE_zero_power_is_zero_l1691_169198

theorem zero_power_is_zero (n : ℕ) : 0^n = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_power_is_zero_l1691_169198


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_l1691_169172

theorem sum_of_odd_powers (x y z a : ℝ) (k : ℕ) 
  (h1 : x + y + z = a) 
  (h2 : x^3 + y^3 + z^3 = a^3) 
  (h3 : Odd k) : 
  x^k + y^k + z^k = a^k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_l1691_169172


namespace NUMINAMATH_CALUDE_smallest_multiple_l1691_169141

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n ≡ 3 [ZMOD 71] ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m ≡ 3 [ZMOD 71])) → 
  n = 1139 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1691_169141


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_l1691_169128

theorem cos_two_thirds_pi : Real.cos (2/3 * Real.pi) = -(1/2) := by sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_l1691_169128


namespace NUMINAMATH_CALUDE_rod_cutting_l1691_169130

theorem rod_cutting (rod_length_m : ℝ) (piece_length_cm : ℝ) : 
  rod_length_m = 38.25 →
  piece_length_cm = 85 →
  ⌊(rod_length_m * 100) / piece_length_cm⌋ = 45 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l1691_169130


namespace NUMINAMATH_CALUDE_complex_number_location_l1691_169160

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1691_169160


namespace NUMINAMATH_CALUDE_perpendicular_point_sets_l1691_169154

-- Definition of a perpendicular point set
def is_perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets M₃ and M₄
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem statement
theorem perpendicular_point_sets :
  is_perpendicular_point_set M₃ ∧ is_perpendicular_point_set M₄ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_sets_l1691_169154


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l1691_169178

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ) (incorrect_sums : ℕ),
    correct_sums + incorrect_sums = total_sums ∧
    (correct_sums : ℤ) * correct_marks - incorrect_sums * incorrect_marks = total_marks ∧
    correct_sums = 25 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l1691_169178


namespace NUMINAMATH_CALUDE_sequence_property_l1691_169118

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l1691_169118


namespace NUMINAMATH_CALUDE_calculate_expression_l1691_169182

theorem calculate_expression : (-1)^2 + (1/3)^0 = 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1691_169182


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1691_169185

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 5, 3]
def base1 : Nat := 8
def den1 : List Nat := [1, 3]
def base2 : Nat := 4
def num2 : List Nat := [1, 4, 4]
def base3 : Nat := 5
def den2 : List Nat := [2, 2]
def base4 : Nat := 3

-- State the theorem
theorem base_conversion_sum :
  (baseToDecimal num1 base1 : Rat) / (baseToDecimal den1 base2 : Rat) +
  (baseToDecimal num2 base3 : Rat) / (baseToDecimal den2 base4 : Rat) =
  30.125 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1691_169185


namespace NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l1691_169117

theorem smallest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ 2^(1 - n)) ∧
  (∀ (m : ℕ), m > 0 → m < n →
    ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > 2^(1 - m)) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_trig_inequality_l1691_169117


namespace NUMINAMATH_CALUDE_max_draws_for_cmwmc_l1691_169168

/-- Represents the number of tiles of each letter in the bag -/
structure TileCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- Represents the number of tiles needed to spell the word -/
structure WordCounts :=
  (c : Nat)
  (m : Nat)
  (w : Nat)

/-- The maximum number of tiles that need to be drawn -/
def maxDraws (bag : TileCounts) (word : WordCounts) : Nat :=
  bag.c + bag.m + bag.w - (word.c - 1) - (word.m - 1) - (word.w - 1)

/-- Theorem stating the maximum number of draws for the given problem -/
theorem max_draws_for_cmwmc :
  let bag := TileCounts.mk 8 8 8
  let word := WordCounts.mk 2 2 1
  maxDraws bag word = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_draws_for_cmwmc_l1691_169168


namespace NUMINAMATH_CALUDE_bank_balance_deduction_l1691_169159

theorem bank_balance_deduction (X : ℝ) (current_balance : ℝ) : 
  current_balance = X * 0.9 ∧ current_balance = 90000 → X = 100000 := by
sorry

end NUMINAMATH_CALUDE_bank_balance_deduction_l1691_169159


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l1691_169143

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 1000 * a + 100 * (b - a) + n % 100 ∧ 
                10 ≤ a ∧ a < 100 ∧ 
                0 ≤ b - a ∧ b - a < 100 ∧
                n = (a + (n % 100))^2

theorem special_numbers_theorem : 
  {n : ℕ | is_special_number n} = {3025, 2025, 9801} := by sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l1691_169143


namespace NUMINAMATH_CALUDE_sequence_inequality_l1691_169125

/-- The sequence a_n defined by n^2 + kn + 2 -/
def a (n : ℕ) (k : ℝ) : ℝ := n^2 + k * n + 2

/-- Theorem stating that if a_n ≥ a_4 for all n ≥ 4, then k is in [-9, -7] -/
theorem sequence_inequality (k : ℝ) :
  (∀ n : ℕ, n ≥ 4 → a n k ≥ a 4 k) →
  k ∈ Set.Icc (-9 : ℝ) (-7 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1691_169125


namespace NUMINAMATH_CALUDE_inequality_proof_l1691_169164

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : Set.Icc 0 2 = {x | f x ((1/m) + (1/(2*n))) ≤ 1}) : 
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1691_169164


namespace NUMINAMATH_CALUDE_logarithm_power_sum_l1691_169197

theorem logarithm_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_power_sum_l1691_169197


namespace NUMINAMATH_CALUDE_min_steps_to_one_l1691_169137

/-- Represents the allowed operations in one step -/
inductive Operation
  | AddOne
  | DivideByTwo
  | DivideByThree

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.DivideByTwo => n / 2
  | Operation.DivideByThree => n / 3

/-- Checks if a sequence of operations is valid -/
def isValidSequence (start : ℕ) (ops : List Operation) : Bool :=
  ops.foldl (fun acc op => applyOperation acc op) start = 1

/-- The minimum number of steps to reach 1 from the starting number -/
def minSteps (start : ℕ) : ℕ :=
  sorry

theorem min_steps_to_one :
  minSteps 19 = 6 :=
sorry

end NUMINAMATH_CALUDE_min_steps_to_one_l1691_169137


namespace NUMINAMATH_CALUDE_haley_initial_lives_l1691_169120

theorem haley_initial_lives : 
  ∀ (initial_lives : ℕ), 
    (initial_lives - 4 + 36 = 46) → 
    initial_lives = 14 := by
  sorry

end NUMINAMATH_CALUDE_haley_initial_lives_l1691_169120


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1691_169112

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales
  (total_tickets : ℕ)
  (first_day_sales : ℕ)
  (third_day_sales : ℕ)
  (h1 : total_tickets = 80)
  (h2 : first_day_sales = 20)
  (h3 : third_day_sales = 28) :
  total_tickets - first_day_sales - third_day_sales = 32 := by
  sorry

#check amanda_ticket_sales

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1691_169112


namespace NUMINAMATH_CALUDE_third_square_is_G_l1691_169109

/-- Represents a 2x2 square -/
structure Square :=
  (label : Char)

/-- Represents the visibility of a square -/
inductive Visibility
  | Full
  | Partial

/-- Represents the position of a square in the 4x4 grid -/
structure Position :=
  (row : Fin 2)
  (col : Fin 2)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Option Square

/-- Represents the sequence of square placements -/
def PlacementSequence := List Square

/-- Determines if a square is in a corner position -/
def isCorner (pos : Position) : Bool :=
  (pos.row = 0 ∨ pos.row = 1) ∧ (pos.col = 0 ∨ pos.col = 1)

/-- The main theorem to prove -/
theorem third_square_is_G 
  (squares : List Square)
  (grid : Grid)
  (sequence : PlacementSequence)
  (visibility : Square → Visibility)
  (position : Square → Position) :
  squares.length = 8 ∧
  (∃ s ∈ squares, s.label = 'E') ∧
  visibility (Square.mk 'E') = Visibility.Full ∧
  (∀ s ∈ squares, s.label ≠ 'E' → visibility s = Visibility.Partial) ∧
  (∃ s ∈ squares, isCorner (position s) ∧ s.label = 'G') ∧
  sequence.length = 8 ∧
  sequence.getLast? = some (Square.mk 'E') →
  (sequence.get? 2 = some (Square.mk 'G')) :=
by sorry

end NUMINAMATH_CALUDE_third_square_is_G_l1691_169109


namespace NUMINAMATH_CALUDE_triangle_polynomial_roots_l1691_169193

theorem triangle_polynomial_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hac : a + c > b) (hbc : b + c > a) :
  ¬ (∃ x y : ℝ, x < 1/3 ∧ y < 1/3 ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_polynomial_roots_l1691_169193


namespace NUMINAMATH_CALUDE_jason_total_games_l1691_169153

/-- The number of football games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of football games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of football games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l1691_169153


namespace NUMINAMATH_CALUDE_problem_grid_square_count_l1691_169183

/-- Represents a square grid with some line segments removed -/
structure SquareGrid :=
  (size : Nat)
  (removed_segments : List (Nat × Nat × Nat × Nat))

/-- Counts the number of squares in a SquareGrid -/
def count_squares (grid : SquareGrid) : Nat :=
  sorry

/-- The specific 4x4 grid with two line segments removed as described in the problem -/
def problem_grid : SquareGrid :=
  { size := 4,
    removed_segments := [(1, 1, 1, 2), (2, 2, 3, 2)] }

/-- Theorem stating that the number of squares in the problem grid is 22 -/
theorem problem_grid_square_count :
  count_squares problem_grid = 22 := by sorry

end NUMINAMATH_CALUDE_problem_grid_square_count_l1691_169183


namespace NUMINAMATH_CALUDE_tagged_fish_count_l1691_169132

/-- The number of tagged fish found in the second catch -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Proof that the number of tagged fish in the second catch is 2 -/
theorem tagged_fish_count :
  let total_fish : ℕ := 1800
  let initially_tagged : ℕ := 60
  let second_catch : ℕ := 60
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_count_l1691_169132


namespace NUMINAMATH_CALUDE_sum_four_digit_numbers_eq_179982_l1691_169148

/-- The sum of all four-digit numbers created using digits 1, 2, and 3 with repetition -/
def sum_four_digit_numbers : ℕ :=
  let digits : List ℕ := [1, 2, 3]
  let total_numbers : ℕ := digits.length ^ 4
  let sum_per_position : ℕ := (digits.sum * total_numbers) / digits.length
  sum_per_position * 1000 + sum_per_position * 100 + sum_per_position * 10 + sum_per_position

theorem sum_four_digit_numbers_eq_179982 :
  sum_four_digit_numbers = 179982 := by
  sorry

#eval sum_four_digit_numbers

end NUMINAMATH_CALUDE_sum_four_digit_numbers_eq_179982_l1691_169148


namespace NUMINAMATH_CALUDE_cost_per_box_l1691_169121

-- Define the box dimensions
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

-- Define the total volume of the collection
def total_volume : ℝ := 1920000

-- Define the minimum total cost for boxes
def min_total_cost : ℝ := 200

-- Theorem to prove
theorem cost_per_box :
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  let cost_per_box := min_total_cost / num_boxes
  cost_per_box = 0.5 := by sorry

end NUMINAMATH_CALUDE_cost_per_box_l1691_169121


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l1691_169194

/-- A line in 2D space represented by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : ParametricLine) : Prop :=
  ∃ m1 m2 : ℝ, (∀ t : ℝ, l1.y t = m1 * l1.x t + (l1.y 0 - m1 * l1.x 0)) ∧
              (∀ s : ℝ, l2.y s = m2 * l2.x s + (l2.y 0 - m2 * l2.x 0)) ∧
              m1 * m2 = -1

theorem perpendicular_lines_k_value :
  ∀ k : ℝ,
  let l1 : ParametricLine := {
    x := λ t => 1 - 2*t,
    y := λ t => 2 + k*t
  }
  let l2 : ParametricLine := {
    x := λ s => s,
    y := λ s => 1 - 2*s
  }
  perpendicular l1 l2 → k = -1 := by
  sorry

#check perpendicular_lines_k_value

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l1691_169194


namespace NUMINAMATH_CALUDE_mike_pears_l1691_169145

theorem mike_pears (jason_pears keith_pears total_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : total_pears = 105)
  (h4 : ∃ mike_pears : ℕ, jason_pears + keith_pears + mike_pears = total_pears) :
  ∃ mike_pears : ℕ, mike_pears = 12 ∧ jason_pears + keith_pears + mike_pears = total_pears := by
sorry

end NUMINAMATH_CALUDE_mike_pears_l1691_169145


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1691_169158

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1691_169158


namespace NUMINAMATH_CALUDE_non_right_triangle_l1691_169131

theorem non_right_triangle : 
  let triangle_sets : List (ℝ × ℝ × ℝ) := 
    [(6, 8, 10), (1, Real.sqrt 3, 2), (5/4, 1, 3/4), (4, 5, 7)]
  ∀ (a b c : ℝ), (a, b, c) ∈ triangle_sets →
    (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ↔ (a, b, c) ≠ (4, 5, 7) :=
by sorry

end NUMINAMATH_CALUDE_non_right_triangle_l1691_169131


namespace NUMINAMATH_CALUDE_jason_has_four_balloons_l1691_169129

/-- The number of violet balloons Jason has now, given his initial count and the number he lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason has 4 violet balloons now. -/
theorem jason_has_four_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_four_balloons_l1691_169129


namespace NUMINAMATH_CALUDE_fraction_simplification_l1691_169190

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 + x) / (x^2 - 1) = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1691_169190


namespace NUMINAMATH_CALUDE_unique_solution_l1691_169181

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 13 ∧ y = 11 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1691_169181


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l1691_169103

/-- Represents a parallelepiped with integer dimensions --/
structure Parallelepiped where
  width : ℕ
  depth : ℕ
  height : ℕ

/-- Represents a square with an integer side length --/
structure Square where
  side : ℕ

/-- Checks if a set of squares can cover a parallelepiped without gaps or overlaps --/
def can_cover (p : Parallelepiped) (squares : List Square) : Prop :=
  let surface_area := 2 * (p.width * p.depth + p.width * p.height + p.depth * p.height)
  let squares_area := squares.map (λ s => s.side * s.side) |>.sum
  surface_area = squares_area

theorem parallelepiped_coverage : 
  let p := Parallelepiped.mk 1 1 4
  let squares := [Square.mk 4, Square.mk 1, Square.mk 1]
  can_cover p squares := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l1691_169103


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1691_169171

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (3 - 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 3 - 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant : discriminant a b c = 9/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1691_169171


namespace NUMINAMATH_CALUDE_opposite_of_seven_l1691_169122

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l1691_169122


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_l1691_169126

theorem quadratic_equation_completion (x k ℓ : ℝ) : 
  (13 * x^2 + 39 * x - 91 = 0) ∧ 
  ((x + k)^2 - |ℓ| = 0) →
  |k + ℓ| = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_l1691_169126


namespace NUMINAMATH_CALUDE_simplify_expression_l1691_169187

theorem simplify_expression (x : ℝ) : (3*x)^4 + (4*x)*(x^5) = 81*x^4 + 4*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1691_169187


namespace NUMINAMATH_CALUDE_fast_food_purchase_cost_l1691_169140

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem fast_food_purchase_cost : total_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_purchase_cost_l1691_169140


namespace NUMINAMATH_CALUDE_work_completion_solution_l1691_169142

/-- Represents the work completion problem -/
structure WorkCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  added_workers : ℕ

/-- Calculates the total days to complete the work -/
def total_days (w : WorkCompletion) : ℚ :=
  let initial_work_rate : ℚ := 1 / (w.initial_workers * w.initial_days)
  let work_done : ℚ := w.worked_days * w.initial_workers * initial_work_rate
  let remaining_work : ℚ := 1 - work_done
  let new_work_rate : ℚ := (w.initial_workers + w.added_workers) * initial_work_rate
  w.worked_days + remaining_work / new_work_rate

/-- Theorem stating the solution to the work completion problem -/
theorem work_completion_solution :
  ∀ w : WorkCompletion,
    w.initial_workers = 12 →
    w.initial_days = 18 →
    w.worked_days = 6 →
    w.added_workers = 4 →
    total_days w = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_solution_l1691_169142


namespace NUMINAMATH_CALUDE_q_div_p_eq_225_l1691_169101

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The ratio of q to p is 225 -/
theorem q_div_p_eq_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_eq_225_l1691_169101


namespace NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1691_169107

theorem p_or_q_necessary_not_sufficient :
  (∀ p q : Prop, (¬p → (p ∨ q))) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(¬p → False)) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_necessary_not_sufficient_l1691_169107


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1691_169144

/-- Given a line passing through points (1, 3) and (3, 7) with equation y = mx + b, 
    the sum of m and b is equal to 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1691_169144


namespace NUMINAMATH_CALUDE_sally_final_count_l1691_169169

def sally_pokemon_cards (initial : ℕ) (from_dan : ℕ) (bought : ℕ) : ℕ :=
  initial + from_dan + bought

theorem sally_final_count :
  sally_pokemon_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sally_final_count_l1691_169169


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_three_i_squared_l1691_169151

theorem imaginary_part_of_one_minus_three_i_squared : 
  Complex.im ((1 - 3*Complex.I)^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_three_i_squared_l1691_169151


namespace NUMINAMATH_CALUDE_function_identity_l1691_169139

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 1/2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1691_169139


namespace NUMINAMATH_CALUDE_cookies_per_neighbor_l1691_169116

/-- Proves the number of cookies each neighbor was supposed to take -/
theorem cookies_per_neighbor
  (total_cookies : ℕ)
  (num_neighbors : ℕ)
  (cookies_left : ℕ)
  (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : num_neighbors = 15)
  (h3 : cookies_left = 8)
  (h4 : sarah_cookies = 12)
  : total_cookies / num_neighbors = 10 := by
  sorry

#check cookies_per_neighbor

end NUMINAMATH_CALUDE_cookies_per_neighbor_l1691_169116


namespace NUMINAMATH_CALUDE_correct_propositions_l1691_169134

theorem correct_propositions :
  -- Proposition 1
  (∀ P : Prop, (¬P ↔ P) → ¬P) ∧
  -- Proposition 2 (negation)
  ¬(∀ a : ℕ → ℝ, a 0 = 2 ∧ (∀ n : ℕ, a (n + 1) = a n + (a 2 - a 0) / 2) ∧
    (∃ q : ℝ, a 2 = a 0 * q ∧ a 3 = a 2 * q) →
    a 1 - a 0 = -1/2) ∧
  -- Proposition 3
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 →
    (2/a + 3/b ≥ 5 + 2 * Real.sqrt 6)) ∧
  -- Proposition 4 (negation)
  ¬(∀ A B C : ℝ, 0 ≤ A ∧ A ≤ π ∧ 0 ≤ B ∧ B ≤ π ∧ 0 ≤ C ∧ C ≤ π ∧ A + B + C = π →
    (Real.sin A)^2 < (Real.sin B)^2 + (Real.sin C)^2 →
    A < π/2 ∧ B < π/2 ∧ C < π/2) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l1691_169134


namespace NUMINAMATH_CALUDE_poly_sequence_properties_l1691_169184

/-- Represents a polynomial sequence generated by the given operation -/
def PolySequence (a : ℝ) (n : ℕ) : List ℝ :=
  sorry

/-- The product of all polynomials in the sequence after n operations -/
def PolyProduct (a : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The sum of all polynomials in the sequence after n operations -/
def PolySum (a : ℝ) (n : ℕ) : ℝ :=
  sorry

theorem poly_sequence_properties (a : ℝ) :
  (∀ a, |a| ≥ 2 → PolyProduct a 2 ≤ 0) ∧
  (∀ n, PolySum a n = 2*a + 2*(n+1)) :=
by sorry

end NUMINAMATH_CALUDE_poly_sequence_properties_l1691_169184


namespace NUMINAMATH_CALUDE_min_l_shapes_5x5_grid_l1691_169156

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- An L-shaped figure made of 3 cells --/
structure LShape where
  x : Fin 5
  y : Fin 5
  orientation : Fin 4

/-- Check if an L-shape is within the grid bounds --/
def LShape.isValid (l : LShape) : Bool :=
  match l.orientation with
  | 0 => l.x < 4 ∧ l.y < 4
  | 1 => l.x > 0 ∧ l.y < 4
  | 2 => l.x < 4 ∧ l.y > 0
  | 3 => l.x > 0 ∧ l.y > 0

/-- Check if two L-shapes overlap --/
def LShape.overlaps (l1 l2 : LShape) : Bool :=
  sorry

/-- Check if a set of L-shapes is valid (non-overlapping and within bounds) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- Check if no more L-shapes can be added to a given set of shapes --/
def isMaximalPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_shapes_5x5_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    isMaximalPlacement shapes ∧
    ∀ (otherShapes : List LShape),
      isValidPlacement otherShapes ∧ isMaximalPlacement otherShapes →
      otherShapes.length ≥ 4 :=
  sorry

end NUMINAMATH_CALUDE_min_l_shapes_5x5_grid_l1691_169156


namespace NUMINAMATH_CALUDE_y_value_at_50_l1691_169174

/-- A line passing through given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Theorem: Y-coordinate when X is 50 on a specific line -/
theorem y_value_at_50 (l : Line) (u : ℝ) : 
  l.x1 = 10 ∧ l.y1 = 30 ∧ 
  l.x2 = 15 ∧ l.y2 = 45 ∧ 
  (∃ y3 : ℝ, y3 = 3 * 20 ∧ Line.mk 10 30 20 y3 = l) ∧
  (∃ y4 : ℝ, y4 = u ∧ Line.mk 10 30 40 y4 = l) →
  (∃ y : ℝ, y = 150 ∧ Line.mk 10 30 50 y = l) :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_50_l1691_169174


namespace NUMINAMATH_CALUDE_skating_rink_visitors_l1691_169175

/-- The number of people at a skating rink at noon, given the initial number of visitors,
    the number of people who left, and the number of new arrivals. -/
def people_at_noon (initial : ℕ) (left : ℕ) (arrived : ℕ) : ℕ :=
  initial - left + arrived

/-- Theorem stating that the number of people at the skating rink at noon is 280,
    given the specific values from the problem. -/
theorem skating_rink_visitors : people_at_noon 264 134 150 = 280 := by
  sorry

end NUMINAMATH_CALUDE_skating_rink_visitors_l1691_169175


namespace NUMINAMATH_CALUDE_ellipse_center_locus_l1691_169177

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Represents a right angle -/
structure RightAngle where
  vertex : Point

/-- Predicate to check if an ellipse touches both sides of a right angle -/
def touches_right_angle (e : Ellipse) (ra : RightAngle) : Prop :=
  sorry

/-- The locus of the center of the ellipse -/
def center_locus (ra : RightAngle) (a b : ℝ) : Set Point :=
  {p : Point | ∃ e : Ellipse, e.center = p ∧ e.semi_major_axis = a ∧ e.semi_minor_axis = b ∧ touches_right_angle e ra}

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on an arc of a circle -/
def on_circle_arc (p : Point) (c : Circle) : Prop :=
  sorry

theorem ellipse_center_locus (ra : RightAngle) (a b : ℝ) :
  ∃ c : Circle, c.center = ra.vertex ∧ ∀ p ∈ center_locus ra a b, on_circle_arc p c :=
sorry

end NUMINAMATH_CALUDE_ellipse_center_locus_l1691_169177


namespace NUMINAMATH_CALUDE_middle_legs_arrangements_adjacent_legs_arrangements_l1691_169106

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of athletes needed for the relay -/
def relay_size : ℕ := 4

/-- The number of ways to arrange n items taken r at a time -/
def permutations (n r : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- The number of ways to choose r items from n items -/
def combinations (n r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Theorem for the number of arrangements with A and B running the middle two legs -/
theorem middle_legs_arrangements : 
  permutations 2 2 * permutations (total_athletes - 2) (relay_size - 2) = 24 := by sorry

/-- Theorem for the number of arrangements with A and B running adjacent legs -/
theorem adjacent_legs_arrangements : 
  permutations 2 2 * combinations (total_athletes - 2) (relay_size - 2) * permutations 3 3 = 72 := by sorry

end NUMINAMATH_CALUDE_middle_legs_arrangements_adjacent_legs_arrangements_l1691_169106


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l1691_169114

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to cross each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (crossing_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * (5/18)
  let length2 := relative_speed_mps * crossing_time - length1
  length2

/-- The length of the second train is approximately 159.97 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 60 40 11.879049676025918 170 - 159.97| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l1691_169114


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fifteen_fourths_l1691_169124

theorem greatest_integer_less_than_negative_fifteen_fourths :
  ⌊-15/4⌋ = -4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fifteen_fourths_l1691_169124


namespace NUMINAMATH_CALUDE_sum_of_digits_is_three_l1691_169146

/-- Represents a 100-digit number with repeating pattern 5050 --/
def a : ℕ := 5050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050

/-- Represents a 100-digit number with repeating pattern 7070 --/
def b : ℕ := 7070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070

/-- The product of a and b --/
def product : ℕ := a * b

/-- Extracts the thousands digit from a number --/
def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

/-- Extracts the units digit from a number --/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the sum of the thousands digit and units digit of the product is 3 --/
theorem sum_of_digits_is_three : 
  thousands_digit product + units_digit product = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_three_l1691_169146


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l1691_169138

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define an interior point
def interior_point (q : Quadrilateral) (O : ℝ × ℝ) : Prop := sorry

-- Define distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define s₃ and s₄
def s₃ (q : Quadrilateral) (O : ℝ × ℝ) : ℝ :=
  distance O q.A + distance O q.B + distance O q.C + distance O q.D

def s₄ (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) (O : ℝ × ℝ) :
  interior_point q O →
  triangle_area O q.A q.B = triangle_area O q.C q.D →
  s₃ q O ≥ (1/2) * s₄ q ∧ s₃ q O ≤ s₄ q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l1691_169138


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l1691_169161

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (drama_or_science_students : ℕ) 
  (h1 : total_students = 500)
  (h2 : drama_students = 150)
  (h3 : science_students = 200)
  (h4 : drama_or_science_students = 300) :
  drama_students + science_students - drama_or_science_students = 50 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l1691_169161


namespace NUMINAMATH_CALUDE_solve_for_m_l1691_169127

theorem solve_for_m : ∃ m : ℝ, (-1 : ℝ) - 2 * m = 9 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1691_169127


namespace NUMINAMATH_CALUDE_jeans_discount_percentage_l1691_169133

theorem jeans_discount_percentage (original_price : ℝ) (coupon : ℝ) (card_discount : ℝ) (total_savings : ℝ) :
  original_price = 125 →
  coupon = 10 →
  card_discount = 0.1 →
  total_savings = 44 →
  ∃ (sale_discount : ℝ),
    sale_discount = 0.2 ∧
    (original_price - sale_discount * original_price - coupon) * (1 - card_discount) = original_price - total_savings :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_percentage_l1691_169133


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1691_169166

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1691_169166


namespace NUMINAMATH_CALUDE_decimal_to_binary_34_l1691_169186

theorem decimal_to_binary_34 : 
  (34 : ℕ) = (1 * 2^5 + 0 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_34_l1691_169186


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1691_169157

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 15
  let bars_per_small_box : ℕ := 25
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 375 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1691_169157


namespace NUMINAMATH_CALUDE_root_product_theorem_l1691_169195

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 + x₁^2 + 1 = 0) → 
  (x₂^5 + x₂^2 + 1 = 0) → 
  (x₃^5 + x₃^2 + 1 = 0) → 
  (x₄^5 + x₄^2 + 1 = 0) → 
  (x₅^5 + x₅^2 + 1 = 0) → 
  (x₁^3 - 2) * (x₂^3 - 2) * (x₃^3 - 2) * (x₄^3 - 2) * (x₅^3 - 2) = -243 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1691_169195


namespace NUMINAMATH_CALUDE_arrange_balls_and_boxes_eq_20_l1691_169102

/-- The number of ways to arrange 5 balls in 5 boxes with exactly two matches -/
def arrange_balls_and_boxes : ℕ :=
  let n : ℕ := 5  -- Total number of balls and boxes
  let k : ℕ := 2  -- Number of matches required
  let derangement_3 : ℕ := 2  -- Number of derangements for 3 elements
  (n.choose k) * derangement_3

/-- Theorem stating that the number of arrangements is 20 -/
theorem arrange_balls_and_boxes_eq_20 : arrange_balls_and_boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_and_boxes_eq_20_l1691_169102


namespace NUMINAMATH_CALUDE_advance_ticket_cost_l1691_169119

/-- The cost of advance tickets is $20, given the specified conditions. -/
theorem advance_ticket_cost (same_day_cost : ℕ) (total_tickets : ℕ) (total_receipts : ℕ) (advance_tickets_sold : ℕ) :
  same_day_cost = 30 →
  total_tickets = 60 →
  total_receipts = 1600 →
  advance_tickets_sold = 20 →
  ∃ (advance_cost : ℕ), advance_cost * advance_tickets_sold + same_day_cost * (total_tickets - advance_tickets_sold) = total_receipts ∧ advance_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_advance_ticket_cost_l1691_169119


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1691_169155

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- Theorem statement
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l1691_169155


namespace NUMINAMATH_CALUDE_reflection_of_M_l1691_169167

/-- Reflection of a point about the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_of_M :
  let M : ℝ × ℝ := (3, 2)
  reflect_x M = (3, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l1691_169167


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1691_169162

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1691_169162


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_minimum_m_for_intersection_l1691_169108

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x > 2} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Part II
-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem minimum_m_for_intersection :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_minimum_m_for_intersection_l1691_169108


namespace NUMINAMATH_CALUDE_student_community_selection_l1691_169199

/-- The number of ways to select communities for students. -/
def ways_to_select (num_students : ℕ) (num_communities : ℕ) : ℕ :=
  num_communities ^ num_students

/-- Theorem: Given 4 students and 3 communities, where each student chooses 1 community,
    the number of different ways of selection is 3^4. -/
theorem student_community_selection :
  ways_to_select 4 3 = 3^4 := by
  sorry

#eval ways_to_select 4 3  -- Should output 81

end NUMINAMATH_CALUDE_student_community_selection_l1691_169199


namespace NUMINAMATH_CALUDE_josh_shopping_cost_l1691_169165

def film_cost : ℕ := 5
def book_cost : ℕ := 4
def cd_cost : ℕ := 3

def num_films : ℕ := 9
def num_books : ℕ := 4
def num_cds : ℕ := 6

theorem josh_shopping_cost : 
  (num_films * film_cost + num_books * book_cost + num_cds * cd_cost) = 79 := by
  sorry

end NUMINAMATH_CALUDE_josh_shopping_cost_l1691_169165


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1691_169150

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r = (a + b - c) / 2 ∧ r > 0 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1691_169150


namespace NUMINAMATH_CALUDE_fibLastDigitsCyclic_l1691_169176

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Sequence of last digits of Fibonacci numbers -/
def fibLastDigits : ℕ → ℕ := λ n => lastDigit (fib n)

/-- Period of a sequence -/
def isPeriodic (f : ℕ → ℕ) (p : ℕ) : Prop :=
  ∀ n, f (n + p) = f n

/-- Theorem: The sequence of last digits of Fibonacci numbers is cyclic -/
theorem fibLastDigitsCyclic : ∃ p : ℕ, p > 0 ∧ isPeriodic fibLastDigits p :=
  sorry

end NUMINAMATH_CALUDE_fibLastDigitsCyclic_l1691_169176


namespace NUMINAMATH_CALUDE_generating_function_value_at_one_intersection_point_on_generating_function_l1691_169104

/-- Linear function -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Generating function of two linear functions -/
def generatingFunction (f₁ f₂ : LinearFunction) (m n : ℝ) (x : ℝ) : ℝ :=
  m * (f₁.a * x + f₁.b) + n * (f₂.a * x + f₂.b)

/-- Theorem: The value of the generating function of y = x + 1 and y = 2x when x = 1 is 2 -/
theorem generating_function_value_at_one :
  ∀ (m n : ℝ), m + n = 1 →
  generatingFunction ⟨1, 1⟩ ⟨2, 0⟩ m n 1 = 2 := by
  sorry

/-- Theorem: The intersection point of two linear functions lies on their generating function -/
theorem intersection_point_on_generating_function (f₁ f₂ : LinearFunction) (m n : ℝ) :
  m + n = 1 →
  ∀ (x y : ℝ),
  (f₁.a * x + f₁.b = y ∧ f₂.a * x + f₂.b = y) →
  generatingFunction f₁ f₂ m n x = y := by
  sorry

end NUMINAMATH_CALUDE_generating_function_value_at_one_intersection_point_on_generating_function_l1691_169104


namespace NUMINAMATH_CALUDE_two_hats_on_first_maximizes_sum_optimal_distribution_l1691_169115

/-- The number of hats in the hat box -/
def total_hats : ℕ := 21

/-- The number of caps in the hat box -/
def total_caps : ℕ := 18

/-- The capacity of the first shelf -/
def first_shelf_capacity : ℕ := 20

/-- The capacity of the second shelf -/
def second_shelf_capacity : ℕ := 19

/-- The percentage of hats on a shelf given the number of hats and total items -/
def hat_percentage (hats : ℕ) (total : ℕ) : ℚ :=
  (hats : ℚ) / (total : ℚ) * 100

/-- The sum of hat percentages for a given distribution -/
def sum_of_percentages (hats_on_first : ℕ) : ℚ :=
  hat_percentage hats_on_first first_shelf_capacity +
  hat_percentage (total_hats - hats_on_first) second_shelf_capacity

/-- Theorem stating that 2 hats on the first shelf maximizes the sum of percentages -/
theorem two_hats_on_first_maximizes_sum :
  ∀ x : ℕ, x ≤ total_hats → sum_of_percentages 2 ≥ sum_of_percentages x :=
sorry

/-- Corollary stating the optimal distribution of hats -/
theorem optimal_distribution :
  sum_of_percentages 2 = hat_percentage 2 first_shelf_capacity +
                         hat_percentage 19 second_shelf_capacity :=
sorry

end NUMINAMATH_CALUDE_two_hats_on_first_maximizes_sum_optimal_distribution_l1691_169115


namespace NUMINAMATH_CALUDE_bad_carrots_l1691_169196

/-- The number of bad carrots in Carol and her mother's carrot picking scenario -/
theorem bad_carrots (carol_carrots : ℕ) (mother_carrots : ℕ) (good_carrots : ℕ) : 
  carol_carrots = 29 → mother_carrots = 16 → good_carrots = 38 →
  carol_carrots + mother_carrots - good_carrots = 7 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_l1691_169196


namespace NUMINAMATH_CALUDE_unique_number_property_l1691_169111

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1691_169111


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l1691_169173

/-- Represents a machine model with its cost and production capacity -/
structure MachineModel where
  cost : ℕ
  production : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelA : ℕ
  modelB : ℕ

def modelA : MachineModel := ⟨60000, 15⟩
def modelB : MachineModel := ⟨40000, 10⟩

def totalMachines : ℕ := 10
def budgetLimit : ℕ := 440000
def requiredProduction : ℕ := 102

def isValidPlan (plan : PurchasePlan) : Prop :=
  plan.modelA + plan.modelB = totalMachines ∧
  plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ budgetLimit ∧
  plan.modelA * modelA.production + plan.modelB * modelB.production ≥ requiredProduction

def isOptimalPlan (plan : PurchasePlan) : Prop :=
  isValidPlan plan ∧
  ∀ (otherPlan : PurchasePlan), 
    isValidPlan otherPlan → 
    plan.modelA * modelA.cost + plan.modelB * modelB.cost ≤ 
    otherPlan.modelA * modelA.cost + otherPlan.modelB * modelB.cost

theorem optimal_purchase_plan :
  ∃ (plan : PurchasePlan), isOptimalPlan plan ∧ plan.modelA = 1 ∧ plan.modelB = 9 := by
  sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l1691_169173


namespace NUMINAMATH_CALUDE_max_value_theorem_l1691_169180

theorem max_value_theorem (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) ≤ 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1691_169180


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1691_169105

theorem min_value_sum_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 10000/29 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1691_169105
