import Mathlib

namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l662_66262

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = T := by sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l662_66262


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l662_66294

theorem rectangle_with_hole_area (x : ℝ) : 
  let large_length : ℝ := 2*x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3*x - 4
  let hole_width : ℝ := x + 1
  large_length * large_width - hole_length * hole_width = -x^2 + 22*x + 52 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l662_66294


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l662_66278

theorem two_numbers_with_given_means : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧
  b = (5 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l662_66278


namespace NUMINAMATH_CALUDE_sum_of_fractions_eq_9900_l662_66231

/-- The sum of all fractions in lowest terms with denominator 3, 
    greater than 10 and less than 100 -/
def sum_of_fractions : ℚ :=
  (Finset.filter (fun n => n % 3 ≠ 0) (Finset.range 269)).sum (fun n => (n + 31 : ℚ) / 3)

/-- Theorem stating that the sum of fractions is equal to 9900 -/
theorem sum_of_fractions_eq_9900 : sum_of_fractions = 9900 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_eq_9900_l662_66231


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l662_66213

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l662_66213


namespace NUMINAMATH_CALUDE_same_terminal_side_l662_66219

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def SameTerminalSide (a b : ℝ) : Prop := ∃ k : ℤ, a - b = k * 360

/-- The angle -510 degrees -/
def angle1 : ℝ := -510

/-- The angle 210 degrees -/
def angle2 : ℝ := 210

/-- Theorem: angle1 and angle2 have the same terminal side -/
theorem same_terminal_side : SameTerminalSide angle1 angle2 := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l662_66219


namespace NUMINAMATH_CALUDE_min_value_of_expression_l662_66216

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 7) :
  (1 / (1 + a) + 4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 7 ∧
    1 / (1 + a₀) + 4 / (2 + b₀) = (13 + 4 * Real.sqrt 3) / 14 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l662_66216


namespace NUMINAMATH_CALUDE_problem_solution_l662_66215

theorem problem_solution : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 15 + 11 = 152 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l662_66215


namespace NUMINAMATH_CALUDE_q_polynomial_expression_l662_66271

theorem q_polynomial_expression (q : ℝ → ℝ) : 
  (∀ x, q x + (2*x^6 + 4*x^4 + 6*x^2 + 2) = 8*x^4 + 27*x^3 + 30*x^2 + 10*x + 3) →
  (∀ x, q x = -2*x^6 + 4*x^4 + 27*x^3 + 24*x^2 + 10*x + 1) := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_expression_l662_66271


namespace NUMINAMATH_CALUDE_electricity_consumption_scientific_notation_l662_66230

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem electricity_consumption_scientific_notation :
  let consumption : ℝ := 36400
  let scientific := toScientificNotation consumption
  scientific.coefficient = 3.64 ∧ scientific.exponent = 4 :=
by sorry

end NUMINAMATH_CALUDE_electricity_consumption_scientific_notation_l662_66230


namespace NUMINAMATH_CALUDE_square_plus_one_gt_x_l662_66297

theorem square_plus_one_gt_x : ∀ x : ℝ, x^2 + 1 > x := by sorry

end NUMINAMATH_CALUDE_square_plus_one_gt_x_l662_66297


namespace NUMINAMATH_CALUDE_unique_solutions_l662_66272

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 16 ∧
  (∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 16) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 16 → n ∈ seq) ∧
  (∀ i, i < 15 → is_perfect_square (seq[i]! + seq[i+1]!))

def solution1 : List ℕ := [16, 9, 7, 2, 14, 11, 5, 4, 12, 13, 3, 6, 10, 15, 1, 8]
def solution2 : List ℕ := [8, 1, 15, 10, 6, 3, 13, 12, 4, 5, 11, 14, 2, 7, 9, 16]

theorem unique_solutions :
  (∀ seq : List ℕ, valid_sequence seq → seq = solution1 ∨ seq = solution2) ∧
  valid_sequence solution1 ∧
  valid_sequence solution2 :=
sorry

end NUMINAMATH_CALUDE_unique_solutions_l662_66272


namespace NUMINAMATH_CALUDE_triangular_pyramid_angle_l662_66275

/-- Represents a triangular pyramid with specific properties -/
structure TriangularPyramid where
  -- The length of the hypotenuse of the base triangle
  c : ℝ
  -- The volume of the pyramid
  V : ℝ
  -- All lateral edges form the same angle with the base plane
  lateral_angle_uniform : True
  -- This angle is equal to one of the acute angles of the right triangle in the base
  angle_matches_base : True
  -- Ensure c and V are positive
  c_pos : c > 0
  V_pos : V > 0

/-- 
Theorem: In a triangular pyramid where all lateral edges form the same angle α 
with the base plane, and this angle is equal to one of the acute angles of the 
right triangle in the base, if the hypotenuse of the base triangle is c and the 
volume of the pyramid is V, then α = arcsin(√(12V/c³)).
-/
theorem triangular_pyramid_angle (p : TriangularPyramid) : 
  ∃ α : ℝ, α = Real.arcsin (Real.sqrt (12 * p.V / p.c^3)) := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_angle_l662_66275


namespace NUMINAMATH_CALUDE_solution_set_implies_ratio_l662_66244

theorem solution_set_implies_ratio (a b : ℝ) (h : ∀ x, (2*a - b)*x + (a + b) > 0 ↔ x > -3) :
  b / a = 5 / 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_ratio_l662_66244


namespace NUMINAMATH_CALUDE_completing_square_step_l662_66296

theorem completing_square_step (x : ℝ) : x^2 - 4*x + 3 = 0 → x^2 - 4*x + (-2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_step_l662_66296


namespace NUMINAMATH_CALUDE_line_mb_value_l662_66214

/-- Proves that for a line y = mx + b passing through points (0, -2) and (1, 1), mb = -10 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) → -- Line equation
  (-2 : ℝ) = b →               -- y-intercept
  (1 : ℝ) = m * 1 + b →        -- Point (1, 1) satisfies the equation
  m * b = -10 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l662_66214


namespace NUMINAMATH_CALUDE_product_remainder_mod_10_l662_66242

theorem product_remainder_mod_10 : (2457 * 7963 * 92324) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_10_l662_66242


namespace NUMINAMATH_CALUDE_contrapositive_zero_product_l662_66289

theorem contrapositive_zero_product (a b : ℝ) :
  (¬(a = 0 ∨ b = 0) → ab ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_zero_product_l662_66289


namespace NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l662_66259

theorem right_triangle_altitude_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : b = (3/2) * a) (d : ℝ) (h_altitude : d^2 = (a*b)/c) :
  (c-d)/d = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_ratio_l662_66259


namespace NUMINAMATH_CALUDE_sphere_radius_from_cross_sections_l662_66240

theorem sphere_radius_from_cross_sections (r : ℝ) (h₁ h₂ : ℝ) : 
  h₁ > h₂ →
  h₁ - h₂ = 1 →
  π * (r^2 - h₁^2) = 5 * π →
  π * (r^2 - h₂^2) = 8 * π →
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_cross_sections_l662_66240


namespace NUMINAMATH_CALUDE_binary_sum_equality_l662_66221

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The sum of the given binary numbers is equal to 11110011₂ -/
theorem binary_sum_equality : 
  let a := bitsToNat [true, false, true, false, true]
  let b := bitsToNat [true, false, true, true]
  let c := bitsToNat [true, true, true, false, false]
  let d := bitsToNat [true, false, true, false, true, false, true]
  let sum := bitsToNat [true, true, true, true, false, false, true, true]
  a + b + c + d = sum := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equality_l662_66221


namespace NUMINAMATH_CALUDE_average_home_runs_l662_66267

theorem average_home_runs : 
  let players_5 := 7
  let players_6 := 5
  let players_7 := 4
  let players_9 := 3
  let players_11 := 1
  let total_players := players_5 + players_6 + players_7 + players_9 + players_11
  let total_home_runs := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11
  (total_home_runs : ℚ) / total_players = 131 / 20 := by
  sorry

end NUMINAMATH_CALUDE_average_home_runs_l662_66267


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l662_66282

/-- Given an ellipse and a hyperbola with specific properties, prove that |ab| = 2√111 -/
theorem ellipse_hyperbola_ab_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 111 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ab_product_l662_66282


namespace NUMINAMATH_CALUDE_max_product_of_arithmetic_sequence_l662_66261

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

theorem max_product_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + 2 * a 6 = 6) :
  (∀ x : ℝ, a 4 * a 6 ≤ x → x ≤ 4) ∧ a 4 * a 6 ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_arithmetic_sequence_l662_66261


namespace NUMINAMATH_CALUDE_product_zero_from_sum_conditions_l662_66276

theorem product_zero_from_sum_conditions (x y z w : ℝ) 
  (sum_condition : x + y + z + w = 0)
  (power_sum_condition : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_from_sum_conditions_l662_66276


namespace NUMINAMATH_CALUDE_area_ratio_hexagon_octagon_l662_66229

noncomputable def hexagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * (11 / 3) * side_length^2

noncomputable def octagon_circumradius (side_length : ℝ) : ℝ :=
  side_length * (2 * Real.sqrt 2) / Real.sqrt (2 - Real.sqrt 2)

noncomputable def octagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * ((octagon_circumradius side_length)^2 - (3 + 2 * Real.sqrt 2) * side_length^2)

theorem area_ratio_hexagon_octagon (side_length : ℝ) (h : side_length > 0) :
  hexagon_area_between_circles side_length / octagon_area_between_circles side_length =
  11 / (3 * ((octagon_circumradius 1)^2 - (3 + 2 * Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_hexagon_octagon_l662_66229


namespace NUMINAMATH_CALUDE_A_minus_2B_specific_value_A_minus_2B_independent_of_x_l662_66200

/-- The algebraic expression A -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

/-- The algebraic expression B -/
def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B equals -20 when x = -2 and y = 3 -/
theorem A_minus_2B_specific_value : A (-2) 3 - 2 * B (-2) 3 = -20 := by sorry

/-- Theorem 2: A - 2B is independent of x when y = 2/5 -/
theorem A_minus_2B_independent_of_x : 
  ∀ x : ℝ, A x (2/5) - 2 * B x (2/5) = A 0 (2/5) - 2 * B 0 (2/5) := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_specific_value_A_minus_2B_independent_of_x_l662_66200


namespace NUMINAMATH_CALUDE_part_one_part_two_l662_66204

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ (- b / a) = (a + b) / 2

/-- Part 1: Prove that 4x - 8/3 = 0 is a midpoint equation -/
theorem part_one : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Part 2: Prove that for 5x + m - 1 = 0 to be a midpoint equation, m = -18/7 -/
theorem part_two : ∃ m : ℚ, is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l662_66204


namespace NUMINAMATH_CALUDE_twelfth_odd_multiple_of_five_l662_66268

theorem twelfth_odd_multiple_of_five : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 0 ∧
  (∃ k : ℕ, k = 12 ∧ 
    n = (Finset.filter (λ x => x % 2 = 1 ∧ x % 5 = 0) (Finset.range n)).card) ∧
  n = 115 := by
sorry

end NUMINAMATH_CALUDE_twelfth_odd_multiple_of_five_l662_66268


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l662_66287

/-- Calculates the pass percentage for a group of students -/
def passPercentage (totalStudents : ℕ) (passedStudents : ℕ) : ℚ :=
  (passedStudents : ℚ) / (totalStudents : ℚ) * 100

theorem exam_pass_percentage :
  let set1 := 40
  let set2 := 50
  let set3 := 60
  let pass1 := 40  -- 100% of 40
  let pass2 := 45  -- 90% of 50
  let pass3 := 48  -- 80% of 60
  let totalStudents := set1 + set2 + set3
  let totalPassed := pass1 + pass2 + pass3
  abs (passPercentage totalStudents totalPassed - 88.67) < 0.01 := by
  sorry

#eval passPercentage (40 + 50 + 60) (40 + 45 + 48)

end NUMINAMATH_CALUDE_exam_pass_percentage_l662_66287


namespace NUMINAMATH_CALUDE_negation_of_existence_l662_66252

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l662_66252


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l662_66293

theorem circle_equation_m_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m - 3)*x + 2*y + 5 = 0 → ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m > 5 ∨ m < 1) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l662_66293


namespace NUMINAMATH_CALUDE_at_most_two_protocols_l662_66232

/-- Represents a skier in the race -/
structure Skier :=
  (number : Nat)
  (startPosition : Nat)
  (finishPosition : Nat)
  (overtakes : Nat)
  (overtakenBy : Nat)

/-- Represents the race conditions -/
structure RaceConditions :=
  (skiers : List Skier)
  (totalSkiers : Nat)
  (h_totalSkiers : totalSkiers = 7)
  (h_startSequence : ∀ s : Skier, s ∈ skiers → s.number = s.startPosition)
  (h_constantSpeed : ∀ s : Skier, s ∈ skiers → s.overtakes + s.overtakenBy = 2)
  (h_uniqueFinish : ∀ s1 s2 : Skier, s1 ∈ skiers → s2 ∈ skiers → s1.finishPosition = s2.finishPosition → s1 = s2)

/-- The theorem to be proved -/
theorem at_most_two_protocols (rc : RaceConditions) : 
  (∃ p1 p2 : List Nat, 
    (∀ p : List Nat, p.length = rc.totalSkiers ∧ (∀ s : Skier, s ∈ rc.skiers → s.finishPosition = p.indexOf s.number + 1) → p = p1 ∨ p = p2) ∧
    p1 ≠ p2) :=
sorry

end NUMINAMATH_CALUDE_at_most_two_protocols_l662_66232


namespace NUMINAMATH_CALUDE_competition_results_l662_66205

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def mode (l : List ℕ) : ℕ := sorry

def average (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 71 / 10) ∧
  (7 > median seventh_grade_scores) ∧
  (7 < median eighth_grade_scores) := by sorry

end NUMINAMATH_CALUDE_competition_results_l662_66205


namespace NUMINAMATH_CALUDE_remainder_divisibility_l662_66279

theorem remainder_divisibility (y : ℤ) : 
  ∃ k : ℤ, y = 288 * k + 45 → ∃ m : ℤ, y = 24 * m + 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l662_66279


namespace NUMINAMATH_CALUDE_count_negative_numbers_l662_66298

theorem count_negative_numbers : 
  let expressions := [-2^2, (-2)^2, -(-2), -|-2|]
  (expressions.filter (λ x => x < 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l662_66298


namespace NUMINAMATH_CALUDE_salary_before_raise_l662_66269

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) 
  (h1 : new_salary = 70)
  (h2 : increase_percentage = 0.40) :
  let original_salary := new_salary / (1 + increase_percentage)
  original_salary = 50 := by
sorry

end NUMINAMATH_CALUDE_salary_before_raise_l662_66269


namespace NUMINAMATH_CALUDE_number_puzzle_l662_66263

theorem number_puzzle (x : ℝ) : (((3/4 * x) - 25) / 7) + 50 = 100 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l662_66263


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l662_66299

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := y = -1/8 * x^2

-- Define the axis of symmetry
def axis_of_symmetry (y : ℝ) : Prop := y = 2

-- Theorem statement
theorem parabola_axis_of_symmetry :
  ∀ x y : ℝ, parabola_eq x y → axis_of_symmetry y := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l662_66299


namespace NUMINAMATH_CALUDE_solid_surface_area_l662_66209

/-- The surface area of a solid composed of a cylinder topped with a hemisphere -/
theorem solid_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 3) :
  2 * π * r * h + 2 * π * r^2 + 2 * π * r^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_solid_surface_area_l662_66209


namespace NUMINAMATH_CALUDE_num_constructible_heights_l662_66227

/-- The number of bricks available --/
def num_bricks : ℕ := 25

/-- The possible height contributions of each brick after normalization and simplification --/
def height_options : List ℕ := [0, 3, 4]

/-- A function that returns the set of all possible tower heights --/
noncomputable def constructible_heights : Finset ℕ :=
  sorry

/-- The theorem stating that the number of constructible heights is 98 --/
theorem num_constructible_heights :
  Finset.card constructible_heights = 98 :=
sorry

end NUMINAMATH_CALUDE_num_constructible_heights_l662_66227


namespace NUMINAMATH_CALUDE_chris_initial_money_l662_66211

def chris_money_problem (initial_money : ℕ) : Prop :=
  let grandmother_gift : ℕ := 25
  let aunt_uncle_gift : ℕ := 20
  let parents_gift : ℕ := 75
  let total_after_gifts : ℕ := 279
  initial_money + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_gifts

theorem chris_initial_money :
  ∃ (initial_money : ℕ), chris_money_problem initial_money ∧ initial_money = 159 :=
by
  sorry

end NUMINAMATH_CALUDE_chris_initial_money_l662_66211


namespace NUMINAMATH_CALUDE_main_theorem_l662_66248

/-- The function y in terms of x and m -/
def y (x m : ℝ) : ℝ := (m + 1) * x^2 - m * x + m - 1

/-- The condition for y < 0 having no solution -/
def no_solution_condition (m : ℝ) : Prop :=
  ∀ x, y x m ≥ 0

/-- The solution set for y ≥ m when m > -2 -/
def solution_set (m : ℝ) : Set ℝ :=
  {x | y x m ≥ m}

theorem main_theorem :
  (∀ m : ℝ, no_solution_condition m ↔ m ≥ 2 * Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, m > -2 →
    (m = -1 → solution_set m = {x | x ≥ 1}) ∧
    (m > -1 → solution_set m = {x | x ≤ -1/(m+1) ∨ x ≥ 1}) ∧
    (-2 < m ∧ m < -1 → solution_set m = {x | 1 ≤ x ∧ x ≤ -1/(m+1)})) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l662_66248


namespace NUMINAMATH_CALUDE_unique_b_value_l662_66225

theorem unique_b_value (a b : ℕ+) (h1 : (3 ^ a.val) ^ b.val = 3 ^ 3) (h2 : 3 ^ a.val * 3 ^ b.val = 81) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l662_66225


namespace NUMINAMATH_CALUDE_total_journey_time_l662_66234

/-- Represents the problem of Joe's journey to school -/
structure JourneyToSchool where
  d : ℝ  -- Total distance from home to school
  walk_speed : ℝ  -- Joe's walking speed
  run_speed : ℝ  -- Joe's running speed
  walk_time : ℝ  -- Time Joe takes to walk 1/3 of the distance

/-- Conditions of the problem -/
def journey_conditions (j : JourneyToSchool) : Prop :=
  j.run_speed = 4 * j.walk_speed ∧
  j.walk_time = 9 ∧
  j.walk_speed * j.walk_time = j.d / 3

/-- The theorem to be proved -/
theorem total_journey_time (j : JourneyToSchool) 
  (h : journey_conditions j) : 
  ∃ (total_time : ℝ), total_time = 13.5 ∧ 
    total_time = j.walk_time + (2 * j.d / 3) / j.run_speed :=
by sorry

end NUMINAMATH_CALUDE_total_journey_time_l662_66234


namespace NUMINAMATH_CALUDE_lychee_harvest_theorem_l662_66273

/-- Represents the lychee harvest data for a single year -/
structure LycheeHarvest where
  red : ℕ
  yellow : ℕ

/-- Calculates the percentage increase between two harvests -/
def percentageIncrease (last : LycheeHarvest) (current : LycheeHarvest) : ℚ :=
  ((current.red - last.red : ℚ) / last.red + (current.yellow - last.yellow : ℚ) / last.yellow) / 2 * 100

/-- Calculates the remaining lychees after selling and family consumption -/
def remainingLychees (harvest : LycheeHarvest) : LycheeHarvest :=
  let redAfterSelling := harvest.red - (2 * harvest.red / 3)
  let yellowAfterSelling := harvest.yellow - (3 * harvest.yellow / 7)
  let redRemaining := redAfterSelling - (3 * redAfterSelling / 5)
  let yellowRemaining := yellowAfterSelling - (4 * yellowAfterSelling / 9)
  { red := redRemaining, yellow := yellowRemaining }

theorem lychee_harvest_theorem (lastYear : LycheeHarvest) (thisYear : LycheeHarvest)
    (h1 : lastYear.red = 350)
    (h2 : lastYear.yellow = 490)
    (h3 : thisYear.red = 500)
    (h4 : thisYear.yellow = 700) :
    percentageIncrease lastYear thisYear = 42.86 ∧
    (remainingLychees thisYear).red = 67 ∧
    (remainingLychees thisYear).yellow = 223 := by
  sorry


end NUMINAMATH_CALUDE_lychee_harvest_theorem_l662_66273


namespace NUMINAMATH_CALUDE_new_year_weather_probability_l662_66254

theorem new_year_weather_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 2  -- desired number of clear days
  let p : ℚ := 3/5  -- probability of snow (complement of 60%)

  -- probability of exactly k clear days out of n days
  (n.choose k : ℚ) * p^(n - k) * (1 - p)^k = 1080/3125 :=
by
  sorry

end NUMINAMATH_CALUDE_new_year_weather_probability_l662_66254


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l662_66238

/-- Given a parabola y = -x² + 1 and three points on it, prove the relationship between their y-coordinates -/
theorem parabola_y_relationship : ∀ (y₁ y₂ y₃ : ℝ),
  ((-2 : ℝ), y₁) ∈ {(x, y) | y = -x^2 + 1} →
  ((-1 : ℝ), y₂) ∈ {(x, y) | y = -x^2 + 1} →
  ((3 : ℝ), y₃) ∈ {(x, y) | y = -x^2 + 1} →
  y₂ > y₁ ∧ y₁ > y₃ :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l662_66238


namespace NUMINAMATH_CALUDE_star_three_neg_four_star_not_commutative_l662_66253

-- Define the new operation "*" for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - 1 + b

-- Theorem 1: 3 * (-4) = 1
theorem star_three_neg_four : star 3 (-4) = 1 := by sorry

-- Theorem 2: 7 * (-3) ≠ (-3) * 7
theorem star_not_commutative : star 7 (-3) ≠ star (-3) 7 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_four_star_not_commutative_l662_66253


namespace NUMINAMATH_CALUDE_special_linear_functions_f_one_l662_66226

/-- Two linear functions satisfying specific properties -/
class SpecialLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  h_sum : ∀ x, f x + g x = 2
  h_comp : ∀ x, f (f x) = g (g x)
  h_f_zero : f 0 = 2022
  h_linear_f : ∃ a b : ℝ, ∀ x, f x = a * x + b
  h_linear_g : ∃ c d : ℝ, ∀ x, g x = c * x + d

/-- The main theorem stating f(1) = 2021 -/
theorem special_linear_functions_f_one
  (S : SpecialLinearFunctions) : S.f 1 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_special_linear_functions_f_one_l662_66226


namespace NUMINAMATH_CALUDE_alpha_minus_beta_equals_pi_over_four_l662_66206

open Real

theorem alpha_minus_beta_equals_pi_over_four
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : tan α = 4/3)
  (h4 : tan β = 1/7) :
  α - β = π/4 := by
sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_equals_pi_over_four_l662_66206


namespace NUMINAMATH_CALUDE_border_area_l662_66247

def photo_height : ℝ := 9
def photo_width : ℝ := 12
def frame_border : ℝ := 3

theorem border_area : 
  let framed_height := photo_height + 2 * frame_border
  let framed_width := photo_width + 2 * frame_border
  let photo_area := photo_height * photo_width
  let framed_area := framed_height * framed_width
  framed_area - photo_area = 162 := by sorry

end NUMINAMATH_CALUDE_border_area_l662_66247


namespace NUMINAMATH_CALUDE_prairie_total_area_l662_66241

/-- The total area of a prairie given the dusted and untouched areas -/
theorem prairie_total_area (dusted_area untouched_area : ℕ) 
  (h1 : dusted_area = 64535)
  (h2 : untouched_area = 522) :
  dusted_area + untouched_area = 65057 := by
  sorry

#check prairie_total_area

end NUMINAMATH_CALUDE_prairie_total_area_l662_66241


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l662_66210

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) : ℝ) / x = 7 / 8 → x ≤ 48 / 7 := by
sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l662_66210


namespace NUMINAMATH_CALUDE_base10_512_equals_base6_2212_l662_66235

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 6 * acc) 0

-- Define the theorem
theorem base10_512_equals_base6_2212 :
  512 = base6ToNat [2, 1, 2, 2] := by
  sorry


end NUMINAMATH_CALUDE_base10_512_equals_base6_2212_l662_66235


namespace NUMINAMATH_CALUDE_abrahams_shopping_budget_l662_66264

/-- Abraham's shopping problem -/
theorem abrahams_shopping_budget (budget : ℕ) 
  (shower_gel_price shower_gel_quantity : ℕ) 
  (toothpaste_price laundry_detergent_price : ℕ) : 
  budget = 60 →
  shower_gel_price = 4 →
  shower_gel_quantity = 4 →
  toothpaste_price = 3 →
  laundry_detergent_price = 11 →
  budget - (shower_gel_price * shower_gel_quantity + toothpaste_price + laundry_detergent_price) = 30 := by
  sorry


end NUMINAMATH_CALUDE_abrahams_shopping_budget_l662_66264


namespace NUMINAMATH_CALUDE_fractional_method_min_experiments_l662_66239

/-- The number of division points in the temperature range -/
def division_points : ℕ := 33

/-- The minimum number of experiments needed -/
def min_experiments : ℕ := 7

/-- Theorem stating the minimum number of experiments needed for the given conditions -/
theorem fractional_method_min_experiments :
  ∃ (n : ℕ), 2^n - 1 ≥ division_points ∧ n = min_experiments :=
sorry

end NUMINAMATH_CALUDE_fractional_method_min_experiments_l662_66239


namespace NUMINAMATH_CALUDE_problem_statement_l662_66236

theorem problem_statement (a : ℝ) (h : a/2 - 2/a = 5) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l662_66236


namespace NUMINAMATH_CALUDE_light_flash_interval_l662_66208

/-- Given a light that flashes 600 times in 1/6 of an hour, prove that the time between each flash is 1 second. -/
theorem light_flash_interval (flashes_per_sixth_hour : ℕ) (h : flashes_per_sixth_hour = 600) :
  (1 / 6 : ℚ) * 3600 / flashes_per_sixth_hour = 1 := by
  sorry

#check light_flash_interval

end NUMINAMATH_CALUDE_light_flash_interval_l662_66208


namespace NUMINAMATH_CALUDE_taxi_speed_theorem_l662_66286

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the taxi and bus departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_theorem :
  taxi_speed * overtake_time = (taxi_speed - 30) * (overtake_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_taxi_speed_theorem_l662_66286


namespace NUMINAMATH_CALUDE_circle_area_is_one_l662_66285

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 / (2 * Real.pi * r) = 2 * r) → Real.pi * r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l662_66285


namespace NUMINAMATH_CALUDE_gcd_1722_966_l662_66245

theorem gcd_1722_966 : Int.gcd 1722 966 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1722_966_l662_66245


namespace NUMINAMATH_CALUDE_min_sum_squares_l662_66258

theorem min_sum_squares (a b : ℝ) : 
  (∃! x, x^2 - 2*a*x + a^2 - a*b + 4 ≤ 0) → 
  (∀ c d : ℝ, (∃! x, x^2 - 2*c*x + c^2 - c*d + 4 ≤ 0) → a^2 + b^2 ≤ c^2 + d^2) →
  a^2 + b^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l662_66258


namespace NUMINAMATH_CALUDE_gcd_of_2750_and_9450_l662_66222

theorem gcd_of_2750_and_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2750_and_9450_l662_66222


namespace NUMINAMATH_CALUDE_reading_pages_in_week_l662_66290

/-- Calculates the total number of pages read in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

/-- Theorem: Reading 5 pages in the morning and 10 pages in the evening for a week results in 105 pages read -/
theorem reading_pages_in_week :
  pages_read_in_week 5 10 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_reading_pages_in_week_l662_66290


namespace NUMINAMATH_CALUDE_mirror_16_is_8_l662_66255

/-- Represents a time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Calculates the mirror image of a given time -/
def mirrorTime (t : Time) : Time :=
  { hour := (24 - t.hour) % 24,
    minute := (60 - t.minute) % 60,
    h_valid := by sorry
    m_valid := by sorry }

/-- Theorem: The mirror image of 16:00 is 08:00 -/
theorem mirror_16_is_8 :
  let t : Time := ⟨16, 0, by norm_num, by norm_num⟩
  mirrorTime t = ⟨8, 0, by norm_num, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_mirror_16_is_8_l662_66255


namespace NUMINAMATH_CALUDE_pipe_b_shut_time_l662_66277

-- Define the rates at which pipes fill the tank
def pipe_a_rate : ℚ := 1
def pipe_b_rate : ℚ := 1 / 15

-- Define the time it takes for the tank to overflow
def overflow_time : ℚ := 1 / 2  -- 30 minutes = 0.5 hours

-- Define the theorem
theorem pipe_b_shut_time :
  let combined_rate := pipe_a_rate + pipe_b_rate
  let volume_filled_together := combined_rate * overflow_time
  let pipe_b_shut_time := 1 - volume_filled_together
  pipe_b_shut_time * 60 = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_b_shut_time_l662_66277


namespace NUMINAMATH_CALUDE_stream_bottom_width_l662_66228

/-- Represents the trapezoidal cross-section of a stream -/
structure StreamCrossSection where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its height -/
def trapezoidAreaFormula (s : StreamCrossSection) : Prop :=
  s.area = (s.topWidth + s.bottomWidth) / 2 * s.depth

theorem stream_bottom_width
  (s : StreamCrossSection)
  (h1 : s.topWidth = 10)
  (h2 : s.depth = 80)
  (h3 : s.area = 640)
  (h4 : trapezoidAreaFormula s) :
  s.bottomWidth = 6 := by
  sorry

end NUMINAMATH_CALUDE_stream_bottom_width_l662_66228


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l662_66246

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℚ, x^(3*m+2) + (-x^2 - 1)^(3*n+1) + 1 = (x^2 + x + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l662_66246


namespace NUMINAMATH_CALUDE_final_price_difference_l662_66295

def total_budget : ℝ := 1500
def tv_budget : ℝ := 1000
def sound_system_budget : ℝ := 500
def tv_discount_flat : ℝ := 150
def tv_discount_percent : ℝ := 0.20
def sound_system_discount_percent : ℝ := 0.15
def tax_rate : ℝ := 0.08

theorem final_price_difference :
  let tv_price := (tv_budget - tv_discount_flat) * (1 - tv_discount_percent)
  let sound_system_price := sound_system_budget * (1 - sound_system_discount_percent)
  let total_before_tax := tv_price + sound_system_price
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount
  total_budget - final_price = 306.60 := by sorry

end NUMINAMATH_CALUDE_final_price_difference_l662_66295


namespace NUMINAMATH_CALUDE_perimeter_of_parallelogram_l662_66280

/-- Triangle PQR with properties -/
structure Triangle (P Q R : ℝ × ℝ) :=
  (pq_eq_pr : dist P Q = dist P R)
  (pq_length : dist P Q = 30)
  (qr_length : dist Q R = 28)

/-- Point on a line segment -/
def PointOnSegment (A B M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B

/-- Parallel lines -/
def Parallel (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)

/-- Perimeter of a quadrilateral -/
def Perimeter (A B C D : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D A

theorem perimeter_of_parallelogram
  (P Q R M N O : ℝ × ℝ)
  (tri : Triangle P Q R)
  (m_on_pq : PointOnSegment P Q M)
  (n_on_qr : PointOnSegment Q R N)
  (o_on_pr : PointOnSegment P R O)
  (mn_parallel_pr : Parallel M N P R)
  (no_parallel_pq : Parallel N O P Q) :
  Perimeter P M N O = 60 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_parallelogram_l662_66280


namespace NUMINAMATH_CALUDE_coopers_age_l662_66274

theorem coopers_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  dante = 2 * cooper →
  maria = dante + 1 →
  cooper = 6 := by
sorry

end NUMINAMATH_CALUDE_coopers_age_l662_66274


namespace NUMINAMATH_CALUDE_simple_interest_problem_l662_66251

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 →
  P = 300 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l662_66251


namespace NUMINAMATH_CALUDE_remaining_wire_length_l662_66237

def total_wire_length : ℝ := 60
def square_side_length : ℝ := 9

theorem remaining_wire_length :
  total_wire_length - 4 * square_side_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l662_66237


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l662_66256

/-- Given a circle with center O and radius 8, where the shaded region is half of the circle plus two radii, 
    the perimeter of the shaded region is 16 + 8π. -/
theorem shaded_region_perimeter (O : Point) (r : ℝ) (h1 : r = 8) : 
  let perimeter := 2 * r + (π * r)
  perimeter = 16 + 8 * π := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l662_66256


namespace NUMINAMATH_CALUDE_fraction_problem_l662_66281

theorem fraction_problem (x : ℚ) : 
  (9 - x = 4.5 * (1/2)) → x = 27/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l662_66281


namespace NUMINAMATH_CALUDE_range_of_t_l662_66283

def A : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}
def B (t : ℝ) : Set ℝ := {x | x ≥ t}

theorem range_of_t (t : ℝ) : A ∪ B t = A → t ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l662_66283


namespace NUMINAMATH_CALUDE_discount_amount_l662_66249

/-- Given a shirt with an original price and a discounted price, 
    the discount amount is the difference between the two prices. -/
theorem discount_amount (original_price discounted_price : ℕ) :
  original_price = 22 →
  discounted_price = 16 →
  original_price - discounted_price = 6 := by
sorry

end NUMINAMATH_CALUDE_discount_amount_l662_66249


namespace NUMINAMATH_CALUDE_total_players_count_l662_66224

/-- The number of players who play kabaddi -/
def kabaddi_players : ℕ := 10

/-- The number of players who play kho-kho only -/
def kho_kho_only_players : ℕ := 20

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabaddi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l662_66224


namespace NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l662_66243

theorem quadratic_roots_and_inequality (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 4 * (a + 1) * x + 3 * a + 3
  (∃ x y : ℝ, x < 2 ∧ y < 2 ∧ x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∀ x : ℝ, (a + 1) * x^2 - a * x + a - 1 < 0) ↔ a < -2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l662_66243


namespace NUMINAMATH_CALUDE_cylinder_triple_volume_radius_l662_66220

/-- Theorem: Tripling the volume of a cylinder while keeping the same height results in a new radius that is √3 times the original radius. -/
theorem cylinder_triple_volume_radius (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let v := π * r^2 * h
  let v_new := 3 * v
  let r_new := Real.sqrt ((3 * π * r^2 * h) / (π * h))
  r_new = r * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_triple_volume_radius_l662_66220


namespace NUMINAMATH_CALUDE_lazy_kingdom_date_l662_66202

-- Define the days of the week in the Lazy Kingdom
inductive LazyDay
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Saturday

-- Define a function to calculate the next day
def nextDay (d : LazyDay) : LazyDay :=
  match d with
  | LazyDay.Sunday => LazyDay.Monday
  | LazyDay.Monday => LazyDay.Tuesday
  | LazyDay.Tuesday => LazyDay.Wednesday
  | LazyDay.Wednesday => LazyDay.Thursday
  | LazyDay.Thursday => LazyDay.Saturday
  | LazyDay.Saturday => LazyDay.Sunday

-- Define a function to calculate the day after n days
def dayAfter (start : LazyDay) (n : Nat) : LazyDay :=
  match n with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

-- Theorem statement
theorem lazy_kingdom_date : 
  dayAfter LazyDay.Sunday 374 = LazyDay.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_lazy_kingdom_date_l662_66202


namespace NUMINAMATH_CALUDE_isosceles_triangle_inscribed_circle_and_orthocenter_l662_66250

/-- An isosceles triangle with unit-length legs -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ := 1

/-- The radius of the inscribed circle of an isosceles triangle -/
noncomputable def inscribedRadius (t : IsoscelesTriangle) : ℝ := sorry

/-- The orthocenter of an isosceles triangle -/
noncomputable def orthocenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- A point on the semicircle drawn on the base of the triangle -/
noncomputable def semicirclePoint (t : IsoscelesTriangle) : ℝ × ℝ := sorry

theorem isosceles_triangle_inscribed_circle_and_orthocenter 
  (t : IsoscelesTriangle) : 
  (∃ (max_t : IsoscelesTriangle), 
    (∀ (other_t : IsoscelesTriangle), inscribedRadius max_t ≥ inscribedRadius other_t) ∧
    max_t.base = Real.sqrt 5 - 1 ∧
    semicirclePoint max_t = orthocenter max_t) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_inscribed_circle_and_orthocenter_l662_66250


namespace NUMINAMATH_CALUDE_train_speed_calculation_l662_66233

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 90)
  (h2 : bridge_length = 200)
  (h3 : crossing_time = 36) :
  ∃ (speed : ℝ), 
    (speed ≥ 28.9 ∧ speed ≤ 29.1) ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l662_66233


namespace NUMINAMATH_CALUDE_smallest_four_digit_with_digit_product_12_l662_66201

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_digit_product_12 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 12 → 1126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_with_digit_product_12_l662_66201


namespace NUMINAMATH_CALUDE_add_one_five_times_l662_66266

theorem add_one_five_times (m : ℕ) : 
  let n := m + 5
  n = m + 5 ∧ n - (m + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_add_one_five_times_l662_66266


namespace NUMINAMATH_CALUDE_box_width_l662_66257

/-- The width of a rectangular box given its dimensions and cube properties -/
theorem box_width (length height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h1 : length = 10)
  (h2 : height = 4)
  (h3 : cube_volume = 12)
  (h4 : min_cubes = 60) :
  (min_cubes : ℝ) * cube_volume / (length * height) = 18 := by
  sorry

end NUMINAMATH_CALUDE_box_width_l662_66257


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l662_66218

theorem min_value_and_inequality_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) ↔ -2 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l662_66218


namespace NUMINAMATH_CALUDE_sequence_always_terminates_l662_66223

def units_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def next_term (n : ℕ) : ℕ :=
  if units_digit n ≤ 5 then remove_last_digit n else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate next_term k a₀) = 0

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

end NUMINAMATH_CALUDE_sequence_always_terminates_l662_66223


namespace NUMINAMATH_CALUDE_appropriate_word_count_appropriate_lengths_l662_66291

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 30 }

/-- The optimal speaking rate in words per minute -/
def OptimalSpeakingRate : ℝ := 135

/-- Calculates the number of words for a given duration at the optimal speaking rate -/
def WordCount (duration : PresentationDuration) : ℝ :=
  duration.val * OptimalSpeakingRate

/-- Theorem stating that the appropriate word count is between 2700 and 4050 -/
theorem appropriate_word_count (duration : PresentationDuration) :
  2700 ≤ WordCount duration ∧ WordCount duration ≤ 4050 := by
  sorry

/-- Theorem stating that 3000 and 3700 words are appropriate lengths for the presentation -/
theorem appropriate_lengths :
  ∃ (d1 d2 : PresentationDuration), WordCount d1 = 3000 ∧ WordCount d2 = 3700 := by
  sorry

end NUMINAMATH_CALUDE_appropriate_word_count_appropriate_lengths_l662_66291


namespace NUMINAMATH_CALUDE_handshake_count_l662_66265

def networking_event (total_people group_a_size group_b_size : ℕ)
  (group_a_fully_acquainted group_a_partially_acquainted : ℕ)
  (group_a_partial_connections : ℕ) : Prop :=
  total_people = group_a_size + group_b_size ∧
  group_a_size = group_a_fully_acquainted + 1 ∧
  group_a_partially_acquainted = 1 ∧
  group_a_partial_connections = 5

theorem handshake_count
  (h : networking_event 40 25 15 24 1 5) :
  (25 * 15) +  -- handshakes between Group A and Group B
  (15 * 14 / 2) +  -- handshakes within Group B
  19  -- handshakes of partially acquainted member in Group A
  = 499 :=
sorry

end NUMINAMATH_CALUDE_handshake_count_l662_66265


namespace NUMINAMATH_CALUDE_gcd_12n_plus_5_7n_plus_3_l662_66284

theorem gcd_12n_plus_5_7n_plus_3 (n : ℕ+) : Nat.gcd (12 * n + 5) (7 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12n_plus_5_7n_plus_3_l662_66284


namespace NUMINAMATH_CALUDE_ceiling_minus_y_is_half_l662_66217

theorem ceiling_minus_y_is_half (x : ℝ) (y : ℝ) 
  (h1 : ⌈x⌉ - ⌊x⌋ = 0) 
  (h2 : y = x + 1/2) : 
  ⌈y⌉ - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_is_half_l662_66217


namespace NUMINAMATH_CALUDE_g_of_f_minus_two_three_l662_66212

/-- Transformation f that takes a pair of integers and negates the second component -/
def f (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

/-- Transformation g that takes a pair of integers and negates both components -/
def g (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

/-- Theorem stating that g[f(-2,3)] = (2,3) -/
theorem g_of_f_minus_two_three : g (f (-2, 3)) = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_minus_two_three_l662_66212


namespace NUMINAMATH_CALUDE_max_value_of_f_l662_66288

open Real

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ y ∈ Set.Ioo 0 (π / 2), 8 * sin y - tan y ≤ max_val :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l662_66288


namespace NUMINAMATH_CALUDE_cat_mouse_problem_l662_66207

/-- Given that 5 cats can catch 5 mice in 5 minutes, prove that 5 cats can catch 100 mice in 500 minutes -/
theorem cat_mouse_problem (cats mice minutes : ℕ) 
  (h1 : cats = 5)
  (h2 : mice = 5)
  (h3 : minutes = 5)
  (h4 : cats * mice = cats * minutes) : 
  cats * 100 = cats * 500 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_problem_l662_66207


namespace NUMINAMATH_CALUDE_smallest_integer_a_l662_66260

theorem smallest_integer_a (a b : ℤ) : 
  (∃ k : ℤ, a > k ∧ a < 21) →
  (b > 19 ∧ b < 31) →
  (a / b : ℚ) ≤ 2/3 →
  (∀ m : ℤ, m < a → m ≤ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_a_l662_66260


namespace NUMINAMATH_CALUDE_digit_equation_solution_l662_66292

theorem digit_equation_solution : ∃ (Θ : ℕ), 
  Θ ≤ 9 ∧ 
  252 / Θ = 40 + 2 * Θ ∧ 
  Θ = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l662_66292


namespace NUMINAMATH_CALUDE_diana_apollo_dice_probability_l662_66270

theorem diana_apollo_dice_probability :
  let diana_die := Finset.range 10
  let apollo_die := Finset.range 6
  let total_outcomes := diana_die.card * apollo_die.card
  let favorable_outcomes := (apollo_die.sum fun a => 
    (diana_die.filter (fun d => d > a)).card)
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 20 := by
sorry

end NUMINAMATH_CALUDE_diana_apollo_dice_probability_l662_66270


namespace NUMINAMATH_CALUDE_cos_probability_l662_66203

/-- The probability that cos(πx/2) is between 0 and 1/2 when x is randomly selected from [-1, 1] -/
theorem cos_probability : 
  ∃ (P : Set ℝ → ℝ), 
    (∀ x ∈ Set.Icc (-1) 1, P {y | 0 ≤ Real.cos (π * y / 2) ∧ Real.cos (π * y / 2) ≤ 1/2} = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_cos_probability_l662_66203
