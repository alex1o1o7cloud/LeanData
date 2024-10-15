import Mathlib

namespace NUMINAMATH_CALUDE_laura_biathlon_l2831_283143

/-- Laura's biathlon training problem -/
theorem laura_biathlon (x : ℝ) : x > 0 → (25 / (3*x + 2) + 4 / x + 8/60 = 140/60) → (6.6*x^2 - 32.6*x - 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_laura_biathlon_l2831_283143


namespace NUMINAMATH_CALUDE_power_division_equality_l2831_283172

theorem power_division_equality : (3 : ℕ)^12 / (9 : ℕ)^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2831_283172


namespace NUMINAMATH_CALUDE_fraction_value_l2831_283180

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2831_283180


namespace NUMINAMATH_CALUDE_complement_of_union_l2831_283167

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  (M ∪ N)ᶜ = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2831_283167


namespace NUMINAMATH_CALUDE_total_students_l2831_283140

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) : 
  boys + girls = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l2831_283140


namespace NUMINAMATH_CALUDE_tangent_lines_at_P_l2831_283153

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point P
def P : ℝ × ℝ := (2, -6)

-- Define the two potential tangent lines
def line1 (x y : ℝ) : Prop := 3*x + y = 0
def line2 (x y : ℝ) : Prop := 24*x - y - 54 = 0

-- Theorem statement
theorem tangent_lines_at_P :
  (∃ t : ℝ, f t = P.2 ∧ f' t * (P.1 - t) = P.2 - f t ∧ (line1 P.1 P.2 ∨ line2 P.1 P.2)) ∧
  (∀ x y : ℝ, (line1 x y ∨ line2 x y) → ∃ t : ℝ, f t = y ∧ f' t * (x - t) = y - f t) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_at_P_l2831_283153


namespace NUMINAMATH_CALUDE_sum_remainder_modulo_11_l2831_283105

theorem sum_remainder_modulo_11 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_modulo_11_l2831_283105


namespace NUMINAMATH_CALUDE_max_stamps_proof_l2831_283157

/-- The price of a stamp in cents -/
def stamp_price : ℕ := 50

/-- The total budget in cents -/
def total_budget : ℕ := 5000

/-- The number of stamps required for discount eligibility -/
def discount_threshold : ℕ := 80

/-- The discount amount per stamp in cents -/
def discount_amount : ℕ := 5

/-- The maximum number of stamps that can be purchased with the given conditions -/
def max_stamps : ℕ := 111

theorem max_stamps_proof :
  ∀ n : ℕ,
  n ≤ max_stamps ∧
  (n > discount_threshold → n * (stamp_price - discount_amount) ≤ total_budget) ∧
  (n ≤ discount_threshold → n * stamp_price ≤ total_budget) ∧
  (max_stamps > discount_threshold → max_stamps * (stamp_price - discount_amount) ≤ total_budget) ∧
  (max_stamps + 1 > discount_threshold → (max_stamps + 1) * (stamp_price - discount_amount) > total_budget) := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_proof_l2831_283157


namespace NUMINAMATH_CALUDE_sum_ge_sum_of_abs_div_three_l2831_283174

theorem sum_ge_sum_of_abs_div_three (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) :
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_ge_sum_of_abs_div_three_l2831_283174


namespace NUMINAMATH_CALUDE_train_passing_time_l2831_283116

/-- The time taken for a train to pass a telegraph post -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 60 →
  train_speed_kmph = 36 →
  (train_length / (train_speed_kmph * (5/18))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2831_283116


namespace NUMINAMATH_CALUDE_expected_bullets_remaining_l2831_283139

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.6

/-- The total number of bullets -/
def total_bullets : ℕ := 4

/-- The expected number of bullets remaining after stopping the shooting -/
def expected_remaining_bullets : ℝ := 2.376

/-- Theorem stating that the expected number of bullets remaining is 2.376 -/
theorem expected_bullets_remaining :
  let p := hit_probability
  let n := total_bullets
  let E := expected_remaining_bullets
  E = (0 * (1 - p)^3 + 1 * p * (1 - p)^2 + 2 * p * (1 - p) + 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_expected_bullets_remaining_l2831_283139


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l2831_283111

/-- Given a line with equation ax + by + c = 0, returns the equation of the line
    symmetric to it with respect to y = x as a triple (a', b', c') representing
    a'x + b'y + c' = 0 -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

theorem symmetric_line_correct :
  let original_line := (1, -3, 5)  -- Represents x - 3y + 5 = 0
  let symm_line := symmetric_line 1 (-3) 5
  symm_line = (3, -1, -5)  -- Represents 3x - y - 5 = 0
  := by sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l2831_283111


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_950_degrees_l2831_283138

theorem angle_with_same_terminal_side_as_negative_950_degrees :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ ∃ k : ℤ, θ = -950 + 360 * k ∧ θ = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_950_degrees_l2831_283138


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2831_283127

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 793 ∧
  100 ≤ n ∧ n < 1000 ∧
  ∃ k₁ : ℕ, n = 9 * k₁ + 1 ∧
  ∃ k₂ : ℕ, n = 5 * k₂ + 3 ∧
  ∃ k₃ : ℕ, n = 7 * k₃ + 2 ∧
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧
    ∃ l₁ : ℕ, m = 9 * l₁ + 1 ∧
    ∃ l₂ : ℕ, m = 5 * l₂ + 3 ∧
    ∃ l₃ : ℕ, m = 7 * l₃ + 2) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2831_283127


namespace NUMINAMATH_CALUDE_equation_solutions_l2831_283185

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (3 * x₁^2 + 3 * x₁ + 6 = |(-20 + 5 * x₁)|) ∧ 
  (3 * x₂^2 + 3 * x₂ + 6 = |(-20 + 5 * x₂)|) ∧ 
  (x₁ ≠ x₂) ∧ 
  (-4 < x₁) ∧ (x₁ < 2) ∧ 
  (-4 < x₂) ∧ (x₂ < 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2831_283185


namespace NUMINAMATH_CALUDE_function_characterization_l2831_283134

def DivisibilityCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ (a b : ℕ+), a + b > 2019 → (a + f b) ∣ (a^2 + b * f a)

theorem function_characterization (f : ℕ+ → ℕ+) 
  (h : DivisibilityCondition f) : 
  ∃ (r : ℕ+), ∀ (x : ℕ+), f x = r * x := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2831_283134


namespace NUMINAMATH_CALUDE_octahedron_flattenable_l2831_283148

/-- Represents a cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)

/-- Represents an octahedron -/
structure Octahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 12)
  faces : Finset (Fin 8)

/-- Defines the relationship between a cube and its corresponding octahedron -/
def correspondingOctahedron (c : Cube) : Octahedron :=
  sorry

/-- Defines what it means for a set of faces to be connected -/
def isConnected {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines what it means for a set of faces to be flattenable -/
def isFlattenable {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines the operation of cutting edges on a polyhedron -/
def cutEdges {α : Type*} (edges : Finset α) (toCut : Finset α) : Finset α :=
  sorry

theorem octahedron_flattenable (c : Cube) (cubeCuts : Finset (Fin 12)) :
  (cubeCuts.card = 7) →
  (isConnected (cutEdges c.edges cubeCuts)) →
  (isFlattenable (cutEdges c.edges cubeCuts)) →
  let o := correspondingOctahedron c
  let octaCuts := c.edges \ cubeCuts
  (isConnected (cutEdges o.edges octaCuts)) ∧
  (isFlattenable (cutEdges o.edges octaCuts)) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_flattenable_l2831_283148


namespace NUMINAMATH_CALUDE_certain_number_proof_l2831_283144

theorem certain_number_proof (x : ℝ) : 0.15 * x + 0.12 * 45 = 9.15 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2831_283144


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l2831_283182

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def binary_number (bits : List Bool) : Nat := binary_to_decimal bits

theorem binary_sum_theorem :
  let a := binary_number [true, false, true, false, true]
  let b := binary_number [true, true, true]
  let c := binary_number [true, false, true, true, true, false]
  let d := binary_number [true, false, true, false, true, true]
  let sum := binary_number [true, true, true, true, false, false, true]
  a + b + c + d = sum := by sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l2831_283182


namespace NUMINAMATH_CALUDE_min_distance_sum_l2831_283177

/-- Given points M(-1,3) and N(2,1), and point P on the x-axis,
    the minimum value of PM+PN is 5. -/
theorem min_distance_sum (M N P : ℝ × ℝ) : 
  M = (-1, 3) → 
  N = (2, 1) → 
  P.2 = 0 → 
  ∃ (min_val : ℝ), (∀ Q : ℝ × ℝ, Q.2 = 0 → 
    Real.sqrt ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) + 
    Real.sqrt ((Q.1 - N.1)^2 + (Q.2 - N.2)^2) ≥ min_val) ∧ 
  min_val = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l2831_283177


namespace NUMINAMATH_CALUDE_largest_two_digit_remainder_2_mod_13_l2831_283188

theorem largest_two_digit_remainder_2_mod_13 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_remainder_2_mod_13_l2831_283188


namespace NUMINAMATH_CALUDE_chandler_saves_26_weeks_l2831_283164

/-- The number of weeks it takes Chandler to save for a mountain bike. -/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 650
  let birthday_money : ℕ := 80 + 35 + 15
  let weekly_earnings : ℕ := 20
  (bike_cost - birthday_money) / weekly_earnings

/-- Theorem stating that it takes 26 weeks for Chandler to save for the mountain bike. -/
theorem chandler_saves_26_weeks : weeks_to_save = 26 := by
  sorry

end NUMINAMATH_CALUDE_chandler_saves_26_weeks_l2831_283164


namespace NUMINAMATH_CALUDE_quadratic_two_unequal_real_roots_l2831_283130

theorem quadratic_two_unequal_real_roots :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (2 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_unequal_real_roots_l2831_283130


namespace NUMINAMATH_CALUDE_negation_of_existential_l2831_283133

theorem negation_of_existential (P : α → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_l2831_283133


namespace NUMINAMATH_CALUDE_exam_logic_l2831_283109

structure Student where
  name : String
  score : ℝ
  grade : String

def exam_rule (s : Student) : Prop :=
  s.score ≥ 0.8 → s.grade = "A"

theorem exam_logic (s : Student) (h : exam_rule s) :
  (s.grade ≠ "A" → s.score < 0.8) ∧
  (s.score ≥ 0.8 → s.grade = "A") := by
  sorry

end NUMINAMATH_CALUDE_exam_logic_l2831_283109


namespace NUMINAMATH_CALUDE_games_given_away_l2831_283198

def initial_games : ℕ := 183
def remaining_games : ℕ := 92

theorem games_given_away : initial_games - remaining_games = 91 := by
  sorry

end NUMINAMATH_CALUDE_games_given_away_l2831_283198


namespace NUMINAMATH_CALUDE_fermat_number_divisibility_l2831_283173

theorem fermat_number_divisibility (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, 2^(2^m) - 1 = (2^(2^n) + 1) * k := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_divisibility_l2831_283173


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_chord_length_implies_line_equation_l2831_283158

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line equation
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for part 1
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for part 2
theorem chord_length_implies_line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 * Real.sqrt 10 / 5)^2) →
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_chord_length_implies_line_equation_l2831_283158


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2831_283115

/-- Two lines y = ax - 2 and y = (a+2)x + 1 are perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 2) + 1 = 0

/-- Theorem: If the lines y = ax - 2 and y = (a+2)x + 1 are perpendicular, then a = -1 -/
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, are_perpendicular a → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2831_283115


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l2831_283142

theorem triangle_tangent_product (A B C : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = (Real.sin B)^2 →
  (Real.tan A) * (Real.tan C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l2831_283142


namespace NUMINAMATH_CALUDE_binomial_eight_choose_two_l2831_283170

theorem binomial_eight_choose_two : (8 : ℕ).choose 2 = 28 := by sorry

end NUMINAMATH_CALUDE_binomial_eight_choose_two_l2831_283170


namespace NUMINAMATH_CALUDE_proportional_relationship_l2831_283137

theorem proportional_relationship (k : ℝ) (x z : ℝ → ℝ) :
  (∀ t, x t = k / (z t * Real.sqrt (z t))) →
  x 9 = 8 →
  x 64 = 27 / 64 := by
sorry

end NUMINAMATH_CALUDE_proportional_relationship_l2831_283137


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2831_283195

theorem trigonometric_expression_equality : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2831_283195


namespace NUMINAMATH_CALUDE_intercollegiate_competition_l2831_283141

theorem intercollegiate_competition (day1 day2 day3 day1_and_2 day2_and_3 only_day1 : ℕ)
  (h1 : day1 = 175)
  (h2 : day2 = 210)
  (h3 : day3 = 150)
  (h4 : day1_and_2 = 80)
  (h5 : day2_and_3 = 70)
  (h6 : only_day1 = 45)
  : ∃ all_days : ℕ,
    day1 = only_day1 + day1_and_2 + all_days ∧
    day2 = day1_and_2 + day2_and_3 + all_days ∧
    day3 = day2_and_3 + all_days ∧
    all_days = 50 := by
  sorry

end NUMINAMATH_CALUDE_intercollegiate_competition_l2831_283141


namespace NUMINAMATH_CALUDE_number_relationship_l2831_283103

theorem number_relationship (a b c : ℤ) : 
  (a + b + c = 264) → 
  (a = 2 * b) → 
  (b = 72) → 
  (c = a - 96) := by
sorry

end NUMINAMATH_CALUDE_number_relationship_l2831_283103


namespace NUMINAMATH_CALUDE_abhay_sameer_speed_comparison_l2831_283145

theorem abhay_sameer_speed_comparison 
  (distance : ℝ) 
  (abhay_speed : ℝ) 
  (time_difference : ℝ) :
  distance = 42 →
  abhay_speed = 7 →
  time_difference = 2 →
  distance / abhay_speed = distance / (distance / (distance / abhay_speed - time_difference)) + time_difference →
  distance / (2 * abhay_speed) = (distance / (distance / (distance / abhay_speed - time_difference))) - 1 :=
by sorry

end NUMINAMATH_CALUDE_abhay_sameer_speed_comparison_l2831_283145


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2831_283160

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_inequality : (a 5 + a 6 + a 7 + a 8) * (a 6 + a 7 + a 8) < 0) :
  |a 6| > |a 7| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2831_283160


namespace NUMINAMATH_CALUDE_gcd_three_numbers_l2831_283184

theorem gcd_three_numbers (a b c : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 18) :
  Nat.gcd a (Nat.gcd b c) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_three_numbers_l2831_283184


namespace NUMINAMATH_CALUDE_simplify_fraction_l2831_283196

theorem simplify_fraction (a : ℚ) (h : a = 2) : 15 * a^4 / (45 * a^3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2831_283196


namespace NUMINAMATH_CALUDE_ellipse_and_hyperbola_properties_l2831_283183

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ
  foci_on_y_axis : Bool

/-- A hyperbola with foci on the y-axis -/
structure Hyperbola where
  real_axis : ℝ
  imaginary_axis : ℝ
  foci_on_y_axis : Bool

/-- Given ellipse properties, prove its equation, foci coordinates, eccentricity, and related hyperbola equation -/
theorem ellipse_and_hyperbola_properties (e : Ellipse) 
    (h1 : e.major_axis = 10) 
    (h2 : e.minor_axis = 8) 
    (h3 : e.foci_on_y_axis = true) : 
  (∃ (x y : ℝ), x^2/16 + y^2/25 = 1) ∧ 
  (∃ (f1 f2 : ℝ × ℝ), f1 = (0, -3) ∧ f2 = (0, 3)) ∧
  (3/5 : ℝ) = (5^2 - 4^2).sqrt / 5 ∧
  (∃ (h : Hyperbola), h.real_axis = 3 ∧ h.imaginary_axis = 4 ∧ h.foci_on_y_axis = true ∧
    ∃ (x y : ℝ), y^2/9 - x^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_hyperbola_properties_l2831_283183


namespace NUMINAMATH_CALUDE_bakery_pie_division_l2831_283163

theorem bakery_pie_division (total_pie : ℚ) (num_friends : ℕ) : 
  total_pie = 5/8 → num_friends = 4 → total_pie / num_friends = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l2831_283163


namespace NUMINAMATH_CALUDE_yao_ming_shots_l2831_283123

/-- Represents the scoring details of a basketball player in a game -/
structure ScoringDetails where
  total_shots_made : ℕ
  total_points : ℕ
  three_pointers_made : ℕ

/-- Calculates the number of 2-point shots and free throws made given the scoring details -/
def calculate_shots (details : ScoringDetails) : ℕ × ℕ :=
  let two_pointers := (details.total_points - 3 * details.three_pointers_made) / 2
  let free_throws := details.total_shots_made - details.three_pointers_made - two_pointers
  (two_pointers, free_throws)

/-- Theorem stating that given Yao Ming's scoring details, he made 8 2-point shots and 3 free throws -/
theorem yao_ming_shots :
  let details : ScoringDetails := {
    total_shots_made := 14,
    total_points := 28,
    three_pointers_made := 3
  }
  calculate_shots details = (8, 3) := by sorry

end NUMINAMATH_CALUDE_yao_ming_shots_l2831_283123


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l2831_283106

theorem systematic_sampling_interval_count
  (total_papers : Nat)
  (selected_papers : Nat)
  (interval_start : Nat)
  (interval_end : Nat)
  (h1 : total_papers = 1000)
  (h2 : selected_papers = 50)
  (h3 : interval_start = 850)
  (h4 : interval_end = 949)
  (h5 : interval_start ≤ interval_end)
  (h6 : interval_end ≤ total_papers) :
  let sample_interval := total_papers / selected_papers
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval = 5 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l2831_283106


namespace NUMINAMATH_CALUDE_bee_colony_loss_rate_l2831_283131

/-- Proves that given a colony of bees with an initial population of 80,000 individuals,
    if after 50 days the population reduces to one-fourth of its initial size,
    then the daily loss rate is 1,200 bees per day. -/
theorem bee_colony_loss_rate (initial_population : ℕ) (days : ℕ) (final_population : ℕ) :
  initial_population = 80000 →
  days = 50 →
  final_population = initial_population / 4 →
  (initial_population - final_population) / days = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bee_colony_loss_rate_l2831_283131


namespace NUMINAMATH_CALUDE_kitten_growth_l2831_283193

/-- The length of a kitten after doubling twice -/
def kitten_length (initial_length : ℝ) : ℝ :=
  initial_length * 2 * 2

/-- Theorem: A kitten with initial length 4 inches will be 16 inches long after doubling twice -/
theorem kitten_growth : kitten_length 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l2831_283193


namespace NUMINAMATH_CALUDE_circle_and_trajectory_l2831_283189

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x - y + 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, -2)

-- Define point D
def D : ℝ × ℝ := (4, 3)

theorem circle_and_trajectory :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ Line ∧
    A ∈ Circle C r ∧
    B ∈ Circle C r ∧
    (∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ (x, y) ∈ Circle C r) ∧
    (∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
      ∃ (E : ℝ × ℝ), E ∈ Circle C r ∧ (x, y) = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_trajectory_l2831_283189


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l2831_283129

/-- Given two workers A and B, their combined work efficiency, and A's individual efficiency,
    prove the ratio of their efficiencies. -/
theorem work_efficiency_ratio 
  (total_days : ℝ) 
  (a_days : ℝ) 
  (h1 : total_days = 12) 
  (h2 : a_days = 16) : 
  (1 / a_days) / ((1 / total_days) - (1 / a_days)) = 3 := by
  sorry

#check work_efficiency_ratio

end NUMINAMATH_CALUDE_work_efficiency_ratio_l2831_283129


namespace NUMINAMATH_CALUDE_auditorium_seats_cost_l2831_283125

theorem auditorium_seats_cost 
  (rows : ℕ) 
  (seats_per_row : ℕ) 
  (cost_per_seat : ℕ) 
  (discount_rate : ℚ) 
  (seats_per_discount_group : ℕ) : 
  rows = 5 → 
  seats_per_row = 8 → 
  cost_per_seat = 30 → 
  discount_rate = 1/10 → 
  seats_per_discount_group = 10 → 
  (rows * seats_per_row * cost_per_seat : ℚ) - 
    ((rows * seats_per_row / seats_per_discount_group : ℚ) * 
     (seats_per_discount_group * cost_per_seat * discount_rate)) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seats_cost_l2831_283125


namespace NUMINAMATH_CALUDE_multiples_of_four_l2831_283176

theorem multiples_of_four (n : ℕ) : n = 16 ↔ 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ m ∈ l, m % 4 = 0 ∧ n ≤ m ∧ m ≤ 112) ∧
    (∀ k : ℕ, n ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l) ∧
    (∀ m : ℕ, n < m → 
      ¬∃ (l' : List ℕ), l'.length = 25 ∧ 
        (∀ m' ∈ l', m' % 4 = 0 ∧ m ≤ m' ∧ m' ≤ 112) ∧
        (∀ k : ℕ, m ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l'))) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_four_l2831_283176


namespace NUMINAMATH_CALUDE_infinite_commuting_functions_l2831_283100

/-- Given a bijective function f from R to R, there exists an infinite number of functions g 
    from R to R such that f(g(x)) = g(f(x)) for all x in R. -/
theorem infinite_commuting_functions 
  (f : ℝ → ℝ) 
  (hf : Function.Bijective f) : 
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ g ∈ S, ∀ x, f (g x) = g (f x) := by
  sorry

end NUMINAMATH_CALUDE_infinite_commuting_functions_l2831_283100


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l2831_283162

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 3) (hx₂ : x₂ = -4) :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x - x₁) * (x - x₂) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l2831_283162


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2831_283154

theorem complex_product_magnitude : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2831_283154


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2831_283108

/-- Two lines ax+2y+1=0 and 3x+(a-1)y+1=0 are parallel if and only if a = -2 -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + (a-1)*y + 1 = 0) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2831_283108


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_l2831_283102

theorem largest_number_from_hcf_lcm (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.gcd a b = 210 →
  Nat.gcd (Nat.gcd a b) c = 210 →
  Nat.lcm (Nat.lcm a b) c = 902910 →
  max a (max b c) = 4830 :=
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_l2831_283102


namespace NUMINAMATH_CALUDE_smallest_y_value_l2831_283165

theorem smallest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 1) : 
  ∀ (z : ℤ), z ≥ -10 ∨ ¬∃ (w : ℤ), w * z + 3 * w + 2 * z = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_value_l2831_283165


namespace NUMINAMATH_CALUDE_equal_pieces_after_exchanges_l2831_283168

theorem equal_pieces_after_exchanges (initial_white : ℕ) (initial_black : ℕ) 
  (exchange_count : ℕ) (pieces_per_exchange : ℕ) :
  initial_white = 80 →
  initial_black = 50 →
  pieces_per_exchange = 3 →
  exchange_count = 5 →
  initial_white - exchange_count * pieces_per_exchange = 
  initial_black + exchange_count * pieces_per_exchange :=
by
  sorry

#check equal_pieces_after_exchanges

end NUMINAMATH_CALUDE_equal_pieces_after_exchanges_l2831_283168


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l2831_283179

theorem smallest_cube_root_with_small_remainder :
  ∃ (m : ℕ) (r : ℝ),
    (∀ (m' : ℕ) (r' : ℝ), m' < m → ¬(∃ (n' : ℕ), m'^(1/3 : ℝ) = n' + r' ∧ 0 < r' ∧ r' < 1/10000)) ∧
    m^(1/3 : ℝ) = 58 + r ∧
    0 < r ∧
    r < 1/10000 ∧
    (∀ (n : ℕ), n < 58 → 
      ¬(∃ (m' : ℕ) (r' : ℝ), m'^(1/3 : ℝ) = n + r' ∧ 0 < r' ∧ r' < 1/10000)) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l2831_283179


namespace NUMINAMATH_CALUDE_ancient_chinese_fruit_problem_l2831_283113

/-- Represents the ancient Chinese fruit problem -/
theorem ancient_chinese_fruit_problem 
  (x y : ℚ) -- x: number of bitter fruits, y: number of sweet fruits
  (h1 : x + y = 1000) -- total number of fruits
  (h2 : 7 * (4 / 7 : ℚ) = 4) -- cost of 7 bitter fruits
  (h3 : 9 * (11 / 9 : ℚ) = 11) -- cost of 9 sweet fruits
  (h4 : (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) -- total cost
  : (x + y = 1000 ∧ (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) :=
by sorry

end NUMINAMATH_CALUDE_ancient_chinese_fruit_problem_l2831_283113


namespace NUMINAMATH_CALUDE_building_shadow_length_l2831_283155

/-- Given a flagstaff and a building with their respective heights and shadow lengths,
    prove that the length of the shadow cast by the building is as calculated. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ)
  (building_height : ℝ) :
  flagstaff_height = 17.5 →
  flagstaff_shadow = 40.25 →
  building_height = 12.5 →
  ∃ (building_shadow : ℝ),
    building_shadow = 28.75 ∧
    flagstaff_height / flagstaff_shadow = building_height / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_building_shadow_length_l2831_283155


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2831_283122

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = x - y ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l2831_283122


namespace NUMINAMATH_CALUDE_square_root_division_l2831_283197

theorem square_root_division (x : ℝ) : (Real.sqrt 3600 / x = 4) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l2831_283197


namespace NUMINAMATH_CALUDE_line_m_equation_l2831_283120

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

theorem line_m_equation (ℓ m : Line) (Q Q'' : Point) :
  ℓ.a = 1 ∧ ℓ.b = 3 ∧ ℓ.c = 7 ∧  -- Equation of line ℓ: x + 3y = 7
  Q.x = 2 ∧ Q.y = 5 ∧  -- Coordinates of Q
  Q''.x = 5 ∧ Q''.y = 0 ∧  -- Coordinates of Q''
  (1 : ℝ) * ℓ.a + 2 * ℓ.b = ℓ.c ∧  -- ℓ passes through (1, 2)
  (1 : ℝ) * m.a + 2 * m.b = m.c ∧  -- m passes through (1, 2)
  Q'' = reflect (reflect Q ℓ) m →  -- Q'' is the result of reflecting Q about ℓ and then m
  m.a = 2 ∧ m.b = -1 ∧ m.c = 2  -- Equation of line m: 2x - y = 2
  := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l2831_283120


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2831_283166

theorem arithmetic_expression_equality : 8 + 12 / 3 - 2^3 + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2831_283166


namespace NUMINAMATH_CALUDE_hash_five_three_l2831_283192

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a + 6 * b

-- Theorem statement
theorem hash_five_three : hash 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_three_l2831_283192


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2831_283171

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : quadratic_function f) 
  (h2 : f 0 = 1) 
  (h3 : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧ 
  Set.Icc (3/4 : ℝ) 3 = {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2831_283171


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2831_283101

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-3) - (-2) + 3 * k - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2831_283101


namespace NUMINAMATH_CALUDE_log_division_simplification_l2831_283147

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2831_283147


namespace NUMINAMATH_CALUDE_x_squared_congruence_l2831_283135

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 10 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l2831_283135


namespace NUMINAMATH_CALUDE_fraction_comparison_l2831_283126

theorem fraction_comparison (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m < n) :
  (m + 3 : ℚ) / (n + 3) > (m : ℚ) / n := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2831_283126


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2831_283156

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2831_283156


namespace NUMINAMATH_CALUDE_arrow_pointing_theorem_l2831_283199

/-- Represents the direction of an arrow -/
inductive Direction
| Left
| Right

/-- Represents an arrangement of arrows -/
def ArrowArrangement (n : ℕ) := Fin n → Direction

/-- The number of arrows pointing to the i-th arrow -/
def pointingTo (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

/-- The number of arrows that the i-th arrow is pointing to -/
def pointingFrom (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

theorem arrow_pointing_theorem (n : ℕ) (h : Odd n) (h1 : n ≥ 1) (arr : ArrowArrangement n) :
  ∃ i : Fin n, pointingTo arr i = pointingFrom arr i := by sorry

end NUMINAMATH_CALUDE_arrow_pointing_theorem_l2831_283199


namespace NUMINAMATH_CALUDE_q_polynomial_form_l2831_283194

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^5 + 5x^4 + 8x^3 + 9x) = (10x^4 + 35x^3 + 50x^2 + 72x + 5),
    prove that q(x) = -2x^5 + 5x^4 + 27x^3 + 50x^2 + 63x + 5 -/
theorem q_polynomial_form (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2*x^5 + 5*x^4 + 8*x^3 + 9*x) = 10*x^4 + 35*x^3 + 50*x^2 + 72*x + 5) :
  ∀ x, q x = -2*x^5 + 5*x^4 + 27*x^3 + 50*x^2 + 63*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l2831_283194


namespace NUMINAMATH_CALUDE_fraction_product_result_l2831_283121

def fraction_product (n : ℕ) : ℚ :=
  let seq (k : ℕ) := 2 + 3 * k
  (seq 0) / (seq n)

theorem fraction_product_result :
  fraction_product 667 = 2 / 2007 := by sorry

end NUMINAMATH_CALUDE_fraction_product_result_l2831_283121


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l2831_283191

/-- A cubic equation with coefficients a and b -/
def cubic_equation (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Condition for the cubic equation to have only one real root -/
def has_one_real_root (a b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a b x = 0

theorem cubic_one_real_root :
  (has_one_real_root (-3) (-3)) ∧
  (∀ b > 2, has_one_real_root (-3) b) ∧
  (has_one_real_root 0 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_one_real_root_l2831_283191


namespace NUMINAMATH_CALUDE_expression_value_l2831_283178

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 7
  let c : ℤ := 2
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2831_283178


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2831_283104

theorem circle_diameter_from_area (A : ℝ) (h : A = 196 * Real.pi) :
  ∃ (d : ℝ), d = 28 ∧ A = Real.pi * (d / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2831_283104


namespace NUMINAMATH_CALUDE_exists_prime_number_of_ones_l2831_283181

/-- A number consisting of q ones in decimal notation -/
def number_of_ones (q : ℕ) : ℕ := (10^q - 1) / 9

/-- Theorem stating that there exists a natural number k such that
    a number consisting of (6k-1) ones is prime -/
theorem exists_prime_number_of_ones :
  ∃ k : ℕ, Nat.Prime (number_of_ones (6*k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_prime_number_of_ones_l2831_283181


namespace NUMINAMATH_CALUDE_functional_equation_proof_l2831_283114

open Real

theorem functional_equation_proof (x : ℝ) (hx : x ≠ 0) :
  let f : ℝ → ℝ := λ x => (x / 3) + (2 / (3 * x))
  2 * f x - f (1 / x) = 1 / x := by sorry

end NUMINAMATH_CALUDE_functional_equation_proof_l2831_283114


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2831_283190

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  (1.15 * l) * (1.25 * w) = 1.4375 * (l * w) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2831_283190


namespace NUMINAMATH_CALUDE_six_balls_removal_ways_l2831_283149

/-- Represents the number of ways to remove n balls from a box, removing at least one at a time. -/
def removalWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else sorry  -- The actual implementation would go here

/-- The number of ways to remove 6 balls is 32. -/
theorem six_balls_removal_ways : removalWays 6 = 32 := by
  sorry  -- The proof would go here

end NUMINAMATH_CALUDE_six_balls_removal_ways_l2831_283149


namespace NUMINAMATH_CALUDE_annika_hike_distance_l2831_283112

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  flat_speed : ℝ  -- minutes per kilometer on flat terrain
  uphill_speed : ℝ  -- minutes per kilometer uphill
  downhill_speed : ℝ  -- minutes per kilometer downhill
  initial_distance : ℝ  -- kilometers hiked initially
  total_time : ℝ  -- total time available to return

/-- Calculates the total distance hiked east given a hiking scenario -/
def total_distance_east (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating the total distance hiked east in the given scenario -/
theorem annika_hike_distance (scenario : HikingScenario) 
  (h1 : scenario.flat_speed = 10)
  (h2 : scenario.uphill_speed = 15)
  (h3 : scenario.downhill_speed = 7)
  (h4 : scenario.initial_distance = 2.5)
  (h5 : scenario.total_time = 35) :
  total_distance_east scenario = 3.0833 :=
sorry

end NUMINAMATH_CALUDE_annika_hike_distance_l2831_283112


namespace NUMINAMATH_CALUDE_two_a_plus_b_values_l2831_283110

theorem two_a_plus_b_values (a b : ℝ) 
  (h1 : |a - 1| = 4)
  (h2 : |-b| = |-7|)
  (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_two_a_plus_b_values_l2831_283110


namespace NUMINAMATH_CALUDE_product_zero_l2831_283150

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l2831_283150


namespace NUMINAMATH_CALUDE_root_product_theorem_l2831_283107

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b)^2 - p*(a^2 + 1/b) + r = 0) →
  ((b^2 + 1/a)^2 - p*(b^2 + 1/a) + r = 0) →
  r = 46/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2831_283107


namespace NUMINAMATH_CALUDE_sqrt_of_four_l2831_283159

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l2831_283159


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2831_283187

theorem polynomial_equation_solution (a a1 a2 a3 a4 : ℝ) : 
  (∀ x, (x + a)^4 = x^4 + a1*x^3 + a2*x^2 + a3*x + a4) →
  (a1 + a2 + a3 = 64) →
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2831_283187


namespace NUMINAMATH_CALUDE_basketball_probability_l2831_283146

theorem basketball_probability (p_free_throw p_high_school p_pro : ℚ) 
  (h1 : p_free_throw = 4/5)
  (h2 : p_high_school = 1/2)
  (h3 : p_pro = 1/3) :
  1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l2831_283146


namespace NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2831_283175

/-- The sum of an arithmetic series with given first term, last term, and common difference -/
def arithmetic_series_sum (a l d : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  (n * (a + l)) / 2

/-- Theorem stating that the sum of the specific arithmetic series is -576 -/
theorem specific_arithmetic_series_sum :
  arithmetic_series_sum (-47) (-1) 2 = -576 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2831_283175


namespace NUMINAMATH_CALUDE_projection_of_vectors_l2831_283151

/-- Given two vectors in ℝ², prove that the projection of one onto the other is as specified. -/
theorem projection_of_vectors (a b : ℝ × ℝ) (h1 : a = (0, 1)) (h2 : b = (1, Real.sqrt 3)) :
  (a • b / (b • b)) • b = (Real.sqrt 3 / 4) • b :=
sorry

end NUMINAMATH_CALUDE_projection_of_vectors_l2831_283151


namespace NUMINAMATH_CALUDE_floor_product_equals_45_l2831_283117

theorem floor_product_equals_45 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 45 ↔ x ∈ Set.Icc (7.5) (7 + 2/3) := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_45_l2831_283117


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l2831_283152

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 1 / b = Real.sqrt (a * b)) :
  (∀ x y, x > 0 → y > 0 → 1 / x + 1 / y = Real.sqrt (x * y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = Real.sqrt (x * y) ∧ 2 * x + 3 * y = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l2831_283152


namespace NUMINAMATH_CALUDE_power_equation_solution_l2831_283119

theorem power_equation_solution (m : ℝ) : (7 : ℝ) ^ (4 * m) = (1 / 7) ^ (2 * m - 18) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2831_283119


namespace NUMINAMATH_CALUDE_factorization_equality_l2831_283161

theorem factorization_equality (x y : ℝ) : x^2 - 1 + 2*x*y + y^2 = (x+y+1)*(x+y-1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l2831_283161


namespace NUMINAMATH_CALUDE_range_of_m_l2831_283124

theorem range_of_m (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2) →
  (∀ x, q x ↔ x^2 - 2*x + (1-m^2) ≤ 0) →
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ x, ¬(p x) ∧ q x :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2831_283124


namespace NUMINAMATH_CALUDE_apples_picked_total_l2831_283186

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 :=
by sorry

end NUMINAMATH_CALUDE_apples_picked_total_l2831_283186


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2831_283132

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 15) = 15) ∧ (x = 42) := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2831_283132


namespace NUMINAMATH_CALUDE_spy_arrangement_exists_l2831_283128

-- Define the board
def Board := Fin 6 → Fin 6 → Bool

-- Define the direction a spy can face
inductive Direction
| North
| East
| South
| West

-- Define a spy's position and direction
structure Spy where
  row : Fin 6
  col : Fin 6
  dir : Direction

-- Define the visibility function for a spy
def canSee (s : Spy) (r : Fin 6) (c : Fin 6) : Prop :=
  match s.dir with
  | Direction.North => 
      (s.row > r && s.row - r ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.East => 
      (s.col < c && c - s.col ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))
  | Direction.South => 
      (s.row < r && r - s.row ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.West => 
      (s.col > c && s.col - c ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))

-- Define a valid arrangement of spies
def validArrangement (spies : List Spy) : Prop :=
  spies.length = 18 ∧
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 →
    ¬(canSee s1 s2.row s2.col) ∧ ¬(canSee s2 s1.row s1.col)

-- Theorem: There exists a valid arrangement of 18 spies
theorem spy_arrangement_exists : ∃ spies : List Spy, validArrangement spies := by
  sorry

end NUMINAMATH_CALUDE_spy_arrangement_exists_l2831_283128


namespace NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l2831_283118

/-- Given a total number of votes and a loss margin, calculate the percentage of votes received by the losing candidate. -/
def calculate_vote_percentage (total_votes : ℕ) (loss_margin : ℕ) : ℚ :=
  let candidate_votes := (total_votes - loss_margin) / 2
  (candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that given 7000 total votes and a loss margin of 2100 votes, the losing candidate received 35% of the votes. -/
theorem losing_candidate_vote_percentage :
  calculate_vote_percentage 7000 2100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_losing_candidate_vote_percentage_l2831_283118


namespace NUMINAMATH_CALUDE_coin_ratio_l2831_283169

/-- Given the total number of coins, the fraction Amalie spends, and the number of coins
    Amalie has left, prove the ratio of Elsa's coins to Amalie's original coins. -/
theorem coin_ratio (total : ℕ) (amalie_spent_fraction : ℚ) (amalie_left : ℕ)
  (h1 : total = 440)
  (h2 : amalie_spent_fraction = 3/4)
  (h3 : amalie_left = 90) :
  (total - (amalie_left / (1 - amalie_spent_fraction))) / (amalie_left / (1 - amalie_spent_fraction)) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_coin_ratio_l2831_283169


namespace NUMINAMATH_CALUDE_gizi_does_not_catch_up_l2831_283136

/-- Represents the work progress of Kató and Gizi -/
structure WorkProgress where
  kato_lines : ℕ
  gizi_lines : ℕ

/-- Represents the copying rates and page capacities -/
structure CopyingParameters where
  kato_lines_per_page : ℕ
  gizi_lines_per_page : ℕ
  kato_rate : ℕ
  gizi_rate : ℕ

def initial_state : WorkProgress :=
  { kato_lines := 80,  -- 4 pages * 20 lines per page
    gizi_lines := 0 }

def copying_params : CopyingParameters :=
  { kato_lines_per_page := 20,
    gizi_lines_per_page := 30,
    kato_rate := 3,
    gizi_rate := 4 }

def setup_time_progress (wp : WorkProgress) : WorkProgress :=
  { kato_lines := wp.kato_lines + 3,  -- 2.5 rounded up to 3
    gizi_lines := wp.gizi_lines }

def update_progress (wp : WorkProgress) (cp : CopyingParameters) : WorkProgress :=
  { kato_lines := wp.kato_lines + cp.kato_rate,
    gizi_lines := wp.gizi_lines + cp.gizi_rate }

def gizi_catches_up (wp : WorkProgress) : Prop :=
  wp.gizi_lines * 4 ≥ wp.kato_lines * 3

theorem gizi_does_not_catch_up :
  ¬∃ n : ℕ, gizi_catches_up (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)) ∧
            (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)).gizi_lines ≤ 150 :=
sorry

end NUMINAMATH_CALUDE_gizi_does_not_catch_up_l2831_283136
