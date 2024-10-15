import Mathlib

namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_prism_l1283_128306

theorem sphere_surface_area_of_circumscribed_prism (h : ℝ) (v : ℝ) (r : ℝ) :
  h = 4 →
  v = 16 →
  v = h * r^2 →
  let d := Real.sqrt (h^2 + 2 * r^2)
  (4 / 3) * π * (d / 2)^3 = (4 / 3) * π * r^2 * h →
  4 * π * (d / 2)^2 = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_prism_l1283_128306


namespace NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1283_128361

theorem sin_alpha_plus_beta (α β t : ℝ) : 
  (Real.exp (α + π/6) - Real.exp (-α - π/6) + Real.cos (5*π/3 + α) = t) →
  (Real.exp (β - π/4) - Real.exp (π/4 - β) + Real.cos (5*π/4 + β) = -t) →
  Real.sin (α + β) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1283_128361


namespace NUMINAMATH_CALUDE_odd_function_sum_l1283_128329

-- Define an odd function f on the real numbers
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : isOddFunction f) (h2 : f 1 = -2) :
  f (-1) + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1283_128329


namespace NUMINAMATH_CALUDE_rectangle_y_coordinate_l1283_128317

/-- Given a rectangle with vertices (-8, 1), (1, 1), (1, y), and (-8, y) in a rectangular coordinate system,
    if the area of the rectangle is 72, then y = 9 -/
theorem rectangle_y_coordinate (y : ℝ) : 
  let vertex1 : ℝ × ℝ := (-8, 1)
  let vertex2 : ℝ × ℝ := (1, 1)
  let vertex3 : ℝ × ℝ := (1, y)
  let vertex4 : ℝ × ℝ := (-8, y)
  let length : ℝ := vertex2.1 - vertex1.1
  let width : ℝ := vertex3.2 - vertex2.2
  let area : ℝ := length * width
  area = 72 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_coordinate_l1283_128317


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l1283_128319

-- Define the sets A and B
def A : Set ℝ := {x | (1 + x) / (2 - x) > 0}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Theorem 1: When a = 1, A ∩ B = {x | 1 ≤ x < 2}
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: When B ⊆ ℝ\A, the range of a is 0 < a ≤ 1/2
theorem range_of_a_when_B_subset_complement_A :
  ∀ a : ℝ, (0 < a ∧ B a ⊆ (Set.univ \ A)) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_B_subset_complement_A_l1283_128319


namespace NUMINAMATH_CALUDE_fraction_simplification_l1283_128383

theorem fraction_simplification : (2 : ℚ) / (1 - 2 / 3) = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1283_128383


namespace NUMINAMATH_CALUDE_age_difference_proof_l1283_128367

theorem age_difference_proof (younger_age elder_age : ℕ) : 
  younger_age = 35 →
  elder_age - 15 = 2 * (younger_age - 15) →
  elder_age - younger_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1283_128367


namespace NUMINAMATH_CALUDE_g_zero_at_neg_three_iff_s_eq_neg_192_l1283_128364

/-- The function g(x) defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

/-- Theorem stating that g(-3) = 0 if and only if s = -192 -/
theorem g_zero_at_neg_three_iff_s_eq_neg_192 :
  ∀ s : ℝ, g (-3) s = 0 ↔ s = -192 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_neg_three_iff_s_eq_neg_192_l1283_128364


namespace NUMINAMATH_CALUDE_root_implies_c_value_l1283_128378

theorem root_implies_c_value (b c : ℝ) :
  (∃ (x : ℂ), x^2 + b*x + c = 0 ∧ x = 1 - Complex.I * Real.sqrt 2) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_c_value_l1283_128378


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l1283_128309

/-- A hyperbola with equation x²/m - y²/4 = 1 and focal distance 6 has m = 5 -/
theorem hyperbola_focal_distance (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m - y^2/4 = 1) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →                -- Focal distance is 6, so c = 3
  m = 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l1283_128309


namespace NUMINAMATH_CALUDE_shifted_line_equation_and_intercept_l1283_128352

/-- A line obtained by shifting a direct proportion function -/
structure ShiftedLine where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_one_two : k * 1 + b = 2 + 5
  shifted_up_five : b = 5

theorem shifted_line_equation_and_intercept (l : ShiftedLine) :
  (l.k = 2 ∧ l.b = 5) ∧ 
  (∃ (x : ℝ), x = -2.5 ∧ l.k * x + l.b = 0) := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_equation_and_intercept_l1283_128352


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1283_128331

theorem arithmetic_expression_evaluation : (8 * 6) - (4 / 2) = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1283_128331


namespace NUMINAMATH_CALUDE_division_simplification_l1283_128335

theorem division_simplification (x : ℝ) (hx : x ≠ 0) :
  (1 + 1/x) / ((x^2 + x)/x) = 1/x := by sorry

end NUMINAMATH_CALUDE_division_simplification_l1283_128335


namespace NUMINAMATH_CALUDE_complex_calculation_l1283_128395

theorem complex_calculation (p q : ℂ) (hp : p = 3 + 2*I) (hq : q = 2 - 3*I) :
  3*p + 4*q = 17 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1283_128395


namespace NUMINAMATH_CALUDE_quadratic_sequence_l1283_128388

theorem quadratic_sequence (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_l1283_128388


namespace NUMINAMATH_CALUDE_myrtle_absence_duration_l1283_128363

/-- Proves that Myrtle was gone for 21 days given the conditions of the problem -/
theorem myrtle_absence_duration (daily_production neighbor_took dropped remaining : ℕ) 
  (h1 : daily_production = 3)
  (h2 : neighbor_took = 12)
  (h3 : dropped = 5)
  (h4 : remaining = 46) :
  ∃ d : ℕ, d * daily_production - neighbor_took - dropped = remaining ∧ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_absence_duration_l1283_128363


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l1283_128315

theorem pencil_eraser_cost :
  ∃ (p e : ℕ), 
    p > 0 ∧ 
    e > 0 ∧ 
    7 * p + 5 * e = 130 ∧ 
    p > e ∧ 
    p + e = 22 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l1283_128315


namespace NUMINAMATH_CALUDE_lcm_sum_inequality_l1283_128356

theorem lcm_sum_inequality (a b c d e : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (1 : ℚ) / Nat.lcm a b + 1 / Nat.lcm b c + 1 / Nat.lcm c d + 1 / Nat.lcm d e ≤ 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_inequality_l1283_128356


namespace NUMINAMATH_CALUDE_problem_solution_l1283_128382

def f (k : ℝ) (x : ℝ) : ℝ := |3*x - 1| + |3*x + k|
def g (x : ℝ) : ℝ := x + 4

theorem problem_solution :
  (∀ x : ℝ, f (-3) x ≥ 4 ↔ (x ≤ 0 ∨ x ≥ 4/3)) ∧
  (∀ k : ℝ, k > -1 → 
    (∀ x : ℝ, x ∈ Set.Icc (-k/3) (1/3) → f k x ≤ g x) →
    k ∈ Set.Ioo (-1) (9/4)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1283_128382


namespace NUMINAMATH_CALUDE_prop_p_or_q_l1283_128374

theorem prop_p_or_q : 
  (∀ x : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x : ℝ, Real.sin x + Real.cos x = 2) :=
sorry

end NUMINAMATH_CALUDE_prop_p_or_q_l1283_128374


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_l1283_128360

theorem triangle_square_perimeter (d : ℕ) : 
  let triangle_side := s + d
  let square_side := s
  (∃ s : ℚ, s > 0 ∧ 3 * triangle_side - 4 * square_side = 1989) →
  d > 663 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_l1283_128360


namespace NUMINAMATH_CALUDE_mikes_typing_speed_reduction_l1283_128376

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Mike's typing speed reduction is 20 words per minute. -/
theorem mikes_typing_speed_reduction :
  typing_speed_reduction 65 810 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_typing_speed_reduction_l1283_128376


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seats_l1283_128332

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : ℕ
  large_seats : ℕ
  small_seat_capacity : ℕ
  people_on_small_seats : ℕ

/-- The number of small seats on the Ferris wheel is 2 -/
theorem ferris_wheel_small_seats (fw : FerrisWheel) 
  (h1 : fw.large_seats = 23)
  (h2 : fw.small_seat_capacity = 14)
  (h3 : fw.people_on_small_seats = 28) :
  fw.small_seats = 2 := by
  sorry

#check ferris_wheel_small_seats

end NUMINAMATH_CALUDE_ferris_wheel_small_seats_l1283_128332


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l1283_128379

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_two_l1283_128379


namespace NUMINAMATH_CALUDE_angle_through_point_l1283_128346

theorem angle_through_point (α : Real) : 
  0 ≤ α ∧ α ≤ 2 * Real.pi → 
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (2 * Real.pi / 3) ∧ 
                      r * Real.sin α = Real.sin (2 * Real.pi / 3)) →
  α = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_through_point_l1283_128346


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1283_128369

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1283_128369


namespace NUMINAMATH_CALUDE_performance_arrangements_l1283_128386

def original_programs : ℕ := 6
def added_programs : ℕ := 3
def available_spaces : ℕ := original_programs - 1

theorem performance_arrangements : 
  (Nat.descFactorial available_spaces added_programs) + 
  (Nat.descFactorial 3 2 * Nat.descFactorial available_spaces 2) + 
  (5 * Nat.descFactorial 3 3) = 210 := by sorry

end NUMINAMATH_CALUDE_performance_arrangements_l1283_128386


namespace NUMINAMATH_CALUDE_coefficient_of_c_l1283_128348

theorem coefficient_of_c (A : ℝ) (c d : ℝ) : 
  (∀ c', c' ≤ 47) → 
  (A * 47 + (d - 12)^2 = 235) → 
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_c_l1283_128348


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1283_128334

theorem ellipse_to_hyperbola (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = Real.sqrt 3 * c) 
  (h4 : a + c = 3 * Real.sqrt 3) 
  (h5 : a^2 = b^2 + c^2) :
  ∃ (x y : ℝ), y^2 / 12 - x^2 / 9 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l1283_128334


namespace NUMINAMATH_CALUDE_valid_numerical_pyramid_exists_l1283_128314

/-- Represents a row in the numerical pyramid --/
structure PyramidRow where
  digits : List ℕ
  result : ℕ

/-- Represents the entire numerical pyramid --/
structure NumericalPyramid where
  row1 : PyramidRow
  row2 : PyramidRow
  row3 : PyramidRow
  row4 : PyramidRow
  row5 : PyramidRow
  row6 : PyramidRow
  row7 : PyramidRow

/-- Function to check if a pyramid satisfies all conditions --/
def is_valid_pyramid (p : NumericalPyramid) : Prop :=
  p.row1.digits = [1, 2] ∧ p.row1.result = 3 ∧
  p.row2.digits = [1, 2, 3] ∧ p.row2.result = 4 ∧
  p.row3.digits = [1, 2, 3, 4] ∧ p.row3.result = 5 ∧
  p.row4.digits = [1, 2, 3, 4, 5] ∧ p.row4.result = 6 ∧
  p.row5.digits = [1, 2, 3, 4, 5, 6] ∧ p.row5.result = 7 ∧
  p.row6.digits = [1, 2, 3, 4, 5, 6, 7] ∧ p.row6.result = 8 ∧
  p.row7.digits = [1, 2, 3, 4, 5, 6, 7, 8] ∧ p.row7.result = 9

/-- Theorem stating that a valid numerical pyramid exists --/
theorem valid_numerical_pyramid_exists : ∃ (p : NumericalPyramid), is_valid_pyramid p := by
  sorry

end NUMINAMATH_CALUDE_valid_numerical_pyramid_exists_l1283_128314


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1283_128311

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2) ≤ 2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a + b)^2 / (a^2 + b^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1283_128311


namespace NUMINAMATH_CALUDE_tan_sum_squared_l1283_128353

theorem tan_sum_squared (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) + Real.tan (b / 2))^2 = 6 ∨ (Real.tan (a / 2) + Real.tan (b / 2))^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_squared_l1283_128353


namespace NUMINAMATH_CALUDE_sets_equality_implies_sum_l1283_128391

-- Define the sets A and B
def A (x y : ℝ) : Set ℝ := {0, |x|, y}
def B (x y : ℝ) : Set ℝ := {x, x*y, Real.sqrt (x-y)}

-- State the theorem
theorem sets_equality_implies_sum (x y : ℝ) : A x y = B x y → x + y = -2 := by
  sorry

end NUMINAMATH_CALUDE_sets_equality_implies_sum_l1283_128391


namespace NUMINAMATH_CALUDE_product_equals_nine_l1283_128362

theorem product_equals_nine : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * 
  (1 + 1/5) * (1 + 1/6) * (1 + 1/7) * (1 + 1/8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_nine_l1283_128362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1283_128328

/-- Given two arithmetic sequences and their sum ratios, prove a specific ratio of their terms -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h : ∀ n : ℕ, S n / T n = (2 * n - 3 : ℚ) / (4 * n - 1 : ℚ)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1283_128328


namespace NUMINAMATH_CALUDE_large_data_logarithm_l1283_128396

theorem large_data_logarithm (m : ℝ) (n : ℕ+) :
  (1 < m) ∧ (m < 10) ∧
  (0.4771 < Real.log 3 / Real.log 10) ∧ (Real.log 3 / Real.log 10 < 0.4772) ∧
  (3 ^ 2000 : ℝ) = m * 10 ^ (n : ℝ) →
  n = 954 := by
  sorry

end NUMINAMATH_CALUDE_large_data_logarithm_l1283_128396


namespace NUMINAMATH_CALUDE_repair_time_is_30_minutes_l1283_128321

/-- The time it takes to replace the buckle on one shoe (in minutes) -/
def buckle_time : ℕ := 5

/-- The time it takes to even out the heel on one shoe (in minutes) -/
def heel_time : ℕ := 10

/-- The number of shoes Melissa is repairing -/
def num_shoes : ℕ := 2

/-- The total time Melissa spends repairing her shoes -/
def total_repair_time : ℕ := (buckle_time + heel_time) * num_shoes

theorem repair_time_is_30_minutes : total_repair_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_repair_time_is_30_minutes_l1283_128321


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l1283_128303

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax : ℝ) (food_tax : ℝ) (other_tax : ℝ) : ℝ :=
  clothing_percent * clothing_tax + food_percent * food_tax + other_percent * other_tax

/-- Theorem stating that the total tax percentage is 5.2% given the specific conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08 = 0.052 := by
  sorry

#eval total_tax_percentage 0.5 0.1 0.4 0.04 0 0.08

end NUMINAMATH_CALUDE_shopping_tax_theorem_l1283_128303


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l1283_128354

/-- The parabola P with equation y = x^2 + 10x -/
def P (x y : ℝ) : Prop := y = x^2 + 10 * x

/-- The point Q (10, 5) -/
def Q : ℝ × ℝ := (10, 5)

/-- A line through Q with slope m -/
def line_through_Q (m : ℝ) (x y : ℝ) : Prop :=
  y - Q.2 = m * (x - Q.1)

/-- The sum of slopes of tangent lines to P passing through Q is 60 -/
theorem sum_of_tangent_slopes :
  ∃ r s : ℝ,
    (∀ m : ℝ, r < m ∧ m < s ↔
      ¬∃ x y : ℝ, P x y ∧ line_through_Q m x y) ∧
    r + s = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l1283_128354


namespace NUMINAMATH_CALUDE_simplify_expression_l1283_128310

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 0) (h3 : a ≠ -1) :
  (a^2 - 2*a + 1) / (a^2 - 1) / (a - 2*a / (a + 1)) = 1 / a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1283_128310


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1283_128392

/-- Given two infinite geometric series with the specified properties, prove that n = 195 --/
theorem geometric_series_problem (n : ℝ) : 
  let first_series_a1 : ℝ := 15
  let first_series_a2 : ℝ := 5
  let second_series_a1 : ℝ := 15
  let second_series_a2 : ℝ := 5 + n
  let first_series_sum := first_series_a1 / (1 - (first_series_a2 / first_series_a1))
  let second_series_sum := second_series_a1 / (1 - (second_series_a2 / second_series_a1))
  second_series_sum = 5 * first_series_sum →
  n = 195 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l1283_128392


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l1283_128381

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity (m : ℝ) :
  let f := fun (x : ℝ) ↦ -2 * x^2 + m * x + 1
  IsMonotonic f (-1) 4 ↔ m ∈ Set.Iic (-4) ∪ Set.Ici 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l1283_128381


namespace NUMINAMATH_CALUDE_evaluate_expression_l1283_128326

theorem evaluate_expression : -(16 / 4 * 11 - 50 + 2^3 * 5) = -34 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1283_128326


namespace NUMINAMATH_CALUDE_correct_calculation_l1283_128385

theorem correct_calculation (a : ℝ) : 3 * a^2 - 2 * a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1283_128385


namespace NUMINAMATH_CALUDE_shooting_game_probability_l1283_128308

-- Define the probability of hitting the target
variable (p : ℝ)

-- Define the number of shooting attempts
def η : ℕ → ℝ
| 1 => p
| 2 => (1 - p) * p
| 3 => (1 - p)^2
| _ => 0

-- Define the expected value of η
def E_η : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p)^2

-- Theorem statement
theorem shooting_game_probability (h1 : 0 < p) (h2 : p < 1) (h3 : E_η > 7/4) :
  p ∈ Set.Ioo 0 (1/2) :=
sorry

end NUMINAMATH_CALUDE_shooting_game_probability_l1283_128308


namespace NUMINAMATH_CALUDE_power_of_six_seven_equals_product_of_seven_sixes_l1283_128358

theorem power_of_six_seven_equals_product_of_seven_sixes :
  6^7 = (List.replicate 7 6).prod := by
  sorry

end NUMINAMATH_CALUDE_power_of_six_seven_equals_product_of_seven_sixes_l1283_128358


namespace NUMINAMATH_CALUDE_family_member_bites_eq_two_l1283_128327

/-- The number of mosquito bites each family member (excluding Cyrus) has, given the conditions in the problem. -/
def family_member_bites : ℕ :=
  let cyrus_arm_leg_bites : ℕ := 14
  let cyrus_body_bites : ℕ := 10
  let cyrus_total_bites : ℕ := cyrus_arm_leg_bites + cyrus_body_bites
  let family_size : ℕ := 6
  let family_total_bites : ℕ := cyrus_total_bites / 2
  family_total_bites / family_size

theorem family_member_bites_eq_two : family_member_bites = 2 := by
  sorry

end NUMINAMATH_CALUDE_family_member_bites_eq_two_l1283_128327


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1283_128368

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℝ) 
  (wrong_mark correct_mark : ℝ) :
  n = 25 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (n * initial_average - wrong_mark + correct_mark) / n = 98 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1283_128368


namespace NUMINAMATH_CALUDE_coffee_consumption_ratio_l1283_128399

/-- Given that Brayan drinks 4 cups of coffee per hour and they drink a total of 30 cups of coffee
    together in 5 hours, prove that the ratio of the amount of coffee Brayan drinks to the amount
    Ivory drinks is 2:1. -/
theorem coffee_consumption_ratio :
  let brayan_per_hour : ℚ := 4
  let total_cups : ℚ := 30
  let total_hours : ℚ := 5
  let ivory_per_hour : ℚ := total_cups / total_hours - brayan_per_hour
  brayan_per_hour / ivory_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_ratio_l1283_128399


namespace NUMINAMATH_CALUDE_specific_parallelogram_area_and_height_l1283_128398

/-- Represents a parallelogram with given properties -/
structure Parallelogram where
  angle : ℝ  -- One angle of the parallelogram in degrees
  side1 : ℝ  -- Length of one side
  side2 : ℝ  -- Length of the adjacent side
  extension : ℝ  -- Length of extension beyond the vertex

/-- Calculates the area and height of a parallelogram with specific properties -/
def parallelogram_area_and_height (p : Parallelogram) : ℝ × ℝ :=
  sorry

/-- Theorem stating the area and height of a specific parallelogram -/
theorem specific_parallelogram_area_and_height :
  let p : Parallelogram := ⟨150, 10, 18, 2⟩
  let (area, height) := parallelogram_area_and_height p
  area = 36 * Real.sqrt 3 ∧ height = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_parallelogram_area_and_height_l1283_128398


namespace NUMINAMATH_CALUDE_profit_difference_l1283_128322

-- Define the types of statues
inductive StatueType
| Giraffe
| Elephant
| Rhinoceros

-- Define the properties of each statue type
def jade_required (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 120
  | StatueType.Elephant => 240
  | StatueType.Rhinoceros => 180

def original_price (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 150
  | StatueType.Elephant => 350
  | StatueType.Rhinoceros => 250

-- Define the bulk discount
def bulk_discount : ℚ := 0.9

-- Define the total jade available
def total_jade : ℕ := 1920

-- Calculate the number of statues that can be made
def num_statues (s : StatueType) : ℕ :=
  total_jade / jade_required s

-- Calculate the revenue for a statue type
def revenue (s : StatueType) : ℚ :=
  if num_statues s > 3 then
    (num_statues s : ℚ) * (original_price s : ℚ) * bulk_discount
  else
    (num_statues s : ℚ) * (original_price s : ℚ)

-- Theorem to prove
theorem profit_difference : 
  revenue StatueType.Elephant - revenue StatueType.Rhinoceros = 270 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_l1283_128322


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1283_128384

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are parallel, then m = -3 -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (1 + m, 1 - m)
  parallel a b → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1283_128384


namespace NUMINAMATH_CALUDE_triangle_solution_l1283_128351

/-- Given a triangle with sides a, b, c, angle γ, and circumscribed circle diameter d,
    if a² - b² = 19, γ = 126°52'12", and d = 21.25,
    then a ≈ 10, b ≈ 9, and c ≈ 17 -/
theorem triangle_solution (a b c : ℝ) (γ : Real) (d : ℝ) : 
  a^2 - b^2 = 19 →
  γ = 126 * π / 180 + 52 * π / (180 * 60) + 12 * π / (180 * 60 * 60) →
  d = 21.25 →
  (abs (a - 10) < 0.5 ∧ abs (b - 9) < 0.5 ∧ abs (c - 17) < 0.5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_solution_l1283_128351


namespace NUMINAMATH_CALUDE_coin_toss_recurrence_l1283_128397

/-- The probability of having a group of length k or more in n tosses of a symmetric coin. -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem coin_toss_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_coin_toss_recurrence_l1283_128397


namespace NUMINAMATH_CALUDE_adams_book_purchase_l1283_128347

/-- Represents a bookcase with a given number of shelves and average books per shelf. -/
structure Bookcase where
  shelves : ℕ
  avgBooksPerShelf : ℕ

/-- Calculates the total capacity of a bookcase. -/
def Bookcase.capacity (b : Bookcase) : ℕ := b.shelves * b.avgBooksPerShelf

theorem adams_book_purchase (
  adam_bookcase : Bookcase
  ) (adam_bookcase_shelves : adam_bookcase.shelves = 4)
    (adam_bookcase_avg : adam_bookcase.avgBooksPerShelf = 20)
    (initial_books : ℕ) (initial_books_count : initial_books = 56)
    (books_left_over : ℕ) (books_left_over_count : books_left_over = 2) :
  adam_bookcase.capacity + books_left_over - initial_books = 26 := by
  sorry

end NUMINAMATH_CALUDE_adams_book_purchase_l1283_128347


namespace NUMINAMATH_CALUDE_campground_distance_l1283_128373

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l1283_128373


namespace NUMINAMATH_CALUDE_max_cows_is_correct_l1283_128387

/-- Represents the maximum number of cows a rancher can buy given specific constraints. -/
def max_cows : ℕ :=
  let budget : ℕ := 1300
  let steer_cost : ℕ := 30
  let cow_cost : ℕ := 33
  30

/-- Theorem stating that max_cows is indeed the maximum number of cows the rancher can buy. -/
theorem max_cows_is_correct :
  ∀ s c : ℕ,
  s > 0 →
  c > 0 →
  c > 2 * s →
  s * 30 + c * 33 ≤ 1300 →
  c ≤ max_cows :=
by sorry

#eval max_cows  -- Should output 30

end NUMINAMATH_CALUDE_max_cows_is_correct_l1283_128387


namespace NUMINAMATH_CALUDE_age_ratio_in_3_years_l1283_128301

def franks_current_age : ℕ := 12
def johns_current_age : ℕ := franks_current_age + 15

def franks_age_in_3_years : ℕ := franks_current_age + 3
def johns_age_in_3_years : ℕ := johns_current_age + 3

theorem age_ratio_in_3_years :
  ∃ (k : ℕ), k > 0 ∧ johns_age_in_3_years = k * franks_age_in_3_years ∧
  johns_age_in_3_years / franks_age_in_3_years = 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_in_3_years_l1283_128301


namespace NUMINAMATH_CALUDE_special_right_triangle_sides_l1283_128343

/-- A right triangle with a special inscribed circle -/
structure SpecialRightTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first leg of the triangle -/
  x : ℝ
  /-- The second leg of the triangle -/
  y : ℝ
  /-- The hypotenuse of the triangle -/
  z : ℝ
  /-- The area of the triangle is 2r^2/3 -/
  area_eq : x * y / 2 = 2 * r^2 / 3
  /-- The triangle is right-angled -/
  pythagoras : x^2 + y^2 = z^2
  /-- The circle touches one leg, the extension of the other leg, and the hypotenuse -/
  circle_property : z = 2*r + x - y

/-- The sides of a special right triangle are r, 4r/3, and 5r/3 -/
theorem special_right_triangle_sides (t : SpecialRightTriangle) : 
  t.x = t.r ∧ t.y = 4 * t.r / 3 ∧ t.z = 5 * t.r / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_sides_l1283_128343


namespace NUMINAMATH_CALUDE_factorial_sum_equals_5040_l1283_128333

theorem factorial_sum_equals_5040 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_5040_l1283_128333


namespace NUMINAMATH_CALUDE_abs_z_equals_five_l1283_128307

theorem abs_z_equals_five (z : ℂ) (h : z - 3 = (3 + I) / I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_five_l1283_128307


namespace NUMINAMATH_CALUDE_largest_number_l1283_128393

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

def a : Nat := base_to_decimal [5, 8] 9
def b : Nat := base_to_decimal [1, 0, 3] 5
def c : Nat := base_to_decimal [1, 0, 0, 1] 2

theorem largest_number : a > b ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1283_128393


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1283_128355

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 10}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1283_128355


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1283_128338

theorem expand_and_simplify (x y : ℝ) : (x + 2*y) * (x - 2*y) - y * (3 - 4*y) = x^2 - 3*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1283_128338


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1283_128375

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 6 ↔ -7/2 < x ∧ x < 5/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1283_128375


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1283_128370

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1283_128370


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_with_150_exterior_l1283_128312

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle₁ : ℝ
  base_angle₂ : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 180 - t.vertex_angle

-- Theorem statement
theorem isosceles_triangle_base_angle_with_150_exterior
  (t : IsoscelesTriangle)
  (h : exterior_angle t = 150) :
  t.base_angle₁ = 30 ∨ t.base_angle₁ = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_with_150_exterior_l1283_128312


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1283_128302

theorem negation_of_proposition (P : ℝ → Prop) : 
  (∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ¬(∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1283_128302


namespace NUMINAMATH_CALUDE_parabola_and_line_problem_l1283_128371

-- Define the parabola and directrix
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := y = x
def l₂ (x y : ℝ) : Prop := y = -x

-- Define the point E
def E : ℝ × ℝ := (4, 1)

-- Define the circle N
def circle_N (center : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = r^2

-- Theorem statement
theorem parabola_and_line_problem :
  -- Part 1: The coordinates of N are (2, 0)
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x y : ℝ), parabola p x y → 
    (∀ (x' : ℝ), directrix p x' → 
      ((x - x')^2 + y^2 = (x - 2)^2 + y^2))) ∧
  -- Part 2: No line l exists satisfying all conditions
  ¬∃ (m b : ℝ), 
    -- Define line l: y = mx + b
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      -- l intersects l₁ and l₂
      (y₁ = m*x₁ + b ∧ l₁ x₁ y₁) ∧
      (y₂ = m*x₂ + b ∧ l₂ x₂ y₂) ∧
      -- Midpoint of intersection points is E
      ((x₁ + x₂)/2 = E.1 ∧ (y₁ + y₂)/2 = E.2) ∧
      -- Chord length on circle N is 2
      (∃ (r : ℝ), 
        circle_N (2, 0) r 2 2 ∧ 
        circle_N (2, 0) r 2 (-2) ∧
        ∃ (x₃ y₃ x₄ y₄ : ℝ),
          y₃ = m*x₃ + b ∧ y₄ = m*x₄ + b ∧
          circle_N (2, 0) r x₃ y₃ ∧
          circle_N (2, 0) r x₄ y₄ ∧
          (x₃ - x₄)^2 + (y₃ - y₄)^2 = 4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_problem_l1283_128371


namespace NUMINAMATH_CALUDE_exact_calculation_equals_rounded_l1283_128339

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  if x % 100 < 50 then x - (x % 100) else x + (100 - (x % 100))

theorem exact_calculation_equals_rounded : round_to_nearest_hundred (63 + 48 - 21) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exact_calculation_equals_rounded_l1283_128339


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1283_128316

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum (1/3) (1/3) n = 26/81 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1283_128316


namespace NUMINAMATH_CALUDE_sally_picked_seven_lemons_l1283_128365

/-- The number of lemons Mary picked -/
def mary_lemons : ℕ := 9

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons Sally picked -/
def sally_lemons : ℕ := total_lemons - mary_lemons

theorem sally_picked_seven_lemons : sally_lemons = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_seven_lemons_l1283_128365


namespace NUMINAMATH_CALUDE_unique_number_property_l1283_128372

theorem unique_number_property : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (n.div 100 + n.mod 100 / 10 + n.mod 10 = 328 - n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l1283_128372


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l1283_128349

theorem students_behind_yoongi (total_students : ℕ) (students_in_front : ℕ) : 
  total_students = 20 → students_in_front = 11 → total_students - (students_in_front + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l1283_128349


namespace NUMINAMATH_CALUDE_rachels_mystery_book_shelves_l1283_128325

theorem rachels_mystery_book_shelves 
  (books_per_shelf : ℕ) 
  (picture_book_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : books_per_shelf = 9)
  (h2 : picture_book_shelves = 2)
  (h3 : total_books = 72) :
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 := by
  sorry

end NUMINAMATH_CALUDE_rachels_mystery_book_shelves_l1283_128325


namespace NUMINAMATH_CALUDE_line_rotation_theorem_l1283_128305

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line counterclockwise around a point --/
def rotateLine (l : Line) (θ : ℝ) (p : ℝ × ℝ) : Line :=
  sorry

/-- Finds the intersection of a line with the x-axis --/
def xAxisIntersection (l : Line) : ℝ × ℝ :=
  sorry

theorem line_rotation_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -4 →
  let p := xAxisIntersection l
  let l' := rotateLine l (π/4) p
  l'.a = 3 ∧ l'.b = 1 ∧ l'.c = -6 :=
sorry

end NUMINAMATH_CALUDE_line_rotation_theorem_l1283_128305


namespace NUMINAMATH_CALUDE_shenzhen_revenue_precision_l1283_128336

/-- Represents a large monetary amount in yuan -/
structure LargeAmount where
  value : ℝ
  unit : String

/-- Defines the precision of a number -/
inductive Precision
  | HundredBillion
  | TenBillion
  | Billion
  | HundredMillion
  | TenMillion
  | Million

/-- Returns the precision of a given LargeAmount -/
def getPrecision (amount : LargeAmount) : Precision :=
  sorry

theorem shenzhen_revenue_precision :
  let revenue : LargeAmount := { value := 21.658, unit := "billion yuan" }
  getPrecision revenue = Precision.HundredMillion := by sorry

end NUMINAMATH_CALUDE_shenzhen_revenue_precision_l1283_128336


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1283_128318

-- Define complex numbers
variable (a b c d e f : ℝ)

-- Define the theorem
theorem complex_sum_problem :
  b = 4 →
  e = -2*a - c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 5*Complex.I →
  d + 2*f = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1283_128318


namespace NUMINAMATH_CALUDE_trajectory_of_B_l1283_128366

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the ellipse equation
def satisfiesEllipseEquation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Main theorem
theorem trajectory_of_B (ABC : Triangle) :
  ABC.A = (-1, 0) →
  ABC.C = (1, 0) →
  isArithmeticSequence (dist ABC.B ABC.C) (dist ABC.C ABC.A) (dist ABC.A ABC.B) →
  ∀ x y, x ≠ 2 ∧ x ≠ -2 →
  ABC.B = (x, y) →
  satisfiesEllipseEquation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_B_l1283_128366


namespace NUMINAMATH_CALUDE_area_between_curves_l1283_128304

/-- The upper function in the integral -/
def f (x : ℝ) : ℝ := 2 * x - x^2 + 3

/-- The lower function in the integral -/
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

/-- The theorem stating that the area between the curves is 9 -/
theorem area_between_curves : ∫ x in (0)..(3), (f x - g x) = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l1283_128304


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1283_128340

/-- Given a book with 400 pages, prove that after reading 20% of it, 320 pages are left to read. -/
theorem pages_left_to_read (total_pages : ℕ) (percentage_read : ℚ) 
  (h1 : total_pages = 400)
  (h2 : percentage_read = 20 / 100) : 
  total_pages - (total_pages * percentage_read).floor = 320 := by
  sorry

#eval (400 : ℕ) - ((400 : ℕ) * (20 / 100 : ℚ)).floor

end NUMINAMATH_CALUDE_pages_left_to_read_l1283_128340


namespace NUMINAMATH_CALUDE_special_triangle_sum_l1283_128350

/-- A triangle with an incircle that evenly trisects a median -/
structure SpecialTriangle where
  -- The side length BC
  a : ℝ
  -- The area of the triangle
  area : ℝ
  -- k and p, where area = k√p
  k : ℕ
  p : ℕ
  -- Conditions
  side_length : a = 24
  area_form : area = k * Real.sqrt p
  p_not_square_divisible : ∀ (q : ℕ), Prime q → ¬(q^2 ∣ p)
  incircle_trisects_median : True  -- This condition is implicit in the structure

/-- The sum of k and p for the special triangle is 51 -/
theorem special_triangle_sum (t : SpecialTriangle) : t.k + t.p = 51 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_l1283_128350


namespace NUMINAMATH_CALUDE_rational_absolute_value_equation_l1283_128337

theorem rational_absolute_value_equation (a : ℚ) : 
  |a - 1| = 4 → (a = 5 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_equation_l1283_128337


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1283_128330

open Real
open EuclideanSpace

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define the vectors
variable (MA MB MC : V)

-- State the theorem
theorem vectors_are_coplanar 
  (h_noncollinear : ¬ ∃ (k : ℝ), MA = k • MB)
  (h_MC_def : MC = 5 • MA - 3 • MB) :
  ∃ (a b c : ℝ), a • MA + b • MB + c • MC = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l1283_128330


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l1283_128342

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 1) - m / (1 - x) = 2

-- State the theorem
theorem equation_positive_root_implies_m_eq_neg_one :
  ∃ (x : ℝ), x > 0 ∧ equation x m → m = -1 :=
sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_eq_neg_one_l1283_128342


namespace NUMINAMATH_CALUDE_smallest_m_correct_l1283_128357

/-- The smallest positive value of m for which the equation 10x^2 - mx + 600 = 0 has consecutive integer solutions -/
def smallest_m : ℕ := 170

/-- Predicate to check if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1 ∨ a = b + 1

theorem smallest_m_correct :
  ∀ m : ℕ,
  (∃ x y : ℤ, consecutive x y ∧ 10 * x^2 - m * x + 600 = 0 ∧ 10 * y^2 - m * y + 600 = 0) →
  m ≥ smallest_m :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l1283_128357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1283_128394

/-- The sum of the first n terms of an arithmetic sequence -/
def S (a d : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- Theorem: If the ratio of S_{4n} to S_n is constant for an arithmetic sequence 
    with common difference 5, then the first term is 5/2 -/
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ c : ℚ, S a 5 (4 * n) / S a 5 n = c) →
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1283_128394


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_l1283_128390

theorem opposite_of_negative_one :
  (∀ x : ℤ, x + (-x) = 0) →
  -(-1) = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_l1283_128390


namespace NUMINAMATH_CALUDE_sample_size_is_70_l1283_128380

/-- Represents the ratio of quantities for products A, B, and C -/
structure ProductRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes for products A, B, and C -/
structure SampleSize where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the sample sizes for each product -/
def totalSampleSize (s : SampleSize) : ℕ := s.a + s.b + s.c

/-- Theorem stating that given the product ratio and the sample size of product A, 
    the total sample size is 70 -/
theorem sample_size_is_70 (ratio : ProductRatio) (sample : SampleSize) :
  ratio = ⟨3, 4, 7⟩ → sample.a = 15 → totalSampleSize sample = 70 := by
  sorry

#check sample_size_is_70

end NUMINAMATH_CALUDE_sample_size_is_70_l1283_128380


namespace NUMINAMATH_CALUDE_golden_raisin_cost_l1283_128345

/-- The cost per scoop of natural seedless raisins -/
def natural_cost : ℝ := 3.45

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The cost per scoop of the mixture -/
def mixture_cost : ℝ := 3

/-- The cost per scoop of golden seedless raisins -/
def golden_cost : ℝ := 2.55

theorem golden_raisin_cost :
  (natural_cost * natural_scoops + golden_cost * golden_scoops) / (natural_scoops + golden_scoops) = mixture_cost :=
sorry

end NUMINAMATH_CALUDE_golden_raisin_cost_l1283_128345


namespace NUMINAMATH_CALUDE_largest_k_inequality_l1283_128341

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (∃ k : ℕ+, k = 4 ∧
    (∀ m : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ (m : ℝ) / (a - c)) → m ≤ k) ∧
    (1 / (a - b) + 1 / (b - c) ≥ (k : ℝ) / (a - c))) :=
sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l1283_128341


namespace NUMINAMATH_CALUDE_ellipse_parameters_and_eccentricity_l1283_128300

/-- Given an ellipse and a line passing through its vertex and focus, prove the ellipse's parameters and eccentricity. -/
theorem ellipse_parameters_and_eccentricity 
  (a b : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x - 2*y + 2 = 0 → (x = 0 ∧ y = 1) ∨ (x = -2 ∧ y = 0))) :
  a^2 = 5 ∧ b^2 = 1 ∧ (a^2 - b^2) / a^2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameters_and_eccentricity_l1283_128300


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1283_128359

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l with slope k passing through (-1, 0) -/
def line_l (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point on parabola C -/
def on_parabola_C (x y : ℝ) : Prop := parabola_C x y

/-- Point on line l -/
def on_line_l (k x y : ℝ) : Prop := line_l k x y

/-- Intersection ratio condition -/
def intersection_ratio (y₁ y₂ : ℝ) : Prop := y₁/y₂ + y₂/y₁ = 18

theorem parabola_line_intersection (k : ℝ) 
  (hk : k > 0)
  (hA : ∃ x₁ y₁, on_parabola_C x₁ y₁ ∧ on_line_l k x₁ y₁)
  (hB : ∃ x₂ y₂, on_parabola_C x₂ y₂ ∧ on_line_l k x₂ y₂)
  (hM : ∃ xₘ yₘ, on_parabola_C xₘ yₘ)
  (hN : ∃ xₙ yₙ, on_parabola_C xₙ yₙ)
  (h_ratio : ∀ y₁ y₂, intersection_ratio y₁ y₂) :
  k = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1283_128359


namespace NUMINAMATH_CALUDE_milk_production_l1283_128323

/-- Given that x cows produce y gallons of milk in z days, 
    calculate the amount of milk w cows produce in v days with 10% daily waste. -/
theorem milk_production (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0) :
  let daily_waste : ℝ := 0.1
  let milk_per_cow_per_day : ℝ := y / (z * x)
  let effective_milk_per_cow_per_day : ℝ := milk_per_cow_per_day * (1 - daily_waste)
  effective_milk_per_cow_per_day * w * v = 0.9 * (w * y * v) / (z * x) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l1283_128323


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1283_128377

theorem scientific_notation_equivalence :
  216000 = 2.16 * (10 ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1283_128377


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1283_128344

/-- An arithmetic sequence with positive first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  a₁_pos : 0 < a₁
  d_pos : 0 < d

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem max_value_of_expression (seq : ArithmeticSequence)
    (h1 : seq.nthTerm 1 + seq.nthTerm 2 ≤ 60)
    (h2 : seq.nthTerm 2 + seq.nthTerm 3 ≤ 100) :
    5 * seq.nthTerm 1 + seq.nthTerm 5 ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1283_128344


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l1283_128320

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℚ)
  (h1 : original_intensity = 15 / 100)
  (h2 : new_intensity = 30 / 100)
  (h3 : replacement_fraction = 3 / 2)
  : ∃ added_intensity : ℚ,
    added_intensity = 40 / 100 ∧
    (original_intensity * (1 / (1 + replacement_fraction)) + 
     added_intensity * (replacement_fraction / (1 + replacement_fraction)) = new_intensity) :=
by sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l1283_128320


namespace NUMINAMATH_CALUDE_equation_solution_l1283_128389

theorem equation_solution : 
  ∃ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1283_128389


namespace NUMINAMATH_CALUDE_jasons_military_career_l1283_128324

theorem jasons_military_career (join_age retire_age : ℕ) 
  (chief_to_master_chief_factor : ℚ) (additional_years : ℕ) :
  join_age = 18 →
  retire_age = 46 →
  chief_to_master_chief_factor = 1.25 →
  additional_years = 10 →
  ∃ (years_to_chief : ℕ),
    years_to_chief + (chief_to_master_chief_factor * years_to_chief) + additional_years = retire_age - join_age ∧
    years_to_chief = 8 := by
  sorry

end NUMINAMATH_CALUDE_jasons_military_career_l1283_128324


namespace NUMINAMATH_CALUDE_prime_roots_sum_fraction_l1283_128313

theorem prime_roots_sum_fraction (p q m : ℕ) : 
  Prime p → Prime q → 
  p^2 - 99*p + m = 0 → 
  q^2 - 99*q + m = 0 → 
  (p : ℚ) / q + (q : ℚ) / p = 9413 / 194 := by
  sorry

end NUMINAMATH_CALUDE_prime_roots_sum_fraction_l1283_128313
