import Mathlib

namespace NUMINAMATH_CALUDE_oil_in_engine_l2321_232105

theorem oil_in_engine (oil_per_cylinder : ℕ) (num_cylinders : ℕ) (additional_oil_needed : ℕ) :
  oil_per_cylinder = 8 →
  num_cylinders = 6 →
  additional_oil_needed = 32 →
  oil_per_cylinder * num_cylinders - additional_oil_needed = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_oil_in_engine_l2321_232105


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2321_232138

theorem polynomial_remainder (x : ℝ) : 
  (4*x^3 - 9*x^2 + 12*x - 14) % (2*x - 4) = 6 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2321_232138


namespace NUMINAMATH_CALUDE_cafeteria_fruit_distribution_l2321_232190

/-- The number of students who wanted fruit in the school cafeteria -/
def students_wanting_fruit : ℕ := 21

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 15

/-- The number of extra apples left after distribution -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of students who wanted fruit is 21 -/
theorem cafeteria_fruit_distribution :
  students_wanting_fruit = red_apples + green_apples :=
by
  sorry

#check cafeteria_fruit_distribution

end NUMINAMATH_CALUDE_cafeteria_fruit_distribution_l2321_232190


namespace NUMINAMATH_CALUDE_max_dominoes_after_removal_l2321_232197

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Nat)
  (removed_black : Nat)
  (removed_white : Nat)

/-- Calculates the maximum number of guaranteed dominoes -/
def max_guaranteed_dominoes (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the maximum number of guaranteed dominoes for the given problem -/
theorem max_dominoes_after_removal :
  ∀ (board : Chessboard),
    board.size = 8 ∧
    board.removed = 10 ∧
    board.removed_black > 0 ∧
    board.removed_white > 0 ∧
    board.removed_black + board.removed_white = board.removed →
    max_guaranteed_dominoes board = 23 :=
  sorry

end NUMINAMATH_CALUDE_max_dominoes_after_removal_l2321_232197


namespace NUMINAMATH_CALUDE_integer_sum_and_square_is_twelve_l2321_232125

theorem integer_sum_and_square_is_twelve : ∃ N : ℕ+, (N : ℤ)^2 + (N : ℤ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_and_square_is_twelve_l2321_232125


namespace NUMINAMATH_CALUDE_max_area_of_remaining_rectangle_l2321_232193

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles in the square -/
structure SquareConfiguration where
  sideLength : ℝ
  rect1 : Rectangle
  rect2 : Rectangle
  rectR : Rectangle

/-- The theorem statement -/
theorem max_area_of_remaining_rectangle (config : SquareConfiguration) :
  config.sideLength ≥ 4 →
  config.rect1.width = 2 ∧ config.rect1.height = 4 →
  config.rect2.width = 2 ∧ config.rect2.height = 2 →
  config.rectR.area ≤ config.sideLength^2 - 12 ∧
  (config.sideLength = 4 → config.rectR.area = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_remaining_rectangle_l2321_232193


namespace NUMINAMATH_CALUDE_blocks_added_to_tower_l2321_232166

/-- The number of blocks added to a tower -/
def blocks_added (initial final : ℝ) : ℝ := final - initial

/-- Proof that 65.0 blocks were added to the tower -/
theorem blocks_added_to_tower : blocks_added 35.0 100 = 65.0 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_to_tower_l2321_232166


namespace NUMINAMATH_CALUDE_modulus_of_complex_square_root_l2321_232102

theorem modulus_of_complex_square_root (w : ℂ) (h : w^2 = -48 + 36*I) : 
  Complex.abs w = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_square_root_l2321_232102


namespace NUMINAMATH_CALUDE_problem_statement_l2321_232154

theorem problem_statement (x y : ℝ) (h : |x - 3| + Real.sqrt (y - 2) = 0) : 
  (y - x)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2321_232154


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_volunteer_arrangement_problem_l2321_232121

theorem volunteer_arrangement_count : Nat → Nat → Nat
  | n, k => if k ≤ n then n.factorial / (n - k).factorial else 0

theorem volunteer_arrangement_problem :
  volunteer_arrangement_count 6 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_volunteer_arrangement_problem_l2321_232121


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l2321_232199

/-- The ratio of the new circumference to the new diameter when the radius is increased by 2 units -/
theorem circle_ratio_after_increase (r : ℝ) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l2321_232199


namespace NUMINAMATH_CALUDE_angle_B_in_arithmetic_sequence_triangle_l2321_232159

/-- In a triangle ABC where the interior angles A, B, and C form an arithmetic sequence, 
    the measure of angle B is 60°. -/
theorem angle_B_in_arithmetic_sequence_triangle : 
  ∀ (A B C : ℝ),
  (0 < A) ∧ (A < 180) ∧
  (0 < B) ∧ (B < 180) ∧
  (0 < C) ∧ (C < 180) ∧
  (A + B + C = 180) ∧
  (2 * B = A + C) →
  B = 60 := by
sorry

end NUMINAMATH_CALUDE_angle_B_in_arithmetic_sequence_triangle_l2321_232159


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l2321_232136

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 45689 ≡ n [ZMOD 23] ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l2321_232136


namespace NUMINAMATH_CALUDE_least_positive_k_for_equation_l2321_232147

theorem least_positive_k_for_equation : ∃ (k : ℕ), 
  (k > 0) ∧ 
  (∃ (x : ℤ), x > 0 ∧ x + 6 + 8*k = k*(x + 8)) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (y : ℤ), y > 0 ∧ y + 6 + 8*j = j*(y + 8)) ∧
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_least_positive_k_for_equation_l2321_232147


namespace NUMINAMATH_CALUDE_complex_distance_bounds_l2321_232145

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*Complex.I) = 1) :
  (∃ w : ℂ, Complex.abs (w + 2 - 2*Complex.I) = 1 ∧ Complex.abs (w - 3 - 2*Complex.I) = 6) ∧
  (∃ v : ℂ, Complex.abs (v + 2 - 2*Complex.I) = 1 ∧ Complex.abs (v - 3 - 2*Complex.I) = 4) ∧
  (∀ u : ℂ, Complex.abs (u + 2 - 2*Complex.I) = 1 → 
    Complex.abs (u - 3 - 2*Complex.I) ≤ 6 ∧ Complex.abs (u - 3 - 2*Complex.I) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_bounds_l2321_232145


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2321_232148

theorem last_two_digits_product (A B : Nat) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (10 * A + B) % 5 = 0 → 
  A + B = 16 → 
  A * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2321_232148


namespace NUMINAMATH_CALUDE_triangle_properties_l2321_232103

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a - 2 * t.c * Real.cos t.B = t.c) 
  (h2 : Real.cos t.B = 1/3) 
  (h3 : t.c = 3) 
  (h4 : 0 < t.A ∧ t.A < Real.pi/2) 
  (h5 : 0 < t.B ∧ t.B < Real.pi/2) 
  (h6 : 0 < t.C ∧ t.C < Real.pi/2) :
  t.b = 2 * Real.sqrt 6 ∧ 
  1/2 < Real.sin t.C ∧ Real.sin t.C < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2321_232103


namespace NUMINAMATH_CALUDE_solve_for_q_l2321_232152

theorem solve_for_q (n m q : ℚ) 
  (eq1 : (3 : ℚ) / 4 = n / 88)
  (eq2 : (3 : ℚ) / 4 = (m + n) / 100)
  (eq3 : (3 : ℚ) / 4 = (q - m) / 150) :
  q = 121.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2321_232152


namespace NUMINAMATH_CALUDE_axis_of_symmetry_compare_points_range_of_t_max_t_value_l2321_232151

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, parabola t x y → (∀ ε > 0, ∃ y₁ y₂ : ℝ, 
    parabola t (t - ε) y₁ ∧ parabola t (t + ε) y₂ ∧ y₁ = y₂) :=
sorry

-- Theorem 2: Comparing points
theorem compare_points (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → n > m :=
sorry

-- Theorem 3: Range of t
theorem range_of_t (t : ℝ) :
  (∀ x₁ y₁ x₂ y₂ : ℝ, -1 ≤ x₁ → x₁ < 3 → x₂ = 3 →
    parabola t x₁ y₁ → parabola t x₂ y₂ → y₁ ≤ y₂) → t ≤ 1 :=
sorry

-- Theorem 4: Maximum value of t
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ → parabola t (2*t-4) y₂ → y₁ ≥ y₂ → t ≤ t_max :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_compare_points_range_of_t_max_t_value_l2321_232151


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l2321_232181

def f (x : ℝ) := x^2 - x - 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l2321_232181


namespace NUMINAMATH_CALUDE_heavens_brother_erasers_l2321_232195

def total_money : ℕ := 100
def sharpener_count : ℕ := 2
def notebook_count : ℕ := 4
def item_price : ℕ := 5
def eraser_price : ℕ := 4
def highlighter_cost : ℕ := 30

theorem heavens_brother_erasers :
  let heaven_spent := sharpener_count * item_price + notebook_count * item_price
  let brother_money := total_money - heaven_spent
  let eraser_money := brother_money - highlighter_cost
  eraser_money / eraser_price = 10 := by sorry

end NUMINAMATH_CALUDE_heavens_brother_erasers_l2321_232195


namespace NUMINAMATH_CALUDE_percentage_loss_l2321_232189

theorem percentage_loss (cost_price selling_price : ℝ) : 
  cost_price = 750 → 
  selling_price = 675 → 
  (cost_price - selling_price) / cost_price * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_l2321_232189


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l2321_232124

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hc : Odd c) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l2321_232124


namespace NUMINAMATH_CALUDE_max_correct_answers_l2321_232128

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 50 →
  correct_points = 5 →
  incorrect_points = 2 →
  total_score = 150 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points - incorrect * incorrect_points = total_score ∧
    correct ≤ 35 ∧
    ∀ (c : ℕ), c > correct →
      ¬(∃ (i u : ℕ), c + i + u = total_questions ∧
        c * correct_points - i * incorrect_points = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2321_232128


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2321_232146

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 3) = k + 3 * x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2321_232146


namespace NUMINAMATH_CALUDE_playground_area_l2321_232142

theorem playground_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 100 → 
  length = 3 * width → 
  2 * length + 2 * width = perimeter → 
  length * width = 468.75 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l2321_232142


namespace NUMINAMATH_CALUDE_exponent_properties_l2321_232161

theorem exponent_properties (a x y : ℝ) (h1 : a^x = 3) (h2 : a^y = 2) :
  a^(x - y) = 3/2 ∧ a^(2*x + y) = 18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_properties_l2321_232161


namespace NUMINAMATH_CALUDE_binomial_12_3_l2321_232106

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l2321_232106


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2321_232163

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 7 - Real.sqrt 13) * (Real.sqrt 7 + Real.sqrt 13) + 
  (Real.sqrt 3 + 1)^2 - (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 + 
  |-(Real.sqrt 3)| = -3 + 3 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 {a : ℝ} (ha : a < 0) : 
  Real.sqrt (4 - (a + 1/a)^2) - Real.sqrt (4 + (a - 1/a)^2) = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2321_232163


namespace NUMINAMATH_CALUDE_incorrect_calculation_D_l2321_232178

theorem incorrect_calculation_D :
  (∀ x : ℝ, x * 0 = 0) ∧
  (∀ x y : ℝ, y ≠ 0 → x / y = x * (1 / y)) ∧
  (∀ x y : ℝ, x * (-y) = -(x * y)) →
  ¬(1 / 3 / (-1) = 3 * (-1)) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_calculation_D_l2321_232178


namespace NUMINAMATH_CALUDE_intersection_points_distance_l2321_232157

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * a * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (-2 + Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

-- Define point P in polar coordinates
def point_P : ℝ × ℝ := (2, Real.pi)

-- Theorem statement
theorem intersection_points_distance (a : ℝ) :
  a > 0 →
  ∃ (M N : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), M = line_l t₁ ∧ N = line_l t₂) ∧
    (∃ (θ₁ θ₂ : ℝ), curve_C a θ₁ = Real.sqrt ((M.1)^2 + (M.2)^2) ∧
                    curve_C a θ₂ = Real.sqrt ((N.1)^2 + (N.2)^2)) ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 5 * Real.sqrt 2 →
  a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_points_distance_l2321_232157


namespace NUMINAMATH_CALUDE_arman_earnings_l2321_232120

/-- Represents the pay rates and working hours for Arman over two weeks -/
structure PayData :=
  (last_week_rate : ℝ)
  (last_week_hours : ℝ)
  (this_week_rate_increase : ℝ)
  (overtime_multiplier : ℝ)
  (weekend_multiplier : ℝ)
  (night_shift_multiplier : ℝ)
  (monday_hours : ℝ)
  (monday_night_hours : ℝ)
  (tuesday_hours : ℝ)
  (tuesday_night_hours : ℝ)
  (wednesday_hours : ℝ)
  (thursday_hours : ℝ)
  (thursday_night_hours : ℝ)
  (thursday_overtime : ℝ)
  (friday_hours : ℝ)
  (saturday_hours : ℝ)
  (sunday_hours : ℝ)
  (sunday_night_hours : ℝ)

/-- Calculates the total earnings for Arman over two weeks -/
def calculate_earnings (data : PayData) : ℝ :=
  let last_week_earnings := data.last_week_rate * data.last_week_hours
  let this_week_rate := data.last_week_rate + data.this_week_rate_increase
  let this_week_earnings :=
    (data.monday_hours - data.monday_night_hours) * this_week_rate +
    data.monday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - data.tuesday_night_hours) * this_week_rate +
    data.tuesday_night_hours * this_week_rate * data.night_shift_multiplier +
    (data.tuesday_hours - 8) * this_week_rate * data.overtime_multiplier +
    data.wednesday_hours * this_week_rate +
    (data.thursday_hours - data.thursday_night_hours - data.thursday_overtime) * this_week_rate +
    data.thursday_night_hours * this_week_rate * data.night_shift_multiplier +
    data.thursday_overtime * this_week_rate * data.overtime_multiplier +
    data.friday_hours * this_week_rate +
    data.saturday_hours * this_week_rate * data.weekend_multiplier +
    (data.sunday_hours - data.sunday_night_hours) * this_week_rate * data.weekend_multiplier +
    data.sunday_night_hours * this_week_rate * data.weekend_multiplier * data.night_shift_multiplier
  last_week_earnings + this_week_earnings

/-- Theorem stating that Arman's total earnings for the two weeks equal $1055.46 -/
theorem arman_earnings (data : PayData)
  (h1 : data.last_week_rate = 10)
  (h2 : data.last_week_hours = 35)
  (h3 : data.this_week_rate_increase = 0.5)
  (h4 : data.overtime_multiplier = 1.5)
  (h5 : data.weekend_multiplier = 1.7)
  (h6 : data.night_shift_multiplier = 1.3)
  (h7 : data.monday_hours = 8)
  (h8 : data.monday_night_hours = 3)
  (h9 : data.tuesday_hours = 10)
  (h10 : data.tuesday_night_hours = 4)
  (h11 : data.wednesday_hours = 8)
  (h12 : data.thursday_hours = 9)
  (h13 : data.thursday_night_hours = 3)
  (h14 : data.thursday_overtime = 1)
  (h15 : data.friday_hours = 5)
  (h16 : data.saturday_hours = 6)
  (h17 : data.sunday_hours = 4)
  (h18 : data.sunday_night_hours = 2) :
  calculate_earnings data = 1055.46 := by sorry

end NUMINAMATH_CALUDE_arman_earnings_l2321_232120


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2321_232185

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence :
  let a₁ := (3 : ℚ) / 4
  let a₂ := (5 : ℚ) / 4
  let a₃ := (7 : ℚ) / 4
  let d := a₂ - a₁
  arithmeticSequence a₁ d 10 = (21 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2321_232185


namespace NUMINAMATH_CALUDE_problem_solution_l2321_232108

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/5 = 5*y) : x = 625/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2321_232108


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2321_232168

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x}
  S = {x : ℝ | -1 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2321_232168


namespace NUMINAMATH_CALUDE_linearDependence_l2321_232187

/-- Two 2D vectors -/
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

/-- The set of vectors is linearly dependent iff there exist non-zero scalars a and b
    such that a * v1 + b * v2 = 0 -/
def isLinearlyDependent (k : ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * v1 i + b * v2 k i = 0)

theorem linearDependence (k : ℝ) : isLinearlyDependent k ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_linearDependence_l2321_232187


namespace NUMINAMATH_CALUDE_inequality_proof_l2321_232131

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2321_232131


namespace NUMINAMATH_CALUDE_existence_of_small_power_l2321_232100

theorem existence_of_small_power (p e : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : e > 0) :
  ∃ n : ℕ, (1 - p) ^ n < e := by
sorry

end NUMINAMATH_CALUDE_existence_of_small_power_l2321_232100


namespace NUMINAMATH_CALUDE_equation_solution_l2321_232139

theorem equation_solution (a b c d : ℝ) : 
  a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + Real.sqrt (a + b + c - 2*d) →
  d = 1 ∨ d = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2321_232139


namespace NUMINAMATH_CALUDE_park_visitors_l2321_232126

theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l2321_232126


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2321_232127

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  a4_eq : a 4 = -2
  S10_eq : S 10 = 25
  arith_seq : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 14) ∧
  (seq.S 4 = -26 ∧ ∀ n : ℕ, seq.S n ≥ -26) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2321_232127


namespace NUMINAMATH_CALUDE_cheyenne_earnings_l2321_232173

def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

theorem cheyenne_earnings : 
  (total_pots - (cracked_fraction * total_pots).num) * price_per_pot = 1920 := by
  sorry

end NUMINAMATH_CALUDE_cheyenne_earnings_l2321_232173


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2321_232192

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 7 + 2 * Real.sqrt 7 ∧ x₂ = 7 - 2 * Real.sqrt 7 ∧
    x₁^2 - 14*x₁ + 21 = 0 ∧ x₂^2 - 14*x₂ + 21 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2 ∧
    x₁^2 - 3*x₁ + 2 = 0 ∧ x₂^2 - 3*x₂ + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2321_232192


namespace NUMINAMATH_CALUDE_total_money_l2321_232169

/-- Given three people A, B, and C with the following conditions:
  1. A and C together have 200 rupees
  2. B and C together have 360 rupees
  3. C has 60 rupees
  Prove that the total amount they have is 500 rupees -/
theorem total_money (A B C : ℕ) 
  (h1 : A + C = 200) 
  (h2 : B + C = 360) 
  (h3 : C = 60) : 
  A + B + C = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2321_232169


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2321_232174

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2321_232174


namespace NUMINAMATH_CALUDE_prism_min_faces_and_pyramid_min_vertices_l2321_232156

/-- A prism is a three-dimensional geometric shape with two parallel polygonal bases and rectangular faces connecting corresponding edges of the bases. -/
structure Prism where
  bases : ℕ -- number of sides in each base
  height : ℝ
  mk_pos : height > 0

/-- A pyramid is a three-dimensional geometric shape with a polygonal base and triangular faces meeting at a point (apex). -/
structure Pyramid where
  base_sides : ℕ -- number of sides in the base
  height : ℝ
  mk_pos : height > 0

/-- The number of faces in a prism. -/
def Prism.num_faces (p : Prism) : ℕ := p.bases + 2

/-- The number of vertices in a pyramid. -/
def Pyramid.num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

theorem prism_min_faces_and_pyramid_min_vertices :
  (∀ p : Prism, p.num_faces ≥ 5) ∧
  (∀ p : Pyramid, p.num_vertices ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_prism_min_faces_and_pyramid_min_vertices_l2321_232156


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2321_232183

theorem termite_ridden_not_collapsing (total_homes : ℚ) 
  (termite_ridden_ratio : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden_ratio = 1/3 →
  collapsing_ratio = 5/8 →
  termite_ridden_ratio - (termite_ridden_ratio * collapsing_ratio) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l2321_232183


namespace NUMINAMATH_CALUDE_unique_modular_integer_l2321_232140

theorem unique_modular_integer : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_integer_l2321_232140


namespace NUMINAMATH_CALUDE_test_problems_count_l2321_232196

theorem test_problems_count :
  let total_points : ℕ := 110
  let computation_problems : ℕ := 20
  let points_per_computation : ℕ := 3
  let points_per_word : ℕ := 5
  let word_problems : ℕ := (total_points - computation_problems * points_per_computation) / points_per_word
  computation_problems + word_problems = 30 := by
  sorry

end NUMINAMATH_CALUDE_test_problems_count_l2321_232196


namespace NUMINAMATH_CALUDE_not_prime_4n_squared_minus_1_l2321_232164

theorem not_prime_4n_squared_minus_1 (n : ℤ) (h : n ≥ 2) : ¬ Prime (4 * n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_4n_squared_minus_1_l2321_232164


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2321_232110

/-- Proves that if 32% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 11875. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 32)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 11875 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2321_232110


namespace NUMINAMATH_CALUDE_absent_percentage_l2321_232115

def total_students : ℕ := 100
def present_students : ℕ := 86

theorem absent_percentage : 
  (total_students - present_students) * 100 / total_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_absent_percentage_l2321_232115


namespace NUMINAMATH_CALUDE_probability_of_desired_event_l2321_232171

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
(penny : CoinFlip)
(nickel : CoinFlip)
(dime : CoinFlip)
(quarter : CoinFlip)
(fifty_cent : CoinFlip)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired event: at least penny, dime, and 50-cent coin are heads -/
def desired_event (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.fifty_cent = CoinFlip.Heads

/-- The number of outcomes satisfying the desired event -/
def successful_outcomes : ℕ := 4

/-- Theorem stating the probability of the desired event -/
theorem probability_of_desired_event :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_of_desired_event_l2321_232171


namespace NUMINAMATH_CALUDE_cos_105_degrees_l2321_232123

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l2321_232123


namespace NUMINAMATH_CALUDE_cookie_production_cost_l2321_232180

/-- The cost to produce one cookie -/
def production_cost : ℝ := sorry

/-- The selling price of one cookie -/
def selling_price : ℝ := 1.2 * production_cost

/-- The number of cookies sold -/
def cookies_sold : ℕ := 50

/-- The total revenue from selling the cookies -/
def total_revenue : ℝ := 60

theorem cookie_production_cost :
  production_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_cookie_production_cost_l2321_232180


namespace NUMINAMATH_CALUDE_nate_running_distance_l2321_232111

/-- The total distance Nate ran given the length of a football field and additional distance -/
def total_distance (field_length : ℝ) (additional_distance : ℝ) : ℝ :=
  4 * field_length + additional_distance

/-- Theorem stating that Nate's total running distance is 1172 meters -/
theorem nate_running_distance :
  total_distance 168 500 = 1172 := by
  sorry

end NUMINAMATH_CALUDE_nate_running_distance_l2321_232111


namespace NUMINAMATH_CALUDE_unique_interior_point_is_median_intersection_l2321_232141

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The intersection point of the medians of a triangle -/
def MedianIntersection (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem unique_interior_point_is_median_intersection (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → (p = t.A ∨ p = t.B ∨ p = t.C))
  (h2 : ∃! O : LatticePoint, IsInside O t) :
  ∃ O : LatticePoint, IsInside O t ∧ O = MedianIntersection t := by
  sorry

end NUMINAMATH_CALUDE_unique_interior_point_is_median_intersection_l2321_232141


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l2321_232158

theorem sector_area_120_deg_sqrt3_radius (π : ℝ) (h_pi : π = Real.pi) : 
  let angle : ℝ := 120 * π / 180
  let radius : ℝ := Real.sqrt 3
  let area : ℝ := 1/2 * angle * radius^2
  area = π := by
sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l2321_232158


namespace NUMINAMATH_CALUDE_probability_more_ones_than_sixes_l2321_232170

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling five dice -/
def FiveDiceRoll := Vector DieOutcome 5

/-- The total number of possible outcomes when rolling five fair six-sided dice -/
def totalOutcomes : Nat := 7776

/-- The number of outcomes where there are more 1's than 6's -/
def favorableOutcomes : Nat := 2676

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def probabilityMoreOnesThanSixes : Rat := favorableOutcomes / totalOutcomes

theorem probability_more_ones_than_sixes :
  probabilityMoreOnesThanSixes = 2676 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_more_ones_than_sixes_l2321_232170


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2321_232160

def M : ℚ :=
  (1 / (3 * 4 * 5 * 6 * 16 * 17 * 18) +
   1 / (4 * 5 * 6 * 7 * 15 * 16 * 17 * 18) +
   1 / (5 * 6 * 7 * 8 * 14 * 15 * 16 * 17 * 18) +
   1 / (6 * 7 * 8 * 9 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (7 * 8 * 9 * 10 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (8 * 9 * 10 * 11 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18) +
   1 / (9 * 10 * 11 * 12 * 10 * 11 * 12 * 13 * 14 * 15 * 16 * 17 * 18)) * (2 * 17 * 18)

theorem greatest_integer_less_than_M_over_100 : 
  ∀ n : ℤ, n ≤ ⌊M / 100⌋ ↔ n ≤ 145 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2321_232160


namespace NUMINAMATH_CALUDE_sum_three_digit_even_numbers_l2321_232155

/-- The sum of all even natural numbers between 100 and 998 (inclusive) is 247050. -/
theorem sum_three_digit_even_numbers : 
  (Finset.range 450).sum (fun i => 100 + 2 * i) = 247050 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_digit_even_numbers_l2321_232155


namespace NUMINAMATH_CALUDE_pqr_value_exists_l2321_232112

theorem pqr_value_exists :
  ∃ (p q r : ℝ), (p * q) * (q * r) * (r * p) = 16 ∧ p * q * r = 4 :=
by sorry

end NUMINAMATH_CALUDE_pqr_value_exists_l2321_232112


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l2321_232137

theorem wire_length_around_square_field (area : ℝ) (n : ℕ) (wire_length : ℝ) : 
  area = 69696 → n = 15 → wire_length = 15840 → 
  wire_length = n * 4 * Real.sqrt area := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l2321_232137


namespace NUMINAMATH_CALUDE_egyptian_art_pieces_l2321_232186

theorem egyptian_art_pieces (total : ℕ) (asian : ℕ) (egyptian : ℕ) : 
  total = 992 → asian = 465 → egyptian = total - asian → egyptian = 527 := by
sorry

end NUMINAMATH_CALUDE_egyptian_art_pieces_l2321_232186


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2321_232198

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y - 2 : ℂ) + (y - 2 : ℂ) * I = 0 → x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2321_232198


namespace NUMINAMATH_CALUDE_shark_teeth_relationship_hammerhead_shark_teeth_fraction_l2321_232191

/-- The number of teeth a tiger shark has -/
def tiger_shark_teeth : ℕ := 180

/-- The number of teeth a great white shark has -/
def great_white_shark_teeth : ℕ := 420

/-- The fraction of teeth a hammerhead shark has compared to a tiger shark -/
def hammerhead_fraction : ℚ := 1 / 6

/-- Theorem stating the relationship between shark teeth counts -/
theorem shark_teeth_relationship : 
  great_white_shark_teeth = 2 * (tiger_shark_teeth + hammerhead_fraction * tiger_shark_teeth) :=
by sorry

/-- Theorem proving the fraction of teeth a hammerhead shark has compared to a tiger shark -/
theorem hammerhead_shark_teeth_fraction : 
  hammerhead_fraction = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_shark_teeth_relationship_hammerhead_shark_teeth_fraction_l2321_232191


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2321_232143

/-- If (a + i) / (1 - i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + I) / (1 - I) = I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2321_232143


namespace NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l2321_232194

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probabilityDivisibleBy10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1/19 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t = 0 ∨ probabilityDivisibleBy10 t ≥ 1/19 :=
sorry

end NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l2321_232194


namespace NUMINAMATH_CALUDE_polynomial_factorization_exists_l2321_232153

theorem polynomial_factorization_exists :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (p q r s : ℤ),
    ∀ (x : ℤ), x * (x - a) * (x - b) * (x - c) + 1 = (x^2 + p*x + q) * (x^2 + r*x + s) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_exists_l2321_232153


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l2321_232113

theorem sqrt_three_squared : Real.sqrt 3 * Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l2321_232113


namespace NUMINAMATH_CALUDE_power_function_inequality_l2321_232134

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/5)

-- State the theorem
theorem power_function_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l2321_232134


namespace NUMINAMATH_CALUDE_journey_remaining_distance_l2321_232150

/-- Represents a journey with two stopovers and a final destination -/
structure Journey where
  total_distance : ℕ
  first_stopover : ℕ
  second_stopover : ℕ

/-- Calculates the remaining distance to the destination after the second stopover -/
def remaining_distance (j : Journey) : ℕ :=
  j.total_distance - (j.first_stopover + j.second_stopover)

/-- Theorem: For the given journey, the remaining distance is 68 miles -/
theorem journey_remaining_distance :
  let j : Journey := {
    total_distance := 436,
    first_stopover := 132,
    second_stopover := 236
  }
  remaining_distance j = 68 := by
  sorry

end NUMINAMATH_CALUDE_journey_remaining_distance_l2321_232150


namespace NUMINAMATH_CALUDE_triangle_count_48_l2321_232107

/-- The number of distinct, non-degenerate triangles with integer side lengths and perimeter n -/
def count_triangles (n : ℕ) : ℕ :=
  let isosceles := (n - 1) / 2 - n / 4
  let scalene := Nat.choose (n - 1) 2 - 3 * Nat.choose (n / 2) 2
  let total := (scalene - 3 * isosceles) / 6
  if n % 3 = 0 then total - 1 else total

theorem triangle_count_48 :
  ∃ n : ℕ, n > 0 ∧ count_triangles n = 48 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_48_l2321_232107


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2321_232118

def f (x : ℝ) : ℝ := x^2 - 3*x + 4

def g (x m : ℝ) : ℝ := 2*x + m

def h (x t : ℝ) : ℝ := f x - (2*t - 3)*x

def F (x m : ℝ) : ℝ := f x - g x m

theorem quadratic_function_properties :
  (f 0 = 4) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ f x₁ = 2*x₁ ∧ f x₂ = 2*x₂) ∧
  (∃ t : ℝ, t = Real.sqrt 2 / 2 ∧ 
    (∀ x : ℝ, x ∈ Set.Icc 0 1 → h x t ≥ 7/2) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ h x t = 7/2)) ∧
  (∀ m : ℝ, m ∈ Set.Ioo (-9/4) (-2) →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ 
      F x₁ m = 0 ∧ F x₂ m = 0)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2321_232118


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2321_232182

theorem baron_munchausen_claim_false : 
  ¬ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ∃ m : ℕ, 0 ≤ m ∧ m ≤ 99 ∧ ∃ k : ℕ, (n * 100 + m) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_false_l2321_232182


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2321_232172

/-- The maximum y-coordinate of a point on the graph of r = cos 2θ is √2/2 -/
theorem max_y_coordinate_polar_curve : 
  let r : ℝ → ℝ := λ θ => Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ => (r θ) * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l2321_232172


namespace NUMINAMATH_CALUDE_shirts_sold_l2321_232129

theorem shirts_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 49 → remaining = 28 → sold = initial - remaining → sold = 21 := by
sorry

end NUMINAMATH_CALUDE_shirts_sold_l2321_232129


namespace NUMINAMATH_CALUDE_vacation_book_selection_l2321_232133

theorem vacation_book_selection (total_books : ℕ) (books_to_bring : ℕ) (favorite_book : ℕ) :
  total_books = 15 →
  books_to_bring = 3 →
  favorite_book = 1 →
  Nat.choose (total_books - favorite_book) (books_to_bring - favorite_book) = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_book_selection_l2321_232133


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2321_232162

/-- Represents a continued fraction with repeating terms a and b -/
def RepeatingContinuedFraction (a b : ℤ) : ℝ :=
  sorry

/-- The other root of a quadratic equation with integer coefficients -/
def OtherRoot (a b : ℤ) : ℝ :=
  sorry

theorem quadratic_roots_theorem (a b : ℤ) :
  ∃ (p q r : ℤ), 
    (p * (RepeatingContinuedFraction a b)^2 + q * (RepeatingContinuedFraction a b) + r = 0) →
    (OtherRoot a b = -1 / (RepeatingContinuedFraction b a)) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2321_232162


namespace NUMINAMATH_CALUDE_jerry_weekly_spending_jerry_specific_case_l2321_232119

/-- Given Jerry's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem jerry_weekly_spending (lawn_money weed_money : ℕ) (weeks : ℕ) : 
  (lawn_money + weed_money) / weeks = (lawn_money + weed_money) / weeks :=
by sorry

/-- Jerry's specific case -/
theorem jerry_specific_case : 
  (14 + 31) / 9 = 5 :=
by sorry

end NUMINAMATH_CALUDE_jerry_weekly_spending_jerry_specific_case_l2321_232119


namespace NUMINAMATH_CALUDE_dogwood_trees_to_cut_l2321_232101

/-- The number of dogwood trees in the first part of the park -/
def trees_part1 : ℝ := 5.0

/-- The number of dogwood trees in the second part of the park -/
def trees_part2 : ℝ := 4.0

/-- The number of dogwood trees that will be left after the work is done -/
def trees_left : ℝ := 2.0

/-- The number of dogwood trees to be cut down -/
def trees_to_cut : ℝ := trees_part1 + trees_part2 - trees_left

theorem dogwood_trees_to_cut :
  trees_to_cut = 7.0 := by sorry

end NUMINAMATH_CALUDE_dogwood_trees_to_cut_l2321_232101


namespace NUMINAMATH_CALUDE_gcd_multiple_smallest_l2321_232184

/-- Given positive integers m and n with gcd(m,n) = 12, 
    the smallest possible value of gcd(12m,18n) is 72 -/
theorem gcd_multiple_smallest (m n : ℕ+) (h : Nat.gcd m n = 12) :
  ∃ (k : ℕ+), ∀ (x : ℕ+), Nat.gcd (12 * m) (18 * n) ≥ k ∧ 
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 12 ∧ Nat.gcd (12 * m') (18 * n') = k) ∧
  k = 72 := by
  sorry

#check gcd_multiple_smallest

end NUMINAMATH_CALUDE_gcd_multiple_smallest_l2321_232184


namespace NUMINAMATH_CALUDE_count_triples_eq_two_l2321_232122

/-- The number of positive integer triples (x, y, z) satisfying x · y = 6 and y · z = 15 -/
def count_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    t.1 * t.2.1 = 6 ∧ t.2.1 * t.2.2 = 15 ∧
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 4) (Finset.range 16)))).card

theorem count_triples_eq_two : count_triples = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_eq_two_l2321_232122


namespace NUMINAMATH_CALUDE_inequality_proof_l2321_232114

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2321_232114


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l2321_232104

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns a tuple of its coefficients (a, b, c) -/
def quadraticCoefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The quadratic equation 5x^2 + 2x - 1 = 0 -/
def givenEquation : ℝ × ℝ × ℝ := (5, 2, -1)

theorem coefficients_of_given_equation :
  quadraticCoefficients 5 2 (-1) = givenEquation :=
by sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l2321_232104


namespace NUMINAMATH_CALUDE_log_identity_l2321_232109

/-- Given real numbers a and b greater than 1 satisfying lg(a + b) = lg(a) + lg(b),
    prove that lg(a - 1) + lg(b - 1) = 0 and lg(1/a + 1/b) = 0 -/
theorem log_identity (a b : ℝ) (ha : a > 1) (hb : b > 1) 
    (h : Real.log (a + b) = Real.log a + Real.log b) :
  Real.log (a - 1) + Real.log (b - 1) = 0 ∧ Real.log (1/a + 1/b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l2321_232109


namespace NUMINAMATH_CALUDE_polynomial_with_positive_integer_roots_l2321_232165

theorem polynomial_with_positive_integer_roots :
  ∀ (a b c : ℝ),
  (∃ (p q r s : ℕ+),
    (∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + b = (x - p)*(x - q)*(x - r)*(x - s)) ∧
    p + q + r + s = -a ∧
    p*q + p*r + p*s + q*r + q*s + r*s = b ∧
    p*q*r + p*q*s + p*r*s + q*r*s = -c ∧
    p*q*r*s = b) →
  ((a = -21 ∧ b = 112 ∧ c = -204) ∨ (a = -12 ∧ b = 48 ∧ c = -80)) := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_positive_integer_roots_l2321_232165


namespace NUMINAMATH_CALUDE_wheel_probability_l2321_232149

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l2321_232149


namespace NUMINAMATH_CALUDE_only_2020_is_very_good_l2321_232135

/-- Represents a four-digit number YEAR --/
structure Year where
  Y : Fin 10
  E : Fin 10
  A : Fin 10
  R : Fin 10

/-- Checks if a Year is in the 21st century --/
def is_21st_century (year : Year) : Prop :=
  2001 ≤ year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ∧ 
  year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ≤ 2100

/-- The system of linear equations for a given Year --/
def system_has_multiple_solutions (year : Year) : Prop :=
  ∃ (x y z w : ℝ) (x' y' z' w' : ℝ),
    (x ≠ x' ∨ y ≠ y' ∨ z ≠ z' ∨ w ≠ w') ∧
    (year.Y * x + year.E * y + year.A * z + year.R * w = year.Y) ∧
    (year.R * x + year.Y * y + year.E * z + year.A * w = year.E) ∧
    (year.A * x + year.R * y + year.Y * z + year.E * w = year.A) ∧
    (year.E * x + year.A * y + year.R * z + year.Y * w = year.R) ∧
    (year.Y * x' + year.E * y' + year.A * z' + year.R * w' = year.Y) ∧
    (year.R * x' + year.Y * y' + year.E * z' + year.A * w' = year.E) ∧
    (year.A * x' + year.R * y' + year.Y * z' + year.E * w' = year.A) ∧
    (year.E * x' + year.A * y' + year.R * z' + year.Y * w' = year.R)

/-- The main theorem stating that 2020 is the only "very good" year in the 21st century --/
theorem only_2020_is_very_good :
  ∀ (year : Year),
    is_21st_century year ∧ system_has_multiple_solutions year ↔
    year.Y = 2 ∧ year.E = 0 ∧ year.A = 2 ∧ year.R = 0 :=
sorry

end NUMINAMATH_CALUDE_only_2020_is_very_good_l2321_232135


namespace NUMINAMATH_CALUDE_line_plane_parallel_l2321_232130

-- Define the types for lines and planes
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relations
variable (subset : L → P → Prop)  -- line is contained in plane
variable (parallel : L → P → Prop)  -- line is parallel to plane
variable (coplanar : L → L → Prop)  -- two lines are coplanar
variable (parallel_lines : L → L → Prop)  -- two lines are parallel

-- State the theorem
theorem line_plane_parallel (m n : L) (α : P) :
  subset m α → parallel n α → coplanar m n → parallel_lines m n := by sorry

end NUMINAMATH_CALUDE_line_plane_parallel_l2321_232130


namespace NUMINAMATH_CALUDE_no_conclusive_deduction_l2321_232116

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bars, Fins, and Grips
variable (Bar Fin Grip : U → Prop)

-- Define the given conditions
variable (some_bars_not_fins : ∃ x, Bar x ∧ ¬Fin x)
variable (no_fins_are_grips : ∀ x, Fin x → ¬Grip x)

-- Define the statements to be proved
def some_bars_not_grips := ∃ x, Bar x ∧ ¬Grip x
def some_grips_not_bars := ∃ x, Grip x ∧ ¬Bar x
def no_bar_is_grip := ∀ x, Bar x → ¬Grip x
def some_bars_are_grips := ∃ x, Bar x ∧ Grip x

-- Theorem stating that none of the above statements can be conclusively deduced
theorem no_conclusive_deduction :
  ¬(some_bars_not_grips U Bar Grip ∨
     some_grips_not_bars U Grip Bar ∨
     no_bar_is_grip U Bar Grip ∨
     some_bars_are_grips U Bar Grip) :=
sorry

end NUMINAMATH_CALUDE_no_conclusive_deduction_l2321_232116


namespace NUMINAMATH_CALUDE_numbers_with_seven_in_range_l2321_232188

/-- The count of natural numbers from 1 to 800 (inclusive) that contain the digit 7 at least once -/
def count_numbers_with_seven : ℕ := 152

/-- The total count of numbers from 1 to 800 -/
def total_count : ℕ := 800

/-- The count of numbers from 1 to 800 without the digit 7 -/
def count_without_seven : ℕ := 648

theorem numbers_with_seven_in_range :
  count_numbers_with_seven = total_count - count_without_seven :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_seven_in_range_l2321_232188


namespace NUMINAMATH_CALUDE_max_xyz_value_l2321_232117

theorem max_xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x * y + 3 * z = (x + 3 * z) * (y + 3 * z)) :
  x * y * z ≤ 1 / 81 :=
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2321_232117


namespace NUMINAMATH_CALUDE_sequence_general_term_l2321_232175

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = (3^n - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2321_232175


namespace NUMINAMATH_CALUDE_count_multiples_l2321_232179

theorem count_multiples (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 14 ≠ 0) (Finset.range 350)).card = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_l2321_232179


namespace NUMINAMATH_CALUDE_painting_width_l2321_232167

/-- Given a wall and a painting with specific dimensions, prove the width of the painting -/
theorem painting_width
  (wall_height : ℝ)
  (wall_width : ℝ)
  (painting_height : ℝ)
  (painting_area_percentage : ℝ)
  (h1 : wall_height = 5)
  (h2 : wall_width = 10)
  (h3 : painting_height = 2)
  (h4 : painting_area_percentage = 0.16)
  : (wall_height * wall_width * painting_area_percentage) / painting_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_painting_width_l2321_232167


namespace NUMINAMATH_CALUDE_december_24_is_sunday_l2321_232177

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) (thanksgivingDOW : DayOfWeek) : DayOfWeek :=
  sorry

theorem december_24_is_sunday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 24)
  (h3 : dayOfWeek thanksgiving DayOfWeek.Friday = DayOfWeek.Friday) :
  dayOfWeek ⟨12, 24⟩ DayOfWeek.Friday = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_december_24_is_sunday_l2321_232177


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2321_232132

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k + 2) * x₁ + 2 * k - 1 = 0 ∧
    x₂^2 - (k + 2) * x₂ + 2 * k - 1 = 0) ∧
  (3^2 - (k + 2) * 3 + 2 * k - 1 = 0 → k = 2 ∧ 1^2 - (k + 2) * 1 + 2 * k - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2321_232132


namespace NUMINAMATH_CALUDE_smallest_m_value_l2321_232176

def count_quadruplets (m : ℕ) : ℕ :=
  sorry

theorem smallest_m_value :
  ∃ (m : ℕ),
    (count_quadruplets m = 125000) ∧
    (∀ (a b c d : ℕ), (Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                       Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m) →
                      (count_quadruplets m = 125000)) ∧
    (∀ (m' : ℕ), m' < m →
      (count_quadruplets m' ≠ 125000 ∨
       ∃ (a b c d : ℕ), Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 125 ∧
                         Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = m' ∧
                         count_quadruplets m' ≠ 125000)) ∧
    m = 9450000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_value_l2321_232176


namespace NUMINAMATH_CALUDE_divisible_by_six_l2321_232144

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n^2 + 5) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l2321_232144
