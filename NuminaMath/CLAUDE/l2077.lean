import Mathlib

namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_1192_l2077_207753

/-- A function that counts the number of positive four-digit integers less than 5000 
    with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let lower_bound := 1000
  let upper_bound := 4999
  sorry

/-- Theorem stating that the count of positive four-digit integers less than 5000 
    with at least two identical digits is 1192 -/
theorem count_integers_with_repeated_digits_is_1192 : 
  count_integers_with_repeated_digits = 1192 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_1192_l2077_207753


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2077_207744

theorem triangle_angle_measure (D E F : ℝ) : 
  -- DEF is a triangle
  D + E + F = 180 →
  -- Measure of angle E is three times the measure of angle F
  E = 3 * F →
  -- Angle F is 15°
  F = 15 →
  -- Then the measure of angle D is 120°
  D = 120 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2077_207744


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2077_207725

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 8*x + y^2 + 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 2*y + 25 = 0}
  ∃ d : ℝ, d = Real.sqrt 97 - 5 ∧ 
    ∀ p1 ∈ circle1, ∀ p2 ∈ circle2, d ≤ dist p1 p2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2077_207725


namespace NUMINAMATH_CALUDE_solution_difference_l2077_207735

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) 
  (hr : equation r) 
  (hs : equation s) 
  (hdistinct : r ≠ s) 
  (horder : r > s) : 
  r - s = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l2077_207735


namespace NUMINAMATH_CALUDE_circle_equation_l2077_207771

/-- Given points A and B, and a circle whose center lies on a line, prove the equation of the circle. -/
theorem circle_equation (A B C : ℝ × ℝ) (r : ℝ) : 
  A = (1, -1) →
  B = (-1, 1) →
  C.1 + C.2 = 2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  ∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 1)^2 + (y - 1)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2077_207771


namespace NUMINAMATH_CALUDE_piggy_bank_ratio_l2077_207760

theorem piggy_bank_ratio (T A S X Y : ℝ) (hT : T = 450) (hA : A = 30) 
  (hS : S > A) (hX : X > S) (hY : Y > X) (hTotal : A + S + X + Y = T) :
  ∃ (r : ℝ), r = (T - A - X - Y) / A ∧ r = S / A :=
sorry

end NUMINAMATH_CALUDE_piggy_bank_ratio_l2077_207760


namespace NUMINAMATH_CALUDE_train_crossing_time_l2077_207778

/-- Represents the time it takes for a train to cross a tree given its length and the time it takes to pass a platform of known length. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 1200)
  (h2 : platform_length = 1000)
  (h3 : platform_crossing_time = 220) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2077_207778


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l2077_207798

theorem rationalize_and_simplify : 
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l2077_207798


namespace NUMINAMATH_CALUDE_fifth_number_is_one_l2077_207710

def random_table : List (List Nat) := [
  [7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
  [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]
]

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 20

def extract_valid_numbers (lst : List Nat) : List Nat :=
  lst.filter (λ n => is_valid_number n)

def select_numbers (table : List (List Nat)) : List Nat :=
  let flattened := table.join
  let valid_numbers := extract_valid_numbers flattened
  valid_numbers.take 5

theorem fifth_number_is_one :
  (select_numbers random_table).get? 4 = some 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_is_one_l2077_207710


namespace NUMINAMATH_CALUDE_min_draws_for_target_color_l2077_207761

/- Define the number of balls for each color -/
def red_balls : ℕ := 34
def green_balls : ℕ := 25
def yellow_balls : ℕ := 23
def blue_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 10

/- Define the target number of balls of a single color -/
def target : ℕ := 20

/- Define the total number of balls -/
def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

/- Theorem statement -/
theorem min_draws_for_target_color :
  ∃ (n : ℕ), n = 100 ∧
  (∀ (m : ℕ), m < n → 
    ∃ (r g y b w k : ℕ), 
      r ≤ red_balls ∧ 
      g ≤ green_balls ∧ 
      y ≤ yellow_balls ∧ 
      b ≤ blue_balls ∧ 
      w ≤ white_balls ∧ 
      k ≤ black_balls ∧
      r + g + y + b + w + k = m ∧
      r < target ∧ g < target ∧ y < target ∧ b < target ∧ w < target ∧ k < target) ∧
  (∀ (r g y b w k : ℕ),
    r ≤ red_balls →
    g ≤ green_balls →
    y ≤ yellow_balls →
    b ≤ blue_balls →
    w ≤ white_balls →
    k ≤ black_balls →
    r + g + y + b + w + k = n →
    r ≥ target ∨ g ≥ target ∨ y ≥ target ∨ b ≥ target ∨ w ≥ target ∨ k ≥ target) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_target_color_l2077_207761


namespace NUMINAMATH_CALUDE_rectangle_area_bounds_l2077_207709

/-- Represents the reported dimension of a rectangular tile -/
structure ReportedDimension where
  value : ℝ
  min : ℝ := value - 1.0
  max : ℝ := value + 1.0

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedRectangle where
  length : ReportedDimension
  width : ReportedDimension

/-- Calculates the minimum area of a reported rectangle -/
def minArea (rect : ReportedRectangle) : ℝ :=
  rect.length.min * rect.width.min

/-- Calculates the maximum area of a reported rectangle -/
def maxArea (rect : ReportedRectangle) : ℝ :=
  rect.length.max * rect.width.max

theorem rectangle_area_bounds :
  let rect : ReportedRectangle := {
    length := { value := 4 },
    width := { value := 6 }
  }
  minArea rect = 15.0 ∧ maxArea rect = 35.0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_bounds_l2077_207709


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_seventeen_tenths_l2077_207715

theorem at_least_one_greater_than_seventeen_tenths
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a * b * c) :
  max a (max b c) > 17/10 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_seventeen_tenths_l2077_207715


namespace NUMINAMATH_CALUDE_shaded_fraction_of_semicircle_l2077_207787

/-- Given a larger semicircle with diameter 4 and a smaller semicircle removed from it,
    where the two semicircles touch at exactly three points, prove that the fraction
    of the larger semicircle that remains shaded is 1/2. -/
theorem shaded_fraction_of_semicircle (R : ℝ) (r : ℝ) : 
  R = 2 →  -- Radius of larger semicircle
  r^2 + r^2 = (R - r)^2 →  -- Condition for touching at three points
  (π * R^2 / 2 - π * r^2 / 2) / (π * R^2 / 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_semicircle_l2077_207787


namespace NUMINAMATH_CALUDE_function_zero_between_consecutive_integers_l2077_207706

theorem function_zero_between_consecutive_integers :
  ∃ (a b : ℤ), 
    (∀ x ∈ Set.Ioo a b, (Real.log x + x - 3 : ℝ) ≠ 0) ∧
    b = a + 1 ∧
    a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_between_consecutive_integers_l2077_207706


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l2077_207756

theorem nested_subtraction_simplification (x : ℝ) :
  1 - (2 - (3 - (4 - (5 - x)))) = 3 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l2077_207756


namespace NUMINAMATH_CALUDE_linda_original_amount_l2077_207702

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_amount : 
  (lucy_original - transfer_amount = linda_original + transfer_amount) →
  linda_original = 10 := by
sorry

end NUMINAMATH_CALUDE_linda_original_amount_l2077_207702


namespace NUMINAMATH_CALUDE_finger_2004_is_index_l2077_207719

def finger_sequence : ℕ → String
| 0 => "pinky"
| 1 => "ring"
| 2 => "middle"
| 3 => "index"
| 4 => "thumb"
| 5 => "index"
| 6 => "middle"
| 7 => "ring"
| n + 8 => finger_sequence n

theorem finger_2004_is_index : finger_sequence 2003 = "index" := by
  sorry

end NUMINAMATH_CALUDE_finger_2004_is_index_l2077_207719


namespace NUMINAMATH_CALUDE_factor_equality_l2077_207795

theorem factor_equality (x y : ℝ) : 9*x^2 - y^2 - 4*y - 4 = (3*x + y + 2)*(3*x - y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_equality_l2077_207795


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2077_207726

theorem quadratic_roots_sum (a b : ℝ) : 
  a^2 - 4*a - 1 = 0 → b^2 - 4*b - 1 = 0 → 2*a^2 + 3/b + 5*b = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2077_207726


namespace NUMINAMATH_CALUDE_simplify_sqrt_m_squared_n_l2077_207777

theorem simplify_sqrt_m_squared_n
  (m n : ℝ)
  (h1 : m < 0)
  (h2 : m^2 * n ≥ 0) :
  Real.sqrt (m^2 * n) = -m * Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_m_squared_n_l2077_207777


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_ln_positive_l2077_207769

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_ln_positive :
  (¬ ∃ x > 0, Real.log x > 0) ↔ (∀ x > 0, Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_ln_positive_l2077_207769


namespace NUMINAMATH_CALUDE_function_inequality_l2077_207775

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f''(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x > f x)

-- State the theorem to be proved
theorem function_inequality : f (Real.log 2015) > 2015 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2077_207775


namespace NUMINAMATH_CALUDE_quartic_roots_equivalence_l2077_207759

theorem quartic_roots_equivalence (x : ℂ) : 
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔ 
  (x + 1/x = (-1 + Real.sqrt 43)/3 ∨ x + 1/x = (-1 - Real.sqrt 43)/3) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_equivalence_l2077_207759


namespace NUMINAMATH_CALUDE_two_roots_iff_a_eq_twenty_l2077_207788

/-- The quadratic equation in x parametrized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The condition for at least two distinct roots -/
def has_at_least_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- The main theorem -/
theorem two_roots_iff_a_eq_twenty :
  ∀ a : ℝ, has_at_least_two_distinct_roots a ↔ a = 20 := by sorry

end NUMINAMATH_CALUDE_two_roots_iff_a_eq_twenty_l2077_207788


namespace NUMINAMATH_CALUDE_exactly_one_correct_l2077_207757

/-- Represents a geometric statement --/
inductive GeometricStatement
  | complement_acute : GeometricStatement
  | equal_vertical : GeometricStatement
  | unique_parallel : GeometricStatement
  | perpendicular_distance : GeometricStatement
  | corresponding_angles : GeometricStatement

/-- Checks if a geometric statement is correct --/
def is_correct (s : GeometricStatement) : Prop :=
  match s with
  | GeometricStatement.complement_acute => True
  | _ => False

/-- The list of all geometric statements --/
def all_statements : List GeometricStatement :=
  [GeometricStatement.complement_acute,
   GeometricStatement.equal_vertical,
   GeometricStatement.unique_parallel,
   GeometricStatement.perpendicular_distance,
   GeometricStatement.corresponding_angles]

/-- Theorem stating that exactly one statement is correct --/
theorem exactly_one_correct :
  ∃! (s : GeometricStatement), s ∈ all_statements ∧ is_correct s :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_l2077_207757


namespace NUMINAMATH_CALUDE_starters_with_twin_l2077_207741

def total_players : Nat := 12
def num_starters : Nat := 5
def num_twins : Nat := 2

theorem starters_with_twin (total_players num_starters num_twins : Nat) :
  total_players = 12 →
  num_starters = 5 →
  num_twins = 2 →
  (Nat.choose total_players num_starters) - (Nat.choose (total_players - num_twins) num_starters) = 540 := by
  sorry

end NUMINAMATH_CALUDE_starters_with_twin_l2077_207741


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2077_207785

theorem circle_line_intersection_range (r : ℝ) (h_r_pos : r > 0) :
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) ∧
    A ≠ B) ∧
  (∃ m : ℝ, ∀ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) →
    (A.1 + B.1)^2 + (A.2 + B.2)^2 ≥ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  1 < r ∧ r ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2077_207785


namespace NUMINAMATH_CALUDE_count_integers_in_range_l2077_207704

theorem count_integers_in_range : 
  ∃! n : ℕ, n = (Finset.filter (fun x : ℕ => 
    50 < x^2 + 6*x + 9 ∧ x^2 + 6*x + 9 < 100) (Finset.range 100)).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l2077_207704


namespace NUMINAMATH_CALUDE_triangle_side_length_l2077_207708

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Theorem: In triangle ABC, if AB = 2, BC = 5, and the perimeter is even, then AC = 5 -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 5)
  (h3 : ∃ n : ℕ, t.perimeter = 2 * n) :
  t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2077_207708


namespace NUMINAMATH_CALUDE_house_sale_revenue_distribution_l2077_207723

theorem house_sale_revenue_distribution (market_value : ℝ) (selling_price_percentage : ℝ) 
  (num_people : ℕ) (tax_rate : ℝ) (individual_share : ℝ) : 
  market_value = 500000 →
  selling_price_percentage = 1.20 →
  num_people = 4 →
  tax_rate = 0.10 →
  individual_share = (market_value * selling_price_percentage * (1 - tax_rate)) / num_people →
  individual_share = 135000 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_revenue_distribution_l2077_207723


namespace NUMINAMATH_CALUDE_compound_proposition_falsehood_l2077_207781

theorem compound_proposition_falsehood (p q : Prop) : 
  ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_falsehood_l2077_207781


namespace NUMINAMATH_CALUDE_orthogonal_vectors_solution_l2077_207732

theorem orthogonal_vectors_solution :
  ∃! y : ℝ, (2 : ℝ) * (-1 : ℝ) + (-1 : ℝ) * y + (3 : ℝ) * (0 : ℝ) + (1 : ℝ) * (-4 : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_solution_l2077_207732


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l2077_207707

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 101)
  (h2 : finished_problems = 47)
  (h3 : remaining_pages = 6)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l2077_207707


namespace NUMINAMATH_CALUDE_sum_equidistant_terms_l2077_207742

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_equidistant_terms 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a7 : a 7 = 12) : 
  a 2 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_equidistant_terms_l2077_207742


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l2077_207731

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l2077_207731


namespace NUMINAMATH_CALUDE_max_area_folded_rectangle_l2077_207743

/-- Given a rectangle ABCD with perimeter 24 and AB > AD, when folded along its diagonal AC
    such that AB meets DC at point P, the maximum area of triangle ADP is 72√2. -/
theorem max_area_folded_rectangle (AB AD : ℝ) (h1 : AB > AD) (h2 : AB + AD = 12) :
  let x := AB
  let a := (x^2 - 12*x + 72) / x
  let DP := (12*x - 72) / x
  let area := 3 * (12 - x) * ((12*x - 72) / x)
  ∃ (max_area : ℝ), (∀ x, 0 < x → x < 12 → area ≤ max_area) ∧ max_area = 72 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_folded_rectangle_l2077_207743


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2077_207716

def p (x : ℝ) : Prop := (x^2 + 6*x + 8) * Real.sqrt (x + 3) ≥ 0

def q (x : ℝ) : Prop := x = -3

theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2077_207716


namespace NUMINAMATH_CALUDE_vectors_not_basis_l2077_207724

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (v w : ℝ × ℝ) : Prop :=
  ∀ (c : ℝ), v ≠ c • w

/-- Two vectors are linearly dependent if one is a scalar multiple of the other -/
def LinearlyDependent (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v = c • w

theorem vectors_not_basis (e₁ e₂ : ℝ × ℝ) (h : NonCollinear e₁ e₂) :
  LinearlyDependent (e₁ + 3 • e₂) (6 • e₂ + 2 • e₁) :=
sorry

end NUMINAMATH_CALUDE_vectors_not_basis_l2077_207724


namespace NUMINAMATH_CALUDE_unique_cyclic_number_l2077_207722

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def same_digits (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (∃ k, a / 10^k % 10 = d) ↔ (∃ k, b / 10^k % 10 = d)

theorem unique_cyclic_number : 
  ∃! N : ℕ, is_six_digit N ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → 
      is_six_digit (k * N) ∧ 
      same_digits N (k * N) ∧ 
      N ≠ k * N) ∧
    N = 142857 :=
sorry

end NUMINAMATH_CALUDE_unique_cyclic_number_l2077_207722


namespace NUMINAMATH_CALUDE_point_in_first_or_third_quadrant_l2077_207765

/-- A point is in the first or third quadrant if the product of its coordinates is positive -/
theorem point_in_first_or_third_quadrant (x y : ℝ) :
  x * y > 0 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_or_third_quadrant_l2077_207765


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2077_207700

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 7) → (Real.sqrt 7 < b) → (a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2077_207700


namespace NUMINAMATH_CALUDE_solve_system_l2077_207752

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2077_207752


namespace NUMINAMATH_CALUDE_possible_d_values_l2077_207751

theorem possible_d_values : 
  ∀ d : ℤ, (∃ e f : ℤ, ∀ x : ℤ, (x - d) * (x - 12) + 1 = (x + e) * (x + f)) → (d = 22 ∨ d = 26) :=
by sorry

end NUMINAMATH_CALUDE_possible_d_values_l2077_207751


namespace NUMINAMATH_CALUDE_parallelogram_with_right_angle_is_rectangle_l2077_207774

-- Define a parallelogram
structure Parallelogram :=
  (has_parallel_sides : Bool)

-- Define a rectangle
structure Rectangle extends Parallelogram :=
  (has_right_angle : Bool)

-- Theorem statement
theorem parallelogram_with_right_angle_is_rectangle 
  (p : Parallelogram) (h : Bool) : 
  (p.has_parallel_sides ∧ h) ↔ ∃ (r : Rectangle), r.has_right_angle ∧ r.has_parallel_sides = p.has_parallel_sides :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_right_angle_is_rectangle_l2077_207774


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2077_207770

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀*x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆) →
  a₁ + a₃ + a₅ = -364 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l2077_207770


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2077_207745

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 4

def alice_choice : ℕ := num_puppies + num_rabbits

/-- The number of ways Alice, Bob, Charlie, and Dana can buy pets and leave the store satisfied. -/
theorem pet_store_combinations : ℕ := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2077_207745


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l2077_207746

-- Define a line as a type
def Line : Type := ℝ → ℝ → Prop

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define the notion of two lines intersecting at a point
def intersect (l1 l2 : Line) (p : Point) : Prop :=
  l1 p.1 p.2 ∧ l2 p.1 p.2

-- Define vertical angles
def vertical_angles (l1 l2 : Line) (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (i : Point), intersect l1 l2 i ∧
  (p1 ≠ i ∧ p2 ≠ i ∧ p3 ≠ i ∧ p4 ≠ i) ∧
  (l1 p1.1 p1.2 ∧ l1 p3.1 p3.2) ∧
  (l2 p2.1 p2.2 ∧ l2 p4.1 p4.2)

-- Define angle measure
def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line) (p1 p2 p3 p4 : Point) :
  vertical_angles l1 l2 p1 p2 p3 p4 →
  angle_measure p1 i p2 = angle_measure p3 i p4 :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l2077_207746


namespace NUMINAMATH_CALUDE_h_has_one_zero_and_inequality_l2077_207758

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := Real.exp x - 1
noncomputable def h (x : ℝ) := f x - g x

theorem h_has_one_zero_and_inequality :
  (∃! x, h x = 0) ∧
  (g (Real.exp 2 - Real.log 2 - 1) > Real.log (Real.exp 2 - Real.log 2)) ∧
  (Real.log (Real.exp 2 - Real.log 2) > 2 - f (Real.log 2)) := by
  sorry

end NUMINAMATH_CALUDE_h_has_one_zero_and_inequality_l2077_207758


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2077_207749

/-- If a point P(a-1, a+2) lies on the x-axis, then P = (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 1 ∧ P.2 = a + 2 ∧ P.2 = 0) → 
  ∃ P : ℝ × ℝ, P = (-3, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2077_207749


namespace NUMINAMATH_CALUDE_batman_game_cost_batman_game_cost_proof_l2077_207764

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def strategy_cost : ℝ := 9.46

theorem batman_game_cost : ℝ := by
  sorry

theorem batman_game_cost_proof : batman_game_cost = 12.04 := by
  sorry

end NUMINAMATH_CALUDE_batman_game_cost_batman_game_cost_proof_l2077_207764


namespace NUMINAMATH_CALUDE_zero_in_interval_l2077_207773

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f x = 0 :=
by
  have h1 : f (1/4) < 0 := by sorry
  have h2 : f (1/2) > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2077_207773


namespace NUMINAMATH_CALUDE_square_area_ratio_l2077_207793

theorem square_area_ratio : 
  let a : ℝ := 36
  let b : ℝ := 42
  let c : ℝ := 54
  (a^2 + b^2) / c^2 = 255 / 243 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2077_207793


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2077_207705

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) :
  A + B = 130 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2077_207705


namespace NUMINAMATH_CALUDE_percentage_male_worker_ants_l2077_207768

theorem percentage_male_worker_ants (total_ants : ℕ) (female_worker_ants : ℕ) 
  (h1 : total_ants = 110)
  (h2 : female_worker_ants = 44) : 
  (((total_ants / 2 - female_worker_ants : ℚ) / (total_ants / 2)) * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_percentage_male_worker_ants_l2077_207768


namespace NUMINAMATH_CALUDE_problem_solution_l2077_207762

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem problem_solution (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (3*x + 10) = f (3*x + 1))
  (h_f1 : f 1 = 5) :
  f 2007 + f 2008 = 5 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2077_207762


namespace NUMINAMATH_CALUDE_three_color_theorem_l2077_207797

theorem three_color_theorem : ∃ f : ℕ → Fin 3,
  (∀ n : ℕ, ∀ x y : ℕ, 2^n ≤ x ∧ x < 2^(n+1) ∧ 2^n ≤ y ∧ y < 2^(n+1) → f x = f y) ∧
  (∀ x y z : ℕ, f x = f y ∧ f y = f z ∧ x + y = z^2 → x = 2 ∧ y = 2 ∧ z = 2) :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l2077_207797


namespace NUMINAMATH_CALUDE_exists_function_satisfying_properties_l2077_207738

/-- A strictly increasing function from natural numbers to natural numbers -/
def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m < n → f m < f n

/-- The property that f(f(f(n))) = n + 2f(n) for all n -/
def TripleCompositionProperty (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (f (f n)) = n + 2 * (f n)

/-- The main theorem stating the existence of a function satisfying both properties -/
theorem exists_function_satisfying_properties :
  ∃ f : ℕ → ℕ, StrictlyIncreasing f ∧ TripleCompositionProperty f :=
sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_properties_l2077_207738


namespace NUMINAMATH_CALUDE_always_even_l2077_207763

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def change_sign (n : ℕ) (k : ℕ) : ℤ :=
  (sum_to_n n : ℤ) - 2 * k

theorem always_even (n : ℕ) (k : ℕ) (h1 : n = 1995) (h2 : k ≤ n) :
  Even (change_sign n k) := by
  sorry

end NUMINAMATH_CALUDE_always_even_l2077_207763


namespace NUMINAMATH_CALUDE_triangle_properties_l2077_207799

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * (1 + Real.cos t.A) = Real.sqrt 3 * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 1) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = (3 : ℝ) * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2077_207799


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2077_207786

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2077_207786


namespace NUMINAMATH_CALUDE_integral_sin_cos_sin_l2077_207720

open Real

theorem integral_sin_cos_sin (x : ℝ) :
  ∃ C : ℝ, ∫ t, sin t * cos (2*t) * sin (5*t) = 
    (1/24) * sin (6*x) - (1/32) * sin (8*x) - (1/8) * sin (2*x) + (1/16) * sin (4*x) + C :=
by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_sin_l2077_207720


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2077_207739

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement to be proved
theorem perpendicular_transitivity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) (h2 : α ≠ β)
  (h3 : perp a α) (h4 : perp a β) (h5 : perp b β) :
  perp b α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2077_207739


namespace NUMINAMATH_CALUDE_intersection_range_l2077_207750

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = 4*x + m

-- Define symmetry with respect to the line
def symmetric_points (x1 y1 x2 y2 m : ℝ) : Prop :=
  line ((x1 + x2)/2) ((y1 + y2)/2) m

-- Theorem statement
theorem intersection_range (m : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    ellipse x1 y1 ∧ 
    ellipse x2 y2 ∧ 
    line x1 y1 m ∧ 
    line x2 y2 m ∧ 
    symmetric_points x1 y1 x2 y2 m) ↔ 
  -2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2077_207750


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l2077_207776

/-- Geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Arithmetic sequence with the given property -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Sum of first n terms of a sequence -/
def sum_of_terms (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

theorem sequence_sum_theorem (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  3 * a 5 - a 3 * a 7 = 0 →
  b 5 = a 5 →
  sum_of_terms b 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l2077_207776


namespace NUMINAMATH_CALUDE_set_operations_l2077_207780

theorem set_operations (M N P : Set ℕ) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2077_207780


namespace NUMINAMATH_CALUDE_circle_line_slope_range_l2077_207703

/-- Given a circle and a line, if there are at least three distinct points on the circle
    with a specific distance from the line, then the slope of the line is within a certain range. -/
theorem circle_line_slope_range (a b : ℝ) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 4*x - 4*y - 10 = 0
  let line := fun (x y : ℝ) => a*x + b*y = 0
  let k := -a/b  -- slope of the line
  let distance_point_to_line := fun (x y : ℝ) => |a*x + b*y| / Real.sqrt (a^2 + b^2)
  (∃ (p q r : ℝ × ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    circle p.1 p.2 ∧ circle q.1 q.2 ∧ circle r.1 r.2 ∧
    distance_point_to_line p.1 p.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line q.1 q.2 = 2 * Real.sqrt 2 ∧
    distance_point_to_line r.1 r.2 = 2 * Real.sqrt 2) →
  2 - Real.sqrt 3 ≤ k ∧ k ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_slope_range_l2077_207703


namespace NUMINAMATH_CALUDE_golden_retriever_adult_weight_l2077_207783

/-- Represents the weight of a golden retriever at different stages of growth -/
structure DogWeight where
  initial : ℕ  -- Weight at 7 weeks
  week9 : ℕ    -- Weight at 9 weeks
  month3 : ℕ   -- Weight at 3 months
  month5 : ℕ   -- Weight at 5 months
  adult : ℕ    -- Adult weight at 1 year

/-- Calculates the adult weight of a golden retriever based on its growth pattern -/
def calculateAdultWeight (w : DogWeight) : ℕ :=
  w.initial * 2 * 2 * 2 + 30

/-- Theorem stating that the adult weight of the golden retriever is 78 pounds -/
theorem golden_retriever_adult_weight (w : DogWeight) 
  (h1 : w.initial = 6)
  (h2 : w.week9 = w.initial * 2)
  (h3 : w.month3 = w.week9 * 2)
  (h4 : w.month5 = w.month3 * 2)
  (h5 : w.adult = w.month5 + 30) :
  w.adult = 78 := by
  sorry


end NUMINAMATH_CALUDE_golden_retriever_adult_weight_l2077_207783


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l2077_207701

theorem units_digit_of_sum_of_powers : (47^4 + 28^4) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l2077_207701


namespace NUMINAMATH_CALUDE_percentage_problem_l2077_207730

theorem percentage_problem (p : ℝ) : p * 50 / 100 = 200 → p = 400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2077_207730


namespace NUMINAMATH_CALUDE_algebra_test_average_l2077_207713

theorem algebra_test_average (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (total_average : ℚ) (male_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 36 →
  male_students = 8 →
  female_students = 28 →
  total_average = 90 →
  male_average = 83 →
  (total_students : ℚ) * total_average = 
    (male_students : ℚ) * male_average + (female_students : ℚ) * ((3240 - 664 : ℚ) / 28) :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2077_207713


namespace NUMINAMATH_CALUDE_flash_catch_up_distance_l2077_207717

theorem flash_catch_up_distance
  (v a x y : ℝ) -- v: Ace's speed, a: Flash's acceleration, x: Flash's initial speed multiplier, y: initial distance behind
  (hx : x > 1)
  (ha : a > 0) :
  let d := y + x * v * (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  let t := (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  d = y + x * v * t + (1/2) * a * t^2 ∧
  d = v * t :=
by sorry

end NUMINAMATH_CALUDE_flash_catch_up_distance_l2077_207717


namespace NUMINAMATH_CALUDE_prime_pair_sum_cube_difference_l2077_207790

theorem prime_pair_sum_cube_difference (p q : ℕ) : 
  Prime p ∧ Prime q ∧ p + q = (p - q)^3 → (p = 5 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_sum_cube_difference_l2077_207790


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2077_207734

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 ∧ percentage = 50 → final = initial * (1 + percentage / 100) → final = 105 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2077_207734


namespace NUMINAMATH_CALUDE_gumball_machine_total_l2077_207755

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.red = 16 ∧
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue

/-- Calculates the total number of gumballs in the machine -/
def totalGumballs (m : GumballMachine) : ℕ :=
  m.red + m.blue + m.green

/-- Theorem stating that a valid gumball machine has 56 gumballs in total -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : validGumballMachine m) : totalGumballs m = 56 := by
  sorry

end NUMINAMATH_CALUDE_gumball_machine_total_l2077_207755


namespace NUMINAMATH_CALUDE_problem_solution_l2077_207711

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (mul : S → S → S)

axiom mul_def : ∀ (a b : S), mul a (mul b a) = b

theorem problem_solution :
  (∀ (b : S), mul b (mul b b) = b) ∧
  (∀ (a b : S), mul (mul a b) (mul b (mul a b)) = b) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2077_207711


namespace NUMINAMATH_CALUDE_one_ounce_bottle_caps_count_l2077_207729

/-- The number of one-ounce bottle caps in a collection -/
def oneOunceBottleCaps (totalWeight : ℕ) (totalCaps : ℕ) : ℕ :=
  totalWeight * 16

/-- Theorem: The number of one-ounce bottle caps is equal to the total weight in ounces -/
theorem one_ounce_bottle_caps_count 
  (totalWeight : ℕ) 
  (totalCaps : ℕ) 
  (h1 : totalWeight = 18) 
  (h2 : totalCaps = 2016) : 
  oneOunceBottleCaps totalWeight totalCaps = totalWeight * 16 :=
by sorry

end NUMINAMATH_CALUDE_one_ounce_bottle_caps_count_l2077_207729


namespace NUMINAMATH_CALUDE_well_diameter_l2077_207733

theorem well_diameter (depth : ℝ) (volume : ℝ) (diameter : ℝ) : 
  depth = 14 →
  volume = 43.982297150257104 →
  volume = Real.pi * (diameter / 2)^2 * depth →
  diameter = 2 := by
sorry

end NUMINAMATH_CALUDE_well_diameter_l2077_207733


namespace NUMINAMATH_CALUDE_abc_inequality_l2077_207789

theorem abc_inequality (a b c : Real) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : Real.exp a = 9 * a * Real.log 11)
  (eq_b : Real.exp b = 10 * b * Real.log 10)
  (eq_c : Real.exp c = 11 * c * Real.log 9) :
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2077_207789


namespace NUMINAMATH_CALUDE_function_equality_condition_l2077_207779

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem function_equality_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a ∈ Set.Ioi (2/3) ∪ Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_function_equality_condition_l2077_207779


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2077_207792

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2077_207792


namespace NUMINAMATH_CALUDE_geometric_mean_a4_a8_l2077_207727

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_mean_a4_a8 :
  let a := geometric_sequence (1/8) 2
  (a 4 * a 8)^(1/2) = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_mean_a4_a8_l2077_207727


namespace NUMINAMATH_CALUDE_hexagonal_glass_side_length_l2077_207791

/-- A glass with regular hexagonal top and bottom, containing three identical spheres -/
structure HexagonalGlass where
  /-- Side length of the hexagonal bottom -/
  sideLength : ℝ
  /-- Volume of the glass -/
  volume : ℝ
  /-- The glass contains three identical spheres each touching every side -/
  spheresFit : True
  /-- The volume of the glass is 108 cm³ -/
  volumeIs108 : volume = 108

/-- The theorem stating the relationship between the glass volume and side length -/
theorem hexagonal_glass_side_length (g : HexagonalGlass) : 
  g.sideLength = 2 / Real.rpow 3 (1/3) :=
sorry

end NUMINAMATH_CALUDE_hexagonal_glass_side_length_l2077_207791


namespace NUMINAMATH_CALUDE_solution_completeness_l2077_207794

def is_integer (q : ℚ) : Prop := ∃ n : ℤ, q = n

def satisfies_conditions (x y z : ℚ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≤ y ∧ y ≤ z ∧
  is_integer (x + y + z) ∧
  is_integer (1/x + 1/y + 1/z) ∧
  is_integer (x * y * z)

def solution_set : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 3, 6), (2, 4, 4), (3, 3, 3)}

theorem solution_completeness :
  ∀ x y z : ℚ, satisfies_conditions x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_completeness_l2077_207794


namespace NUMINAMATH_CALUDE_russian_chess_championship_games_l2077_207782

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 18 players, 153 games are played -/
theorem russian_chess_championship_games : 
  num_games 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_russian_chess_championship_games_l2077_207782


namespace NUMINAMATH_CALUDE_fraction_simplification_l2077_207784

theorem fraction_simplification (x y : ℝ) (h : x / y = 2 / 5) :
  (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2077_207784


namespace NUMINAMATH_CALUDE_closer_to_one_than_four_closer_to_zero_than_ax_l2077_207736

-- Part 1
theorem closer_to_one_than_four (x : ℝ) :
  |x^2 - 1| < |4 - 1| → x ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

-- Part 2
theorem closer_to_zero_than_ax (x a : ℝ) :
  a > 0 → |x^2 + a| < |(a + 1) * x| →
  (0 < a ∧ a < 1 → x ∈ Set.Ioo (-1 : ℝ) (-a) ∪ Set.Ioo a 1) ∧
  (a = 1 → False) ∧
  (a > 1 → x ∈ Set.Ioo (-a : ℝ) (-1) ∪ Set.Ioo 1 a) :=
sorry

end NUMINAMATH_CALUDE_closer_to_one_than_four_closer_to_zero_than_ax_l2077_207736


namespace NUMINAMATH_CALUDE_m_range_l2077_207728

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → (m - 1 < x ∧ x < m + 1)) → 
  (1 ≤ m ∧ m ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_m_range_l2077_207728


namespace NUMINAMATH_CALUDE_triangle_construction_cases_l2077_207712

/-- A triangle with side lengths a, b, c and angles A, B, C. --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The height of a triangle from vertex A to side BC. --/
def height_A (t : Triangle) : ℝ := sorry

/-- The height of a triangle from vertex C to side AB. --/
def height_C (t : Triangle) : ℝ := sorry

/-- Constructs triangles given side AB, height CC₁, and angle A. --/
def construct_ABC_CC1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height CC₁, and angle C. --/
def construct_ABC_CC1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle A. --/
def construct_ABC_AA1_A (c : ℝ) (h : ℝ) (α : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle B. --/
def construct_ABC_AA1_B (c : ℝ) (h : ℝ) (β : ℝ) : Set Triangle := sorry

/-- Constructs triangles given side AB, height AA₁, and angle C. --/
def construct_ABC_AA1_C (c : ℝ) (h : ℝ) (γ : ℝ) : Set Triangle := sorry

/-- The total number of distinct triangles that can be constructed from all cases. --/
def total_distinct_triangles : ℕ := sorry

theorem triangle_construction_cases :
  ∀ (c h α β γ : ℝ),
    c > 0 → h > 0 → 0 < α < π → 0 < β < π → 0 < γ < π →
    total_distinct_triangles = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_construction_cases_l2077_207712


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2077_207748

theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 0 → (6 / (x - 1) - (x + 5) / (x^2 - x) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2077_207748


namespace NUMINAMATH_CALUDE_field_area_theorem_l2077_207766

/-- Represents a rectangular field with a given length and breadth. -/
structure RectangularField where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular field. -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.length * field.breadth

/-- Theorem: The area of a rectangular field with breadth 60% of its length
    and perimeter 800 m is 37500 square meters. -/
theorem field_area_theorem :
  ∃ (field : RectangularField),
    field.breadth = 0.6 * field.length ∧
    perimeter field = 800 ∧
    area field = 37500 := by
  sorry

end NUMINAMATH_CALUDE_field_area_theorem_l2077_207766


namespace NUMINAMATH_CALUDE_complex_calculation_proof_l2077_207796

theorem complex_calculation_proof :
  let expr1 := (1) - (3^3) * ((-1/3)^2) - 24 * (3/4 - 1/6 + 3/8)
  let expr2 := (2) - (1^100) - (3/4) / (((-2)^2) * ((-1/4)^2) - 1/2)
  (expr1 = -26) ∧ (expr2 = 2) := by
sorry

end NUMINAMATH_CALUDE_complex_calculation_proof_l2077_207796


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l2077_207767

/-- A quadratic function f(x) = x^2 + bx + c with b = 2a-1 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := λ x => x^2 + (2*a - 1)*x + 3

/-- The function is increasing on the interval (1, +∞) -/
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x < f y

theorem quadratic_increasing_implies_a_bound (a : ℝ) :
  IsIncreasingOn (QuadraticFunction a) → a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_bound_l2077_207767


namespace NUMINAMATH_CALUDE_range_of_shifted_f_l2077_207721

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x, f x ∈ Set.Icc 1 2)
variable (h2 : Set.range f = Set.Icc 1 2)

-- State the theorem
theorem range_of_shifted_f :
  Set.range (fun x ↦ f (x + 1)) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_shifted_f_l2077_207721


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_l2077_207747

theorem integral_one_plus_sin : ∫ x in -Real.pi..Real.pi, (1 + Real.sin x) = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_l2077_207747


namespace NUMINAMATH_CALUDE_safari_count_l2077_207772

theorem safari_count (antelopes : ℕ) (h1 : antelopes = 80) : ∃ (rabbits hyenas wild_dogs leopards giraffes lions elephants zebras hippos : ℕ),
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards * 2 = rabbits ∧
  giraffes = antelopes + 15 ∧
  lions = leopards + giraffes ∧
  elephants = 3 * lions ∧
  4 * zebras = 3 * antelopes ∧
  hippos = zebras + zebras / 10 ∧
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants + zebras + hippos = 1334 :=
by
  sorry


end NUMINAMATH_CALUDE_safari_count_l2077_207772


namespace NUMINAMATH_CALUDE_rice_price_calculation_l2077_207718

def initial_amount : ℝ := 500
def wheat_flour_price : ℝ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℝ := 150
def soda_quantity : ℕ := 1
def rice_quantity : ℕ := 2
def remaining_balance : ℝ := 235

theorem rice_price_calculation : 
  ∃ (rice_price : ℝ), 
    initial_amount - 
    (rice_price * rice_quantity + 
     wheat_flour_price * wheat_flour_quantity + 
     soda_price * soda_quantity) = remaining_balance ∧ 
    rice_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_rice_price_calculation_l2077_207718


namespace NUMINAMATH_CALUDE_soft_drink_cost_l2077_207740

/-- The cost of a 12-pack of soft drinks in dollars -/
def pack_cost : ℚ := 299 / 100

/-- The number of cans in a pack -/
def cans_per_pack : ℕ := 12

/-- The cost per can of soft drink -/
def cost_per_can : ℚ := pack_cost / cans_per_pack

/-- Rounding function to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ := round (100 * x) / 100

theorem soft_drink_cost :
  round_to_cent cost_per_can = 25 / 100 :=
sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l2077_207740


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2077_207714

/-- Given a line segment with one endpoint at (1, -3) and midpoint at (3, 5),
    the sum of the coordinates of the other endpoint is 18. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (3 = (x + 1) / 2) →  -- Midpoint x-coordinate condition
    (5 = (y - 3) / 2) →  -- Midpoint y-coordinate condition
    x + y = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2077_207714


namespace NUMINAMATH_CALUDE_three_digit_integer_with_specific_remainders_l2077_207737

theorem three_digit_integer_with_specific_remainders :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
    n % 7 = 3 ∧ n % 8 = 6 ∧ n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integer_with_specific_remainders_l2077_207737


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_proof_l2077_207754

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 := by
  sorry

-- Theorem for part (2)
theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (h1 : f (a + 1) < 1) (h2 : f (b + 1) < 1) :
  f (a * b) / |a| > f (b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_proof_l2077_207754
