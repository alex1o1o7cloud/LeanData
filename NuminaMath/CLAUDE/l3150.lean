import Mathlib

namespace NUMINAMATH_CALUDE_min_shift_sine_graph_l3150_315013

theorem min_shift_sine_graph (φ : ℝ) : 
  (φ > 0 ∧ ∀ x, Real.sin (2*x + 2*φ + π/3) = Real.sin (2*x)) → φ ≥ 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_sine_graph_l3150_315013


namespace NUMINAMATH_CALUDE_dividend_calculation_l3150_315087

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86)
  (hd : d = 52.7)
  (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3150_315087


namespace NUMINAMATH_CALUDE_f_composition_half_l3150_315085

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x|

-- State the theorem
theorem f_composition_half : f (f (1/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l3150_315085


namespace NUMINAMATH_CALUDE_expansion_terms_l3150_315073

-- Define the exponent
def n : ℕ := 2016

-- Define the function that represents the number of terms
def num_terms (n : ℕ) : ℕ :=
  4 * n + 1

-- Theorem statement
theorem expansion_terms : num_terms n = 4033 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_l3150_315073


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l3150_315020

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_36 : sum_of_factors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l3150_315020


namespace NUMINAMATH_CALUDE_cube_sum_geq_product_sum_l3150_315093

theorem cube_sum_geq_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_product_sum_l3150_315093


namespace NUMINAMATH_CALUDE_ball_bounce_height_l3150_315007

/-- Theorem: For a ball that rises with each bounce exactly one-half as high as it had fallen,
    and bounces 4 times, if the total distance traveled is 44.5 meters,
    then the initial height from which the ball was dropped is 9.9 meters. -/
theorem ball_bounce_height (h : ℝ) : 
  (h + 2*h + h + (1/2)*h + (1/4)*h = 44.5) → h = 9.9 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l3150_315007


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3150_315026

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  {x : ℝ | f x = 0} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3150_315026


namespace NUMINAMATH_CALUDE_circle_under_translation_l3150_315035

/-- A parallel translation in a 2D plane. -/
structure ParallelTranslation where
  shift : ℝ × ℝ

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The result of applying a parallel translation to a circle. -/
def translateCircle (c : Circle) (t : ParallelTranslation) : Circle :=
  { center := (c.center.1 + t.shift.1, c.center.2 + t.shift.2),
    radius := c.radius }

/-- Theorem: A circle remains a circle under parallel translation. -/
theorem circle_under_translation (c : Circle) (t : ParallelTranslation) :
  ∃ (c' : Circle), c' = translateCircle c t ∧ c'.radius = c.radius :=
by sorry

end NUMINAMATH_CALUDE_circle_under_translation_l3150_315035


namespace NUMINAMATH_CALUDE_no_real_roots_l3150_315072

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 + 4 * x + (5/4) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3150_315072


namespace NUMINAMATH_CALUDE_piggy_bank_theorem_specific_piggy_bank_case_l3150_315095

/-- Represents a configuration of piggy banks and their keys -/
structure PiggyBankConfig (n : ℕ) where
  keys : Fin n → Fin n
  injective : Function.Injective keys

/-- The probability of opening all remaining piggy banks given n total and k broken -/
def openProbability (n k : ℕ) : ℚ :=
  if k ≤ n then k / n else 0

theorem piggy_bank_theorem (n k : ℕ) (h : k ≤ n) :
  openProbability n k = k / n := by sorry

theorem specific_piggy_bank_case :
  openProbability 30 2 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_piggy_bank_theorem_specific_piggy_bank_case_l3150_315095


namespace NUMINAMATH_CALUDE_triangle_median_properties_l3150_315004

/-- Properties of triangle medians -/
theorem triangle_median_properties
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (P p : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_perimeter : P = a + b + c)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_median_a : 4 * ma^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h_median_b : 4 * mb^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h_median_c : 4 * mc^2 = 2 * a^2 + 2 * b^2 - c^2) :
  (ma + mb ≤ 3/4 * P) ∧ (ma + mb ≥ 3/4 * p) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_properties_l3150_315004


namespace NUMINAMATH_CALUDE_inequality_proof_l3150_315054

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3150_315054


namespace NUMINAMATH_CALUDE_extreme_points_sum_condition_l3150_315064

open Real

noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 + a * log x - (a + 1) * x

noncomputable def F (a x : ℝ) : ℝ := f a x + (a - 1) * x

theorem extreme_points_sum_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x > 0 → F a x ≤ max (F a x₁) (F a x₂)) ∧
    F a x₁ + F a x₂ > -2/exp 1 - 2) →
  0 < a ∧ a < 1/exp 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_sum_condition_l3150_315064


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l3150_315047

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) :
  total_students = male_students + female_students →
  total_students = 700 →
  male_students = 400 →
  female_students = 300 →
  sample_size = 35 →
  (male_students * sample_size) / total_students = 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l3150_315047


namespace NUMINAMATH_CALUDE_percentage_loss_l3150_315099

theorem percentage_loss (cost_price selling_price : ℚ) : 
  cost_price = 2300 →
  selling_price = 1610 →
  (cost_price - selling_price) / cost_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_l3150_315099


namespace NUMINAMATH_CALUDE_triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l3150_315043

open Real

/-- Triangle interior angles in radians -/
structure TriangleAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_eq_pi : A + B + C = π
  all_positive : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_angle_sin_sum_bounds (t : TriangleAngles) :
  -2 < sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ∧
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ≤ 3/2 * Real.sqrt 3 :=
sorry

theorem triangle_angle_sin_sum_equality_condition (t : TriangleAngles) :
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) = 3/2 * Real.sqrt 3 ↔
  t.A = 7*π/18 ∧ t.B = π/9 ∧ t.C = π/9 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l3150_315043


namespace NUMINAMATH_CALUDE_sequence_property_l3150_315011

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (r > s) →
  (s ≥ 2) →
  (a r = a 1) →
  (a s = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l3150_315011


namespace NUMINAMATH_CALUDE_x_value_proof_l3150_315055

theorem x_value_proof (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_x_lt_y : x < y)
  (h_eq1 : Real.sqrt x + Real.sqrt y = 4)
  (h_eq2 : Real.sqrt (x + 2) + Real.sqrt (y + 2) = 5) :
  x = 49 / 36 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3150_315055


namespace NUMINAMATH_CALUDE_black_squares_on_33x33_board_l3150_315006

/-- Represents a checkerboard with alternating colors and black corners -/
structure Checkerboard where
  size : Nat
  has_black_corners : Bool
  is_alternating : Bool

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  sorry

/-- The theorem stating that a 33x33 checkerboard with alternating colors and black corners has 545 black squares -/
theorem black_squares_on_33x33_board :
  ∀ (board : Checkerboard),
    board.size = 33 →
    board.has_black_corners = true →
    board.is_alternating = true →
    count_black_squares board = 545 :=
  sorry

end NUMINAMATH_CALUDE_black_squares_on_33x33_board_l3150_315006


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l3150_315040

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 960) (h2 : Nat.gcd a c = 324) :
  ∃ (d : ℕ), d = Nat.gcd b c ∧ d = 12 ∧ ∀ (e : ℕ), e = Nat.gcd b c → e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l3150_315040


namespace NUMINAMATH_CALUDE_power_expression_equality_l3150_315042

theorem power_expression_equality (c d : ℝ) 
  (h1 : (80 : ℝ) ^ c = 4)
  (h2 : (80 : ℝ) ^ d = 5) :
  (16 : ℝ) ^ ((1 - c - d) / (2 * (1 - d))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_equality_l3150_315042


namespace NUMINAMATH_CALUDE_xy_squared_l3150_315075

theorem xy_squared (x y : ℝ) (h1 : x + y = 20) (h2 : 2*x + y = 27) : (x + y)^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_l3150_315075


namespace NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_figure_l3150_315023

/-- A geometric figure with six angles that form a quadrilateral -/
structure QuadrilateralFigure where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  F : ℝ
  G : ℝ

/-- The sum of angles in a quadrilateral is 360° -/
theorem sum_of_angles_in_quadrilateral_figure (q : QuadrilateralFigure) :
  q.A + q.B + q.C + q.D + q.F + q.G = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_figure_l3150_315023


namespace NUMINAMATH_CALUDE_lexis_cement_is_10_l3150_315094

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_cement : ℝ := 15.1 - 5.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexis_cement_is_10 : lexis_cement = 10 := by
  sorry

end NUMINAMATH_CALUDE_lexis_cement_is_10_l3150_315094


namespace NUMINAMATH_CALUDE_harolds_money_l3150_315058

theorem harolds_money (x : ℚ) : 
  (x / 2 + 5) +  -- Ticket and candies
  ((x / 2 - 5) / 2 + 10) +  -- Newspaper
  (((x / 2 - 5) / 2 - 10) / 2) +  -- Bus fare
  15 +  -- Beggar
  5  -- Remaining money
  = x  -- Total initial money
  → x = 210 := by sorry

end NUMINAMATH_CALUDE_harolds_money_l3150_315058


namespace NUMINAMATH_CALUDE_star_calculation_l3150_315044

-- Define the * operation
def star (a b : ℚ) : ℚ := (a + 2*b) / 3

-- State the theorem
theorem star_calculation : star (star 4 6) 9 = 70 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3150_315044


namespace NUMINAMATH_CALUDE_tan_sum_equation_l3150_315029

theorem tan_sum_equation : ∀ (x y : Real),
  x + y = 60 * π / 180 →
  Real.tan (60 * π / 180) = Real.sqrt 3 →
  Real.tan x + Real.tan y + Real.sqrt 3 * Real.tan x * Real.tan y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equation_l3150_315029


namespace NUMINAMATH_CALUDE_perfect_squares_difference_four_digit_sqrt_difference_l3150_315016

theorem perfect_squares_difference (m n p : ℕ) 
  (h1 : m > n) 
  (h2 : Real.sqrt m - Real.sqrt n = p) : 
  ∃ (a b : ℕ), m = a^2 ∧ n = b^2 := by
  sorry

theorem four_digit_sqrt_difference : 
  ∃! (abcd : ℕ), 
    1000 ≤ abcd ∧ abcd < 10000 ∧
    ∃ (a b c d : ℕ),
      abcd = 1000 * a + 100 * b + 10 * c + d ∧
      100 * a + 10 * c + d < abcd ∧
      Real.sqrt (abcd) - Real.sqrt (100 * a + 10 * c + d) = 11 * b := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_four_digit_sqrt_difference_l3150_315016


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l3150_315060

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 6 = 1 ∧ a > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a F₁.1 F₁.2 ∧ hyperbola a F₂.1 F₂.2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola a A.1 A.2 ∧ hyperbola a B.1 B.2

-- Define the distance condition
def distance_condition (A F₁ : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = 2 * a

-- Define the angle condition
def angle_condition (F₁ A F₂ : ℝ × ℝ) : Prop :=
  Real.arccos (
    ((F₁.1 - A.1) * (F₂.1 - A.1) + (F₁.2 - A.2) * (F₂.2 - A.2)) /
    (Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) * Real.sqrt ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2))
  ) = 2 * Real.pi / 3

-- State the theorem
theorem hyperbola_triangle_area
  (a : ℝ)
  (F₁ F₂ A B : ℝ × ℝ) :
  foci F₁ F₂ a →
  intersection_points A B a →
  distance_condition A F₁ a →
  angle_condition F₁ A F₂ →
  Real.sqrt 3 * (Real.sqrt ((F₁.1 - B.1)^2 + (F₁.2 - B.2)^2) * Real.sqrt ((F₂.1 - B.1)^2 + (F₂.2 - B.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) / 4 = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l3150_315060


namespace NUMINAMATH_CALUDE_two_real_roots_for_radical_equation_l3150_315017

-- Define the function f(x) derived from the original equation
def f (a b c x : ℝ) : ℝ :=
  3 * x^2 - 2 * (a + b + c) * x - (a^2 + b^2 + c^2) + 2 * (a * b + b * c + c * a)

-- Main theorem statement
theorem two_real_roots_for_radical_equation (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, f a b c x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_for_radical_equation_l3150_315017


namespace NUMINAMATH_CALUDE_ways_to_put_five_balls_three_boxes_l3150_315062

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem ways_to_put_five_balls_three_boxes : ways_to_put_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_put_five_balls_three_boxes_l3150_315062


namespace NUMINAMATH_CALUDE_false_statements_exist_l3150_315045

theorem false_statements_exist : ∃ (a b c d : ℝ),
  (a > b ∧ c ≠ 0 ∧ a * c ≤ b * c) ∧
  (a > b ∧ b > 0 ∧ c > d ∧ a * c ≤ b * d) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end NUMINAMATH_CALUDE_false_statements_exist_l3150_315045


namespace NUMINAMATH_CALUDE_unrestricted_x_l3150_315084

theorem unrestricted_x (x y z w : ℤ) 
  (h1 : (x + 2) / (y - 1) < -(z + 3) / (w - 2))
  (h2 : (y - 1) * (w - 2) ≠ 0) :
  ∃ (x_pos x_neg x_zero : ℤ), 
    (x_pos > 0 ∧ (x_pos + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_neg < 0 ∧ (x_neg + 2) / (y - 1) < -(z + 3) / (w - 2)) ∧
    (x_zero = 0 ∧ (x_zero + 2) / (y - 1) < -(z + 3) / (w - 2)) :=
by sorry

end NUMINAMATH_CALUDE_unrestricted_x_l3150_315084


namespace NUMINAMATH_CALUDE_card_arrangement_unique_l3150_315049

def CardArrangement (arrangement : List Nat) : Prop :=
  arrangement.length = 9 ∧
  arrangement.toFinset = Finset.range 9 ∧
  ∀ i, i + 2 < arrangement.length →
    ¬(arrangement[i]! < arrangement[i+1]! ∧ arrangement[i+1]! < arrangement[i+2]!) ∧
    ¬(arrangement[i]! > arrangement[i+1]! ∧ arrangement[i+1]! > arrangement[i+2]!)

theorem card_arrangement_unique :
  ∀ arrangement : List Nat,
    CardArrangement arrangement →
    arrangement[3]! = 5 ∧
    arrangement[5]! = 2 ∧
    arrangement[8]! = 9 :=
by sorry

end NUMINAMATH_CALUDE_card_arrangement_unique_l3150_315049


namespace NUMINAMATH_CALUDE_intersection_constraint_l3150_315034

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_constraint_l3150_315034


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l3150_315022

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 250^2 + 360^2) (129^2 + 249^2 + 361^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l3150_315022


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3150_315028

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, n > 0 → a n = a 1 * q^(n-1) ∧ S n = (a 1 * (1 - q^n)) / (1 - q)

/-- The theorem stating that for a geometric sequence with q = 2 and S_4 = 60, a_2 = 8 -/
theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a 2 S) 
  (h_sum : S 4 = 60) : 
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3150_315028


namespace NUMINAMATH_CALUDE_oranges_per_box_l3150_315015

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3150_315015


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l3150_315039

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the relationship between f(2^x) and f(3^x) -/
theorem quadratic_function_inequality (f : QuadraticFunction) :
  ∀ x : ℝ, f.a * (3^x)^2 + f.b * (3^x) + f.c > f.a * (2^x)^2 + f.b * (2^x) + f.c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l3150_315039


namespace NUMINAMATH_CALUDE_add_base6_example_l3150_315008

/-- Represents a number in base 6 --/
def Base6 : Type := Fin 6 → ℕ

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : Base6 := sorry

/-- Converts a base 6 number to its natural number representation --/
def fromBase6 (b : Base6) : ℕ := sorry

/-- Adds two base 6 numbers --/
def addBase6 (a b : Base6) : Base6 := sorry

/-- The number 5 in base 6 --/
def five_base6 : Base6 := toBase6 5

/-- The number 23 in base 6 --/
def twentythree_base6 : Base6 := toBase6 23

/-- The number 32 in base 6 --/
def thirtytwo_base6 : Base6 := toBase6 32

theorem add_base6_example : addBase6 five_base6 twentythree_base6 = thirtytwo_base6 := by
  sorry

end NUMINAMATH_CALUDE_add_base6_example_l3150_315008


namespace NUMINAMATH_CALUDE_prob_different_colors_value_l3150_315092

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  
  let prob_not_blue : ℚ := 1 - prob_blue
  let prob_not_red : ℚ := 1 - prob_red
  let prob_not_yellow : ℚ := 1 - prob_yellow
  let prob_not_green : ℚ := 1 - prob_green
  
  prob_blue * prob_not_blue +
  prob_red * prob_not_red +
  prob_yellow * prob_not_yellow +
  prob_green * prob_not_green

theorem prob_different_colors_value :
  prob_different_colors = 119 / 162 :=
sorry

end NUMINAMATH_CALUDE_prob_different_colors_value_l3150_315092


namespace NUMINAMATH_CALUDE_triangle_properties_l3150_315052

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 →
  a = Real.sqrt 3 ∧
  (a * b * Real.sin C / 2 = Real.sqrt 3 / 2 → a + b + c = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3150_315052


namespace NUMINAMATH_CALUDE_honda_red_percentage_l3150_315065

theorem honda_red_percentage (total_cars : ℕ) (honda_cars : ℕ) 
  (total_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  total_red_percentage = 60 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (honda_cars : ℚ) * (90 / 100) + 
    (total_cars - honda_cars : ℚ) * non_honda_red_percentage = 
    (total_cars : ℚ) * total_red_percentage :=
by sorry

end NUMINAMATH_CALUDE_honda_red_percentage_l3150_315065


namespace NUMINAMATH_CALUDE_lola_poptarts_count_l3150_315031

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

theorem lola_poptarts_count :
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies = total_pastries :=
by sorry

end NUMINAMATH_CALUDE_lola_poptarts_count_l3150_315031


namespace NUMINAMATH_CALUDE_problem1_l3150_315030

theorem problem1 (x y : ℝ) (h : y ≠ 0) :
  ((x + 3 * y) * (x - 3 * y) - x^2) / (9 * y) = -y := by sorry

end NUMINAMATH_CALUDE_problem1_l3150_315030


namespace NUMINAMATH_CALUDE_sequence_range_l3150_315025

/-- Given an infinite sequence {a_n} satisfying the recurrence relation
    a_{n+1} = p * a_n + 1 / a_n for n ∈ ℕ*, where p is a positive real number,
    a_1 = 2, and {a_n} is monotonically decreasing, prove that p ∈ (1/2, 3/4). -/
theorem sequence_range (p : ℝ) (a : ℕ+ → ℝ) 
  (h_pos : p > 0)
  (h_rec : ∀ n : ℕ+, a (n + 1) = p * a n + 1 / a n)
  (h_init : a 1 = 2)
  (h_decr : ∀ n : ℕ+, a (n + 1) ≤ a n) :
  p > 1/2 ∧ p < 3/4 := by
sorry


end NUMINAMATH_CALUDE_sequence_range_l3150_315025


namespace NUMINAMATH_CALUDE_folded_square_FG_length_l3150_315077

/-- A folded square sheet of paper with side length 1 -/
structure FoldedSquare where
  /-- The point where corners B and D meet after folding -/
  E : ℝ × ℝ
  /-- The point F on side AB -/
  F : ℝ × ℝ
  /-- The point G on side AD -/
  G : ℝ × ℝ
  /-- E lies on the diagonal AC -/
  E_on_diagonal : E.1 = E.2
  /-- F is on side AB -/
  F_on_AB : F.2 = 0 ∧ 0 ≤ F.1 ∧ F.1 ≤ 1
  /-- G is on side AD -/
  G_on_AD : G.1 = 0 ∧ 0 ≤ G.2 ∧ G.2 ≤ 1

/-- The theorem stating that the length of FG in a folded unit square is 2√2 - 2 -/
theorem folded_square_FG_length (s : FoldedSquare) : 
  Real.sqrt ((s.F.1 - s.G.1)^2 + (s.F.2 - s.G.2)^2) = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_square_FG_length_l3150_315077


namespace NUMINAMATH_CALUDE_circle_properties_l3150_315091

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- State the theorem
theorem circle_properties :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- The center is on the negative half of the x-axis
  ∃ (a : ℝ), a < 0 ∧ circle_equation a 0 ∧
  -- The radius is 2
  ∀ (x y : ℝ), circle_equation x y → (x + 2)^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l3150_315091


namespace NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l3150_315080

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ (∀ x > 0, ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x > 0, x^2 > 0) ↔ (∀ x > 0, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l3150_315080


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l3150_315076

/-- Given Jerry's breakfast composition and total calories, prove the calories per pancake. -/
theorem jerrys_breakfast_calories (pancakes : ℕ) (bacon_strips : ℕ) (bacon_calories : ℕ) 
  (cereal_calories : ℕ) (total_calories : ℕ) (calories_per_pancake : ℕ) :
  pancakes = 6 →
  bacon_strips = 2 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  total_calories = pancakes * calories_per_pancake + bacon_strips * bacon_calories + cereal_calories →
  calories_per_pancake = 120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l3150_315076


namespace NUMINAMATH_CALUDE_white_area_is_40_l3150_315079

/-- Represents a rectangular bar with given width and height -/
structure Bar where
  width : ℕ
  height : ℕ

/-- Represents a letter composed of rectangular bars -/
structure Letter where
  bars : List Bar

def sign_width : ℕ := 18
def sign_height : ℕ := 6

def letter_F : Letter := ⟨[{width := 4, height := 1}, {width := 4, height := 1}, {width := 1, height := 6}]⟩
def letter_O : Letter := ⟨[{width := 1, height := 6}, {width := 1, height := 6}, {width := 4, height := 1}, {width := 4, height := 1}]⟩
def letter_D : Letter := ⟨[{width := 1, height := 6}, {width := 4, height := 1}, {width := 1, height := 4}]⟩

def word : List Letter := [letter_F, letter_O, letter_O, letter_D]

def total_sign_area : ℕ := sign_width * sign_height

def letter_area (l : Letter) : ℕ :=
  l.bars.map (fun b => b.width * b.height) |> List.sum

def total_black_area : ℕ :=
  word.map letter_area |> List.sum

theorem white_area_is_40 : total_sign_area - total_black_area = 40 := by
  sorry

end NUMINAMATH_CALUDE_white_area_is_40_l3150_315079


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_25_l3150_315057

theorem largest_four_digit_congruent_to_15_mod_25 : ∃ (n : ℕ), 
  n ≤ 9990 ∧ 
  1000 ≤ n ∧ 
  n < 10000 ∧ 
  n ≡ 15 [MOD 25] ∧
  ∀ (m : ℕ), (1000 ≤ m ∧ m < 10000 ∧ m ≡ 15 [MOD 25]) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_15_mod_25_l3150_315057


namespace NUMINAMATH_CALUDE_johns_annual_profit_l3150_315024

/-- John's apartment subletting profit calculation --/
theorem johns_annual_profit :
  ∀ (num_subletters : ℕ) 
    (subletter_payment : ℕ) 
    (rent_cost : ℕ) 
    (months_in_year : ℕ),
  num_subletters = 3 →
  subletter_payment = 400 →
  rent_cost = 900 →
  months_in_year = 12 →
  (num_subletters * subletter_payment - rent_cost) * months_in_year = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l3150_315024


namespace NUMINAMATH_CALUDE_compound_weight_proof_l3150_315027

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of iodine atoms in the compound -/
def num_I_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 408

/-- Theorem stating that the molecular weight of the compound with 1 Al atom and 3 I atoms 
    is approximately equal to 408 g/mol -/
theorem compound_weight_proof : 
  ∃ ε > 0, |atomic_weight_Al + num_I_atoms * atomic_weight_I - molecular_weight| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l3150_315027


namespace NUMINAMATH_CALUDE_distance_between_points_l3150_315036

theorem distance_between_points : ∃ d : ℝ, 
  let A : ℝ × ℝ := (13, 5)
  let B : ℝ × ℝ := (5, -10)
  d = ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt ∧ d = 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3150_315036


namespace NUMINAMATH_CALUDE_three_digit_congruence_solutions_l3150_315088

theorem three_digit_congruence_solutions : 
  let count := Finset.filter (fun x => 100 ≤ x ∧ x ≤ 999 ∧ (4573 * x + 502) % 23 = 1307 % 23) (Finset.range 1000)
  Finset.card count = 39 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_solutions_l3150_315088


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l3150_315051

theorem quartic_equation_roots : 
  let f (x : ℝ) := 4*x^4 - 28*x^3 + 53*x^2 - 28*x + 4
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 ∨ x = (1/4 : ℝ) ∨ x = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l3150_315051


namespace NUMINAMATH_CALUDE_calculate_example_not_commutative_l3150_315083

-- Define the new operation
def otimes (a b : ℤ) : ℤ := a * b + a - b

-- Theorem 1: Calculate ((-2) ⊗ 5) ⊗ 6
theorem calculate_example : otimes (otimes (-2) 5) 6 = -125 := by
  sorry

-- Theorem 2: The operation is not commutative
theorem not_commutative : ∃ a b : ℤ, otimes a b ≠ otimes b a := by
  sorry

end NUMINAMATH_CALUDE_calculate_example_not_commutative_l3150_315083


namespace NUMINAMATH_CALUDE_larger_number_of_product_56_sum_15_l3150_315037

theorem larger_number_of_product_56_sum_15 (x y : ℕ) : 
  x * y = 56 → x + y = 15 → max x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_product_56_sum_15_l3150_315037


namespace NUMINAMATH_CALUDE_polynomial_root_difference_sum_l3150_315009

theorem polynomial_root_difference_sum (a b c d : ℝ) (x₁ x₂ : ℝ) : 
  a + b + c = 0 →
  a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 →
  a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 →
  x₁ = 1 →
  a ≥ b →
  b ≥ c →
  a > 0 →
  c < 0 →
  ∃ (min_val max_val : ℝ),
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≥ min_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = min_val) ∧
    (∀ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 → |x₁^2 - x₂^2| ≤ max_val) ∧
    (∃ x₂, a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧ |x₁^2 - x₂^2| = max_val) ∧
    min_val + max_val = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_difference_sum_l3150_315009


namespace NUMINAMATH_CALUDE_striped_quadrilateral_area_l3150_315066

/-- Represents a quadrilateral cut from striped gift wrapping paper -/
structure StripedQuadrilateral where
  /-- The combined area of the grey stripes in the quadrilateral -/
  greyArea : ℝ
  /-- The stripes are equally wide -/
  equalStripes : Bool

/-- Theorem stating that if the grey stripes have an area of 10 in a quadrilateral
    cut from equally striped paper, then the total area of the quadrilateral is 20 -/
theorem striped_quadrilateral_area
  (quad : StripedQuadrilateral)
  (h1 : quad.greyArea = 10)
  (h2 : quad.equalStripes = true) :
  quad.greyArea * 2 = 20 := by
  sorry

#check striped_quadrilateral_area

end NUMINAMATH_CALUDE_striped_quadrilateral_area_l3150_315066


namespace NUMINAMATH_CALUDE_max_a_value_l3150_315018

/-- The function f as defined in the problem -/
def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5*a*k + 3)*x + 7

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (k : ℝ), k ∈ Set.Icc 0 2 → 
    ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a) → x₂ ∈ Set.Icc (k+2*a) (k+4*a) → 
      f x₁ k a ≥ f x₂ k a) ∧
  (∀ (a' : ℝ), a' > a → 
    ∃ (k : ℝ), k ∈ Set.Icc 0 2 ∧ 
      ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k+a') ∧ x₂ ∈ Set.Icc (k+2*a') (k+4*a') ∧ 
        f x₁ k a' < f x₂ k a') ∧
  a = (2 * Real.sqrt 6 - 4) / 5 := by
sorry

end NUMINAMATH_CALUDE_max_a_value_l3150_315018


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l3150_315038

/-- The ellipse with equation 4x^2/49 + y^2/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (4 * p.1^2) / 49 + p.2^2 / 6 = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point P on the ellipse satisfying the given ratio condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  distance P F1 / distance P F2 = 4/3 →
  triangleArea P F1 F2 = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l3150_315038


namespace NUMINAMATH_CALUDE_greatest_divisor_of_sequence_l3150_315056

theorem greatest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_sequence_l3150_315056


namespace NUMINAMATH_CALUDE_triangle_pqr_area_l3150_315014

/-- Triangle PQR with given properties -/
structure Triangle where
  inradius : ℝ
  circumradius : ℝ
  angle_relation : ℝ → ℝ → ℝ → Prop

/-- The area of a triangle given its inradius and semiperimeter -/
def triangle_area (r : ℝ) (s : ℝ) : ℝ := r * s

/-- Theorem: Area of triangle PQR with given properties -/
theorem triangle_pqr_area (T : Triangle) 
  (h_inradius : T.inradius = 6)
  (h_circumradius : T.circumradius = 17)
  (h_angle : T.angle_relation = fun P Q R => 3 * Real.cos Q = Real.cos P + Real.cos R) :
  ∃ (s : ℝ), triangle_area T.inradius s = (102 * Real.sqrt 47) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pqr_area_l3150_315014


namespace NUMINAMATH_CALUDE_performance_orders_count_l3150_315021

/-- The number of programs available --/
def total_programs : ℕ := 8

/-- The number of programs to be selected --/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) --/
def special_programs : ℕ := 2

/-- Calculate the number of different performance orders --/
def calculate_orders : ℕ :=
  -- First category: only one of A or B is selected
  (special_programs.choose 1) * ((total_programs - special_programs).choose (selected_programs - 1)) * (selected_programs.factorial) +
  -- Second category: both A and B are selected
  ((total_programs - special_programs).choose (selected_programs - special_programs)) * (special_programs.factorial) * ((selected_programs - special_programs).factorial)

/-- The theorem to be proved --/
theorem performance_orders_count : calculate_orders = 1140 := by
  sorry

end NUMINAMATH_CALUDE_performance_orders_count_l3150_315021


namespace NUMINAMATH_CALUDE_feet_in_garden_l3150_315048

theorem feet_in_garden (num_dogs num_ducks : ℕ) (dog_feet duck_feet : ℕ) :
  num_dogs = 6 → num_ducks = 2 → dog_feet = 4 → duck_feet = 2 →
  num_dogs * dog_feet + num_ducks * duck_feet = 28 := by
sorry

end NUMINAMATH_CALUDE_feet_in_garden_l3150_315048


namespace NUMINAMATH_CALUDE_inequality_proof_l3150_315046

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3150_315046


namespace NUMINAMATH_CALUDE_max_length_sum_l3150_315081

/-- The length of a positive integer is the number of prime factors (not necessarily distinct) in its prime factorization. -/
def length (n : ℕ) : ℕ := sorry

/-- The maximum sum of lengths of x and y given the constraints. -/
theorem max_length_sum : 
  ∃ (x y : ℕ), 
    x > 1 ∧ 
    y > 1 ∧ 
    x + 3*y < 940 ∧ 
    ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 940 → length x + length y ≥ length a + length b ∧
    length x + length y = 15 :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l3150_315081


namespace NUMINAMATH_CALUDE_min_draws_for_fifteen_l3150_315000

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_draws_for_fifteen (counts : BallCounts) :
  counts.red = 30 ∧ counts.green = 25 ∧ counts.yellow = 23 ∧
  counts.blue = 14 ∧ counts.white = 13 ∧ counts.black = 10 →
  minDraws counts 15 = 80 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_fifteen_l3150_315000


namespace NUMINAMATH_CALUDE_gum_distribution_l3150_315082

theorem gum_distribution (num_cousins : Nat) (gum_per_cousin : Nat) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l3150_315082


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3150_315067

/-- Two points P and Q in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to be proved -/
theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨a, 1⟩
  let q : Point := ⟨2, b⟩
  symmetricAboutXAxis p q → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3150_315067


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l3150_315068

-- Part 1
theorem trig_identity : Real.cos (30 * π / 180) * Real.tan (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 3/2 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem quadratic_equation_solution (x : ℝ) : 3 * x^2 - 1 = -2 * x ↔ x = 1/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_equation_solution_l3150_315068


namespace NUMINAMATH_CALUDE_function_inequality_l3150_315050

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) : 
  f 2 + g 1 > f 1 + g 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3150_315050


namespace NUMINAMATH_CALUDE_count_polynomials_l3150_315070

-- Define a function to check if an expression is a polynomial
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "-7" => true
  | "m" => true
  | "x^3y^2" => true
  | "1/a" => false
  | "2x+3y" => true
  | _ => false

-- Define the list of expressions
def expressions : List String := ["-7", "m", "x^3y^2", "1/a", "2x+3y"]

-- Theorem to prove
theorem count_polynomials :
  (expressions.filter isPolynomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_polynomials_l3150_315070


namespace NUMINAMATH_CALUDE_apple_distribution_l3150_315033

/-- Represents the number of apples each person receives when evenly distributing
    a given number of apples among a given number of people. -/
def apples_per_person (total_apples : ℕ) (num_people : ℕ) : ℕ :=
  total_apples / num_people

/-- Theorem stating that when 15 apples are evenly distributed among 3 people,
    each person receives 5 apples. -/
theorem apple_distribution :
  apples_per_person 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3150_315033


namespace NUMINAMATH_CALUDE_max_abs_sum_of_coeffs_l3150_315032

/-- A quadratic polynomial p(x) = ax^2 + bx + c -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that |p(x)| ≤ 1 for all x in [0,1] -/
def BoundedOnInterval (p : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → |p x| ≤ 1

/-- The theorem stating that the maximum value of |a|+|b|+|c| is 4 -/
theorem max_abs_sum_of_coeffs (a b c : ℝ) :
  BoundedOnInterval (QuadraticPolynomial a b c) →
  |a| + |b| + |c| ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_of_coeffs_l3150_315032


namespace NUMINAMATH_CALUDE_prob_zeros_not_adjacent_is_point_six_l3150_315019

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when arranging num_ones ones and num_zeros zeros in a row -/
def prob_zeros_not_adjacent : ℚ :=
  1 - (2 * (Nat.factorial (total_elements - 1))) / (Nat.factorial total_elements)

theorem prob_zeros_not_adjacent_is_point_six :
  prob_zeros_not_adjacent = 3/5 :=
sorry

end NUMINAMATH_CALUDE_prob_zeros_not_adjacent_is_point_six_l3150_315019


namespace NUMINAMATH_CALUDE_triangle_inequality_l3150_315041

theorem triangle_inequality (a b ma mb : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ma > 0) (h4 : mb > 0) (h5 : a > b) :
  a * ma = b * mb →
  a^2010 + ma^2010 ≥ b^2010 + mb^2010 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3150_315041


namespace NUMINAMATH_CALUDE_chord_length_l3150_315059

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (a b c : ℝ) (r : ℝ) (h1 : r > 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let d := |c| / Real.sqrt (a^2 + b^2)
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  (∀ p ∈ line ∩ circle, True) →
  chord_length = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3150_315059


namespace NUMINAMATH_CALUDE_factorization_example_l3150_315063

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x * h x ∧ (∃ c, g x = c * x ∨ g x = x)

/-- The given equation represents factorization from left to right -/
theorem factorization_example :
  is_factorization_left_to_right
    (λ a : ℝ => 2 * a^2 + a)
    (λ a : ℝ => a)
    (λ a : ℝ => 2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l3150_315063


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3150_315098

theorem arithmetic_calculations : 
  ((62 + 38) / 4 = 25) ∧ 
  ((34 + 19) * 7 = 371) ∧ 
  (1500 - 125 * 8 = 500) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3150_315098


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3150_315071

theorem polynomial_factorization (x m : ℝ) : 
  (x^2 + 6*x + 5 = (x+5)*(x+1)) ∧ (m^2 - m - 12 = (m+3)*(m-4)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3150_315071


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3150_315012

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling is valid (uses each number exactly once) -/
def isValidLabeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron as a set of three vertex indices -/
def TetrahedronFace := Fin 3 → Fin 4

/-- The four faces of a tetrahedron -/
def tetrahedronFaces : Fin 4 → TetrahedronFace := sorry

/-- The sum of labels on a face -/
def faceSum (l : TetrahedronLabeling) (f : TetrahedronFace) : Nat :=
  (f 0).val + 1 + (f 1).val + 1 + (f 2).val + 1

/-- Theorem: No valid labeling exists such that all face sums are equal -/
theorem no_valid_tetrahedron_labeling :
  ¬∃ (l : TetrahedronLabeling),
    isValidLabeling l ∧
    ∃ (s : Nat), ∀ (f : Fin 4), faceSum l (tetrahedronFaces f) = s :=
  sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l3150_315012


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3150_315089

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (d h w r : ℝ) (h1 : d = 20) (h2 : w = 4) (h3 : r = d / 2) :
  3 * (2 * π * r) * w = 240 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l3150_315089


namespace NUMINAMATH_CALUDE_sequence_bounds_l3150_315086

theorem sequence_bounds (n : ℕ+) (a : ℕ → ℚ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bounds_l3150_315086


namespace NUMINAMATH_CALUDE_train_length_l3150_315010

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 2.7497800175985923 → 
  speed_kmh * (5/18) * time_s = 110.9912007039437 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3150_315010


namespace NUMINAMATH_CALUDE_triangle_area_range_line_equation_l3150_315090

/-- Ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 6 + y^2 / 4 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Point on ellipse C₁ -/
def point_on_C₁ (P : ℝ × ℝ) : Prop := C₁ P.1 P.2

/-- Line passing through (-1, 0) -/
def line_through_M (l : ℝ → ℝ) : Prop := l 0 = -1

/-- Intersection points of line l with C₁ and C₂ -/
def intersection_points (l : ℝ → ℝ) (A B C D : ℝ × ℝ) : Prop :=
  point_on_C₁ A ∧ point_on_C₁ D ∧ C₂ B.1 B.2 ∧ C₂ C.1 C.2 ∧
  A.2 > B.2 ∧ B.2 > C.2 ∧ C.2 > D.2 ∧
  (∀ y, l y = A.1 ↔ y = A.2) ∧ (∀ y, l y = B.1 ↔ y = B.2) ∧
  (∀ y, l y = C.1 ↔ y = C.2) ∧ (∀ y, l y = D.1 ↔ y = D.2)

/-- Theorem 1: Range of triangle area -/
theorem triangle_area_range :
  ∀ P : ℝ × ℝ, point_on_C₁ P →
  ∃ S : ℝ, 1 ≤ S ∧ S ≤ Real.sqrt 2 ∧
  (∃ Q : ℝ × ℝ, C₂ Q.1 Q.2 ∧ S = (1/2) * Real.sqrt ((P.1^2 + P.2^2) * 2 - (P.1 * Q.1 + P.2 * Q.2)^2)) :=
sorry

/-- Theorem 2: Equation of line l -/
theorem line_equation :
  ∀ l : ℝ → ℝ, line_through_M l →
  (∃ A B C D : ℝ × ℝ, intersection_points l A B C D ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) →
  (∀ y, l y = -1) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_range_line_equation_l3150_315090


namespace NUMINAMATH_CALUDE_two_integer_pairs_satisfy_equation_l3150_315003

theorem two_integer_pairs_satisfy_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 2 ∧ 
  (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_two_integer_pairs_satisfy_equation_l3150_315003


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3150_315002

theorem perfect_square_trinomial (x y : ℝ) : x^2 + 4*y^2 - 4*x*y = (x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3150_315002


namespace NUMINAMATH_CALUDE_range_of_t_l3150_315097

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a^2 - a*b + b^2
  ∃ (x : ℝ), t = x ∧ 1/3 ≤ x ∧ x ≤ 3 ∧
  ∀ (y : ℝ), (∃ (a' b' : ℝ), a'^2 + a'*b' + b'^2 = 1 ∧ a'^2 - a'*b' + b'^2 = y) → 1/3 ≤ y ∧ y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l3150_315097


namespace NUMINAMATH_CALUDE_motorcycle_license_combinations_l3150_315061

def letter_choices : ℕ := 3
def digit_choices : ℕ := 10
def license_length : ℕ := 4

theorem motorcycle_license_combinations : 
  letter_choices * digit_choices ^ license_length = 30000 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_license_combinations_l3150_315061


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3150_315069

theorem quadratic_discriminant (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3150_315069


namespace NUMINAMATH_CALUDE_window_installation_time_l3150_315053

theorem window_installation_time 
  (total_windows : ℕ) 
  (installed_windows : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_windows = 9)
  (h2 : installed_windows = 6)
  (h3 : remaining_time = 18)
  (h4 : installed_windows < total_windows) :
  (remaining_time : ℚ) / (total_windows - installed_windows : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_window_installation_time_l3150_315053


namespace NUMINAMATH_CALUDE_cos_36_degrees_l3150_315001

theorem cos_36_degrees : Real.cos (36 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l3150_315001


namespace NUMINAMATH_CALUDE_sarah_won_30_games_l3150_315005

/-- Sarah's tic-tac-toe game results -/
structure TicTacToeResults where
  total_games : ℕ
  tied_games : ℕ
  total_money : ℤ
  win_money : ℕ
  tie_money : ℕ
  lose_money : ℕ

/-- Theorem: Sarah won 30 games -/
theorem sarah_won_30_games (results : TicTacToeResults)
  (h1 : results.total_games = 100)
  (h2 : results.tied_games = 40)
  (h3 : results.total_money = -30)
  (h4 : results.win_money = 1)
  (h5 : results.tie_money = 0)
  (h6 : results.lose_money = 2) :
  ∃ (won_games lost_games : ℕ),
    won_games + results.tied_games + lost_games = results.total_games ∧
    won_games * results.win_money - lost_games * results.lose_money = results.total_money ∧
    won_games = 30 :=
  sorry

end NUMINAMATH_CALUDE_sarah_won_30_games_l3150_315005


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l3150_315096

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_day1 : ℝ) (miles_day2 : ℝ) : ℝ :=
  consumption_rate * (miles_day1 + miles_day2)

/-- Proves that given the specified conditions, the total gas consumption is 4000 gallons -/
theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_day1 : ℝ := 400
  let miles_day2 : ℝ := miles_day1 + 200
  total_gas_consumption consumption_rate miles_day1 miles_day2 = 4000 :=
by
  sorry

#eval total_gas_consumption 4 400 600

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l3150_315096


namespace NUMINAMATH_CALUDE_triangle_properties_l3150_315074

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def altitude_BC (x y : ℝ) : Prop := x - 2*y - 1 = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0
def point_B : ℝ × ℝ := (2, 1)

-- Define the theorem
theorem triangle_properties (ABC : Triangle) :
  altitude_BC ABC.A.1 ABC.A.2 ∧
  altitude_BC ABC.C.1 ABC.C.2 ∧
  angle_bisector_A ABC.A.2 ∧
  ABC.B = point_B →
  ABC.A = (1, 0) ∧
  ABC.C = (4, -3) ∧
  ∀ (x y : ℝ), y = x - 1 ↔ (x = ABC.A.1 ∧ y = ABC.A.2) ∨ (x = ABC.C.1 ∧ y = ABC.C.2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l3150_315074


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l3150_315078

theorem lawn_mowing_problem (initial_people : ℕ) (initial_hours : ℕ) (target_hours : ℕ) :
  initial_people = 8 →
  initial_hours = 5 →
  target_hours = 3 →
  ∃ (additional_people : ℕ),
    additional_people = 6 ∧
    (initial_people + additional_people) * target_hours = initial_people * initial_hours :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l3150_315078
