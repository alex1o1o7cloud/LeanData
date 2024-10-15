import Mathlib

namespace NUMINAMATH_CALUDE_digit_sum_2017_power_l2084_208421

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The theorem to prove -/
theorem digit_sum_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end NUMINAMATH_CALUDE_digit_sum_2017_power_l2084_208421


namespace NUMINAMATH_CALUDE_preceding_sum_40_times_l2084_208497

theorem preceding_sum_40_times (n : ℕ) : 
  (n ≠ 0) → ((n * (n - 1)) / 2 = 40 * n) → n = 81 := by
  sorry

end NUMINAMATH_CALUDE_preceding_sum_40_times_l2084_208497


namespace NUMINAMATH_CALUDE_train_passing_time_l2084_208449

/-- Prove that a train with given length and speed will pass a fixed point in the calculated time -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 275 →
  train_speed_kmh = 90 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  passing_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2084_208449


namespace NUMINAMATH_CALUDE_cos2α_plus_sin2α_for_point_l2084_208476

theorem cos2α_plus_sin2α_for_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = -3 ∧ r * Real.sin α = 4) →
  Real.cos (2 * α) + Real.sin (2 * α) = -31/25 := by
  sorry

end NUMINAMATH_CALUDE_cos2α_plus_sin2α_for_point_l2084_208476


namespace NUMINAMATH_CALUDE_marks_trees_l2084_208479

theorem marks_trees (initial_trees : ℕ) (planted_trees : ℕ) : 
  initial_trees = 13 → planted_trees = 12 → initial_trees + planted_trees = 25 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l2084_208479


namespace NUMINAMATH_CALUDE_triangle_tangent_equality_l2084_208455

theorem triangle_tangent_equality (A B : ℝ) (a b : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : A + B < π) :
  a * Real.tan A + b * Real.tan B = (a + b) * Real.tan ((A + B) / 2) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_equality_l2084_208455


namespace NUMINAMATH_CALUDE_food_product_range_l2084_208424

/-- Represents the net content of a food product -/
structure NetContent where
  nominal : ℝ
  tolerance : ℝ

/-- Represents a range of values -/
structure Range where
  lower : ℝ
  upper : ℝ

/-- Calculates the qualified net content range for a given net content -/
def qualifiedRange (nc : NetContent) : Range :=
  { lower := nc.nominal - nc.tolerance,
    upper := nc.nominal + nc.tolerance }

/-- Theorem: The qualified net content range for a product labeled "500g ± 5g" is 495g to 505g -/
theorem food_product_range :
  let nc : NetContent := { nominal := 500, tolerance := 5 }
  let range := qualifiedRange nc
  range.lower = 495 ∧ range.upper = 505 := by
  sorry

end NUMINAMATH_CALUDE_food_product_range_l2084_208424


namespace NUMINAMATH_CALUDE_walmart_ground_beef_sales_l2084_208487

theorem walmart_ground_beef_sales (thursday_sales : ℕ) (friday_sales : ℕ) (saturday_sales : ℕ) 
  (h1 : thursday_sales = 210)
  (h2 : friday_sales = 2 * thursday_sales)
  (h3 : (thursday_sales + friday_sales + saturday_sales) / 3 = 260) :
  saturday_sales = 150 := by
sorry

end NUMINAMATH_CALUDE_walmart_ground_beef_sales_l2084_208487


namespace NUMINAMATH_CALUDE_set_union_problem_l2084_208427

theorem set_union_problem (a b l : ℝ) :
  let A : Set ℝ := {-2, a}
  let B : Set ℝ := {2015^a, b}
  A ∩ B = {l} →
  A ∪ B = {-2, 1, 2015} :=
by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2084_208427


namespace NUMINAMATH_CALUDE_unique_solution_l2084_208400

def system_solution (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ x - 2 * y = -1

theorem unique_solution : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2084_208400


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2084_208465

theorem min_value_sum_fractions (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  ((a + b + k) / c + (a + c + k) / b + (b + c + k) / a = 9 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2084_208465


namespace NUMINAMATH_CALUDE_expression_value_l2084_208418

theorem expression_value (y d : ℝ) (h1 : y > 0) 
  (h2 : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2084_208418


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_12_l2084_208445

theorem cos_squared_minus_sin_squared_pi_12 : 
  Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_pi_12_l2084_208445


namespace NUMINAMATH_CALUDE_f_properties_l2084_208410

-- Define the function f(x) = x^3 - 6x + 5
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the theorem for the extreme points and the range of k
theorem f_properties :
  -- Part I: Extreme points
  (∃ (x_max x_min : ℝ), x_max = -Real.sqrt 2 ∧ x_min = Real.sqrt 2 ∧
    (∀ (x : ℝ), f x ≤ f x_max) ∧
    (∀ (x : ℝ), f x ≥ f x_min)) ∧
  -- Part II: Range of k
  (∀ (k : ℝ), (∀ (x : ℝ), x > 1 → f x ≥ k * (x - 1)) ↔ k ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2084_208410


namespace NUMINAMATH_CALUDE_log_equation_solution_l2084_208451

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (4 * x) / Real.log 3 → x = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2084_208451


namespace NUMINAMATH_CALUDE_marble_ratio_l2084_208464

theorem marble_ratio (blue red : ℕ) (h1 : blue = red + 24) (h2 : red = 6) :
  blue / red = 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2084_208464


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2084_208467

theorem quadratic_roots_theorem (a b c : ℝ) :
  (∃ x y : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ∧ y^2 - (a+b)*y + (a*b-c^2) = 0) ∧
  (∃! x : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ↔ a = b ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2084_208467


namespace NUMINAMATH_CALUDE_min_value_sum_cubic_ratios_l2084_208426

theorem min_value_sum_cubic_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_cubic_ratios_l2084_208426


namespace NUMINAMATH_CALUDE_function_value_at_2012_l2084_208423

theorem function_value_at_2012 (f : ℝ → ℝ) 
  (h1 : f 0 = 2012)
  (h2 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x)
  (h3 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x) :
  f 2012 = 2^2012 + 2011 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2012_l2084_208423


namespace NUMINAMATH_CALUDE_sequence_formula_l2084_208474

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem sequence_formula :
  let a₁ : ℝ := 20
  let d : ℝ := -9
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -9 * n + 29 := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2084_208474


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2084_208483

/-- Given a line y = mx + b, prove that mb < -1 --/
theorem line_slope_intercept_product (m b : ℝ) : m * b < -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2084_208483


namespace NUMINAMATH_CALUDE_ratio_problem_l2084_208440

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 3 / 4) :
  x / y = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2084_208440


namespace NUMINAMATH_CALUDE_trig_identity_l2084_208411

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2084_208411


namespace NUMINAMATH_CALUDE_company_growth_rate_equation_l2084_208469

/-- Represents the average annual growth rate of a company's payment. -/
def average_annual_growth_rate (initial_payment final_payment : ℝ) (years : ℕ) : ℝ → Prop :=
  λ x => initial_payment * (1 + x) ^ years = final_payment

/-- Theorem stating that the equation 40(1 + x)^2 = 48.4 correctly represents
    the average annual growth rate of the company's payment. -/
theorem company_growth_rate_equation :
  average_annual_growth_rate 40 48.4 2 = λ x => 40 * (1 + x)^2 = 48.4 := by
  sorry

end NUMINAMATH_CALUDE_company_growth_rate_equation_l2084_208469


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l2084_208416

theorem intersection_point_theorem (α β : ℝ) :
  (∃ x y : ℝ, 
    x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1 ∧
    x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1 ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l2084_208416


namespace NUMINAMATH_CALUDE_percentage_not_red_roses_is_92_percent_l2084_208482

/-- Represents the number of flowers of each type in the garden -/
structure GardenFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers in the garden -/
def totalFlowers (g : GardenFlowers) : ℕ :=
  g.roses + g.tulips + g.daisies + g.lilies + g.sunflowers

/-- Calculates the number of red roses in the garden -/
def redRoses (g : GardenFlowers) : ℕ :=
  g.roses / 2

/-- Calculates the percentage of flowers that are not red roses -/
def percentageNotRedRoses (g : GardenFlowers) : ℚ :=
  (totalFlowers g - redRoses g : ℚ) / (totalFlowers g : ℚ) * 100

/-- Theorem stating that 92% of flowers in the given garden are not red roses -/
theorem percentage_not_red_roses_is_92_percent (g : GardenFlowers) 
  (h1 : g.roses = 25)
  (h2 : g.tulips = 40)
  (h3 : g.daisies = 60)
  (h4 : g.lilies = 15)
  (h5 : g.sunflowers = 10) :
  percentageNotRedRoses g = 92 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_red_roses_is_92_percent_l2084_208482


namespace NUMINAMATH_CALUDE_range_of_m_l2084_208442

/-- Condition p: |1 - (x-1)/3| < 2 -/
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2

/-- Condition q: (x-1)^2 < m^2 -/
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

/-- q is a sufficient condition for p -/
def q_sufficient (m : ℝ) : Prop := ∀ x, q x m → p x

/-- q is not a necessary condition for p -/
def q_not_necessary (m : ℝ) : Prop := ∃ x, p x ∧ ¬q x m

theorem range_of_m :
  (∀ m, q_sufficient m ∧ q_not_necessary m) →
  (∀ m, m ∈ Set.Icc (-3 : ℝ) 3) ∧ 
  (∃ m₁ m₂, m₁ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₂ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₁ ≠ m₂) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2084_208442


namespace NUMINAMATH_CALUDE_cube_colorings_correct_dodecahedron_colorings_correct_l2084_208499

/-- The number of rotational symmetries of a cube -/
def cubeSymmetries : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedronSymmetries : ℕ := 60

/-- The number of geometrically distinct colorings of a cube with 6 different colors -/
def cubeColorings : ℕ := 30

/-- The number of geometrically distinct colorings of a dodecahedron with 12 different colors -/
def dodecahedronColorings : ℕ := (Nat.factorial 11) / 5

theorem cube_colorings_correct :
  cubeColorings = (Nat.factorial 6) / cubeSymmetries :=
sorry

theorem dodecahedron_colorings_correct :
  dodecahedronColorings = (Nat.factorial 12) / dodecahedronSymmetries :=
sorry

end NUMINAMATH_CALUDE_cube_colorings_correct_dodecahedron_colorings_correct_l2084_208499


namespace NUMINAMATH_CALUDE_sum_of_digits_consecutive_numbers_l2084_208472

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sum_of_digits_consecutive_numbers 
  (N : ℕ) 
  (h1 : sumOfDigits N + sumOfDigits (N + 1) = 200)
  (h2 : sumOfDigits (N + 2) + sumOfDigits (N + 3) = 105) :
  sumOfDigits (N + 1) + sumOfDigits (N + 2) = 202 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_consecutive_numbers_l2084_208472


namespace NUMINAMATH_CALUDE_max_marks_proof_l2084_208461

def math_pass_percentage : ℚ := 45/100
def science_pass_percentage : ℚ := 1/2
def math_score : ℕ := 267
def math_shortfall : ℕ := 45
def science_score : ℕ := 292
def science_shortfall : ℕ := 38

def total_marks : ℕ := 1354

theorem max_marks_proof :
  let math_total := (math_score + math_shortfall) / math_pass_percentage
  let science_total := (science_score + science_shortfall) / science_pass_percentage
  ⌈math_total⌉ + science_total = total_marks := by
  sorry

end NUMINAMATH_CALUDE_max_marks_proof_l2084_208461


namespace NUMINAMATH_CALUDE_sequence_property_l2084_208490

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ (p q : ℕ), a (p + q) = a p + a q) 
  (h2 : a 2 = 4) : 
  a 9 = 18 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2084_208490


namespace NUMINAMATH_CALUDE_prime_square_sum_l2084_208471

theorem prime_square_sum (p q n : ℕ) : 
  Prime p → Prime q → n^2 = p^2 + q^2 + p^2 * q^2 → 
  ((p = 2 ∧ q = 3 ∧ n = 7) ∨ (p = 3 ∧ q = 2 ∧ n = 7)) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_l2084_208471


namespace NUMINAMATH_CALUDE_number_relationships_l2084_208450

theorem number_relationships : 
  (10 * 10000 = 100000) ∧
  (10 * 1000000 = 10000000) ∧
  (10 * 10000000 = 100000000) ∧
  (100000000 / 10000 = 10000) := by
  sorry

end NUMINAMATH_CALUDE_number_relationships_l2084_208450


namespace NUMINAMATH_CALUDE_total_recovery_time_l2084_208485

/-- Calculates the total recovery time for James after a hand burn, considering initial healing,
    post-surgery recovery, physical therapy sessions, and medication effects. -/
theorem total_recovery_time (initial_healing : ℝ) (A : ℝ) : 
  initial_healing = 4 →
  let post_surgery := initial_healing * 1.5
  let total_before_reduction := post_surgery
  let therapy_reduction := total_before_reduction * (0.1 * A)
  let medication_reduction := total_before_reduction * 0.2
  total_before_reduction - therapy_reduction - medication_reduction = 4.8 - 0.6 * A := by
  sorry

end NUMINAMATH_CALUDE_total_recovery_time_l2084_208485


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l2084_208463

/-- The quadratic equation x^2 + 2x - 1 = 0 is equivalent to (x+1)^2 = 2 -/
theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l2084_208463


namespace NUMINAMATH_CALUDE_mlb_game_misses_l2084_208407

theorem mlb_game_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  hits + misses = 200 → 
  misses = 150 := by
sorry

end NUMINAMATH_CALUDE_mlb_game_misses_l2084_208407


namespace NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircles_l2084_208405

/-- Given a square with side length 4 and four semicircles with centers at the midpoints of the square's sides, 
    prove that the area not covered by the semicircles is 8 - 2π. -/
theorem shaded_area_in_square_with_semicircles (square_side : ℝ) (semicircle_radius : ℝ) : 
  square_side = 4 → 
  semicircle_radius = Real.sqrt 2 →
  (4 : ℝ) * (π / 2 * semicircle_radius^2) = 2 * π →
  square_side^2 - (4 : ℝ) * (π / 2 * semicircle_radius^2) = 8 - 2 * π := by
  sorry

#align shaded_area_in_square_with_semicircles shaded_area_in_square_with_semicircles

end NUMINAMATH_CALUDE_shaded_area_in_square_with_semicircles_l2084_208405


namespace NUMINAMATH_CALUDE_probability_not_adjacent_correct_l2084_208491

/-- The number of chairs in a row -/
def total_chairs : ℕ := 10

/-- The number of available chairs (excluding the last one) -/
def available_chairs : ℕ := total_chairs - 1

/-- The probability that two people don't sit next to each other 
    when randomly selecting from the first 9 chairs of 10 -/
def probability_not_adjacent : ℚ := 7 / 9

/-- Theorem stating the probability of two people not sitting adjacent 
    when randomly selecting from 9 out of 10 chairs -/
theorem probability_not_adjacent_correct : 
  probability_not_adjacent = 1 - (2 * available_chairs - 2) / (available_chairs * (available_chairs - 1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_correct_l2084_208491


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2084_208438

theorem abs_inequality_solution_set (x : ℝ) :
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2084_208438


namespace NUMINAMATH_CALUDE_vieta_cubic_formulas_l2084_208409

theorem vieta_cubic_formulas (a b c d x₁ x₂ x₃ : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)) →
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
  sorry

end NUMINAMATH_CALUDE_vieta_cubic_formulas_l2084_208409


namespace NUMINAMATH_CALUDE_trig_simplification_l2084_208454

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2084_208454


namespace NUMINAMATH_CALUDE_max_angle_ratio_theorem_l2084_208468

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 8 + 2 * Real.sqrt 3 = 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-2 * Real.sqrt 3, 0) ∧ F₂ = (2 * Real.sqrt 3, 0)

-- Define the point P on the line
def point_on_line (P : ℝ × ℝ) : Prop :=
  line P.1 P.2

-- Define the angle F₁PF₂
def angle_F₁PF₂ (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_ratio_theorem 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : ellipse F₁.1 F₁.2 ∧ ellipse F₂.1 F₂.2)
  (h_foci : foci F₁ F₂)
  (h_point : point_on_line P)
  (h_max_angle : ∀ Q, point_on_line Q → angle_F₁PF₂ F₁ F₂ P ≥ angle_F₁PF₂ F₁ F₂ Q) :
  distance P F₁ / distance P F₂ = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_angle_ratio_theorem_l2084_208468


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l2084_208412

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, x^2/16 + y^2/m = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = 16 - m ∧ 2*c = 2*Real.sqrt 7) →
  m = 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l2084_208412


namespace NUMINAMATH_CALUDE_younger_brother_height_l2084_208430

def father_height : ℕ := 172
def height_diff_father_minkyung : ℕ := 35
def height_diff_minkyung_brother : ℕ := 28

theorem younger_brother_height :
  father_height - height_diff_father_minkyung - height_diff_minkyung_brother = 109 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_height_l2084_208430


namespace NUMINAMATH_CALUDE_sandwiches_per_student_l2084_208404

theorem sandwiches_per_student
  (students_per_group : ℕ)
  (total_groups : ℕ)
  (total_bread_pieces : ℕ)
  (bread_per_sandwich : ℕ)
  (h1 : students_per_group = 6)
  (h2 : total_groups = 5)
  (h3 : total_bread_pieces = 120)
  (h4 : bread_per_sandwich = 2) :
  total_bread_pieces / (bread_per_sandwich * (students_per_group * total_groups)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sandwiches_per_student_l2084_208404


namespace NUMINAMATH_CALUDE_max_value_on_circle_l2084_208414

theorem max_value_on_circle (x y : ℝ) : 
  Complex.abs (x - 2 + y * Complex.I) = 1 →
  (∃ (x' y' : ℝ), Complex.abs (x' - 2 + y' * Complex.I) = 1 ∧ 
    |3 * x' - y'| ≥ |3 * x - y|) →
  |3 * x - y| ≤ 6 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l2084_208414


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2084_208402

/-- The ellipse with equation x²/4 + y² = 1 and eccentricity √3/2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- The line x = ky - 1 -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 = k * p.2 - 1}

/-- Point M is the reflection of A across the x-axis -/
def ReflectAcrossXAxis (A M : ℝ × ℝ) : Prop :=
  M.1 = A.1 ∧ M.2 = -A.2

/-- The line passing through two points -/
def LineThroughPoints (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem ellipse_intersection_fixed_point (k : ℝ) 
  (A B : ℝ × ℝ) (hA : A ∈ Ellipse ∩ Line k) (hB : B ∈ Ellipse ∩ Line k) 
  (M : ℝ × ℝ) (hM : ReflectAcrossXAxis A M) (hAB : A ≠ B) :
  ∃ P : ℝ × ℝ, P ∈ LineThroughPoints M B ∧ P.1 = -4 ∧ P.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2084_208402


namespace NUMINAMATH_CALUDE_symmetry_axis_sine_function_l2084_208470

theorem symmetry_axis_sine_function (φ : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (3 * x + φ)) →
  (|φ| < Real.pi / 2) →
  (∀ x : ℝ, Real.sin (3 * x + φ) = Real.sin (3 * (3 * Real.pi / 2 - x) + φ)) →
  φ = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_sine_function_l2084_208470


namespace NUMINAMATH_CALUDE_no_three_distinct_real_roots_l2084_208494

theorem no_three_distinct_real_roots (c : ℝ) : 
  ¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧ 
    (x₁^3 + 4*x₁^2 + 6*x₁ + c = 0) ∧ 
    (x₂^3 + 4*x₂^2 + 6*x₂ + c = 0) ∧ 
    (x₃^3 + 4*x₃^2 + 6*x₃ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_three_distinct_real_roots_l2084_208494


namespace NUMINAMATH_CALUDE_sum_of_G_from_2_to_100_l2084_208473

-- Define G(n) as the number of solutions to sin x = sin (n^2 x) on [0, 2π]
def G (n : ℕ) : ℕ := 
  if n > 1 then 2 * n^2 + 1 else 0

-- Theorem statement
theorem sum_of_G_from_2_to_100 : 
  (Finset.range 99).sum (fun i => G (i + 2)) = 676797 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_G_from_2_to_100_l2084_208473


namespace NUMINAMATH_CALUDE_x_intercept_distance_l2084_208456

/-- Given two lines intersecting at (8, 20), one with slope 4 and the other with slope -3,
    the distance between their x-intercepts is 35/3. -/
theorem x_intercept_distance (line1 line2 : (ℝ → ℝ)) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -3 * x + 44) →
  line1 8 = 20 →
  line2 8 = 20 →
  |((0 - (-12)) / 4) - ((0 - 44) / (-3))| = 35/3 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l2084_208456


namespace NUMINAMATH_CALUDE_min_value_problem_l2084_208439

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2084_208439


namespace NUMINAMATH_CALUDE_multiple_reals_less_than_negative_one_l2084_208431

theorem multiple_reals_less_than_negative_one :
  ∃ (x y : ℝ), x < -1 ∧ y < -1 ∧ x ≠ y :=
sorry

end NUMINAMATH_CALUDE_multiple_reals_less_than_negative_one_l2084_208431


namespace NUMINAMATH_CALUDE_local_extrema_of_f_l2084_208498

open Real

/-- The function f(x) = x^3 - 3x^2 - 9x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem local_extrema_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 2 ∧
  IsLocalMax f x ∧
  f x = 5 ∧
  (∀ y ∈ Set.Ioo (-2 : ℝ) 2, ¬IsLocalMin f y) := by
  sorry

#check local_extrema_of_f

end NUMINAMATH_CALUDE_local_extrema_of_f_l2084_208498


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2084_208493

/-- Theorem: For a parabola y² = 4x with focus F(1, 0), and a point P(x₀, y₀) on the parabola 
    such that |PF| = 3/2 * x₀, the value of x₀ is 2. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 4*x₀ →                             -- P(x₀, y₀) is on the parabola
  (x₀ - 1)^2 + y₀^2 = (3/2 * x₀)^2 →        -- |PF| = 3/2 * x₀
  x₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2084_208493


namespace NUMINAMATH_CALUDE_rounding_bounds_l2084_208478

def rounded_value : ℕ := 1300000

theorem rounding_bounds :
  ∀ n : ℕ,
  (n + 50000) / 100000 * 100000 = rounded_value →
  n ≤ 1304999 ∧ n ≥ 1295000 :=
by sorry

end NUMINAMATH_CALUDE_rounding_bounds_l2084_208478


namespace NUMINAMATH_CALUDE_min_value_fraction_l2084_208406

theorem min_value_fraction (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = 2 → 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 2 → 8 / ((x + 2) * (y + 4)) ≤ 8 / ((a + 2) * (b + 4))) →
  8 / ((x + 2) * (y + 4)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2084_208406


namespace NUMINAMATH_CALUDE_ivan_petrovich_savings_l2084_208460

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  let principal : ℝ := 750000
  let rate : ℝ := 0.08
  let time : ℝ := 12
  simple_interest principal rate time = 1470000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_petrovich_savings_l2084_208460


namespace NUMINAMATH_CALUDE_fourth_section_area_l2084_208425

/-- Represents a regular hexagon divided into four sections by three line segments -/
structure DividedHexagon where
  total_area : ℝ
  section1_area : ℝ
  section2_area : ℝ
  section3_area : ℝ
  section4_area : ℝ
  is_regular : total_area = 6 * (section1_area + section2_area + section3_area + section4_area) / 6
  sum_of_parts : total_area = section1_area + section2_area + section3_area + section4_area

/-- The theorem stating that if three sections of a divided regular hexagon have areas 2, 3, and 4,
    then the fourth section has an area of 11 -/
theorem fourth_section_area (h : DividedHexagon) 
    (h2 : h.section1_area = 2) 
    (h3 : h.section2_area = 3) 
    (h4 : h.section3_area = 4) : 
    h.section4_area = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_section_area_l2084_208425


namespace NUMINAMATH_CALUDE_triangle_midpoint_vector_l2084_208441

/-- Given a triangle ABC with vertices A(-1, 0), B(0, 2), and C(2, 0),
    and D is the midpoint of BC, prove that vector AD equals (2, 1) -/
theorem triangle_midpoint_vector (A B C D : ℝ × ℝ) : 
  A = (-1, 0) → B = (0, 2) → C = (2, 0) → D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (D.1 - A.1, D.2 - A.2) = (2, 1) := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_vector_l2084_208441


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l2084_208444

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≤ 0, f_inv (f x) = x) ∧
    (∀ y ≥ 2, f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l2084_208444


namespace NUMINAMATH_CALUDE_regular_star_points_l2084_208489

/-- Represents an n-pointed regular star with alternating angles --/
structure RegularStar where
  n : ℕ
  A : ℕ → ℝ
  B : ℕ → ℝ
  A_congruent : ∀ i j, A i = A j
  B_congruent : ∀ i j, B i = B j
  angle_difference : ∀ i, B i - A i = 20

/-- Theorem stating that the only possible number of points for the given conditions is 18 --/
theorem regular_star_points (star : RegularStar) : star.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l2084_208489


namespace NUMINAMATH_CALUDE_area_of_region_l2084_208434

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 - 4 * x - x * y + 2 * y = 0

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2 ∧ 0 ≤ p.2}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l2084_208434


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_equals_three_l2084_208475

theorem no_intersection_implies_k_equals_three (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_equals_three_l2084_208475


namespace NUMINAMATH_CALUDE_remainder_problem_l2084_208480

theorem remainder_problem (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2084_208480


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2084_208453

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line2D) 
  (point : Point2D) 
  (h1 : given_line.a = 1 ∧ given_line.b = 2 ∧ given_line.c = 1) 
  (h2 : point.x = 1 ∧ point.y = 1) : 
  ∃ (l : Line2D), 
    pointOnLine point l ∧ 
    perpendicularLines l given_line ∧ 
    l.a = 2 ∧ l.b = -1 ∧ l.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2084_208453


namespace NUMINAMATH_CALUDE_probability_same_length_segments_l2084_208420

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides + diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of diagonals of the first length in a regular hexagon -/
def num_diagonals_length1 : ℕ := 3

/-- The number of diagonals of the second length in a regular hexagon -/
def num_diagonals_length2 : ℕ := 6

/-- The probability of selecting two segments of the same length from a regular hexagon -/
theorem probability_same_length_segments :
  (Nat.choose num_sides 2 + Nat.choose num_diagonals_length1 2 + Nat.choose num_diagonals_length2 2) /
  Nat.choose total_segments 2 = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_length_segments_l2084_208420


namespace NUMINAMATH_CALUDE_greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l2084_208417

/-- A function that returns all integer substrings of a given positive integer -/
def integerSubstrings (n : ℕ+) : Finset ℕ :=
  sorry

/-- A function that checks if any element in a finite set is divisible by 9 -/
def anyDivisibleBy9 (s : Finset ℕ) : Prop :=
  sorry

theorem greatest_integer_no_substring_divisible_by_9 :
  ∀ n : ℕ+, n > 88888888 → anyDivisibleBy9 (integerSubstrings n) :=
  sorry

theorem all_substrings_of_88888888_not_divisible_by_9 :
  ¬ anyDivisibleBy9 (integerSubstrings 88888888) :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l2084_208417


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l2084_208432

theorem distance_between_circle_centers (a b c : ℝ) (h_a : a = 17) (h_b : b = 15) (h_c : c = 10) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let AI := Real.sqrt (16 + (K / s) ^ 2)
  20 * AI = 20 * Real.sqrt (16 + 5544 / 441) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l2084_208432


namespace NUMINAMATH_CALUDE_product_of_conjugates_l2084_208484

theorem product_of_conjugates (P Q R S : ℝ) : 
  P = Real.sqrt 2023 + Real.sqrt 2024 →
  Q = -Real.sqrt 2023 - Real.sqrt 2024 →
  R = Real.sqrt 2023 - Real.sqrt 2024 →
  S = Real.sqrt 2024 - Real.sqrt 2023 →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_conjugates_l2084_208484


namespace NUMINAMATH_CALUDE_problem_solution_l2084_208428

/-- The graph of y = x + m - 2 does not pass through the second quadrant -/
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

/-- The equation x^2 + y^2 / (1-m) = 1 represents an ellipse with its focus on the x-axis -/
def q (m : ℝ) : Prop := 0 < 1 - m ∧ 1 - m < 1

theorem problem_solution (m : ℝ) :
  (∀ m, q m → p m) ∧ ¬(∀ m, p m → q m) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2084_208428


namespace NUMINAMATH_CALUDE_oreo_milk_purchases_l2084_208446

/-- The number of different flavors of oreos --/
def oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def milk_flavors : ℕ := 4

/-- The total number of products Alpha and Beta purchased --/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store --/
def purchase_combinations : ℕ := 2561

/-- Theorem stating the number of ways Alpha and Beta could have left the store --/
theorem oreo_milk_purchases :
  (oreo_flavors = 6) →
  (milk_flavors = 4) →
  (total_products = 4) →
  purchase_combinations = 2561 :=
by sorry

end NUMINAMATH_CALUDE_oreo_milk_purchases_l2084_208446


namespace NUMINAMATH_CALUDE_age_difference_is_twelve_l2084_208495

/-- The ages of three people A, B, and C, where C is 12 years younger than A -/
structure Ages where
  A : ℕ
  B : ℕ
  C : ℕ
  h : C = A - 12

/-- The difference between the total age of A and B and the total age of B and C -/
def ageDifference (ages : Ages) : ℕ := ages.A + ages.B - (ages.B + ages.C)

/-- Theorem stating that the age difference is always 12 years -/
theorem age_difference_is_twelve (ages : Ages) : ageDifference ages = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_twelve_l2084_208495


namespace NUMINAMATH_CALUDE_star_equation_solution_l2084_208422

/-- Custom binary operation ⭐ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 5 ⭐ x = 40, then x = 45/8 -/
theorem star_equation_solution :
  star 5 x = 40 → x = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2084_208422


namespace NUMINAMATH_CALUDE_six_seat_colorings_eq_66_l2084_208443

/-- Represents the number of ways to paint n seats in a circular arrangement
    with the first seat fixed as red, using three colors (red, blue, green)
    such that adjacent seats have different colors. -/
def S : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 2
| 3 => 2
| (n + 2) => S (n + 1) + 2 * S n

/-- The number of ways to paint six seats in a circular arrangement
    using three colors (red, blue, green) such that adjacent seats
    have different colors. -/
def six_seat_colorings : ℕ := 3 * S 6

theorem six_seat_colorings_eq_66 : six_seat_colorings = 66 := by
  sorry

end NUMINAMATH_CALUDE_six_seat_colorings_eq_66_l2084_208443


namespace NUMINAMATH_CALUDE_max_gcd_lcm_value_l2084_208452

theorem max_gcd_lcm_value (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) : 
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a' b' c' : ℕ), Nat.gcd (Nat.lcm a' b') c' = 10 ∧ 
    Nat.gcd (Nat.lcm a' b') c' * Nat.lcm (Nat.gcd a' b') c' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_value_l2084_208452


namespace NUMINAMATH_CALUDE_carrot_count_l2084_208403

theorem carrot_count (initial picked_later thrown_out : ℕ) :
  initial ≥ thrown_out →
  initial - thrown_out + picked_later = initial + picked_later - thrown_out :=
by sorry

end NUMINAMATH_CALUDE_carrot_count_l2084_208403


namespace NUMINAMATH_CALUDE_car_speed_proof_l2084_208433

/-- Proves that a car traveling at speed v km/h takes 2 seconds longer to travel 1 kilometer
    than it would at 225 km/h if and only if v = 200 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 225 * 3600 + 2) ↔ v = 200 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2084_208433


namespace NUMINAMATH_CALUDE_divisibility_in_chosen_numbers_l2084_208429

theorem divisibility_in_chosen_numbers (n : ℕ+) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2*n + 1) → S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ b % a = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_chosen_numbers_l2084_208429


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2084_208437

/-- Given a rational function decomposition, prove the value of B -/
theorem partial_fraction_decomposition (x A B C : ℝ) : 
  (2 : ℝ) / (x^3 + 5*x^2 - 13*x - 35) = A / (x-7) + B / (x+1) + C / (x+1)^2 →
  x^3 + 5*x^2 - 13*x - 35 = (x-7)*(x+1)^2 →
  B = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2084_208437


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2084_208448

theorem cos_squared_alpha_minus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l2084_208448


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l2084_208436

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 840 →
  absent_children = 420 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l2084_208436


namespace NUMINAMATH_CALUDE_two_rolls_probability_l2084_208415

/-- A fair six-sided die --/
def FairDie := Fin 6

/-- The probability of rolling a specific number on a fair die --/
def prob_single_roll : ℚ := 1 / 6

/-- The sum of two die rolls --/
def sum_of_rolls (a b : FairDie) : ℕ := a.val + b.val + 2

/-- Whether a number is prime --/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The probability that the sum of two rolls is prime --/
def prob_sum_is_prime : ℚ := 15 / 36

theorem two_rolls_probability (rolls : ℕ) : 
  (rolls = 2 ∧ prob_sum_is_prime = 0.41666666666666663) → rolls = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_rolls_probability_l2084_208415


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l2084_208447

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.log (Real.sqrt (x^2 + 1) + x)) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧
                (M + N = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l2084_208447


namespace NUMINAMATH_CALUDE_rogers_allowance_theorem_l2084_208401

/-- Roger's weekly allowance problem -/
theorem rogers_allowance_theorem (B : ℝ) (m s p : ℝ) : 
  (m = (1/4) * (B - s)) → 
  (s = (1/10) * (B - m)) → 
  (p = (1/10) * (m + s)) → 
  (m + s + p) / B = 22 / 65 := by
  sorry

end NUMINAMATH_CALUDE_rogers_allowance_theorem_l2084_208401


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2084_208419

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpLine a b) 
  (h2 : perpPlane a α) : 
  subset b α ∨ parallel b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2084_208419


namespace NUMINAMATH_CALUDE_complex_number_equality_l2084_208459

theorem complex_number_equality : (1 + Complex.I)^2 * (1 - Complex.I) = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2084_208459


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2084_208492

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem book_selection_theorem :
  let biology_ways := choose 10 3
  let chemistry_ways := choose 8 2
  let physics_ways := choose 5 1
  biology_ways * chemistry_ways * physics_ways = 16800 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2084_208492


namespace NUMINAMATH_CALUDE_equation_solution_l2084_208458

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 2 * (1 / (2 - x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2084_208458


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2084_208457

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2*x^2 - 4*x - 1 = 0) ∧
  (∃ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0) ∧
  (∀ x : ℝ, 2*x^2 - 4*x - 1 = 0 → x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) ∧
  (∀ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0 → x = 1 ∨ x = 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2084_208457


namespace NUMINAMATH_CALUDE_probability_even_sum_le_8_l2084_208481

def dice_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 12

theorem probability_even_sum_le_8 : 
  (favorable_outcomes : ℚ) / dice_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_even_sum_le_8_l2084_208481


namespace NUMINAMATH_CALUDE_number_of_best_friends_l2084_208413

theorem number_of_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) 
  (h1 : total_cards = 455) 
  (h2 : cards_per_friend = 91) : 
  total_cards / cards_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_best_friends_l2084_208413


namespace NUMINAMATH_CALUDE_c_equals_zero_l2084_208488

theorem c_equals_zero (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a*b + b - 9) : c = 0 := by
  sorry

end NUMINAMATH_CALUDE_c_equals_zero_l2084_208488


namespace NUMINAMATH_CALUDE_min_sum_squares_l2084_208477

/-- A random variable with normal distribution N(1, σ²) -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than or equal to a -/
def P_le (σ : ℝ) (X : X σ) (a : ℝ) : ℝ := sorry

/-- The probability that X is greater than or equal to b -/
def P_ge (σ : ℝ) (X : X σ) (b : ℝ) : ℝ := sorry

/-- The theorem stating that the minimum value of a² + b² is 2 -/
theorem min_sum_squares (σ : ℝ) (X : X σ) (a b : ℝ) 
  (h : P_le σ X a = P_ge σ X b) : 
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), P_le σ X x = P_ge σ X y → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2084_208477


namespace NUMINAMATH_CALUDE_lei_lei_sheep_count_l2084_208462

/-- The number of sheep Lei Lei bought -/
def num_sheep : ℕ := 10

/-- The initial average price per sheep in yuan -/
def initial_avg_price : ℚ := sorry

/-- The total price of all sheep and goats -/
def total_price : ℚ := sorry

/-- The number of goats Lei Lei bought -/
def num_goats : ℕ := sorry

theorem lei_lei_sheep_count :
  (total_price + 2 * (initial_avg_price + 60) = (num_sheep + 2) * (initial_avg_price + 60)) ∧
  (total_price - 2 * (initial_avg_price - 90) = (num_sheep - 2) * (initial_avg_price - 90)) →
  num_sheep = 10 := by sorry

end NUMINAMATH_CALUDE_lei_lei_sheep_count_l2084_208462


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2084_208435

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2084_208435


namespace NUMINAMATH_CALUDE_triangle_properties_l2084_208466

/-- Triangle ABC with vertices A(5,1), B(1,3), and C(4,4) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from AB in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 4 = 0

/-- The circumcircle of triangle ABC -/
def circumcircle (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => (p.1 - 3)^2 + (p.2 - 2)^2 = 5

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (5, 1)) 
  (h2 : t.B = (1, 3)) 
  (h3 : t.C = (4, 4)) : 
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 4 = 0) ∧ 
  (∀ p, circumcircle t p ↔ (p.1 - 3)^2 + (p.2 - 2)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2084_208466


namespace NUMINAMATH_CALUDE_leigh_has_16_shells_l2084_208408

-- Define the number of seashells each person has
def mimi_shells : ℕ := 24  -- 2 dozen seashells
def kyle_shells : ℕ := 2 * mimi_shells
def leigh_shells : ℕ := kyle_shells / 3

-- Theorem to prove
theorem leigh_has_16_shells : leigh_shells = 16 := by
  sorry

end NUMINAMATH_CALUDE_leigh_has_16_shells_l2084_208408


namespace NUMINAMATH_CALUDE_mn_squared_equals_half_sum_l2084_208496

/-- Represents a quadrilateral ABCD with a segment MN parallel to CD -/
structure QuadrilateralWithSegment where
  /-- Length of segment from A parallel to CD intersecting BC -/
  a : ℝ
  /-- Length of segment from B parallel to CD intersecting AD -/
  b : ℝ
  /-- Length of CD -/
  c : ℝ
  /-- Length of MN -/
  mn : ℝ
  /-- MN is parallel to CD -/
  mn_parallel_cd : True
  /-- M lies on BC and N lies on AD -/
  m_on_bc_n_on_ad : True
  /-- MN divides the quadrilateral ABCD into two equal areas -/
  mn_divides_equally : True

/-- Theorem stating the relationship between MN, a, b, and c -/
theorem mn_squared_equals_half_sum (q : QuadrilateralWithSegment) :
  q.mn ^ 2 = (q.a * q.b + q.c ^ 2) / 2 := by sorry

end NUMINAMATH_CALUDE_mn_squared_equals_half_sum_l2084_208496


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l2084_208486

/-- The volume of a cylindrical swimming pool -/
theorem swimming_pool_volume (diameter : ℝ) (depth : ℝ) (volume : ℝ) :
  diameter = 16 →
  depth = 4 →
  volume = π * (diameter / 2)^2 * depth →
  volume = 256 * π := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l2084_208486
