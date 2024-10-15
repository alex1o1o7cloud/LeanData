import Mathlib

namespace NUMINAMATH_CALUDE_cyclist_wait_time_correct_l802_80244

/-- The time (in minutes) the cyclist stops to wait after passing the hiker -/
def cyclist_wait_time : ℝ := 3.6667

/-- The hiker's speed in miles per hour -/
def hiker_speed : ℝ := 4

/-- The cyclist's speed in miles per hour -/
def cyclist_speed : ℝ := 15

/-- The time (in minutes) the cyclist waits for the hiker to catch up -/
def catch_up_time : ℝ := 13.75

theorem cyclist_wait_time_correct :
  cyclist_wait_time * (cyclist_speed / 60) = catch_up_time * (hiker_speed / 60) := by
  sorry

#check cyclist_wait_time_correct

end NUMINAMATH_CALUDE_cyclist_wait_time_correct_l802_80244


namespace NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l802_80218

theorem cube_plus_minus_one_divisible_by_seven (a : ℤ) (h : ¬ 7 ∣ a) :
  7 ∣ (a^3 + 1) ∨ 7 ∣ (a^3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l802_80218


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l802_80261

/-- The repeating decimal 0.4̄67 as a real number -/
def repeating_decimal : ℚ := 0.4 + (2/3) / 100

/-- The fraction 4621/9900 as a rational number -/
def target_fraction : ℚ := 4621 / 9900

/-- Theorem stating that the repeating decimal 0.4̄67 is equal to the fraction 4621/9900 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l802_80261


namespace NUMINAMATH_CALUDE_antoinette_weight_l802_80289

/-- Proves that Antoinette weighs 79 kilograms given the conditions of the problem -/
theorem antoinette_weight :
  ∀ (rupert antoinette charles : ℝ),
  antoinette = 2 * rupert - 7 →
  charles = (antoinette + rupert) / 2 + 5 →
  rupert + antoinette + charles = 145 →
  antoinette = 79 := by
sorry

end NUMINAMATH_CALUDE_antoinette_weight_l802_80289


namespace NUMINAMATH_CALUDE_proposition_truth_l802_80246

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 > 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

-- Theorem to prove
theorem proposition_truth : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_proposition_truth_l802_80246


namespace NUMINAMATH_CALUDE_angle_B_measure_l802_80239

-- Define the hexagon PROBLEMS
structure Hexagon where
  P : ℝ
  R : ℝ
  O : ℝ
  B : ℝ
  L : ℝ
  S : ℝ

-- Define the conditions
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.P = h.R ∧ h.P = h.B ∧ 
  h.O + h.S = 180 ∧ 
  h.L = 90 ∧
  h.P + h.R + h.O + h.B + h.L + h.S = 720

-- State the theorem
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l802_80239


namespace NUMINAMATH_CALUDE_inequality_proof_l802_80201

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a*b*c*d)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l802_80201


namespace NUMINAMATH_CALUDE_circumcircle_radius_isosceles_triangle_l802_80280

/-- Given a triangle with two sides of length a and one side of length b,
    the radius of its circumcircle is a²/√(4a² - b²). -/
theorem circumcircle_radius_isosceles_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ R : ℝ, R = a^2 / Real.sqrt (4 * a^2 - b^2) ∧ 
  R > 0 ∧ 
  R * Real.sqrt (4 * a^2 - b^2) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_isosceles_triangle_l802_80280


namespace NUMINAMATH_CALUDE_apple_purchase_l802_80222

theorem apple_purchase (cecile_apples diane_apples : ℕ) : 
  diane_apples = cecile_apples + 20 →
  cecile_apples + diane_apples = 50 →
  cecile_apples = 15 := by
sorry

end NUMINAMATH_CALUDE_apple_purchase_l802_80222


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l802_80284

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  Prime n → ∀ p, Prime p ∧ p > 31 → ¬(p ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l802_80284


namespace NUMINAMATH_CALUDE_discount_percentage_l802_80251

/-- Proves that given a cost price of 66.5, a marked price of 87.5, and a profit of 25% on the cost price, the percentage deducted from the list price is 5%. -/
theorem discount_percentage (cost_price : ℝ) (marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 66.5 →
  marked_price = 87.5 →
  profit_percentage = 25 →
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount_percentage := (marked_price - selling_price) / marked_price * 100
  discount_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l802_80251


namespace NUMINAMATH_CALUDE_average_difference_l802_80225

theorem average_difference (a c x : ℝ) 
  (h1 : (a + x) / 2 = 40)
  (h2 : (x + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l802_80225


namespace NUMINAMATH_CALUDE_smallest_non_negative_solution_l802_80268

theorem smallest_non_negative_solution (x : ℕ) : 
  (x + 7263 : ℤ) ≡ 3507 [ZMOD 15] ↔ x = 9 ∨ (x > 9 ∧ (x : ℤ) ≡ 9 [ZMOD 15]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_negative_solution_l802_80268


namespace NUMINAMATH_CALUDE_larger_integer_problem_l802_80228

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : y - x = 6) 
  (h2 : x * y = 135) : 
  y = 15 := by sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l802_80228


namespace NUMINAMATH_CALUDE_complex_equality_theorem_l802_80287

theorem complex_equality_theorem :
  ∃ (x : ℝ), 
    (Complex.mk (Real.sin x ^ 2) (Real.cos (2 * x)) = Complex.mk (Real.sin x ^ 2) (Real.cos x)) ∧ 
    ((Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.I) ∨ 
     (Complex.mk (Real.sin x ^ 2) (Real.cos x) = Complex.mk (3/4) (-1/2))) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_theorem_l802_80287


namespace NUMINAMATH_CALUDE_f_min_value_negative_reals_l802_80241

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem f_min_value_negative_reals 
  (a b : ℝ) 
  (h_max : ∀ x > 0, f a b x ≤ 5) :
  ∀ x < 0, f a b x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_negative_reals_l802_80241


namespace NUMINAMATH_CALUDE_system_of_inequalities_l802_80215

theorem system_of_inequalities (x : ℝ) : 
  (-3 * x^2 + 7 * x + 6 > 0 ∧ 4 * x - 4 * x^2 > -3) ↔ (-1/2 < x ∧ x < 3/2) := by
sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l802_80215


namespace NUMINAMATH_CALUDE_wall_bricks_count_l802_80249

theorem wall_bricks_count :
  -- Define the variables
  -- x: total number of bricks in the wall
  -- r1: rate of first bricklayer (bricks per hour)
  -- r2: rate of second bricklayer (bricks per hour)
  -- rc: combined rate after reduction (bricks per hour)
  ∀ (x r1 r2 rc : ℚ),
  -- Conditions
  (r1 = x / 7) →  -- First bricklayer's rate
  (r2 = x / 11) →  -- Second bricklayer's rate
  (rc = r1 + r2 - 12) →  -- Combined rate after reduction
  (6 * rc = x) →  -- Time to complete the wall after planning
  -- Conclusion
  x = 179 := by
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l802_80249


namespace NUMINAMATH_CALUDE_total_beads_needed_l802_80252

/-- The number of green beads in one pattern repeat -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern repeat -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern repeat -/
def red_beads : ℕ := 2 * green_beads

/-- The total number of beads in one pattern repeat -/
def beads_per_repeat : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repeats in one bracelet -/
def repeats_per_bracelet : ℕ := 3

/-- The number of pattern repeats in one necklace -/
def repeats_per_necklace : ℕ := 5

/-- The number of bracelets to make -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to make -/
def num_necklaces : ℕ := 10

theorem total_beads_needed : 
  beads_per_repeat * (repeats_per_bracelet * num_bracelets + 
  repeats_per_necklace * num_necklaces) = 742 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_needed_l802_80252


namespace NUMINAMATH_CALUDE_difference_61st_terms_arithmetic_sequences_l802_80264

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_61st_terms_arithmetic_sequences :
  let C := arithmetic_sequence 45 15
  let D := arithmetic_sequence 45 (-15)
  |C 61 - D 61| = 1800 := by
sorry

end NUMINAMATH_CALUDE_difference_61st_terms_arithmetic_sequences_l802_80264


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l802_80209

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < y / 17 → y ≥ 13 := by
  sorry

theorem thirteen_satisfies : (8 : ℚ) / 11 < 13 / 17 := by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < y / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < z / 17 → z ≥ y) ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l802_80209


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l802_80272

-- Define the parabola function
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem statement
theorem parabola_has_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l802_80272


namespace NUMINAMATH_CALUDE_johnnys_third_job_rate_l802_80298

/-- Given Johnny's work schedule and earnings, prove the hourly rate of his third job. -/
theorem johnnys_third_job_rate (hours_job1 hours_job2 hours_job3 : ℕ)
                               (rate_job1 rate_job2 : ℕ)
                               (days : ℕ)
                               (total_earnings : ℕ) :
  hours_job1 = 3 →
  hours_job2 = 2 →
  hours_job3 = 4 →
  rate_job1 = 7 →
  rate_job2 = 10 →
  days = 5 →
  total_earnings = 445 →
  ∃ (rate_job3 : ℕ), 
    rate_job3 = 12 ∧
    total_earnings = (hours_job1 * rate_job1 + hours_job2 * rate_job2 + hours_job3 * rate_job3) * days :=
by sorry

end NUMINAMATH_CALUDE_johnnys_third_job_rate_l802_80298


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_fermat_like_equation_l802_80285

theorem no_integer_solutions_for_fermat_like_equation (n : ℕ) (hn : n ≥ 2) :
  ¬∃ (x y z : ℤ), x^2 + y^2 = z^n := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_fermat_like_equation_l802_80285


namespace NUMINAMATH_CALUDE_estimate_passing_papers_l802_80255

theorem estimate_passing_papers (total_papers : ℕ) (sample_size : ℕ) (passing_in_sample : ℕ) 
  (h1 : total_papers = 5000)
  (h2 : sample_size = 400)
  (h3 : passing_in_sample = 360) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 4500 := by
sorry

end NUMINAMATH_CALUDE_estimate_passing_papers_l802_80255


namespace NUMINAMATH_CALUDE_class_average_problem_l802_80235

theorem class_average_problem (first_group_percentage : Real) 
                               (second_group_percentage : Real) 
                               (first_group_average : Real) 
                               (second_group_average : Real) 
                               (overall_average : Real) :
  first_group_percentage = 0.25 →
  second_group_percentage = 0.50 →
  first_group_average = 0.80 →
  second_group_average = 0.65 →
  overall_average = 0.75 →
  let remainder_percentage := 1 - first_group_percentage - second_group_percentage
  let remainder_average := (overall_average - first_group_percentage * first_group_average - 
                            second_group_percentage * second_group_average) / remainder_percentage
  remainder_average = 0.90 := by
sorry

end NUMINAMATH_CALUDE_class_average_problem_l802_80235


namespace NUMINAMATH_CALUDE_system_solution_l802_80206

theorem system_solution : ∃ (x y : ℝ), 
  (x^2 - 6 * Real.sqrt (3 - 2*x) - y + 11 = 0) ∧ 
  (y^2 - 4 * Real.sqrt (3*y - 2) + 4*x + 16 = 0) ∧
  (x = -3) ∧ (y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l802_80206


namespace NUMINAMATH_CALUDE_no_common_solution_l802_80208

theorem no_common_solution :
  ¬ ∃ x : ℝ, (8 * x^2 + 6 * x = 5) ∧ (3 * x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l802_80208


namespace NUMINAMATH_CALUDE_jason_shorts_expenditure_l802_80274

theorem jason_shorts_expenditure (total : ℝ) (jacket : ℝ) (shorts : ℝ) : 
  total = 19.02 → jacket = 4.74 → total = jacket + shorts → shorts = 14.28 := by
  sorry

end NUMINAMATH_CALUDE_jason_shorts_expenditure_l802_80274


namespace NUMINAMATH_CALUDE_contrapositive_zero_product_l802_80258

theorem contrapositive_zero_product (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_zero_product_l802_80258


namespace NUMINAMATH_CALUDE_fraction_problem_l802_80269

theorem fraction_problem (F : ℚ) : 
  3 + F * (1/3) * (1/5) * 90 = (1/15) * 90 → F = 1/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l802_80269


namespace NUMINAMATH_CALUDE_x_fifth_plus_64x_l802_80275

theorem x_fifth_plus_64x (x : ℝ) (h : x^2 + 4*x = 8) : x^5 + 64*x = 768*x - 1024 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_64x_l802_80275


namespace NUMINAMATH_CALUDE_sine_sum_equality_l802_80247

theorem sine_sum_equality (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l802_80247


namespace NUMINAMATH_CALUDE_max_intersection_points_l802_80230

/-- The maximum number of intersection points given 8 planes in 3D space -/
theorem max_intersection_points (n : ℕ) (h : n = 8) : 
  (Nat.choose n 3 : ℕ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l802_80230


namespace NUMINAMATH_CALUDE_gcd_problem_l802_80279

theorem gcd_problem (n : ℕ) : 
  70 ≤ n ∧ n ≤ 80 → Nat.gcd 15 n = 5 → n = 70 ∨ n = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l802_80279


namespace NUMINAMATH_CALUDE_constant_terms_are_like_terms_l802_80281

/-- Two algebraic terms are considered "like terms" if they have the same variables with the same exponents. -/
def like_terms (term1 term2 : String) : Prop := sorry

/-- A constant term is a number without variables. -/
def is_constant_term (term : String) : Prop := sorry

theorem constant_terms_are_like_terms (a b : String) :
  is_constant_term a ∧ is_constant_term b → like_terms a b := by sorry

end NUMINAMATH_CALUDE_constant_terms_are_like_terms_l802_80281


namespace NUMINAMATH_CALUDE_solution_pairs_l802_80223

theorem solution_pairs : 
  {p : ℕ × ℕ | let (m, n) := p; m^2 + 2 * 3^n = m * (2^(n+1) - 1)} = 
  {(9, 3), (6, 3), (9, 5), (54, 5)} :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l802_80223


namespace NUMINAMATH_CALUDE_evaluate_expression_l802_80245

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l802_80245


namespace NUMINAMATH_CALUDE_spaceship_reach_boundary_l802_80262

/-- A path in 3D space --/
structure Path3D where
  points : List (ℝ × ℝ × ℝ)

/-- Distance of a point from a plane --/
def distanceFromPlane (point : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) : ℝ :=
  sorry

/-- Length of a path --/
def pathLength (path : Path3D) : ℝ :=
  sorry

/-- Check if a path reaches the boundary plane --/
def reachesBoundary (path : Path3D) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) : Prop :=
  sorry

/-- The main theorem --/
theorem spaceship_reach_boundary (a : ℝ) (startPoint : ℝ × ℝ × ℝ) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) 
    (h : distanceFromPlane startPoint boundaryPlane = a) :
    ∃ (path : Path3D), pathLength path ≤ 14 * a ∧ reachesBoundary path boundaryPlane :=
  sorry

end NUMINAMATH_CALUDE_spaceship_reach_boundary_l802_80262


namespace NUMINAMATH_CALUDE_dividing_line_ratio_l802_80286

/-- A trapezoid with given dimensions and a dividing line -/
structure Trapezoid :=
  (base1 : ℝ)
  (base2 : ℝ)
  (leg1 : ℝ)
  (leg2 : ℝ)
  (dividing_ratio : ℝ × ℝ)

/-- The condition that the dividing line creates equal perimeters -/
def equal_perimeters (t : Trapezoid) : Prop :=
  let (m, n) := t.dividing_ratio
  let x := t.base1 + (t.base2 - t.base1) * (m / (m + n))
  t.base1 + m + x + t.leg1 * (m / (m + n)) =
  t.base2 + n + x + t.leg1 * (n / (m + n))

/-- The theorem stating the ratio of the dividing line -/
theorem dividing_line_ratio (t : Trapezoid) 
    (h1 : t.base1 = 3) 
    (h2 : t.base2 = 9) 
    (h3 : t.leg1 = 4) 
    (h4 : t.leg2 = 6) 
    (h5 : equal_perimeters t) : 
    t.dividing_ratio = (4, 1) := by
  sorry


end NUMINAMATH_CALUDE_dividing_line_ratio_l802_80286


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_lines_l802_80237

-- Define the original lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define the distance between parallel lines
def distance : ℝ := 7

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (a b c : ℝ), (∀ x y, a * x + b * y + c = 0 ↔ 3 * x - y + 3 = 0) ∧
  (∀ x y, line1 x y → (a * x + b * y + c = 0 → a * 3 + b = 0)) ∧
  (a * point_P.1 + b * point_P.2 + c = 0) :=
sorry

-- Theorem for the parallel lines
theorem parallel_lines :
  ∃ (c1 c2 : ℝ), 
  (∀ x y, 3 * x + 4 * y + c1 = 0 ∨ 3 * x + 4 * y + c2 = 0 ↔ 
    (3 * x + 4 * y + 23 = 0 ∨ 3 * x + 4 * y - 47 = 0)) ∧
  (∀ x y, line2 x y → 
    (|c1 + 12| / Real.sqrt 25 = distance ∧ |c2 + 12| / Real.sqrt 25 = distance)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_lines_l802_80237


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l802_80232

theorem jelly_bean_ratio : 
  let napoleon_beans : ℕ := 17
  let sedrich_beans : ℕ := napoleon_beans + 4
  let mikey_beans : ℕ := 19
  let total_beans : ℕ := napoleon_beans + sedrich_beans
  2 * total_beans = 4 * mikey_beans := by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l802_80232


namespace NUMINAMATH_CALUDE_common_point_theorem_l802_80257

/-- Represents a line with equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the coefficients of a line form an arithmetic progression with common difference a/2 -/
def Line.isArithmeticProgression (l : Line) : Prop :=
  l.b = l.a + l.a/2 ∧ l.c = l.a + 2*(l.a/2)

/-- Checks if a point (x, y) lies on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

theorem common_point_theorem :
  ∀ l : Line, l.isArithmeticProgression → l.containsPoint 0 (4/3) :=
sorry

end NUMINAMATH_CALUDE_common_point_theorem_l802_80257


namespace NUMINAMATH_CALUDE_square_root_division_l802_80216

theorem square_root_division (x : ℝ) : (Real.sqrt 1936) / x = 4 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l802_80216


namespace NUMINAMATH_CALUDE_wall_building_time_l802_80253

/-- Represents the time taken to build a wall given the number of workers -/
def build_time (workers : ℕ) : ℝ :=
  sorry

/-- The number of workers in the initial scenario -/
def initial_workers : ℕ := 20

/-- The number of days taken in the initial scenario -/
def initial_days : ℝ := 6

/-- The number of workers in the new scenario -/
def new_workers : ℕ := 30

theorem wall_building_time :
  (build_time initial_workers = initial_days) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  (build_time new_workers = 4.0) :=
sorry

end NUMINAMATH_CALUDE_wall_building_time_l802_80253


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_24_l802_80200

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem smallest_three_digit_with_digit_product_24 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 24 → 146 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_24_l802_80200


namespace NUMINAMATH_CALUDE_system_solution_l802_80277

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 10 - 4*x)
  (eq2 : x + z = -16 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  3*x + 3*y + 3*z = 1.5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l802_80277


namespace NUMINAMATH_CALUDE_athlete_seating_arrangements_l802_80221

def number_of_arrangements (n : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial n) * (team_sizes.map Nat.factorial).prod

theorem athlete_seating_arrangements :
  number_of_arrangements 4 [4, 3, 2, 3] = 20736 := by
  sorry

end NUMINAMATH_CALUDE_athlete_seating_arrangements_l802_80221


namespace NUMINAMATH_CALUDE_lucy_apples_per_week_l802_80292

/-- Given the following conditions:
  - Chandler eats 23 apples per week
  - They order 168 apples per month
  - There are 4 weeks in a month
  Prove that Lucy can eat 19 apples per week. -/
theorem lucy_apples_per_week :
  ∀ (chandler_apples_per_week : ℕ) 
    (total_apples_per_month : ℕ) 
    (weeks_per_month : ℕ),
  chandler_apples_per_week = 23 →
  total_apples_per_month = 168 →
  weeks_per_month = 4 →
  ∃ (lucy_apples_per_week : ℕ),
    lucy_apples_per_week = 19 ∧
    lucy_apples_per_week * weeks_per_month + 
    chandler_apples_per_week * weeks_per_month = 
    total_apples_per_month :=
by sorry

end NUMINAMATH_CALUDE_lucy_apples_per_week_l802_80292


namespace NUMINAMATH_CALUDE_sum_distances_constant_l802_80224

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- Add necessary fields for a regular tetrahedron

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance from a point to a plane in 3D space -/
def distanceToPlane (p : Point3D) (plane : Set Point3D) : ℝ :=
  sorry

/-- Predicate to check if a point is inside a regular tetrahedron -/
def isInside (p : Point3D) (t : RegularTetrahedron) : Prop :=
  sorry

/-- The faces of a regular tetrahedron -/
def faces (t : RegularTetrahedron) : Finset (Set Point3D) :=
  sorry

/-- Theorem: The sum of distances from any point inside a regular tetrahedron to all its faces is constant -/
theorem sum_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : Point3D, isInside p t →
    (faces t).sum (λ face => distanceToPlane p face) = c :=
  sorry

end NUMINAMATH_CALUDE_sum_distances_constant_l802_80224


namespace NUMINAMATH_CALUDE_correct_equation_l802_80273

/-- Represents the situation described in the problem -/
structure Situation where
  x : ℕ  -- number of people
  total_cost : ℕ  -- total cost of the item

/-- The condition when each person contributes 8 coins -/
def condition_8 (s : Situation) : Prop :=
  8 * s.x = s.total_cost + 3

/-- The condition when each person contributes 7 coins -/
def condition_7 (s : Situation) : Prop :=
  7 * s.x + 4 = s.total_cost

/-- The theorem stating that the equation 8x - 3 = 7x + 4 correctly represents the situation -/
theorem correct_equation (s : Situation) :
  condition_8 s ∧ condition_7 s ↔ 8 * s.x - 3 = 7 * s.x + 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l802_80273


namespace NUMINAMATH_CALUDE_least_froods_for_more_points_l802_80202

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n Froods -/
def eating_points (n : ℕ) : ℕ := 5 * n

/-- Proposition: 10 is the least positive integer n for which 
    dropping n Froods earns more points than eating them -/
theorem least_froods_for_more_points : 
  ∀ n : ℕ, n > 0 → (
    (n < 10 → sum_first_n n ≤ eating_points n) ∧
    (sum_first_n 10 > eating_points 10)
  ) := by sorry

end NUMINAMATH_CALUDE_least_froods_for_more_points_l802_80202


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l802_80242

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 11 * k) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (∃ j : ℕ, m = 11 * j) → n ≤ m) ∧
  n = 110 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l802_80242


namespace NUMINAMATH_CALUDE_trigonometric_identity_l802_80276

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (2 * c) * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (5 * c)) =
  1 / Real.sin (2 * Real.pi / 13) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l802_80276


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l802_80266

theorem partial_fraction_decomposition (x : ℝ) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (x^2 - 10*x + 16) / ((x - 2) * (x - 3) * (x - 4)) =
  2 / (x - 2) + 5 / (x - 3) + 0 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l802_80266


namespace NUMINAMATH_CALUDE_total_pancakes_l802_80260

/-- Represents the number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- Represents the number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- Represents the number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- Represents the number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- Theorem stating the total number of pancakes Hank needs to make -/
theorem total_pancakes : 
  short_stack_orders * short_stack + big_stack_orders * big_stack = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_pancakes_l802_80260


namespace NUMINAMATH_CALUDE_arthur_hamburgers_l802_80236

/-- The number of hamburgers Arthur bought on the first day -/
def hamburgers_day1 : ℕ := 1

/-- The price of a hamburger in dollars -/
def hamburger_price : ℚ := 6

/-- The price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Total cost of Arthur's purchase on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Total cost of Arthur's purchase on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

/-- Number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

theorem arthur_hamburgers :
  (hamburgers_day1 : ℚ) * hamburger_price + (hotdogs_day1 : ℚ) * hotdog_price = total_cost_day1 ∧
  (hamburgers_day2 : ℚ) * hamburger_price + (hotdogs_day2 : ℚ) * hotdog_price = total_cost_day2 :=
sorry

end NUMINAMATH_CALUDE_arthur_hamburgers_l802_80236


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l802_80210

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : total_students > 0)
  (students_more_than_100 : ℕ) 
  (h2 : students_more_than_100 = (18 * total_students) / 100)
  (h3 : (75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100 = 
        (72 * total_students) / 100) :
  (72 * total_students) / 100 = total_students - 
    ((75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100) :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l802_80210


namespace NUMINAMATH_CALUDE_sector_area_sexagesimal_l802_80220

/-- The area of a sector with radius 4 and central angle 625/6000 of a full circle is 5π/3 -/
theorem sector_area_sexagesimal (π : ℝ) (h : π > 0) : 
  let r : ℝ := 4
  let angle_fraction : ℝ := 625 / 6000
  let sector_area := (1/2) * (angle_fraction * 2 * π) * r^2
  sector_area = 5 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_area_sexagesimal_l802_80220


namespace NUMINAMATH_CALUDE_total_watching_time_l802_80263

-- Define the TV series data
def pride_and_prejudice_episodes : ℕ := 6
def pride_and_prejudice_duration : ℕ := 50
def breaking_bad_episodes : ℕ := 62
def breaking_bad_duration : ℕ := 47
def stranger_things_episodes : ℕ := 33
def stranger_things_duration : ℕ := 51

-- Calculate total watching time in minutes
def total_minutes : ℕ := 
  pride_and_prejudice_episodes * pride_and_prejudice_duration +
  breaking_bad_episodes * breaking_bad_duration +
  stranger_things_episodes * stranger_things_duration

-- Convert minutes to hours and round to nearest whole number
def total_hours : ℕ := (total_minutes + 30) / 60

-- Theorem to prove
theorem total_watching_time : total_hours = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_watching_time_l802_80263


namespace NUMINAMATH_CALUDE_inequality_proof_l802_80233

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2)) + (y / (x^2 + y^4)) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l802_80233


namespace NUMINAMATH_CALUDE_carla_games_won_l802_80297

theorem carla_games_won (total_games : ℕ) (frankie_games : ℕ) (carla_games : ℕ) : 
  total_games = 30 →
  frankie_games = carla_games / 2 →
  frankie_games + carla_games = total_games →
  carla_games = 20 := by
  sorry

end NUMINAMATH_CALUDE_carla_games_won_l802_80297


namespace NUMINAMATH_CALUDE_grid_polygon_segment_sums_equal_l802_80256

-- Define a type for grid points
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a type for polygons on a grid
structure GridPolygon where
  vertices : List GridPoint
  convex : Bool
  verticesOnGrid : Bool
  sidesNotAligned : Bool

-- Define a function to calculate the sum of vertical segment lengths
def sumVerticalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Define a function to calculate the sum of horizontal segment lengths
def sumHorizontalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem grid_polygon_segment_sums_equal (p : GridPolygon) :
  p.convex ∧ p.verticesOnGrid ∧ p.sidesNotAligned →
  sumVerticalSegments p = sumHorizontalSegments p :=
sorry

end NUMINAMATH_CALUDE_grid_polygon_segment_sums_equal_l802_80256


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l802_80291

theorem units_digit_of_sum (a b : ℕ) : (24^4 + 42^4) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l802_80291


namespace NUMINAMATH_CALUDE_min_value_theorem_l802_80227

/-- Given a function y = a^x + b where b > 0 and the graph passes through point (1,3),
    the minimum value of 4/(a-1) + 1/b is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a^1 + b = 3) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 → 4/(x-1) + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 ∧ 4/(x-1) + 1/y = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l802_80227


namespace NUMINAMATH_CALUDE_middle_three_average_l802_80299

theorem middle_three_average (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five different positive integers
  (a + b + c + d + e) / 5 = 5 →  -- average is 5
  e - a = 14 →  -- maximum possible difference
  (b + c + d) / 3 = 3 := by  -- average of middle three is 3
sorry

end NUMINAMATH_CALUDE_middle_three_average_l802_80299


namespace NUMINAMATH_CALUDE_shaded_area_of_circle_with_rectangles_l802_80250

/-- The shaded area of a circle with two inscribed rectangles -/
theorem shaded_area_of_circle_with_rectangles :
  let rectangle_width : ℝ := 10
  let rectangle_length : ℝ := 24
  let overlap_side : ℝ := 10
  let circle_radius : ℝ := (rectangle_width ^ 2 + rectangle_length ^ 2).sqrt / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let total_rectangle_area : ℝ := 2 * rectangle_area
  let overlap_area : ℝ := overlap_side ^ 2
  circle_area - total_rectangle_area + overlap_area = 169 * π - 380 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_circle_with_rectangles_l802_80250


namespace NUMINAMATH_CALUDE_cube_edge_is_60_l802_80282

-- Define the volume of the rectangular cuboid-shaped cabinet
def cuboid_volume : ℝ := 420000

-- Define the volume difference between the cabinets
def volume_difference : ℝ := 204000

-- Define the volume of the cube-shaped cabinet
def cube_volume : ℝ := cuboid_volume - volume_difference

-- Define the function to calculate the edge length of a cube given its volume
def cube_edge_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Theorem statement
theorem cube_edge_is_60 :
  cube_edge_length cube_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_is_60_l802_80282


namespace NUMINAMATH_CALUDE_bank_teller_bills_count_l802_80207

theorem bank_teller_bills_count :
  ∀ (num_20_dollar_bills : ℕ),
  (20 * 5 + num_20_dollar_bills * 20 = 780) →
  (20 + num_20_dollar_bills = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_bank_teller_bills_count_l802_80207


namespace NUMINAMATH_CALUDE_school_parade_l802_80265

theorem school_parade (a b : ℕ+) : 
  ∃ k : ℕ, a.val * b.val * (a.val^2 - b.val^2) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_school_parade_l802_80265


namespace NUMINAMATH_CALUDE_proportion_equality_l802_80278

theorem proportion_equality (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x * y ≠ 0) : x / y = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l802_80278


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l802_80288

/-- Calculates the simple interest given principal, time, and rate. -/
def simpleInterest (principal : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  principal * time * rate / 100

theorem interest_rate_calculation (loanB_principal loanC_principal totalInterest : ℚ)
  (loanB_time loanC_time : ℚ) :
  loanB_principal = 5000 →
  loanC_principal = 3000 →
  loanB_time = 2 →
  loanC_time = 4 →
  totalInterest = 3300 →
  ∃ rate : ℚ, 
    simpleInterest loanB_principal loanB_time rate +
    simpleInterest loanC_principal loanC_time rate = totalInterest ∧
    rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l802_80288


namespace NUMINAMATH_CALUDE_fourth_root_16_times_fifth_root_32_l802_80283

theorem fourth_root_16_times_fifth_root_32 : (16 : ℝ) ^ (1/4) * (32 : ℝ) ^ (1/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_fifth_root_32_l802_80283


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l802_80290

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 136 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l802_80290


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l802_80212

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k ∧ Nat.Prime 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l802_80212


namespace NUMINAMATH_CALUDE_lowest_temperature_record_l802_80214

/-- The lowest temperature ever recorded in the world -/
def lowest_temperature : ℝ := -89.2

/-- The location where the lowest temperature was recorded -/
def record_location : String := "Vostok Station, Antarctica"

/-- How the temperature is written -/
def temperature_written : String := "-89.2 °C"

/-- How the temperature is read -/
def temperature_read : String := "negative eighty-nine point two degrees Celsius"

/-- Theorem stating the lowest recorded temperature and its representation -/
theorem lowest_temperature_record :
  lowest_temperature = -89.2 ∧
  record_location = "Vostok Station, Antarctica" ∧
  temperature_written = "-89.2 °C" ∧
  temperature_read = "negative eighty-nine point two degrees Celsius" :=
by sorry

end NUMINAMATH_CALUDE_lowest_temperature_record_l802_80214


namespace NUMINAMATH_CALUDE_matrix_transformation_proof_l802_80213

theorem matrix_transformation_proof : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (a b c d : ℝ),
    N * !![a, b; c, d] = !![3*a, b; 3*c, d] :=
by
  sorry

end NUMINAMATH_CALUDE_matrix_transformation_proof_l802_80213


namespace NUMINAMATH_CALUDE_additional_cars_needed_danica_car_arrangement_l802_80248

theorem additional_cars_needed (initial_cars : ℕ) (cars_per_row : ℕ) : ℕ :=
  cars_per_row - (initial_cars % cars_per_row) % cars_per_row

theorem danica_car_arrangement : additional_cars_needed 39 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_danica_car_arrangement_l802_80248


namespace NUMINAMATH_CALUDE_nancy_crayons_l802_80294

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayons_l802_80294


namespace NUMINAMATH_CALUDE_female_managers_count_l802_80231

theorem female_managers_count (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : male_associates = 160) :
  total_managers = total_employees - female_employees - male_associates :=
by
  sorry

end NUMINAMATH_CALUDE_female_managers_count_l802_80231


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l802_80229

theorem unique_n_satisfying_conditions :
  ∃! n : ℤ,
    0 ≤ n ∧ n ≤ 8 ∧
    ∃ x : ℤ,
      x > 0 ∧
      (-4567 + x ≥ 0) ∧
      (∀ y : ℤ, y > 0 ∧ -4567 + y ≥ 0 → x ≤ y) ∧
      n ≡ -4567 + x [ZMOD 9] ∧
    n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l802_80229


namespace NUMINAMATH_CALUDE_special_triangle_area_l802_80211

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- The height of the triangle -/
  height : ℝ
  /-- The smaller part of the base -/
  small_base : ℝ
  /-- The ratio of the divided angle -/
  angle_ratio : ℝ
  /-- The height divides the angle in the given ratio -/
  height_divides_angle : angle_ratio = 2
  /-- The height is 2 cm -/
  height_is_two : height = 2
  /-- The smaller part of the base is 1 cm -/
  small_base_is_one : small_base = 1

/-- The theorem stating the area of the special triangle -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1 / 2 : ℝ) * t.height * (t.small_base + 5 / 3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l802_80211


namespace NUMINAMATH_CALUDE_five_to_fifth_sum_five_times_l802_80203

theorem five_to_fifth_sum_five_times (n : ℕ) : 5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_five_to_fifth_sum_five_times_l802_80203


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_smiths_bakery_pies_proof_l802_80270

theorem smiths_bakery_pies : ℕ → ℕ → Prop :=
  fun mcgees_pies smiths_pies =>
    mcgees_pies = 16 →
    smiths_pies = mcgees_pies^2 + mcgees_pies^2 / 2 →
    smiths_pies = 384

-- The proof would go here, but we're skipping it as requested
theorem smiths_bakery_pies_proof : smiths_bakery_pies 16 384 := by
  sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_smiths_bakery_pies_proof_l802_80270


namespace NUMINAMATH_CALUDE_tangent_property_of_sine_equation_l802_80296

theorem tangent_property_of_sine_equation (k : ℝ) (α β : ℝ) :
  (∃ (k : ℝ), k > 0 ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 Real.pi → (|Real.sin x| / x = k ↔ x = α ∨ x = β)) ∧
    α ∈ Set.Ioo 0 Real.pi ∧
    β ∈ Set.Ioo 0 Real.pi ∧
    α < β) →
  Real.tan (β + Real.pi / 4) = (1 + β) / (1 - β) :=
by sorry

end NUMINAMATH_CALUDE_tangent_property_of_sine_equation_l802_80296


namespace NUMINAMATH_CALUDE_zero_not_in_positive_integers_l802_80293

theorem zero_not_in_positive_integers : 0 ∉ {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_positive_integers_l802_80293


namespace NUMINAMATH_CALUDE_a2_value_l802_80226

def sequence_sum (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem a2_value (k : ℕ) (h1 : k > 0) 
  (h2 : ∃ (n : ℕ), ∀ (m : ℕ), sequence_sum m k ≤ sequence_sum n k)
  (h3 : ∃ (n : ℕ), sequence_sum n k = 8) :
  sequence_sum 2 k - sequence_sum 1 k = 5/2 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l802_80226


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l802_80240

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 → 
  (a = b) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l802_80240


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l802_80254

-- Define the quadratic equation
def quadratic (x k : ℝ) : Prop := x^2 - 3*x + k = 0

-- Define the condition on the roots
def root_condition (x₁ x₂ : ℝ) : Prop := x₁*x₂ + 2*x₁ + 2*x₂ = 1

-- Theorem statement
theorem quadratic_root_sum_product 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : quadratic x₁ k) 
  (h2 : quadratic x₂ k) 
  (h3 : x₁ ≠ x₂) 
  (h4 : root_condition x₁ x₂) : 
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l802_80254


namespace NUMINAMATH_CALUDE_danny_found_eighteen_caps_l802_80259

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initial : ℕ) (total : ℕ) : ℕ := total - initial

/-- Theorem: Danny found 18 bottle caps at the park -/
theorem danny_found_eighteen_caps : 
  let initial := 37
  let total := 55
  bottleCapsFound initial total = 18 := by
sorry

end NUMINAMATH_CALUDE_danny_found_eighteen_caps_l802_80259


namespace NUMINAMATH_CALUDE_nested_squares_difference_l802_80271

/-- Given four nested squares with side lengths S₁ > S₂ > S₃ > S₄,
    where the differences between consecutive square side lengths are 11, 5, and 13 (from largest to smallest),
    prove that S₁ - S₄ = 29. -/
theorem nested_squares_difference (S₁ S₂ S₃ S₄ : ℝ) 
  (h₁ : S₁ = S₂ + 11)
  (h₂ : S₂ = S₃ + 5)
  (h₃ : S₃ = S₄ + 13) :
  S₁ - S₄ = 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_squares_difference_l802_80271


namespace NUMINAMATH_CALUDE_boys_percentage_l802_80205

theorem boys_percentage (total : ℕ) (boys : ℕ) (girls : ℕ) (additional_boys : ℕ) : 
  total = 50 →
  boys + girls = total →
  additional_boys = 50 →
  girls = (total + additional_boys) / 20 →
  (boys : ℚ) / total = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_boys_percentage_l802_80205


namespace NUMINAMATH_CALUDE_smallest_number_l802_80204

theorem smallest_number (a b c d : ℚ) 
  (ha : a = 0) 
  (hb : b = -3) 
  (hc : c = 1/3) 
  (hd : d = 1) : 
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l802_80204


namespace NUMINAMATH_CALUDE_sum_of_roots_l802_80219

/-- Given a quadratic function f(x) = x^2 - 2016x + 2015 and two distinct points a and b
    where f(a) = f(b), prove that a + b = 2016 -/
theorem sum_of_roots (a b : ℝ) (ha : a ≠ b) :
  (a^2 - 2016*a + 2015 = b^2 - 2016*b + 2015) →
  a + b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l802_80219


namespace NUMINAMATH_CALUDE_expression_equals_one_l802_80234

theorem expression_equals_one 
  (m n k : ℝ) 
  (h : m = 1 / (n * k)) : 
  1 / (1 + m + m * n) + 1 / (1 + n + n * k) + 1 / (1 + k + k * m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l802_80234


namespace NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l802_80243

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ)
  (h_length : field_length = 75)
  (h_width : field_width = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_unit : cost_per_unit = 2) :
  let area := path_area field_length field_width path_width
  let cost := construction_cost area cost_per_unit
  area = 675 ∧ cost = 1350 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l802_80243


namespace NUMINAMATH_CALUDE_sausage_problem_l802_80217

/-- Calculates the total pounds of spicy meat mix used to make sausages -/
def total_meat_mix (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) : ℚ :=
  let remaining_links := initial_links - eaten_links
  let ounces_per_link := remaining_ounces / remaining_links
  let total_ounces := initial_links * ounces_per_link
  total_ounces / 16

/-- Theorem stating that given the conditions, the total meat mix used was 10 pounds -/
theorem sausage_problem (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) 
  (h1 : initial_links = 40)
  (h2 : eaten_links = 12)
  (h3 : remaining_ounces = 112) :
  total_meat_mix initial_links eaten_links remaining_ounces = 10 := by
  sorry

#eval total_meat_mix 40 12 112

end NUMINAMATH_CALUDE_sausage_problem_l802_80217


namespace NUMINAMATH_CALUDE_probability_white_ball_l802_80267

/-- The probability of drawing a white ball from a box with red and white balls -/
theorem probability_white_ball (red_balls white_balls : ℕ) :
  red_balls = 5 →
  white_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_ball_l802_80267


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l802_80295

theorem smallest_number_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧
  (n % 13 = 12) ∧
  (n % 11 = 10) ∧
  (n % 7 = 6) ∧
  (n % 5 = 4) ∧
  (n % 3 = 2) ∧
  (∀ m : ℕ, m > 0 → 
    (m % 13 = 12) ∧
    (m % 11 = 10) ∧
    (m % 7 = 6) ∧
    (m % 5 = 4) ∧
    (m % 3 = 2) → 
    n ≤ m) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l802_80295


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l802_80238

/-- The speed of a boat in still water, given its downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 45)
  (h3 : downstream_time = 1)
  : ∃ (boat_speed : ℝ), boat_speed = 40 ∧ 
    downstream_distance = downstream_time * (boat_speed + stream_speed) :=
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l802_80238
