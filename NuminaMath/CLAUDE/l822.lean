import Mathlib

namespace NUMINAMATH_CALUDE_tower_remainder_l822_82231

/-- Represents the number of towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 1 => if n ≥ 2 then 3 * T n else 2 * T n

/-- The main theorem stating the remainder when T(9) is divided by 500 -/
theorem tower_remainder : T 9 % 500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_tower_remainder_l822_82231


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l822_82232

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 3 / 7 < 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l822_82232


namespace NUMINAMATH_CALUDE_board_numbers_count_l822_82240

theorem board_numbers_count (M : ℝ) : ∃! k : ℕ,
  k > 0 ∧
  (∃ S : ℝ, M = S / k) ∧
  (S + 15) / (k + 1) = M + 2 ∧
  (S + 16) / (k + 2) = M + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_board_numbers_count_l822_82240


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_eq_21_l822_82225

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 9
  side_b : b = 12
  side_c : c = 15

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ :=
  t.a + t.b

theorem sum_longest_altitudes_eq_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_eq_21_l822_82225


namespace NUMINAMATH_CALUDE_rectangle_length_l822_82241

/-- Given a rectangle with width 5 feet and perimeter 22 feet, prove that its length is 6 feet. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 5 → perimeter = 22 → 2 * (length + width) = perimeter → length = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l822_82241


namespace NUMINAMATH_CALUDE_triangle_parallel_ratio_bounds_l822_82229

/-- Given a triangle ABC with sides a, b, c and an interior point O, 
    the ratios formed by lines through O parallel to the sides satisfy
    the given inequalities. -/
theorem triangle_parallel_ratio_bounds 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ)
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (h_sum : a' / a + b' / b + c' / c = 3) :
  (max (a' / a) (max (b' / b) (c' / c)) ≥ 2 / 3) ∧ 
  (min (a' / a) (min (b' / b) (c' / c)) ≤ 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_parallel_ratio_bounds_l822_82229


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l822_82275

theorem multiplication_mistake_difference : 
  let correct_number : ℕ := 139
  let correct_multiplier : ℕ := 43
  let mistaken_multiplier : ℕ := 34
  (correct_number * correct_multiplier) - (correct_number * mistaken_multiplier) = 1251 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l822_82275


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l822_82261

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l822_82261


namespace NUMINAMATH_CALUDE_mothers_offer_l822_82280

def bike_cost : ℕ := 600
def maria_savings : ℕ := 120
def maria_earnings : ℕ := 230

theorem mothers_offer :
  bike_cost - (maria_savings + maria_earnings) = 250 :=
by sorry

end NUMINAMATH_CALUDE_mothers_offer_l822_82280


namespace NUMINAMATH_CALUDE_ratio_equality_l822_82267

theorem ratio_equality (a b c x y z : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l822_82267


namespace NUMINAMATH_CALUDE_weighted_average_problem_l822_82230

/-- Given numbers 4, 6, 8, p, q with a weighted average of 20,
    where p and q each have twice the weight of 4, 6, 8,
    prove that the average of p and q is 30.5 -/
theorem weighted_average_problem (p q : ℝ) : 
  (4 + 6 + 8 + 2*p + 2*q) / 7 = 20 →
  (p + q) / 2 = 30.5 := by
sorry

end NUMINAMATH_CALUDE_weighted_average_problem_l822_82230


namespace NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l822_82254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x

noncomputable def g (x : ℝ) : ℝ := 1 / x - 1 / Real.exp (x - 1)

theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l822_82254


namespace NUMINAMATH_CALUDE_square_area_error_l822_82286

/-- 
If the side of a square is measured with a 1% excess error, 
the resulting error in the calculated area of the square is 2.01%.
-/
theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.01)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 2.01 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l822_82286


namespace NUMINAMATH_CALUDE_tan_value_of_sequences_l822_82282

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem tan_value_of_sequences (a b : ℕ → ℝ) :
  is_geometric_sequence a →
  is_arithmetic_sequence b →
  a 1 - a 6 - a 11 = -3 * Real.sqrt 3 →
  b 1 + b 6 + b 11 = 7 * Real.pi →
  Real.tan ((b 3 + b 9) / (1 - a 4 - a 3)) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_value_of_sequences_l822_82282


namespace NUMINAMATH_CALUDE_alcohol_bottle_problem_l822_82294

/-- The amount of alcohol originally in the bottle -/
def original_amount : ℝ := 750

/-- The amount poured back in after the first pour -/
def amount_added : ℝ := 40

/-- The amount poured out in the third pour -/
def third_pour : ℝ := 180

/-- The amount remaining after all pours -/
def final_amount : ℝ := 60

theorem alcohol_bottle_problem :
  let first_pour := original_amount * (1/3)
  let after_first_pour := original_amount - first_pour + amount_added
  let second_pour := after_first_pour * (5/9)
  let after_second_pour := after_first_pour - second_pour
  after_second_pour - third_pour = final_amount :=
sorry


end NUMINAMATH_CALUDE_alcohol_bottle_problem_l822_82294


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l822_82289

theorem sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l822_82289


namespace NUMINAMATH_CALUDE_percentage_of_120_to_80_l822_82251

theorem percentage_of_120_to_80 : ∃ (p : ℝ), p = (120 : ℝ) / 80 * 100 ∧ p = 150 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_80_l822_82251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l822_82262

theorem arithmetic_sequence_solution :
  ∀ (y : ℚ),
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y
  (a₂ - a₁ = a₃ - a₂) →  -- arithmetic sequence condition
  y = -19/8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l822_82262


namespace NUMINAMATH_CALUDE_max_cos_product_l822_82210

theorem max_cos_product (α β γ : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : 0 < γ ∧ γ < π/2) 
  (h4 : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  Real.cos α * Real.cos β * Real.cos γ ≤ 2 * Real.sqrt 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cos_product_l822_82210


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_rational_expression_l822_82281

-- Problem 1
theorem simplify_sqrt_expression :
  3 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = 8 * Real.sqrt 3 := by sorry

-- Problem 2
theorem simplify_rational_expression (m : ℝ) (h : m^2 + 3*m - 4 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_rational_expression_l822_82281


namespace NUMINAMATH_CALUDE_cyclic_inequality_l822_82216

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l822_82216


namespace NUMINAMATH_CALUDE_min_growth_rate_doubles_coverage_l822_82208

-- Define the initial forest coverage area
variable (a : ℝ)
-- Define the natural growth rate
def natural_growth_rate : ℝ := 0.02
-- Define the time period in years
def years : ℕ := 10
-- Define the target multiplier for forest coverage
def target_multiplier : ℝ := 2

-- Define the function for forest coverage area after x years with natural growth
def forest_coverage (x : ℕ) : ℝ := a * (1 + natural_growth_rate) ^ x

-- Define the minimum required growth rate
def min_growth_rate : ℝ := 0.072

-- Theorem statement
theorem min_growth_rate_doubles_coverage :
  ∀ p : ℝ, p ≥ min_growth_rate →
  a * (1 + p) ^ years ≥ target_multiplier * a :=
by sorry

end NUMINAMATH_CALUDE_min_growth_rate_doubles_coverage_l822_82208


namespace NUMINAMATH_CALUDE_quadratic_equation_with_irrational_root_l822_82284

theorem quadratic_equation_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = 2 * Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -11 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_irrational_root_l822_82284


namespace NUMINAMATH_CALUDE_x_y_negative_l822_82273

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_y_negative_l822_82273


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l822_82298

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with a given size -/
structure Population where
  size : ℕ

/-- Represents a sample drawn from a population -/
structure Sample where
  size : ℕ
  population : Population
  method : SamplingMethod

/-- The probability of an individual being selected in a given sample -/
def selectionProbability (s : Sample) : ℚ :=
  s.size / s.population.size

/-- Theorem: In a population of 2008 parts, after eliminating 8 parts randomly
    and then selecting 20 parts using systematic sampling, the probability
    of each part being selected is 20/2008 -/
theorem systematic_sampling_probability :
  let initialPopulation : Population := ⟨2008⟩
  let eliminatedPopulation : Population := ⟨2000⟩
  let sample : Sample := ⟨20, eliminatedPopulation, SamplingMethod.Systematic⟩
  selectionProbability sample = 20 / 2008 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l822_82298


namespace NUMINAMATH_CALUDE_evaluate_expression_l822_82238

-- Define x in terms of b
def x (b : ℝ) : ℝ := b + 9

-- Theorem to prove
theorem evaluate_expression (b : ℝ) : x b - b + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l822_82238


namespace NUMINAMATH_CALUDE_book_pages_calculation_l822_82202

theorem book_pages_calculation (num_books : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) (num_sheets : ℕ) : 
  num_books = 2 → 
  pages_per_side = 4 → 
  sides_per_sheet = 2 → 
  num_sheets = 150 → 
  (num_sheets * pages_per_side * sides_per_sheet) / num_books = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l822_82202


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_condition_l822_82287

/-- Given a hyperbola mx^2 + y^2 = 1 where m < -1, if its eccentricity is exactly
    the geometric mean of the lengths of the real and imaginary axes, then m = -7 - 4√3 -/
theorem hyperbola_eccentricity_condition (m : ℝ) : 
  m < -1 →
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →
  (∃ e a b : ℝ, e^2 = 4 * a * b ∧ a = 1 ∧ b^2 = -1/m) →
  m = -7 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_condition_l822_82287


namespace NUMINAMATH_CALUDE_parabola_equation_and_min_ratio_l822_82260

/-- Represents a parabola with focus F and parameter p -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  h : p > 0

/-- A point on the parabola -/
def PointOnParabola (para : Parabola) (P : ℝ × ℝ) : Prop :=
  P.2^2 = 2 * para.p * P.1

theorem parabola_equation_and_min_ratio 
  (para : Parabola) 
  (P : ℝ × ℝ) 
  (h_P_on_parabola : PointOnParabola para P)
  (h_P_ordinate : P.2 = 4)
  (h_PF_distance : Real.sqrt ((P.1 - para.F.1)^2 + (P.2 - para.F.2)^2) = 4) :
  -- 1. The equation of the parabola is y^2 = 8x
  (∀ (x y : ℝ), PointOnParabola para (x, y) ↔ y^2 = 8*x) ∧
  -- 2. Minimum value of |MF| / |AB| is 1/2
  (∀ (A B : ℝ × ℝ) (h_A_on_parabola : PointOnParabola para A) 
                    (h_B_on_parabola : PointOnParabola para B)
                    (h_A_ne_B : A ≠ B)
                    (h_A_ne_P : A ≠ P)
                    (h_B_ne_P : B ≠ P),
   ∃ (M : ℝ × ℝ),
     -- Angle bisector of ∠APB is perpendicular to x-axis
     (∃ (k : ℝ), (A.2 - P.2) / (A.1 - P.1) = k ∧ (B.2 - P.2) / (B.1 - P.1) = -1/k) →
     -- M is on x-axis and perpendicular bisector of AB
     (M.2 = 0 ∧ (M.1 - (A.1 + B.1)/2) * ((B.2 - A.2)/(B.1 - A.1)) + (M.2 - (A.2 + B.2)/2) = 0) →
     -- Minimum value of |MF| / |AB| is 1/2
     (Real.sqrt ((M.1 - para.F.1)^2 + (M.2 - para.F.2)^2) / 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_min_ratio_l822_82260


namespace NUMINAMATH_CALUDE_fifth_number_is_24_l822_82285

/-- Definition of the sequence function -/
def f (n : ℕ) : ℕ := n^2 - 1

/-- Theorem stating that the fifth number in the sequence is 24 -/
theorem fifth_number_is_24 : f 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_is_24_l822_82285


namespace NUMINAMATH_CALUDE_triangle_area_l822_82291

theorem triangle_area (a b c : ℝ) (h1 : a = 21) (h2 : b = 72) (h3 : c = 75) : 
  (1/2 : ℝ) * a * b = 756 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l822_82291


namespace NUMINAMATH_CALUDE_square_sum_division_theorem_l822_82295

theorem square_sum_division_theorem (a b : ℕ+) :
  let q : ℕ := (a.val^2 + b.val^2) / (a.val + b.val)
  let r : ℕ := (a.val^2 + b.val^2) % (a.val + b.val)
  q^2 + r = 1977 →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_division_theorem_l822_82295


namespace NUMINAMATH_CALUDE_population_after_two_years_l822_82274

def initial_population : ℕ := 10000
def first_year_rate : ℚ := 1.05
def second_year_rate : ℚ := 0.95

theorem population_after_two_years :
  (↑initial_population * first_year_rate * second_year_rate).floor = 9975 := by
  sorry

end NUMINAMATH_CALUDE_population_after_two_years_l822_82274


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l822_82259

-- Define the quadratic equation
def quadratic_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y + 2

-- State the theorem
theorem parabola_y_intercepts :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ quadratic_equation y₁ = 0 ∧ quadratic_equation y₂ = 0 ∧
  ∀ (y : ℝ), quadratic_equation y = 0 → y = y₁ ∨ y = y₂ :=
sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l822_82259


namespace NUMINAMATH_CALUDE_order_of_abc_l822_82255

theorem order_of_abc : 
  let a := Real.log 1.01
  let b := 2 / 201
  let c := Real.sqrt 1.02 - 1
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l822_82255


namespace NUMINAMATH_CALUDE_square_of_product_pow_two_l822_82249

theorem square_of_product_pow_two (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_pow_two_l822_82249


namespace NUMINAMATH_CALUDE_tony_school_years_l822_82256

/-- The number of years Tony spent getting his initial science degree -/
def initial_degree_years : ℕ := 4

/-- The number of additional degrees Tony obtained -/
def additional_degrees : ℕ := 2

/-- The number of years each additional degree took -/
def years_per_additional_degree : ℕ := 4

/-- The number of years Tony spent getting his graduate degree in physics -/
def graduate_degree_years : ℕ := 2

/-- The total number of years Tony spent in school to become an astronaut -/
def total_years : ℕ := initial_degree_years + additional_degrees * years_per_additional_degree + graduate_degree_years

theorem tony_school_years : total_years = 14 := by
  sorry

end NUMINAMATH_CALUDE_tony_school_years_l822_82256


namespace NUMINAMATH_CALUDE_julian_needs_80_more_legos_l822_82293

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_airplanes : ℕ := 2

/-- The number of legos required for each airplane model -/
def legos_per_airplane : ℕ := 240

/-- The number of additional legos Julian needs -/
def additional_legos : ℕ := num_airplanes * legos_per_airplane - julian_legos

theorem julian_needs_80_more_legos : additional_legos = 80 :=
by sorry

end NUMINAMATH_CALUDE_julian_needs_80_more_legos_l822_82293


namespace NUMINAMATH_CALUDE_birthday_party_friends_l822_82270

theorem birthday_party_friends (total_bill : ℝ) : 
  (∃ n : ℕ, (total_bill / (n + 2 : ℝ) = 12) ∧ (total_bill / n = 16)) → 
  (∃ n : ℕ, (total_bill / (n + 2 : ℝ) = 12) ∧ (total_bill / n = 16) ∧ n = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_party_friends_l822_82270


namespace NUMINAMATH_CALUDE_min_value_expression_l822_82296

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 4*x*y + 4*y^2 + 4*z^2 ≥ 192 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 64 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 4*z₀^2 = 192 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l822_82296


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l822_82233

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l822_82233


namespace NUMINAMATH_CALUDE_cyclic_power_inequality_l822_82247

theorem cyclic_power_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_power_inequality_l822_82247


namespace NUMINAMATH_CALUDE_quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l822_82272

/-- Represents a quadratic function f(x) = ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic function --/
def roots (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

/-- The x-intercepts of a quadratic function --/
def xIntercepts (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

theorem quadratic_roots_are_x_intercepts (f : QuadraticFunction) :
  roots f = xIntercepts f := by sorry

theorem ac_sign_not_guaranteed (f : QuadraticFunction) :
  ¬∀ f : QuadraticFunction, f.a * f.c < 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l822_82272


namespace NUMINAMATH_CALUDE_work_completion_time_l822_82215

theorem work_completion_time (a b c : ℝ) (h1 : a = 2 * b) (h2 : c = 3 * b) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / 18) : b = 33 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l822_82215


namespace NUMINAMATH_CALUDE_bouncing_ball_height_l822_82297

/-- Represents the height of a bouncing ball -/
def BouncingBall (h : ℝ) : Prop :=
  -- The ball rebounds to 50% of its previous height
  let h₁ := h / 2
  let h₂ := h₁ / 2
  -- The total travel distance when it touches the floor for the third time is 200 cm
  h + 2 * h₁ + 2 * h₂ = 200

/-- Theorem stating that the original height of the ball is 80 cm -/
theorem bouncing_ball_height :
  ∃ h : ℝ, BouncingBall h ∧ h = 80 :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_height_l822_82297


namespace NUMINAMATH_CALUDE_jane_nail_polish_drying_time_l822_82266

/-- The total drying time for Jane's nail polish -/
def total_drying_time : ℕ :=
  let base_coat := 4
  let first_color := 5
  let second_color := 6
  let third_color := 7
  let first_nail_art := 8
  let second_nail_art := 10
  let top_coat := 9
  base_coat + first_color + second_color + third_color + first_nail_art + second_nail_art + top_coat

theorem jane_nail_polish_drying_time :
  total_drying_time = 49 := by sorry

end NUMINAMATH_CALUDE_jane_nail_polish_drying_time_l822_82266


namespace NUMINAMATH_CALUDE_prob_adjacent_20_3_l822_82288

/-- The number of people sitting at the round table -/
def n : ℕ := 20

/-- The number of specific people we're interested in -/
def k : ℕ := 3

/-- The probability of at least two out of three specific people sitting next to each other
    in a random seating arrangement of n people at a round table -/
def prob_adjacent (n k : ℕ) : ℚ :=
  17/57

/-- Theorem stating the probability for the given problem -/
theorem prob_adjacent_20_3 : prob_adjacent n k = 17/57 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_20_3_l822_82288


namespace NUMINAMATH_CALUDE_positive_expression_l822_82218

theorem positive_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) :
  0 < b + a^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l822_82218


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l822_82257

/-- The function f(x) = x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 - √13 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f_comp c r₁ = 0 ∧ f_comp c r₂ = 0 ∧ f_comp c r₃ = 0) ↔ 
  c = 1 - Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l822_82257


namespace NUMINAMATH_CALUDE_roses_in_vase_l822_82227

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 10 initial roses and 8 added roses, the total is 18 -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l822_82227


namespace NUMINAMATH_CALUDE_fred_grew_38_cantelopes_l822_82239

/-- The number of cantelopes Tim grew -/
def tims_cantelopes : ℕ := 44

/-- The total number of cantelopes Fred and Tim grew together -/
def total_cantelopes : ℕ := 82

/-- The number of cantelopes Fred grew -/
def freds_cantelopes : ℕ := total_cantelopes - tims_cantelopes

/-- Theorem stating that Fred grew 38 cantelopes -/
theorem fred_grew_38_cantelopes : freds_cantelopes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_38_cantelopes_l822_82239


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l822_82235

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n = a 1 * q ^ (n - 1)) →  -- Definition of geometric sequence
  a 1 = 1 →                        -- First term is 1
  a 5 = 16 →                       -- Last term is 16
  a 2 * a 3 * a 4 = 64 :=           -- Product of middle three terms is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l822_82235


namespace NUMINAMATH_CALUDE_min_ttetrominoes_on_chessboard_l822_82200

/-- Represents a chessboard as an 8x8 grid -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a T-tetromino -/
structure TTetromino where
  center : Fin 8 × Fin 8
  orientation : Fin 4

/-- Checks if a T-tetromino can be placed on the board -/
def canPlaceTTetromino (board : Chessboard) (t : TTetromino) : Bool :=
  sorry

/-- Places a T-tetromino on the board -/
def placeTTetromino (board : Chessboard) (t : TTetromino) : Chessboard :=
  sorry

/-- Checks if any T-tetromino can be placed on the board -/
def canPlaceAnyTTetromino (board : Chessboard) : Bool :=
  sorry

/-- The main theorem stating that 7 is the minimum number of T-tetrominoes -/
theorem min_ttetrominoes_on_chessboard :
  ∀ (n : Nat),
    (∃ (board : Chessboard) (tetrominoes : List TTetromino),
      tetrominoes.length = n ∧
      (∀ t ∈ tetrominoes, canPlaceTTetromino board t) ∧
      ¬canPlaceAnyTTetromino (tetrominoes.foldl placeTTetromino board)) →
    n ≥ 7 :=
  sorry

end NUMINAMATH_CALUDE_min_ttetrominoes_on_chessboard_l822_82200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l822_82209

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a, if a₁ + 3a₈ + a₁₅ = 120, then a₈ = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l822_82209


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l822_82277

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 * (x - 1)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = min) ∧
    max = 4 ∧ min = -2 := by
  sorry


end NUMINAMATH_CALUDE_f_max_min_on_interval_l822_82277


namespace NUMINAMATH_CALUDE_integral_inverse_cube_l822_82264

theorem integral_inverse_cube (x : ℝ) (h : x ≠ 0) :
  ∫ (t : ℝ) in Set.Ioo 0 x, 1 / (t^3) = -1 / (2 * x^2) + 1 / 2 := by sorry

end NUMINAMATH_CALUDE_integral_inverse_cube_l822_82264


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l822_82221

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℚ := 13/4

/-- Represents the number of hours between 4 PM on March 21 and 12 PM on March 28 -/
def totalHours : ℕ := 7 * 24 + 20

/-- Calculates the positive correction in minutes needed for the watch -/
def positiveCorrection : ℚ :=
  (timeLossPerDay * (totalHours : ℚ)) / 24

theorem watch_correction_theorem :
  positiveCorrection = 25 + 17/96 := by sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l822_82221


namespace NUMINAMATH_CALUDE_sine_angle_tangent_inequality_l822_82248

theorem sine_angle_tangent_inequality (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  Real.sin α < α ∧ α < Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_sine_angle_tangent_inequality_l822_82248


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l822_82234

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 0, -3]
  A * B = !![6, 21; -2, -17] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l822_82234


namespace NUMINAMATH_CALUDE_omega_sum_l822_82250

theorem omega_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^15 + ω^20 + ω^25 + ω^30 + ω^35 + ω^40 + ω^45 + ω^50 = 8 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_l822_82250


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l822_82253

theorem greatest_common_multiple_under_150 :
  ∃ (n : ℕ), n = 120 ∧ 
  n % 15 = 0 ∧ 
  n % 20 = 0 ∧ 
  n < 150 ∧ 
  ∀ (m : ℕ), m % 15 = 0 → m % 20 = 0 → m < 150 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l822_82253


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l822_82201

/-- An isosceles trapezoid with the given properties has an area of 54000/3 square centimeters -/
theorem isosceles_trapezoid_area : 
  ∀ (leg diagonal longer_base : ℝ),
  leg = 40 →
  diagonal = 50 →
  longer_base = 60 →
  ∃ (area : ℝ),
  area = 54000 / 3 ∧
  area = (longer_base + (longer_base - 2 * (Real.sqrt (leg^2 - ((100/3)^2))))) * (100/3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l822_82201


namespace NUMINAMATH_CALUDE_functional_equation_solution_l822_82292

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y + z) = f x * f y + f z

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l822_82292


namespace NUMINAMATH_CALUDE_exists_negative_irrational_greater_than_neg_four_l822_82243

theorem exists_negative_irrational_greater_than_neg_four :
  ∃ x : ℝ, x < 0 ∧ Irrational x ∧ -4 < x := by
sorry

end NUMINAMATH_CALUDE_exists_negative_irrational_greater_than_neg_four_l822_82243


namespace NUMINAMATH_CALUDE_group_communication_l822_82290

theorem group_communication (n k : ℕ) : 
  n > 0 → 
  k > 0 → 
  k * (n - 1) * n = 440 → 
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_group_communication_l822_82290


namespace NUMINAMATH_CALUDE_equal_circumradii_not_imply_equal_distances_l822_82204

-- Define the points A_i and B_i in 3D space
variable (A B : ℕ → ℝ × ℝ × ℝ)

-- Define the radius of the circumscribed circle
def circumradius (p q r : ℝ × ℝ × ℝ) : ℝ := sorry

-- A_1 is the circumcenter of a triangle (we don't need to use this directly in the theorem)
-- def A_1_is_circumcenter : sorry := sorry

-- The equality of circumradii for all triangles formed by A_i and B_i
def equal_circumradii (A B : ℕ → ℝ × ℝ × ℝ) : Prop :=
  ∀ i j k, circumradius (A i) (A j) (A k) = circumradius (B i) (B j) (B k)

-- The distance between two points
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

-- The theorem stating that equal circumradii do not necessarily imply equal distances
theorem equal_circumradii_not_imply_equal_distances :
  ∃ A B : ℕ → ℝ × ℝ × ℝ,
    equal_circumradii A B ∧ ∃ i j, distance (A i) (A j) ≠ distance (B i) (B j) :=
sorry

end NUMINAMATH_CALUDE_equal_circumradii_not_imply_equal_distances_l822_82204


namespace NUMINAMATH_CALUDE_qr_length_l822_82271

/-- Triangle PQR with specific properties -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Point Y on QR such that PY = PQ -/
  Y : ℝ × ℝ
  /-- QY has integer length -/
  QY_integer : ℤ
  /-- RY has integer length -/
  RY_integer : ℤ
  /-- PQ equals 95 -/
  PQ_eq : PQ = 95
  /-- PR equals 103 -/
  PR_eq : PR = 103

/-- The length of QR in the special triangle PQR is 132 -/
theorem qr_length (t : TrianglePQR) : Real.sqrt ((t.Y.1 - 0)^2 + (t.Y.2 - 0)^2) = 132 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l822_82271


namespace NUMINAMATH_CALUDE_rabbit_measurement_probability_l822_82219

theorem rabbit_measurement_probability :
  let total_rabbits : ℕ := 5
  let measured_rabbits : ℕ := 3
  let selected_rabbits : ℕ := 3
  let favorable_outcomes : ℕ := (measured_rabbits.choose 2) * ((total_rabbits - measured_rabbits).choose 1)
  let total_outcomes : ℕ := total_rabbits.choose selected_rabbits
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_rabbit_measurement_probability_l822_82219


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l822_82268

theorem max_sum_of_factors (diamond heart : ℕ) : 
  diamond * heart = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ diamond + heart) → diamond + heart = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l822_82268


namespace NUMINAMATH_CALUDE_infinite_inequality_occurrences_l822_82242

theorem infinite_inequality_occurrences (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 1 + a n > a (n - 1) * (2 : ℝ)^(1/5) :=
sorry

end NUMINAMATH_CALUDE_infinite_inequality_occurrences_l822_82242


namespace NUMINAMATH_CALUDE_kamals_english_marks_l822_82217

/-- Proves that given Kamal's marks in four subjects and his average across five subjects, his marks in the fifth subject (English) are 66. -/
theorem kamals_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (h1 : math_marks = 65)
  (h2 : physics_marks = 77)
  (h3 : chemistry_marks = 62)
  (h4 : biology_marks = 75)
  (h5 : average_marks = 69)
  (h6 : average_marks * 5 = math_marks + physics_marks + chemistry_marks + biology_marks + english_marks) :
  english_marks = 66 := by
  sorry

#check kamals_english_marks

end NUMINAMATH_CALUDE_kamals_english_marks_l822_82217


namespace NUMINAMATH_CALUDE_shekar_average_marks_l822_82220

theorem shekar_average_marks :
  let math_score := 76
  let science_score := 65
  let social_studies_score := 82
  let english_score := 67
  let biology_score := 95
  let total_score := math_score + science_score + social_studies_score + english_score + biology_score
  let num_subjects := 5
  (total_score / num_subjects : ℚ) = 77 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l822_82220


namespace NUMINAMATH_CALUDE_division_in_base4_l822_82213

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Represents division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10ToBase4 ((base4ToBase10 a) / (base4ToBase10 b))

theorem division_in_base4 :
  divBase4 [3, 1, 2, 2] [3, 1] = [3, 5] := by sorry

end NUMINAMATH_CALUDE_division_in_base4_l822_82213


namespace NUMINAMATH_CALUDE_proportional_sampling_l822_82278

theorem proportional_sampling :
  let total_population : ℕ := 162
  let elderly_population : ℕ := 27
  let middle_aged_population : ℕ := 54
  let young_population : ℕ := 81
  let sample_size : ℕ := 36

  let elderly_sample : ℕ := 6
  let middle_aged_sample : ℕ := 12
  let young_sample : ℕ := 18

  elderly_population + middle_aged_population + young_population = total_population →
  elderly_sample + middle_aged_sample + young_sample = sample_size →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population ∧
  (middle_aged_sample : ℚ) / sample_size = (middle_aged_population : ℚ) / total_population ∧
  (young_sample : ℚ) / sample_size = (young_population : ℚ) / total_population :=
by
  sorry

end NUMINAMATH_CALUDE_proportional_sampling_l822_82278


namespace NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l822_82205

/-- The number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 15

/-- The initial number of men in the garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 54

/-- The number of men in the reinforcement -/
def reinforcement : ℕ := 1900

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem reinforcement_arrival_theorem :
  initial_garrison * (initial_provision_days - reinforcement_arrival_day) =
  (initial_garrison + reinforcement) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l822_82205


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sinusoid_l822_82245

open Real

theorem axis_of_symmetry_sinusoid (x : ℝ) :
  let f := fun x => Real.sin (1/2 * x - π/6)
  ∃ k : ℤ, f (4*π/3 + x) = f (4*π/3 - x) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sinusoid_l822_82245


namespace NUMINAMATH_CALUDE_fruit_to_grain_value_fruit_worth_in_grains_l822_82236

-- Define the exchange rates
def fruit_to_vegetable : ℚ := 3 / 4
def vegetable_to_grain : ℚ := 5

-- Theorem statement
theorem fruit_to_grain_value :
  fruit_to_vegetable * vegetable_to_grain = 15 / 4 :=
by sorry

-- Corollary to express the result as a mixed number
theorem fruit_worth_in_grains :
  fruit_to_vegetable * vegetable_to_grain = 3 + 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fruit_to_grain_value_fruit_worth_in_grains_l822_82236


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l822_82206

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l822_82206


namespace NUMINAMATH_CALUDE_subset_A_l822_82226

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end NUMINAMATH_CALUDE_subset_A_l822_82226


namespace NUMINAMATH_CALUDE_lisa_baby_spoons_l822_82223

/-- Given the total number of spoons, number of children, number of decorative spoons,
    and number of spoons in the new cutlery set, calculate the number of baby spoons per child. -/
def baby_spoons_per_child (total_spoons : ℕ) (num_children : ℕ) (decorative_spoons : ℕ) (new_cutlery_spoons : ℕ) : ℕ :=
  (total_spoons - decorative_spoons - new_cutlery_spoons) / num_children

/-- Prove that given Lisa's specific situation, each child had 3 baby spoons. -/
theorem lisa_baby_spoons : baby_spoons_per_child 39 4 2 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lisa_baby_spoons_l822_82223


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l822_82276

/-- A quadrilateral inscribed in a circle with given properties -/
structure InscribedQuadrilateral where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- The theorem stating the properties of the specific inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 6)
  (h_side1 : q.side1 = 100)
  (h_side2 : q.side2 = 200)
  (h_side3 : q.side3 = 200) :
  q.side4 = 100 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l822_82276


namespace NUMINAMATH_CALUDE_new_person_weight_l822_82207

theorem new_person_weight
  (initial_count : ℕ)
  (average_increase : ℝ)
  (replaced_weight : ℝ)
  (hcount : initial_count = 7)
  (hincrease : average_increase = 3.5)
  (hreplaced : replaced_weight = 75)
  : ℝ :=
by
  -- The weight of the new person
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l822_82207


namespace NUMINAMATH_CALUDE_grisha_hat_color_l822_82222

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a person in the game -/
structure Person where
  name : String
  hatColor : HatColor
  canSee : List String

/-- The game setup -/
structure GameSetup where
  totalHats : Nat
  whiteHats : Nat
  blackHats : Nat
  persons : List Person
  remainingHats : Nat

/-- Predicate to check if a person can determine their hat color -/
def canDetermineColor (setup : GameSetup) (person : Person) : Prop := sorry

/-- The main theorem -/
theorem grisha_hat_color (setup : GameSetup) 
  (h1 : setup.totalHats = 5)
  (h2 : setup.whiteHats = 2)
  (h3 : setup.blackHats = 3)
  (h4 : setup.remainingHats = 2)
  (h5 : setup.persons.length = 3)
  (h6 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ zhenya.canSee = ["Lyova", "Grisha"])
  (h7 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ lyova.canSee = ["Grisha"])
  (h8 : ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.canSee = [])
  (h9 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ ¬canDetermineColor setup zhenya)
  (h10 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ ¬canDetermineColor setup lyova) :
  ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.hatColor = HatColor.Black ∧ canDetermineColor setup grisha :=
sorry

end NUMINAMATH_CALUDE_grisha_hat_color_l822_82222


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l822_82279

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.im z = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l822_82279


namespace NUMINAMATH_CALUDE_cyclists_equal_distance_l822_82263

/-- Represents a cyclist with a given speed -/
structure Cyclist where
  speed : ℝ
  speed_pos : speed > 0

/-- The problem setup -/
def cyclistProblem (c1 c2 c3 : Cyclist) (totalTime : ℝ) : Prop :=
  c1.speed = 12 ∧ 
  c2.speed = 16 ∧ 
  c3.speed = 24 ∧
  totalTime = 3 ∧
  ∃ (t1 t2 t3 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧
    t1 + t2 + t3 = totalTime ∧
    c1.speed * t1 = c2.speed * t2 ∧
    c2.speed * t2 = c3.speed * t3

/-- The theorem to prove -/
theorem cyclists_equal_distance 
  (c1 c2 c3 : Cyclist) (totalTime : ℝ) 
  (h : cyclistProblem c1 c2 c3 totalTime) : 
  c1.speed * (totalTime / 3) = 16 := by
  sorry


end NUMINAMATH_CALUDE_cyclists_equal_distance_l822_82263


namespace NUMINAMATH_CALUDE_complex_sum_equals_z_l822_82283

theorem complex_sum_equals_z (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 = z := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_z_l822_82283


namespace NUMINAMATH_CALUDE_euro_puzzle_l822_82212

theorem euro_puzzle (E M n : ℕ) : 
  (M + 3 = n * (E - 3)) →
  (E + n = 3 * (M - n)) →
  n > 0 →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_euro_puzzle_l822_82212


namespace NUMINAMATH_CALUDE_solve_equation_l822_82211

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l822_82211


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l822_82214

theorem geometric_sequence_condition (a b c : ℝ) : 
  (b^2 ≠ a*c → ¬(∃ r : ℝ, b = a*r ∧ c = b*r)) ∧ 
  (∃ a b c : ℝ, ¬(∃ r : ℝ, b = a*r ∧ c = b*r) ∧ b^2 = a*c) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l822_82214


namespace NUMINAMATH_CALUDE_b_completion_time_l822_82269

/-- Given two workers a and b, this theorem proves how long it takes b to complete a job alone. -/
theorem b_completion_time (work : ℝ) (a b : ℝ → ℝ) :
  (∀ t, a t + b t = work / 16) →  -- a and b together complete the work in 16 days
  (∀ t, a t = work / 20) →        -- a alone completes the work in 20 days
  (∀ t, b t = work / 80) :=       -- b alone completes the work in 80 days
by sorry

end NUMINAMATH_CALUDE_b_completion_time_l822_82269


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l822_82252

theorem min_x_prime_factorization (x y : ℕ+) (a b : ℕ) (c d : ℕ) 
  (h1 : 4 * x^7 = 13 * y^17)
  (h2 : x = a^c * b^d)
  (h3 : Nat.Prime a)
  (h4 : Nat.Prime b)
  (h5 : ∀ (w z : ℕ+) (e f : ℕ) (p q : ℕ), 
        4 * w^7 = 13 * z^17 → 
        w = p^e * q^f → 
        Nat.Prime p → 
        Nat.Prime q → 
        w ≤ x) : 
  a + b + c + d = 19 := by
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l822_82252


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l822_82258

theorem sin_2x_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x)) →
  (deriv f) x = 2 * Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l822_82258


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l822_82246

theorem polynomial_factor_theorem (c : ℚ) :
  let P : ℚ → ℚ := λ x => x^3 + 2*x^2 + c*x + 8
  (∃ q : ℚ → ℚ, ∀ x, P x = (x - 3) * q x) →
  c = -53/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l822_82246


namespace NUMINAMATH_CALUDE_fraction_modification_l822_82244

theorem fraction_modification (p q r s x : ℚ) : 
  p ≠ q → q ≠ 0 → p = 3 → q = 5 → r = 7 → s = 9 → (p + x) / (q - x) = r / s → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l822_82244


namespace NUMINAMATH_CALUDE_solve_for_b_l822_82203

theorem solve_for_b (m a b c k : ℝ) (h : m = (c^2 * a * b) / (a - k * b)) :
  b = m * a / (c^2 * a + m * k) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_b_l822_82203


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_base6_l822_82265

/-- Represents a number in base 6 --/
structure Base6 :=
  (value : ℕ)
  (isValid : value < 6)

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List Base6 :=
  sorry

/-- Arithmetic sequence in base 6 --/
def arithmeticSequenceBase6 (a l d : Base6) : List Base6 :=
  sorry

/-- Sum of a list of Base6 numbers --/
def sumBase6 (lst : List Base6) : List Base6 :=
  sorry

theorem arithmetic_sequence_sum_base6 :
  let a := Base6.mk 1 (by norm_num)
  let l := Base6.mk 5 (by norm_num) -- 41 in base 6 is 5 * 6 + 5 = 35
  let d := Base6.mk 2 (by norm_num)
  let sequence := arithmeticSequenceBase6 a l d
  sumBase6 sequence = toBase6 441 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_base6_l822_82265


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l822_82237

theorem incorrect_height_calculation (n : ℕ) (initial_avg actual_avg actual_height : ℝ) :
  n = 20 ∧
  initial_avg = 175 ∧
  actual_avg = 173 ∧
  actual_height = 111 →
  ∃ incorrect_height : ℝ,
    incorrect_height = n * initial_avg - (n - 1) * actual_avg - actual_height :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l822_82237


namespace NUMINAMATH_CALUDE_odd_function_a_indeterminate_l822_82224

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_a_indeterminate (f : ℝ → ℝ) (h : OddFunction f) :
  ¬ ∃ a : ℝ, ∀ g : ℝ → ℝ, OddFunction g → g = f :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_indeterminate_l822_82224


namespace NUMINAMATH_CALUDE_flour_spill_ratio_l822_82299

def initial_flour : ℕ := 500
def used_flour : ℕ := 240
def needed_flour : ℕ := 370

theorem flour_spill_ratio :
  let flour_after_baking := initial_flour - used_flour
  let flour_after_spill := initial_flour - needed_flour
  let spilled_flour := flour_after_baking - flour_after_spill
  (spilled_flour : ℚ) / flour_after_baking = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_flour_spill_ratio_l822_82299


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l822_82228

/-- A rectangle with perimeter 72 meters and length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l822_82228
