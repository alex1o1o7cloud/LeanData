import Mathlib

namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1558_155878

theorem vector_subtraction_scalar_multiplication (a b : ℝ × ℝ) :
  a = (3, -8) → b = (2, -6) → a - 5 • b = (-7, 22) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1558_155878


namespace NUMINAMATH_CALUDE_M_is_empty_l1558_155856

def M : Set ℝ := {x | x^4 + 4*x^2 - 12*x + 8 = 0 ∧ x > 0}

theorem M_is_empty : M = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_is_empty_l1558_155856


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l1558_155819

theorem smallest_denominator_between_fractions :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (a : ℚ) / b > 6 / 17 →
    (a : ℚ) / b < 9 / 25 →
    b ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l1558_155819


namespace NUMINAMATH_CALUDE_f_has_two_zeros_h_range_l1558_155894

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp (x - a) - x * Real.log x + (1 - a) * x

-- Define the set of a values for g
def A : Set ℝ := Set.Ioo 1 (3 - Real.log 3)

-- Define h(a) as the local minimum of g(x) for a given a
noncomputable def h (a : ℝ) : ℝ := 
  let x₂ := Real.exp a
  2 * x₂ - x₂^2

-- Theorem statements
theorem f_has_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a > 1 := by sorry

theorem h_range :
  Set.range h = Set.Icc (-3) 1 := by sorry

end

end NUMINAMATH_CALUDE_f_has_two_zeros_h_range_l1558_155894


namespace NUMINAMATH_CALUDE_composite_expression_l1558_155817

theorem composite_expression (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 3^(2*n+1) - 2^(2*n+1) - 6^n := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l1558_155817


namespace NUMINAMATH_CALUDE_imo_42_inequality_l1558_155811

theorem imo_42_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_imo_42_inequality_l1558_155811


namespace NUMINAMATH_CALUDE_apples_removed_by_ricki_l1558_155801

/-- The number of apples Ricki removed -/
def rickis_apples : ℕ := 14

/-- The initial number of apples in the basket -/
def initial_apples : ℕ := 74

/-- The final number of apples in the basket -/
def final_apples : ℕ := 32

/-- Samson removed twice as many apples as Ricki -/
def samsons_apples : ℕ := 2 * rickis_apples

theorem apples_removed_by_ricki :
  rickis_apples = 14 ∧
  initial_apples = final_apples + rickis_apples + samsons_apples :=
by sorry

end NUMINAMATH_CALUDE_apples_removed_by_ricki_l1558_155801


namespace NUMINAMATH_CALUDE_disk_at_nine_oclock_l1558_155872

/-- Represents a circular clock face with a smaller disk rolling on it. -/
structure ClockWithDisk where
  clock_radius : ℝ
  disk_radius : ℝ
  start_position : ℝ -- in radians, 0 represents 3 o'clock
  rotation_direction : Bool -- true for clockwise

/-- Calculates the position of the disk after one full rotation -/
def position_after_rotation (c : ClockWithDisk) : ℝ :=
  sorry

/-- Theorem stating that the disk will be at 9 o'clock after one full rotation -/
theorem disk_at_nine_oclock (c : ClockWithDisk) 
  (h1 : c.clock_radius = 30)
  (h2 : c.disk_radius = 15)
  (h3 : c.start_position = 0)
  (h4 : c.rotation_direction = true) :
  position_after_rotation c = π := by
  sorry

end NUMINAMATH_CALUDE_disk_at_nine_oclock_l1558_155872


namespace NUMINAMATH_CALUDE_unique_valid_triple_l1558_155895

/-- Represents an ordered triple of integers (a, b, c) satisfying the given conditions -/
structure ValidTriple where
  a : ℕ
  b : ℕ
  c : ℕ
  a_ge_2 : a ≥ 2
  b_ge_1 : b ≥ 1
  log_cond : (Real.log b) / (Real.log a) = c^2
  sum_cond : a + b + c = 100

/-- There exists exactly one ordered triple of integers satisfying the given conditions -/
theorem unique_valid_triple : ∃! t : ValidTriple, True := by sorry

end NUMINAMATH_CALUDE_unique_valid_triple_l1558_155895


namespace NUMINAMATH_CALUDE_carrot_problem_l1558_155865

theorem carrot_problem (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) :
  carol_carrots = 29 →
  mom_carrots = 16 →
  good_carrots = 38 →
  carol_carrots + mom_carrots - good_carrots = 7 :=
by sorry

end NUMINAMATH_CALUDE_carrot_problem_l1558_155865


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l1558_155881

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (a^2 + b^2 + c^2 + d^2 ≥ μ * a * b + b * c + 2 * c * d) → μ ≤ 13/2) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ 13/2 * a * b + b * c + 2 * c * d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l1558_155881


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1558_155886

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x = 2 * y ∧
  2 * (x + y) = 2 * (x * y) ∧
  x = 3 ∧ y = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1558_155886


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l1558_155844

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -4378 [ZMOD 6] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l1558_155844


namespace NUMINAMATH_CALUDE_min_correct_answers_to_pass_l1558_155891

/-- Represents the Fire Safety quiz selection -/
structure FireSafetyQuiz where
  total_questions : Nat
  correct_score : Int
  incorrect_score : Int
  passing_score : Int

/-- Calculates the total score based on the number of correct answers -/
def calculate_score (quiz : FireSafetyQuiz) (correct_answers : Nat) : Int :=
  (quiz.correct_score * correct_answers) + 
  (quiz.incorrect_score * (quiz.total_questions - correct_answers))

/-- Theorem: The minimum number of correct answers needed to pass the Fire Safety quiz is 12 -/
theorem min_correct_answers_to_pass (quiz : FireSafetyQuiz) 
  (h1 : quiz.total_questions = 20)
  (h2 : quiz.correct_score = 10)
  (h3 : quiz.incorrect_score = -5)
  (h4 : quiz.passing_score = 80) :
  ∀ n : Nat, calculate_score quiz n ≥ quiz.passing_score → n ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_pass_l1558_155891


namespace NUMINAMATH_CALUDE_power_negative_two_m_squared_cubed_l1558_155814

theorem power_negative_two_m_squared_cubed (m : ℝ) : (-2 * m^2)^3 = -8 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_two_m_squared_cubed_l1558_155814


namespace NUMINAMATH_CALUDE_farmland_width_l1558_155867

/-- Represents a rectangular plot of farmland -/
structure FarmPlot where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Conversion factor from acres to square feet -/
def acreToSqFt : ℝ := 43560

/-- Theorem stating the width of the farmland plot -/
theorem farmland_width (plot : FarmPlot) 
  (h1 : plot.length = 360)
  (h2 : plot.area = 10 * acreToSqFt)
  (h3 : plot.area = plot.length * plot.width) :
  plot.width = 1210 := by
  sorry

end NUMINAMATH_CALUDE_farmland_width_l1558_155867


namespace NUMINAMATH_CALUDE_log_equation_solution_l1558_155855

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 16) + (Real.log x / Real.log 4) + (Real.log x / Real.log 2) = 7 →
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1558_155855


namespace NUMINAMATH_CALUDE_floor_x_width_l1558_155805

/-- Represents a rectangular floor with a length and width in feet. -/
structure RectangularFloor where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular floor. -/
def area (floor : RectangularFloor) : ℝ :=
  floor.length * floor.width

theorem floor_x_width
  (x y : RectangularFloor)
  (h1 : area x = area y)
  (h2 : x.length = 18)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_width_l1558_155805


namespace NUMINAMATH_CALUDE_unique_value_of_a_l1558_155880

theorem unique_value_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x :=
sorry

end NUMINAMATH_CALUDE_unique_value_of_a_l1558_155880


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l1558_155825

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l1558_155825


namespace NUMINAMATH_CALUDE_exchange_rate_20_percent_increase_same_digits_l1558_155839

/-- Represents an exchange rate as a pair of integers (whole, fraction) -/
structure ExchangeRate where
  whole : ℕ
  fraction : ℕ
  h_fraction : fraction < 100

/-- Checks if two exchange rates have the same digits in different order -/
def same_digits_different_order (x y : ExchangeRate) : Prop := sorry

/-- Calculates the 20% increase of an exchange rate -/
def increase_by_20_percent (x : ExchangeRate) : ExchangeRate := sorry

/-- Main theorem: There exists an exchange rate that, when increased by 20%,
    results in a new rate with the same digits in a different order -/
theorem exchange_rate_20_percent_increase_same_digits :
  ∃ (x : ExchangeRate), same_digits_different_order x (increase_by_20_percent x) := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_20_percent_increase_same_digits_l1558_155839


namespace NUMINAMATH_CALUDE_simplify_expression_l1558_155808

theorem simplify_expression : 
  Real.sqrt 6 * (Real.sqrt 2 + Real.sqrt 3) - 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1558_155808


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1558_155882

/-- The line mx - y + 2m + 1 = 0 passes through the point (-2, 1) for any real m -/
theorem fixed_point_on_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1558_155882


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1558_155832

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -1)
  (c_ge : c ≥ -2) :
  (∃ x y z : ℝ, x + y + z = 2 ∧ x ≥ -1/2 ∧ y ≥ -1 ∧ z ≥ -2 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) = Real.sqrt 66) ∧
  (∀ x y z : ℝ, x + y + z = 2 → x ≥ -1/2 → y ≥ -1 → z ≥ -2 →
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) ≤ Real.sqrt 66) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1558_155832


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1558_155871

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1558_155871


namespace NUMINAMATH_CALUDE_number_of_trees_l1558_155800

/-- The number of trees around the house. -/
def n : ℕ := 118

/-- The difference between Alexander's and Timur's starting points. -/
def start_diff : ℕ := 33 - 12

/-- The theorem stating the number of trees around the house. -/
theorem number_of_trees :
  ∃ k : ℕ, n + k = 105 - 12 + 8 ∧ start_diff = 33 - 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trees_l1558_155800


namespace NUMINAMATH_CALUDE_stating_time_is_seven_thirty_two_l1558_155803

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents the time in minutes after 7:00 a.m. -/
def minutesAfterSeven (x : ℚ) : ℚ := 8 * x

/-- Represents the time in minutes before 8:00 a.m. -/
def minutesBeforeEight (x : ℚ) : ℚ := 7 * x

/-- 
Theorem stating that if a time is 8x minutes after 7:00 a.m. and 7x minutes before 8:00 a.m.,
then the time is 32 minutes after 7:00 a.m. (which is 7:32 a.m.)
-/
theorem time_is_seven_thirty_two (x : ℚ) :
  minutesAfterSeven x + minutesBeforeEight x = minutesInHour →
  minutesAfterSeven x = 32 :=
by sorry

end NUMINAMATH_CALUDE_stating_time_is_seven_thirty_two_l1558_155803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1558_155815

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (hd : d ≠ 0)
  (ha : arithmeticSequence a d)
  (hineq : (a 3)^2 ≠ (a 1) * (a 9)) :
  (a 3) / (a 6) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1558_155815


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_m_l1558_155812

-- Define the sets A, B, and C
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = ∅) → (m ∈ Set.Icc (-4) 0) := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_m_l1558_155812


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_180_choose_90_l1558_155861

theorem largest_two_digit_prime_factor_of_180_choose_90 : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    10 ≤ p ∧ 
    p < 100 ∧ 
    p ∣ Nat.choose 180 90 ∧ 
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 180 90 → q ≤ p ∧
    p = 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_180_choose_90_l1558_155861


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_724_l1558_155862

theorem quadratic_expression_equals_724 
  (x y : ℝ) 
  (h1 : 4 * x + y = 18) 
  (h2 : x + 4 * y = 20) : 
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_724_l1558_155862


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l1558_155888

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 281

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the proposition that the coach's handshakes are minimized -/
def coach_handshakes_minimized (k : ℕ) : Prop :=
  ∃ (n : ℕ), 
    gymnast_handshakes n + k = total_handshakes ∧
    ∀ (m : ℕ), m > n → gymnast_handshakes m > total_handshakes

/-- The main theorem stating that the minimum number of coach's handshakes is 5 -/
theorem min_coach_handshakes : 
  ∃ (k : ℕ), k = 5 ∧ coach_handshakes_minimized k :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l1558_155888


namespace NUMINAMATH_CALUDE_line_b_production_l1558_155841

/-- Given three production lines A, B, and C forming an arithmetic sequence,
    prove that Line B produced 4400 units out of a total of 13200 units. -/
theorem line_b_production (total : ℕ) (a b c : ℕ) : 
  total = 13200 →
  a + b + c = total →
  ∃ (d : ℤ), a = b - d ∧ c = b + d →
  b = 4400 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l1558_155841


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1558_155828

/-- The polynomial function defined by x^3(x-3)^2(2+x) -/
def f (x : ℝ) : ℝ := x^3 * (x-3)^2 * (2+x)

/-- The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 -/
def roots : Set ℝ := {x : ℝ | f x = 0}

/-- Theorem: The set of roots of the polynomial equation x^3(x-3)^2(2+x) = 0 is {0, 3, -2} -/
theorem roots_of_polynomial : roots = {0, 3, -2} := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1558_155828


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l1558_155897

/-- Given a quadratic function y = (m-2)x^2 + 2mx - (3-m), prove that the range of m
    satisfying all conditions is 2 < m < 3 --/
theorem quadratic_function_m_range (m : ℝ) : 
  let f (x : ℝ) := (m - 2) * x^2 + 2 * m * x - (3 - m)
  let vertex_x := -m / (m - 2)
  let vertex_y := (-5 * m + 6) / (m - 2)
  (∀ x, (m - 2) * x^2 + 2 * m * x - (3 - m) = f x) →
  (vertex_x < 0 ∧ vertex_y < 0) →
  (m - 2 > 0) →
  (-(3 - m) < 0) →
  (2 < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l1558_155897


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1558_155853

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 6 meters, width 5 meters, and depth 2 meters is 104 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 6 5 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1558_155853


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l1558_155807

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (3 + 4*I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (3 + 4*I)) = 1 ∧ ∀ (u : ℂ), Complex.abs (u - (3 + 4*I)) = 1 → Complex.abs u ≤ Complex.abs w ∧ Complex.abs w = 6 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l1558_155807


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1558_155823

/-- Given two points A and B symmetric with respect to the y-axis, prove that the sum of their x-coordinates is the negative of the difference of their y-coordinates. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, -3) ∧ B = (2, n) ∧ 
   (A.1 = -B.1) ∧ (A.2 = B.2)) → 
  m + n = -5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1558_155823


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l1558_155859

theorem largest_coefficient_binomial_expansion :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 6 →
  (Nat.choose 6 k) * (2^k) ≤ (Nat.choose 6 4) * (2^4) :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l1558_155859


namespace NUMINAMATH_CALUDE_trig_product_identities_l1558_155843

open Real

theorem trig_product_identities (α : ℝ) :
  (1 + sin α = 2 * sin ((π/2 + α)/2) * cos ((π/2 - α)/2)) ∧
  (1 - sin α = 2 * cos ((π/2 + α)/2) * sin ((π/2 - α)/2)) ∧
  (1 + 2 * sin α = 4 * sin ((π/6 + α)/2) * cos ((π/6 - α)/2)) ∧
  (1 - 2 * sin α = 4 * cos ((π/6 + α)/2) * sin ((π/6 - α)/2)) ∧
  (1 + 2 * cos α = 4 * cos ((π/3 + α)/2) * cos ((π/3 - α)/2)) ∧
  (1 - 2 * cos α = -4 * sin ((π/3 + α)/2) * sin ((π/3 - α)/2)) :=
by sorry


end NUMINAMATH_CALUDE_trig_product_identities_l1558_155843


namespace NUMINAMATH_CALUDE_conference_handshakes_count_l1558_155852

/-- The number of unique handshakes in a conference with specified conditions -/
def conferenceHandshakes (numCompanies : ℕ) (repsPerCompany : ℕ) : ℕ :=
  let totalPeople := numCompanies * repsPerCompany
  let handshakesPerPerson := totalPeople - repsPerCompany - 1
  (totalPeople * handshakesPerPerson) / 2

/-- Theorem: The number of handshakes in the specified conference is 250 -/
theorem conference_handshakes_count :
  conferenceHandshakes 5 5 = 250 := by
  sorry

#eval conferenceHandshakes 5 5

end NUMINAMATH_CALUDE_conference_handshakes_count_l1558_155852


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l1558_155829

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def in_band (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem quadratic_function_max_value
  (a b c : ℝ)  -- Coefficients of f(x) = ax^2 + bx + c
  (h1 : in_band (f a b c (-2) + 2) 0 4)
  (h2 : in_band (f a b c 0 + 2) 0 4)
  (h3 : in_band (f a b c 2 + 2) 0 4)
  (h4 : ∀ t : ℝ, in_band (t + 1) (-1) 3 → in_band (f a b c t) (-5/2) (5/2)) :
  ∃ t : ℝ, |f a b c t| = 5/2 ∧ ∀ s : ℝ, |f a b c s| ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l1558_155829


namespace NUMINAMATH_CALUDE_wheel_revolution_distance_l1558_155868

/-- Proves that given specific wheel sizes and revolution difference, the distance traveled is 315 feet -/
theorem wheel_revolution_distance 
  (back_wheel_perimeter : ℝ) 
  (front_wheel_perimeter : ℝ) 
  (revolution_difference : ℝ) 
  (h1 : back_wheel_perimeter = 9) 
  (h2 : front_wheel_perimeter = 7) 
  (h3 : revolution_difference = 10) :
  (front_wheel_perimeter⁻¹ - back_wheel_perimeter⁻¹)⁻¹ * revolution_difference = 315 :=
by sorry

end NUMINAMATH_CALUDE_wheel_revolution_distance_l1558_155868


namespace NUMINAMATH_CALUDE_square_recurrence_cube_recurrence_l1558_155870

-- Define the sequences
def a (n : ℕ) : ℕ := n^2
def b (n : ℕ) : ℕ := n^3

-- Theorem for the linear recurrence relation of a_n = n^2
theorem square_recurrence (n : ℕ) (h : n ≥ 3) :
  a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) := by
  sorry

-- Theorem for the linear recurrence relation of a_n = n^3
theorem cube_recurrence (n : ℕ) (h : n ≥ 4) :
  b n = 4 * b (n - 1) - 6 * b (n - 2) + 4 * b (n - 3) - b (n - 4) := by
  sorry

end NUMINAMATH_CALUDE_square_recurrence_cube_recurrence_l1558_155870


namespace NUMINAMATH_CALUDE_circle_plus_self_twice_l1558_155883

/-- Definition of the ⊕ operation -/
def circle_plus (x y : ℝ) : ℝ := x^3 + 2*x - y

/-- Theorem stating that k ⊕ (k ⊕ k) = k -/
theorem circle_plus_self_twice (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_self_twice_l1558_155883


namespace NUMINAMATH_CALUDE_specific_cylinder_properties_l1558_155893

/-- Represents a cylinder with height and surface area as parameters. -/
structure Cylinder where
  height : ℝ
  surfaceArea : ℝ

/-- Calculates the radius of the base circle of a cylinder. -/
def baseRadius (c : Cylinder) : ℝ :=
  sorry

/-- Calculates the volume of a cylinder. -/
def volume (c : Cylinder) : ℝ :=
  sorry

/-- Theorem stating the properties of a specific cylinder. -/
theorem specific_cylinder_properties :
  let c := Cylinder.mk 8 (130 * Real.pi)
  baseRadius c = 5 ∧ volume c = 200 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_specific_cylinder_properties_l1558_155893


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l1558_155830

theorem smallest_integer_satisfying_conditions : 
  ∃ n : ℤ, (∀ m : ℤ, (m + 15 ≥ 16 ∧ -5 * m < -10) → n ≤ m) ∧ 
           (n + 15 ≥ 16 ∧ -5 * n < -10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l1558_155830


namespace NUMINAMATH_CALUDE_car_replacement_problem_l1558_155864

theorem car_replacement_problem :
  let initial_fleet : ℕ := 20
  let new_cars_per_year : ℕ := 6
  let years : ℕ := 2
  ∃ (x : ℕ),
    x > 0 ∧
    initial_fleet - years * x < initial_fleet / 2 ∧
    ∀ (y : ℕ), y > 0 ∧ initial_fleet - years * y < initial_fleet / 2 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_car_replacement_problem_l1558_155864


namespace NUMINAMATH_CALUDE_A_intersect_B_l1558_155837

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2*a - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1558_155837


namespace NUMINAMATH_CALUDE_cubic_equation_roots_sum_l1558_155836

theorem cubic_equation_roots_sum (a b c : ℝ) : 
  (a^3 - 6*a^2 + 11*a - 6 = 0) → 
  (b^3 - 6*b^2 + 11*b - 6 = 0) → 
  (c^3 - 6*c^2 + 11*c - 6 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^3 + 1/b^3 + 1/c^3 = 251/216) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_sum_l1558_155836


namespace NUMINAMATH_CALUDE_sequence_sum_l1558_155826

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧  -- increasing positive integers
  b - a = c - b ∧                  -- arithmetic progression
  c * c = b * d ∧                  -- geometric progression
  d - a = 42                       -- difference between first and fourth terms
  → a + b + c + d = 123 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1558_155826


namespace NUMINAMATH_CALUDE_derivative_equality_l1558_155822

-- Define the function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

-- Define the derivative of f
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Theorem statement
theorem derivative_equality (a b c : ℝ) :
  (f' a b 2 = 2) → (f' a b (-2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_equality_l1558_155822


namespace NUMINAMATH_CALUDE_calculate_expression_l1558_155884

theorem calculate_expression : (-1)^2022 + Real.sqrt 9 - 2 * Real.sin (30 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1558_155884


namespace NUMINAMATH_CALUDE_tv_weekly_cost_l1558_155820

/-- Calculate the cost in cents to run a TV for a week -/
theorem tv_weekly_cost (tv_power : ℝ) (daily_usage : ℝ) (electricity_cost : ℝ) : 
  tv_power = 125 →
  daily_usage = 4 →
  electricity_cost = 14 →
  (tv_power * daily_usage * 7 / 1000 * electricity_cost) = 49 := by
  sorry

end NUMINAMATH_CALUDE_tv_weekly_cost_l1558_155820


namespace NUMINAMATH_CALUDE_original_price_from_decreased_price_l1558_155818

/-- 
If an article's price after a 50% decrease is 620 (in some currency unit),
then its original price was 1240 (in the same currency unit).
-/
theorem original_price_from_decreased_price (decreased_price : ℝ) 
  (h : decreased_price = 620) : 
  ∃ (original_price : ℝ), 
    original_price * 0.5 = decreased_price ∧ 
    original_price = 1240 :=
by sorry

end NUMINAMATH_CALUDE_original_price_from_decreased_price_l1558_155818


namespace NUMINAMATH_CALUDE_museum_trip_ratio_l1558_155874

theorem museum_trip_ratio : 
  ∀ (p1 p2 p3 p4 : ℕ),
  p1 = 12 →
  p3 = p2 - 6 →
  p4 = p1 + 9 →
  p1 + p2 + p3 + p4 = 75 →
  p2 / p1 = 2 := by
sorry

end NUMINAMATH_CALUDE_museum_trip_ratio_l1558_155874


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l1558_155899

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Represents the rolling properties of the cone -/
structure ConeRolling (cone : RightCircularCone) where
  rotations : ℕ
  no_slipping : Bool

theorem cone_rolling_ratio (cone : RightCircularCone) (rolling : ConeRolling cone) :
  rolling.rotations = 19 ∧ rolling.no_slipping = true →
  cone.h / cone.r = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l1558_155899


namespace NUMINAMATH_CALUDE_arc_length_for_45_degree_angle_l1558_155813

/-- Given a circle with circumference 90 meters and a central angle of 45°,
    the length of the corresponding arc is 11.25 meters. -/
theorem arc_length_for_45_degree_angle (D : Real) (E F : Real) : 
  D = 90 →  -- circumference of circle D is 90 meters
  (E - F) = 45 * π / 180 →  -- central angle ∠EDF is 45° (converted to radians)
  D * (E - F) / (2 * π) = 11.25 :=  -- length of arc EF
by sorry

end NUMINAMATH_CALUDE_arc_length_for_45_degree_angle_l1558_155813


namespace NUMINAMATH_CALUDE_cross_product_scalar_m_l1558_155877

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the cross product operation
variable (cross : V → V → V)

-- Axioms for cross product
variable (cross_anticomm : ∀ a b : V, cross a b = - cross b a)
variable (cross_distributive : ∀ a b c : V, cross a (b + c) = cross a b + cross a c)
variable (cross_zero : ∀ a : V, cross a a = 0)

-- The main theorem
theorem cross_product_scalar_m (m : ℝ) : 
  (∀ u v w : V, u + v + w = 0 → 
    m • (cross v u) + cross v w + cross w u = cross v u) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_cross_product_scalar_m_l1558_155877


namespace NUMINAMATH_CALUDE_factory_cost_l1558_155821

/-- Calculates the total cost of employing all workers for one day --/
def total_cost (total_employees : ℕ) 
  (group1_count : ℕ) (group1_rate : ℚ) (group1_regular_hours : ℕ)
  (group2_count : ℕ) (group2_rate : ℚ) (group2_regular_hours : ℕ)
  (group3_count : ℕ) (group3_rate : ℚ) (group3_regular_hours : ℕ) (group3_flat_rate : ℚ)
  (total_hours : ℕ) : ℚ :=
  let group1_cost := group1_count * (
    group1_rate * group1_regular_hours +
    group1_rate * 1.5 * (total_hours - group1_regular_hours)
  )
  let group2_cost := group2_count * (
    group2_rate * group2_regular_hours +
    group2_rate * 2 * (total_hours - group2_regular_hours)
  )
  let group3_cost := group3_count * (
    group3_rate * group3_regular_hours + group3_flat_rate
  )
  group1_cost + group2_cost + group3_cost

/-- Theorem stating the total cost for the given problem --/
theorem factory_cost : 
  total_cost 500 300 15 8 100 18 10 100 20 8 50 12 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_factory_cost_l1558_155821


namespace NUMINAMATH_CALUDE_initial_shells_l1558_155802

theorem initial_shells (initial_amount added_amount total_amount : ℕ) 
  (h1 : added_amount = 12)
  (h2 : total_amount = 17)
  (h3 : initial_amount + added_amount = total_amount) :
  initial_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_l1558_155802


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1558_155887

/-- The x-coordinate of the intersection point of two lines -/
theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, k * x + b = b * x + k ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1558_155887


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1558_155835

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1558_155835


namespace NUMINAMATH_CALUDE_graduates_parents_l1558_155824

theorem graduates_parents (graduates : ℕ) (teachers : ℕ) (total_chairs : ℕ)
  (h_graduates : graduates = 50)
  (h_teachers : teachers = 20)
  (h_total_chairs : total_chairs = 180) :
  (total_chairs - (graduates + teachers + teachers / 2)) / graduates = 2 := by
  sorry

end NUMINAMATH_CALUDE_graduates_parents_l1558_155824


namespace NUMINAMATH_CALUDE_existence_of_four_numbers_l1558_155848

theorem existence_of_four_numbers (x y : ℝ) : 
  ∃ (a₁ a₂ a₃ a₄ : ℝ), x = a₁ + a₂ + a₃ + a₄ ∧ y = 1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_numbers_l1558_155848


namespace NUMINAMATH_CALUDE_fraction_power_rule_l1558_155827

theorem fraction_power_rule (a b : ℝ) (hb : b ≠ 0) : (a / b) ^ 4 = a ^ 4 / b ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_rule_l1558_155827


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l1558_155860

/-- The line equation 3y - 4x = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * y - 4 * x = 12

/-- The x-axis equation y = 0 -/
def x_axis (y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-3, 0)

theorem line_intersects_x_axis :
  let (x, y) := intersection_point
  line_equation x y ∧ x_axis y := by sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l1558_155860


namespace NUMINAMATH_CALUDE_solve_equation_l1558_155834

theorem solve_equation (x : ℝ) :
  (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) → x = -15 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1558_155834


namespace NUMINAMATH_CALUDE_sequence_term_relation_l1558_155866

theorem sequence_term_relation (k : ℕ) (h_k : k > 1) :
  ∃ (a : ℕ → ℝ),
    (∀ n, a n ≥ a (n + 1)) ∧
    (∑' n, a n) = 1 ∧
    a 1 = 1 / (2 * k) ∧
    ∃ i₁ i₂ : Fin k, 
      ∀ j : Fin k, 
        a (i₁ : ℕ) ≥ a ((j : ℕ) + 1) ∧ 
        a ((i₂ : ℕ) + 1) > (1/2) * a (i₁ : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_relation_l1558_155866


namespace NUMINAMATH_CALUDE_initial_group_size_l1558_155845

theorem initial_group_size (X : ℕ) : 
  X - 6 + 5 - 2 + 3 = 13 → X = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_group_size_l1558_155845


namespace NUMINAMATH_CALUDE_pf_length_l1558_155850

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) where
  right_angled : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3
  pr_length : Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) = 3 * Real.sqrt 3

-- Define the altitude PL and median RM
def altitude (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def median (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point F
def intersectionPoint (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem pf_length (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let F := intersectionPoint P Q R
  Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = 0.857 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_pf_length_l1558_155850


namespace NUMINAMATH_CALUDE_A_less_than_B_l1558_155804

theorem A_less_than_B (x y : ℝ) : 
  let A := -y^2 + 4*x - 3
  let B := x^2 + 2*x + 2*y
  A < B := by sorry

end NUMINAMATH_CALUDE_A_less_than_B_l1558_155804


namespace NUMINAMATH_CALUDE_lisa_patricia_ratio_l1558_155842

/-- Represents the money each person has -/
structure Money where
  patricia : ℕ
  lisa : ℕ
  charlotte : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (m : Money) : Prop :=
  m.patricia = 6 ∧
  m.lisa = 2 * m.charlotte ∧
  m.patricia + m.lisa + m.charlotte = 51

theorem lisa_patricia_ratio (m : Money) :
  baseball_card_problem m →
  m.lisa / m.patricia = 5 := by
  sorry

end NUMINAMATH_CALUDE_lisa_patricia_ratio_l1558_155842


namespace NUMINAMATH_CALUDE_area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l1558_155816

/-- The area of the quadrilateral formed by the circumcenters of four equilateral triangles
    erected on the sides of a unit square (one inside, three outside) -/
theorem area_of_circumcenter_quadrilateral : ℝ :=
  let square_side_length : ℝ := 1
  let triangle_side_length : ℝ := 1
  let inside_triangle_count : ℕ := 1
  let outside_triangle_count : ℕ := 3
  (3 + Real.sqrt 3) / 6

/-- Proof of the area of the quadrilateral ABCD -/
theorem area_of_ABCD_proof :
  area_of_circumcenter_quadrilateral = (3 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l1558_155816


namespace NUMINAMATH_CALUDE_banana_division_existence_l1558_155869

theorem banana_division_existence :
  ∃ (n : ℕ) (b₁ b₂ b₃ b₄ : ℕ),
    n = b₁ + b₂ + b₃ + b₄ ∧
    (5 * (5 * b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     3 * (b₁ + 10 * b₂ + 8 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     2 * (b₁ + 4 * b₂ + 9 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     (b₁ + 4 * b₂ + 8 * b₃ + 12 * b₄)) ∧
    (15 ∣ b₁) ∧ (15 ∣ b₂) ∧ (27 ∣ b₃) ∧ (36 ∣ b₄) :=
by sorry

#check banana_division_existence

end NUMINAMATH_CALUDE_banana_division_existence_l1558_155869


namespace NUMINAMATH_CALUDE_semicircle_chord_projection_l1558_155896

/-- Given a semicircle with diameter 2R and a chord intersecting the semicircle and its tangent,
    prove that the condition AC^2 + CD^2 + BD^2 = 4a^2 has a solution for the projection of C on AB
    if and only if a^2 ≥ R^2, and that this solution is unique. -/
theorem semicircle_chord_projection (R a : ℝ) (h : R > 0) :
  ∃! x, x > 0 ∧ x < 2*R ∧ 
    2*R*x + (4*R^2*(2*R - x)^2)/x^2 + (4*R^2*(2*R - x))/x = 4*a^2 ↔ 
  a^2 ≥ R^2 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_chord_projection_l1558_155896


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l1558_155833

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √3(a - c cos B) = b sin C -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.a - t.c * Real.cos t.B) = t.b * Real.sin t.C

theorem angle_C_is_pi_third (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem max_area_when_a_b_equal_c (t : Triangle) (h : condition t) (hc : t.c = 2) :
  (∀ t' : Triangle, condition t' → t'.c = 2 → t.a * t.b ≥ t'.a * t'.b) →
  t.a = 2 ∧ t.b = 2 := by
  sorry

theorem max_area_is_sqrt_three (t : Triangle) (h : condition t) (hc : t.c = 2)
  (hmax : t.a = 2 ∧ t.b = 2) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l1558_155833


namespace NUMINAMATH_CALUDE_school_seats_cost_l1558_155879

/-- Calculate the total cost of seats with a group discount -/
def totalCostWithDiscount (rows : ℕ) (seatsPerRow : ℕ) (costPerSeat : ℕ) (discountPercent : ℕ) : ℕ :=
  let totalSeats := rows * seatsPerRow
  let fullGroupsOf10 := totalSeats / 10
  let costPer10Seats := 10 * costPerSeat
  let discountPer10Seats := costPer10Seats * discountPercent / 100
  let costPer10SeatsAfterDiscount := costPer10Seats - discountPer10Seats
  fullGroupsOf10 * costPer10SeatsAfterDiscount

theorem school_seats_cost :
  totalCostWithDiscount 5 8 30 10 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_school_seats_cost_l1558_155879


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1558_155847

theorem modular_arithmetic_problem (n : ℕ) : 
  n < 19 ∧ (5 * n) % 19 = 1 → ((3^n)^2 - 3) % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1558_155847


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1558_155858

/-- Given that the line x + y = b is a perpendicular bisector of the line segment from (2,4) to (6,8), prove that b = 10. -/
theorem perpendicular_bisector_b_value : 
  ∀ b : ℝ, 
  (∀ x y : ℝ, x + y = b ↔ 
    (x - 4)^2 + (y - 6)^2 = (2 - 4)^2 + (4 - 6)^2 ∧ 
    (x - 4)^2 + (y - 6)^2 = (6 - 4)^2 + (8 - 6)^2) → 
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1558_155858


namespace NUMINAMATH_CALUDE_optimal_bottle_volume_l1558_155898

theorem optimal_bottle_volume (vol1 vol2 vol3 : ℕ) 
  (h1 : vol1 = 4200) (h2 : vol2 = 3220) (h3 : vol3 = 2520) :
  Nat.gcd vol1 (Nat.gcd vol2 vol3) = 140 := by
  sorry

end NUMINAMATH_CALUDE_optimal_bottle_volume_l1558_155898


namespace NUMINAMATH_CALUDE_inequality_proof_l1558_155863

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1558_155863


namespace NUMINAMATH_CALUDE_magician_reappeared_count_l1558_155854

/-- Represents the magician's performance statistics --/
structure MagicianStats where
  total_shows : ℕ
  min_audience : ℕ
  max_audience : ℕ
  disappear_ratio : ℕ
  no_reappear_prob : ℚ
  double_reappear_prob : ℚ
  triple_reappear_prob : ℚ

/-- Calculates the total number of people who reappeared in the magician's performances --/
def total_reappeared (stats : MagicianStats) : ℕ :=
  sorry

/-- Theorem stating that given the magician's performance statistics, 
    the total number of people who reappeared is 640 --/
theorem magician_reappeared_count (stats : MagicianStats) 
  (h1 : stats.total_shows = 100)
  (h2 : stats.min_audience = 50)
  (h3 : stats.max_audience = 500)
  (h4 : stats.disappear_ratio = 50)
  (h5 : stats.no_reappear_prob = 1/10)
  (h6 : stats.double_reappear_prob = 1/5)
  (h7 : stats.triple_reappear_prob = 1/20) :
  total_reappeared stats = 640 :=
sorry

end NUMINAMATH_CALUDE_magician_reappeared_count_l1558_155854


namespace NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_l1558_155876

/-- Two events are complementary if one event occurs if and only if the other does not occur -/
def complementary_events (Ω : Type*) (A B : Set Ω) : Prop :=
  A = (Bᶜ)

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (Ω : Type*) (A B : Set Ω) : Prop :=
  A ∩ B = ∅

/-- The probability of an event is a number between 0 and 1 inclusive -/
axiom probability_range (Ω : Type*) (A : Set Ω) :
  ∃ (P : Set Ω → ℝ), 0 ≤ P A ∧ P A ≤ 1

theorem complementary_implies_mutually_exclusive (Ω : Type*) (A B : Set Ω) :
  complementary_events Ω A B → mutually_exclusive Ω A B :=
sorry

end NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_l1558_155876


namespace NUMINAMATH_CALUDE_smallest_b_value_l1558_155873

/-- Given real numbers a and b where 2 < a < b, and no triangle with positive area
    has side lengths 2, a, and b or 1/b, 1/a, and 1/2, the smallest possible value of b is 6. -/
theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2))
  (h4 : ¬ (1/b + 1/a > 1/2 ∧ 1/b + 1/2 > 1/a ∧ 1/a + 1/2 > 1/b)) :
  6 ≤ b ∧ ∀ c, (2 < c → c < b → 
    ¬(2 + c > b ∧ 2 + b > c ∧ c + b > 2) → 
    ¬(1/b + 1/c > 1/2 ∧ 1/b + 1/2 > 1/c ∧ 1/c + 1/2 > 1/b) → 
    6 ≤ c) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1558_155873


namespace NUMINAMATH_CALUDE_largest_value_l1558_155809

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def digits_85_9 : List Nat := [8, 5]
def digits_111111_2 : List Nat := [1, 1, 1, 1, 1, 1]
def digits_1000_4 : List Nat := [1, 0, 0, 0]
def digits_210_6 : List Nat := [2, 1, 0]

theorem largest_value :
  let a := to_decimal digits_85_9 9
  let b := to_decimal digits_111111_2 2
  let c := to_decimal digits_1000_4 4
  let d := to_decimal digits_210_6 6
  d > a ∧ d > b ∧ d > c := by sorry

end NUMINAMATH_CALUDE_largest_value_l1558_155809


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1558_155840

def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) - f x = 2 * x + 1) ∧
  (∀ x, f (-x) = f x) ∧
  (f 0 = 1)

theorem quadratic_function_properties (f : ℝ → ℝ) (h : QuadraticFunction f) :
  (∀ x, f x = x^2 + 1) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1558_155840


namespace NUMINAMATH_CALUDE_complex_square_root_expression_l1558_155889

theorem complex_square_root_expression : 71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_expression_l1558_155889


namespace NUMINAMATH_CALUDE_locus_of_point_P_l1558_155831

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- A line with slope 1 -/
def line_slope_1 (x y x' y' : ℝ) : Prop := y = x + (y' - x')

/-- The locus equation -/
def locus (x y : ℝ) : Prop := 148*x^2 + 13*y^2 + 64*x*y - 20 = 0

/-- The theorem statement -/
theorem locus_of_point_P :
  ∀ (x' y' x1 y1 x2 y2 : ℝ),
  ellipse x1 y1 ∧ ellipse x2 y2 ∧  -- A and B are on the ellipse
  line_slope_1 x1 y1 x' y' ∧ line_slope_1 x2 y2 x' y' ∧  -- A, B, and P are on a line with slope 1
  x' = (x1 + 2*x2) / 3 ∧  -- AP = 2PB condition
  x1 < x2 →  -- Ensure A and B are distinct points
  locus x' y' :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l1558_155831


namespace NUMINAMATH_CALUDE_bottom_row_bricks_l1558_155857

/-- Represents a brick wall with decreasing number of bricks in each row -/
structure BrickWall where
  totalRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat
  decreaseRate : Nat
  rowsDecreasing : bottomRowBricks ≥ (totalRows - 1) * decreaseRate

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : Nat) (an : Nat) (n : Nat) : Nat :=
  n * (a1 + an) / 2

/-- Theorem stating that a brick wall with given properties has 43 bricks in the bottom row -/
theorem bottom_row_bricks (wall : BrickWall)
  (h1 : wall.totalRows = 10)
  (h2 : wall.totalBricks = 385)
  (h3 : wall.decreaseRate = 1)
  : wall.bottomRowBricks = 43 := by
  sorry

end NUMINAMATH_CALUDE_bottom_row_bricks_l1558_155857


namespace NUMINAMATH_CALUDE_unique_solution_l1558_155851

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - 23*y + 66*z + 612 = 0 ∧
  y^2 + 62*x - 20*z + 296 = 0 ∧
  z^2 - 22*x + 67*y + 505 = 0

/-- The theorem stating that (-20, -22, -23) is the unique solution to the system -/
theorem unique_solution :
  ∃! (x y z : ℝ), system x y z ∧ x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1558_155851


namespace NUMINAMATH_CALUDE_gift_worth_l1558_155885

/-- Calculates the worth of each gift given the company's structure and budget --/
theorem gift_worth (num_blocks : ℕ) (workers_per_block : ℕ) (total_budget : ℕ) : 
  num_blocks = 15 → 
  workers_per_block = 200 → 
  total_budget = 6000 → 
  (total_budget : ℚ) / (num_blocks * workers_per_block : ℚ) = 2 := by
  sorry

#check gift_worth

end NUMINAMATH_CALUDE_gift_worth_l1558_155885


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1558_155846

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) + 
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1558_155846


namespace NUMINAMATH_CALUDE_profit_percentage_l1558_155838

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 - 0.81) / 0.81 * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1558_155838


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l1558_155875

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℕ+), n > 10 → ¬∃ (a b c : ℕ+), 
    n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 5*a + 5*b + 5*c - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l1558_155875


namespace NUMINAMATH_CALUDE_yo_yo_count_l1558_155810

theorem yo_yo_count 
  (x y z w : ℕ) 
  (h1 : x + y + w = 80)
  (h2 : (3/5 : ℚ) * 300 + (1/5 : ℚ) * 300 = x + y + z + w + 15)
  (h3 : x + y + z + w = 300 - ((3/5 : ℚ) * 300 + (1/5 : ℚ) * 300)) :
  z = 145 := by
  sorry

#check yo_yo_count

end NUMINAMATH_CALUDE_yo_yo_count_l1558_155810


namespace NUMINAMATH_CALUDE_certain_number_problem_l1558_155890

theorem certain_number_problem (x : ℝ) (y : ℝ) : 
  x = y + 0.5 * y → x = 132 → y = 88 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1558_155890


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l1558_155892

/-- Given a train of length 1200 m that crosses a tree in 120 sec,
    prove that it takes 190 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 1200 →
    platform_length = 700 →
    tree_crossing_time = 120 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 190 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l1558_155892


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l1558_155806

/-- Given two quadratic equations with a common root, this theorem proves
    properties about the sum and product of the other two roots. -/
theorem quadratic_equations_common_root
  (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a ≠ b)
  (h_common : ∃ x₀ : ℝ, x₀^2 + a*x₀ + b = 0 ∧ x₀^2 + b*x₀ + a = 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : x₁^2 + a*x₁ + b = 0)
  (h_x₂ : x₂^2 + b*x₂ + a = 0) :
  (x₁ + x₂ = -1) ∧ (x₁ * x₂ ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l1558_155806


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l1558_155849

/-- Represents the ratio of two integers as a pair of integers -/
def Ratio := ℤ × ℤ

/-- Somu's present age in years -/
def somu_age : ℕ := 10

/-- Calculates the father's age given Somu's age -/
def father_age (s : ℕ) : ℕ :=
  5 * (s - 5) + 5

/-- Simplifies a ratio by dividing both numbers by their greatest common divisor -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := r.1.gcd r.2
  (r.1 / gcd, r.2 / gcd)

theorem somu_father_age_ratio :
  simplify_ratio (somu_age, father_age somu_age) = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l1558_155849
