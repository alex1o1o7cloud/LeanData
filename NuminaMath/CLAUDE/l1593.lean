import Mathlib

namespace NUMINAMATH_CALUDE_infinite_series_sum_l1593_159370

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1593_159370


namespace NUMINAMATH_CALUDE_line_through_AB_l1593_159362

-- Define the lines and points
def line1 (a₁ b₁ x y : ℝ) : Prop := a₁ * x + b₁ * y + 1 = 0
def line2 (a₂ b₂ x y : ℝ) : Prop := a₂ * x + b₂ * y + 1 = 0
def point_P : ℝ × ℝ := (2, 3)
def point_A (a₁ b₁ : ℝ) : ℝ × ℝ := (a₁, b₁)
def point_B (a₂ b₂ : ℝ) : ℝ × ℝ := (a₂, b₂)

-- Define the theorem
theorem line_through_AB (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : line1 a₁ b₁ (point_P.1) (point_P.2))
  (h2 : line2 a₂ b₂ (point_P.1) (point_P.2))
  (h3 : a₁ ≠ a₂) :
  ∃ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (y - b₁) / (x - a₁) = (b₂ - b₁) / (a₂ - a₁) :=
by sorry

end NUMINAMATH_CALUDE_line_through_AB_l1593_159362


namespace NUMINAMATH_CALUDE_age_difference_l1593_159317

theorem age_difference (A B C : ℤ) 
  (h1 : A + B = B + C + 15) 
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1593_159317


namespace NUMINAMATH_CALUDE_derivative_limit_theorem_l1593_159392

open Real

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_limit_theorem (h : HasDerivAt f (-3) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ →
    |((f (x₀ + h) - f (x₀ - 3 * h)) / h) - (-12)| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_limit_theorem_l1593_159392


namespace NUMINAMATH_CALUDE_three_roots_range_l1593_159366

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem three_roots_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) →
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_roots_range_l1593_159366


namespace NUMINAMATH_CALUDE_tangent_ratio_equals_three_l1593_159303

theorem tangent_ratio_equals_three (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_equals_three_l1593_159303


namespace NUMINAMATH_CALUDE_perpendicular_point_k_range_l1593_159345

/-- Given points A(1,0) and B(3,0), if there exists a point P on the line y = kx + 1
    such that PA ⊥ PB, then -4/3 ≤ k ≤ 0. -/
theorem perpendicular_point_k_range (k : ℝ) :
  (∃ P : ℝ × ℝ, P.2 = k * P.1 + 1 ∧
    ((P.1 - 1) * (P.1 - 3) + P.2^2 = 0)) →
  -4/3 ≤ k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_point_k_range_l1593_159345


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_100_l1593_159301

theorem remainder_98_power_50_mod_100 : 98^50 % 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_100_l1593_159301


namespace NUMINAMATH_CALUDE_textbook_savings_l1593_159375

/-- Calculates the savings when buying textbooks from alternative bookshops instead of the school bookshop -/
theorem textbook_savings 
  (math_school_price : ℝ) 
  (science_school_price : ℝ) 
  (literature_school_price : ℝ)
  (math_discount : ℝ) 
  (science_discount : ℝ) 
  (literature_discount : ℝ)
  (school_tax_rate : ℝ)
  (alt_tax_rate : ℝ)
  (shipping_cost : ℝ)
  (h1 : math_school_price = 45)
  (h2 : science_school_price = 60)
  (h3 : literature_school_price = 35)
  (h4 : math_discount = 0.2)
  (h5 : science_discount = 0.25)
  (h6 : literature_discount = 0.15)
  (h7 : school_tax_rate = 0.07)
  (h8 : alt_tax_rate = 0.06)
  (h9 : shipping_cost = 10) :
  let school_total := (math_school_price + science_school_price + literature_school_price) * (1 + school_tax_rate)
  let alt_total := (math_school_price * (1 - math_discount) + 
                    science_school_price * (1 - science_discount) + 
                    literature_school_price * (1 - literature_discount)) * (1 + alt_tax_rate) + shipping_cost
  school_total - alt_total = 22.4 := by
  sorry


end NUMINAMATH_CALUDE_textbook_savings_l1593_159375


namespace NUMINAMATH_CALUDE_nine_digit_prime_square_product_l1593_159314

/-- Represents a nine-digit number of the form a₁a₂a₃b₁b₂b₃a₁a₂a₃ --/
def NineDigitNumber (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : ℕ :=
  a₁ * 100000000 + a₂ * 10000000 + a₃ * 1000000 + 
  b₁ * 100000 + b₂ * 10000 + b₃ * 1000 + 
  a₁ * 100 + a₂ * 10 + a₃

/-- Condition: ⎯⎯⎯⎯⎯b₁b₂b₃ = 2 * ⎯⎯⎯⎯⎯(a₁a₂a₃) --/
def MiddleIsDoubleFirst (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  b₁ * 100 + b₂ * 10 + b₃ = 2 * (a₁ * 100 + a₂ * 10 + a₃)

/-- The number is the product of the squares of four different prime numbers --/
def IsProductOfFourPrimeSquares (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n = p₁^2 * p₂^2 * p₃^2 * p₄^2

theorem nine_digit_prime_square_product :
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ : ℕ,
    a₁ ≠ 0 ∧
    MiddleIsDoubleFirst a₁ a₂ a₃ b₁ b₂ b₃ ∧
    IsProductOfFourPrimeSquares (NineDigitNumber a₁ a₂ a₃ b₁ b₂ b₃) :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_prime_square_product_l1593_159314


namespace NUMINAMATH_CALUDE_most_frequent_digit_l1593_159386

/-- The digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n - 1) % 9 + 1

/-- The count of occurrences of each digit (1-9) in the digital roots of numbers from 1 to 1,000,000 -/
def digitCounts : Fin 9 → ℕ
| ⟨i, _⟩ => if i = 0 then 111112 else 111111

theorem most_frequent_digit :
  ∃ (d : Fin 9), ∀ (d' : Fin 9), digitCounts d ≥ digitCounts d' ∧
  (d = ⟨0, by norm_num⟩ ∨ digitCounts d > digitCounts d') :=
sorry

end NUMINAMATH_CALUDE_most_frequent_digit_l1593_159386


namespace NUMINAMATH_CALUDE_power_sum_equality_l1593_159358

theorem power_sum_equality (a : ℕ) (h : 2^50 = a) :
  2^50 + 2^51 + 2^52 + 2^53 + 2^54 + 2^55 + 2^56 + 2^57 + 2^58 + 2^59 +
  2^60 + 2^61 + 2^62 + 2^63 + 2^64 + 2^65 + 2^66 + 2^67 + 2^68 + 2^69 +
  2^70 + 2^71 + 2^72 + 2^73 + 2^74 + 2^75 + 2^76 + 2^77 + 2^78 + 2^79 +
  2^80 + 2^81 + 2^82 + 2^83 + 2^84 + 2^85 + 2^86 + 2^87 + 2^88 + 2^89 +
  2^90 + 2^91 + 2^92 + 2^93 + 2^94 + 2^95 + 2^96 + 2^97 + 2^98 + 2^99 + 2^100 = 2*a^2 - a := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1593_159358


namespace NUMINAMATH_CALUDE_linear_function_slope_l1593_159363

theorem linear_function_slope (x₁ x₂ y₁ y₂ m : ℝ) :
  x₁ > x₂ →
  y₁ > y₂ →
  y₁ = (m - 3) * x₁ - 4 →
  y₂ = (m - 3) * x₂ - 4 →
  m > 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_slope_l1593_159363


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1593_159388

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_constraint : a + b + c + d + e = 3015) : 
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧ 
   ∀ N : ℕ, (N = max (a + b) (max (b + c) (max (c + d) (d + e))) → M ≤ N) ∧
   M = 755) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1593_159388


namespace NUMINAMATH_CALUDE_marks_money_theorem_l1593_159329

/-- The amount of money Mark's father gave him. -/
def fathers_money : ℕ := 85

/-- The number of books Mark bought. -/
def num_books : ℕ := 10

/-- The cost of each book in dollars. -/
def book_cost : ℕ := 5

/-- The amount of money Mark has left after buying the books. -/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Mark's father gave him is correct. -/
theorem marks_money_theorem :
  fathers_money = num_books * book_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_marks_money_theorem_l1593_159329


namespace NUMINAMATH_CALUDE_forest_area_relationship_l1593_159316

/-- 
Given forest areas a, b, c for three consecutive years,
with constant growth rate in the last two years,
prove that ac = b²
-/
theorem forest_area_relationship (a b c : ℝ) 
  (h : ∃ x : ℝ, b = a * (1 + x) ∧ c = b * (1 + x)) : 
  a * c = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_forest_area_relationship_l1593_159316


namespace NUMINAMATH_CALUDE_rals_current_age_l1593_159390

-- Define Suri's and Ral's ages as natural numbers
def suris_age : ℕ := sorry
def rals_age : ℕ := sorry

-- State the theorem
theorem rals_current_age : 
  (rals_age = 3 * suris_age) →   -- Ral is three times as old as Suri
  (suris_age + 3 = 16) →         -- In 3 years, Suri's current age will be 16
  rals_age = 39 :=               -- Ral's current age is 39
by sorry

end NUMINAMATH_CALUDE_rals_current_age_l1593_159390


namespace NUMINAMATH_CALUDE_female_salmon_count_l1593_159326

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l1593_159326


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_work_completion_time_l1593_159360

/-- If a person can complete a piece of work in a given number of days,
    then the time to complete a multiple of that work is proportional to the multiple. -/
theorem work_completion_time_proportional
  (original_days : ℕ) (work_multiple : ℕ) :
  original_days * work_multiple = original_days * work_multiple :=
by sorry

/-- Aarti's work completion time -/
theorem aarti_work_completion_time :
  let original_days : ℕ := 6
  let work_multiple : ℕ := 3
  original_days * work_multiple = 18 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_aarti_work_completion_time_l1593_159360


namespace NUMINAMATH_CALUDE_complex_power_four_l1593_159335

theorem complex_power_four (i : ℂ) : i^2 = -1 → 2 * i^4 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l1593_159335


namespace NUMINAMATH_CALUDE_min_m_value_l1593_159365

theorem min_m_value (x y m : ℝ) 
  (hx : 2 ≤ x ∧ x ≤ 3) 
  (hy : 3 ≤ y ∧ y ≤ 6) 
  (h : ∀ x y, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x*y + y^2 ≥ 0) : 
  m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1593_159365


namespace NUMINAMATH_CALUDE_function_coefficient_l1593_159309

theorem function_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 - 2*x) →
  f (-1) = 4 →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_function_coefficient_l1593_159309


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1593_159361

/-- A natural number greater than 25 is semi-prime if it is the sum of two different prime numbers -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive natural numbers that are semi-prime is 5 -/
theorem max_consecutive_semi_primes :
  ∀ k : ℕ, (∀ n : ℕ, ∀ i : ℕ, i < k → IsSemiPrime (n + i)) → k ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1593_159361


namespace NUMINAMATH_CALUDE_find_multiple_of_q_l1593_159396

theorem find_multiple_of_q (q : ℤ) (m : ℤ) : 
  let x := 55 + 2*q
  let y := m*q + 41
  (q = 7 → x = y) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_multiple_of_q_l1593_159396


namespace NUMINAMATH_CALUDE_typing_problem_solution_l1593_159310

/-- Represents the typing speed and time for two typists -/
structure TypistData where
  x : ℝ  -- Time taken by first typist to type entire manuscript
  y : ℝ  -- Time taken by second typist to type entire manuscript

/-- Checks if the given typing times satisfy the manuscript typing conditions -/
def satisfiesConditions (d : TypistData) : Prop :=
  let totalPages : ℝ := 80
  let pagesTypedIn5Hours : ℝ := 65
  let timeDiff : ℝ := 3
  (totalPages / d.y - totalPages / d.x = timeDiff) ∧
  (5 * (totalPages / d.x + totalPages / d.y) = pagesTypedIn5Hours)

/-- Theorem stating the solution to the typing problem -/
theorem typing_problem_solution :
  ∃ d : TypistData, satisfiesConditions d ∧ d.x = 10 ∧ d.y = 16 := by
  sorry


end NUMINAMATH_CALUDE_typing_problem_solution_l1593_159310


namespace NUMINAMATH_CALUDE_total_packs_is_35_l1593_159359

/-- The number of packs sold by Lucy -/
def lucy_packs : ℕ := 19

/-- The number of packs sold by Robyn -/
def robyn_packs : ℕ := 16

/-- The total number of packs sold by Robyn and Lucy -/
def total_packs : ℕ := lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold is 35 -/
theorem total_packs_is_35 : total_packs = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_is_35_l1593_159359


namespace NUMINAMATH_CALUDE_line_parallel_to_y_axis_l1593_159311

/-- A line parallel to the y-axis passing through a point has a constant x-coordinate -/
theorem line_parallel_to_y_axis (x₀ y₀ : ℝ) :
  let L := {p : ℝ × ℝ | p.1 = x₀}
  ((-1, 3) ∈ L) → (∀ p ∈ L, ∀ q ∈ L, p.2 ≠ q.2 → p.1 = q.1) →
  (∀ p ∈ L, p.1 = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_y_axis_l1593_159311


namespace NUMINAMATH_CALUDE_angle_y_value_l1593_159369

-- Define the angles in the problem
def angle_ACB : ℝ := 45
def angle_ABC : ℝ := 90
def angle_CDE : ℝ := 72

-- Define the theorem
theorem angle_y_value : 
  ∀ (angle_BAC angle_ADE angle_AED angle_DEB : ℝ),
  -- Triangle ABC
  angle_BAC + angle_ACB + angle_ABC = 180 →
  -- Angle ADC is a straight angle
  angle_ADE + angle_CDE = 180 →
  -- Triangle AED
  angle_AED + angle_ADE + angle_BAC = 180 →
  -- Angle AEB is a straight angle
  angle_AED + angle_DEB = 180 →
  -- Conclusion
  angle_DEB = 153 :=
by sorry

end NUMINAMATH_CALUDE_angle_y_value_l1593_159369


namespace NUMINAMATH_CALUDE_inequality_system_solution_fractional_equation_no_solution_l1593_159382

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  (1 - x ≤ 2) ∧ ((x + 1) / 2 + (x - 1) / 3 < 1)

theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 1} :=
sorry

-- Part 2: Fractional equation
def fractional_equation (x : ℝ) : Prop :=
  (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4)

theorem fractional_equation_no_solution :
  ¬∃ x : ℝ, fractional_equation x :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_fractional_equation_no_solution_l1593_159382


namespace NUMINAMATH_CALUDE_charlie_max_success_ratio_l1593_159313

theorem charlie_max_success_ratio 
  (alpha_first_two : ℚ)
  (alpha_last_two : ℚ)
  (charlie_daily : ℕ → ℚ)
  (charlie_attempted : ℕ → ℕ)
  (h1 : alpha_first_two = 120 / 200)
  (h2 : alpha_last_two = 80 / 200)
  (h3 : ∀ i ∈ Finset.range 4, 0 < charlie_daily i)
  (h4 : ∀ i ∈ Finset.range 4, charlie_daily i < 1)
  (h5 : ∀ i ∈ Finset.range 2, charlie_daily i < alpha_first_two)
  (h6 : ∀ i ∈ Finset.range 2, charlie_daily (i + 2) < alpha_last_two)
  (h7 : ∀ i ∈ Finset.range 4, charlie_attempted i > 0)
  (h8 : charlie_attempted 0 + charlie_attempted 1 < 200)
  (h9 : (charlie_attempted 0 + charlie_attempted 1 + charlie_attempted 2 + charlie_attempted 3) = 400)
  : (charlie_daily 0 * charlie_attempted 0 + charlie_daily 1 * charlie_attempted 1 + 
     charlie_daily 2 * charlie_attempted 2 + charlie_daily 3 * charlie_attempted 3) / 400 ≤ 239 / 400 :=
sorry

end NUMINAMATH_CALUDE_charlie_max_success_ratio_l1593_159313


namespace NUMINAMATH_CALUDE_shorter_lateral_side_length_l1593_159349

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- One angle of the trapezoid -/
  angle : ℝ
  /-- The midline (median) of the trapezoid -/
  midline : ℝ
  /-- One of the bases of the trapezoid -/
  base : ℝ
  /-- The angle is 30 degrees -/
  angle_is_30 : angle = 30
  /-- The lines containing the lateral sides intersect at a right angle -/
  lateral_sides_right_angle : True
  /-- The midline is 10 -/
  midline_is_10 : midline = 10
  /-- One base is 8 -/
  base_is_8 : base = 8

/-- The theorem stating the length of the shorter lateral side -/
theorem shorter_lateral_side_length (t : SpecialTrapezoid) : 
  ∃ (shorter_side : ℝ), shorter_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_shorter_lateral_side_length_l1593_159349


namespace NUMINAMATH_CALUDE_quadratic_inequality_bound_l1593_159376

theorem quadratic_inequality_bound (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x - 3) < d ↔ -5/2 < x ∧ x < 3) ↔ d = 39 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bound_l1593_159376


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1593_159315

/-- Calculates the selling price of an article given the gain and gain percentage -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) : 
  gain = 30 ∧ gain_percentage = 20 → 
  (gain / (gain_percentage / 100)) + gain = 180 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1593_159315


namespace NUMINAMATH_CALUDE_business_trip_distance_l1593_159340

theorem business_trip_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 8 →
  speed1 = 70 →
  speed2 = 85 →
  (total_time / 2 * speed1) + (total_time / 2 * speed2) = 620 := by
  sorry

end NUMINAMATH_CALUDE_business_trip_distance_l1593_159340


namespace NUMINAMATH_CALUDE_polar_equations_and_intersection_range_l1593_159330

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define the curve C
def curve_C (x y α : ℝ) : Prop := x = Real.cos α ∧ y = 1 + Real.sin α

-- Define the polar coordinates
def polar_coords (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_range :
  ∀ (x y ρ θ α β : ℝ),
  (0 < β ∧ β < Real.pi / 2) →
  (line_l x →
    ∃ (ρ_l : ℝ), polar_coords x y ρ_l θ ∧ ρ_l * Real.cos θ = 2) ∧
  (curve_C x y α →
    ∃ (ρ_c : ℝ), polar_coords x y ρ_c θ ∧ ρ_c = 2 * Real.sin θ) ∧
  (∃ (ρ_p ρ_m : ℝ),
    polar_coords x y ρ_p β ∧
    curve_C x y α ∧
    polar_coords x y ρ_m β ∧
    line_l x ∧
    0 < ρ_p / ρ_m ∧ ρ_p / ρ_m ≤ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_polar_equations_and_intersection_range_l1593_159330


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_condition_l1593_159343

/-- A quadratic function f(x) = ax^2 + bx + c is monotonically increasing on [1, +∞)
    if and only if b ≥ -2a, where a > 0. -/
theorem quadratic_monotone_increasing_condition (a b c : ℝ) (ha : a > 0) :
  (∀ x y, x ∈ Set.Ici (1 : ℝ) → y ∈ Set.Ici (1 : ℝ) → x ≤ y →
    a * x^2 + b * x + c ≤ a * y^2 + b * y + c) ↔
  b ≥ -2 * a :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_condition_l1593_159343


namespace NUMINAMATH_CALUDE_inequality_proof_l1593_159342

theorem inequality_proof (a b c : ℝ) : 
  a = Real.sqrt ((1 - Real.cos (110 * π / 180)) / 2) →
  b = (Real.sqrt 2 / 2) * (Real.sin (20 * π / 180) + Real.cos (20 * π / 180)) →
  c = (1 + Real.tan (20 * π / 180)) / (1 - Real.tan (20 * π / 180)) →
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1593_159342


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_14_l1593_159348

def expression (x : ℝ) : ℝ := 2 * (x - 6) + 5 * (3 - 3 * x^2 + 6 * x) - 6 * (3 * x - 5)

theorem coefficient_of_x_is_14 : 
  ∃ a b c : ℝ, ∀ x : ℝ, expression x = a * x^2 + 14 * x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_14_l1593_159348


namespace NUMINAMATH_CALUDE_three_over_x_equals_one_l1593_159377

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_over_x_equals_one_l1593_159377


namespace NUMINAMATH_CALUDE_total_chairs_is_528_l1593_159323

/-- Calculates the total number of chairs carried to the hall by Kingsley and her friends -/
def total_chairs : ℕ :=
  let kingsley_chairs := 7
  let friend_chairs := [6, 8, 5, 9, 7]
  let trips := List.range 6 |>.map (λ i => 10 + i)
  (kingsley_chairs :: friend_chairs).zip trips
  |>.map (λ (chairs, trip) => chairs * trip)
  |>.sum

/-- Theorem stating that the total number of chairs carried is 528 -/
theorem total_chairs_is_528 : total_chairs = 528 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_is_528_l1593_159323


namespace NUMINAMATH_CALUDE_son_age_problem_l1593_159352

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l1593_159352


namespace NUMINAMATH_CALUDE_solve_equation_l1593_159372

theorem solve_equation : ∃ x : ℝ, 60 + 5 * x / (180 / 3) = 61 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1593_159372


namespace NUMINAMATH_CALUDE_cone_sphere_volume_l1593_159324

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and its vertex and base circumference lying on a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →                  -- lateral surface radius
  r = l / 2 →                            -- base radius
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  2 * R * h = l^2 →                      -- sphere diameter relation
  (4 / 3) * π * R^3 = (32 * π) / 3 :=    -- sphere volume
by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_l1593_159324


namespace NUMINAMATH_CALUDE_no_rain_no_snow_probability_l1593_159320

theorem no_rain_no_snow_probability
  (rain_prob : ℚ)
  (snow_prob : ℚ)
  (rain_prob_def : rain_prob = 4 / 10)
  (snow_prob_def : snow_prob = 1 / 5)
  (events_independent : True) :
  (1 - rain_prob) * (1 - snow_prob) = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_no_snow_probability_l1593_159320


namespace NUMINAMATH_CALUDE_power_equality_l1593_159389

theorem power_equality (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1593_159389


namespace NUMINAMATH_CALUDE_ice_cream_problem_l1593_159353

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def quarters : ℕ := 26
def double_scoop_cost : ℚ := 3
def leftover : ℚ := 48/100

def total_amount : ℚ := 
  pennies * 1/100 + nickels * 5/100 + dimes * 10/100 + quarters * 25/100

theorem ice_cream_problem : 
  ∃ (n : ℕ), n * double_scoop_cost = total_amount - leftover ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_problem_l1593_159353


namespace NUMINAMATH_CALUDE_tournament_committee_count_l1593_159395

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the winning team -/
def winning_team_selection : ℕ := 3

/-- The number of members selected from each non-winning team -/
def other_team_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of possible tournament committees -/
def num_committees : ℕ := 4917248

theorem tournament_committee_count :
  num_committees = 
    num_teams * (Nat.choose team_size winning_team_selection) * 
    (Nat.choose team_size other_team_selection) ^ (num_teams - 1) := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l1593_159395


namespace NUMINAMATH_CALUDE_k_range_l1593_159354

theorem k_range (x y k : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : ∀ x y, x > 0 → y > 0 → Real.sqrt x + 3 * Real.sqrt y < k * Real.sqrt (x + y)) : 
  k > Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l1593_159354


namespace NUMINAMATH_CALUDE_donuts_per_box_l1593_159306

theorem donuts_per_box (total_boxes : Nat) (boxes_given : Nat) (extra_donuts_given : Nat) (donuts_left : Nat) :
  total_boxes = 4 →
  boxes_given = 1 →
  extra_donuts_given = 6 →
  donuts_left = 30 →
  ∃ (donuts_per_box : Nat), 
    donuts_per_box * total_boxes = 
      donuts_per_box * boxes_given + extra_donuts_given + donuts_left ∧
    donuts_per_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_donuts_per_box_l1593_159306


namespace NUMINAMATH_CALUDE_book_sale_revenue_l1593_159304

theorem book_sale_revenue (total_books : ℕ) (sold_price : ℕ) (remaining_books : ℕ) : 
  (2 : ℚ) / 3 * total_books = total_books - remaining_books ∧
  remaining_books = 50 ∧
  sold_price = 5 →
  (total_books - remaining_books) * sold_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l1593_159304


namespace NUMINAMATH_CALUDE_flower_beds_count_l1593_159383

theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) : 
  seeds_per_bed = 10 → total_seeds = 60 → num_beds * seeds_per_bed = total_seeds → num_beds = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l1593_159383


namespace NUMINAMATH_CALUDE_prob_white_given_popped_l1593_159336

/-- Represents the color of a kernel -/
inductive KernelColor
  | White
  | Yellow
  | Blue

/-- The probability of selecting a kernel of a given color -/
def selectProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 2/5
  | KernelColor.Yellow => 1/5
  | KernelColor.Blue => 2/5

/-- The probability of a kernel popping given its color -/
def popProb (c : KernelColor) : ℚ :=
  match c with
  | KernelColor.White => 1/4
  | KernelColor.Yellow => 3/4
  | KernelColor.Blue => 1/2

/-- The probability that a randomly selected kernel that popped was white -/
theorem prob_white_given_popped :
  (selectProb KernelColor.White * popProb KernelColor.White) /
  (selectProb KernelColor.White * popProb KernelColor.White +
   selectProb KernelColor.Yellow * popProb KernelColor.Yellow +
   selectProb KernelColor.Blue * popProb KernelColor.Blue) = 2/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_white_given_popped_l1593_159336


namespace NUMINAMATH_CALUDE_towel_length_decrease_l1593_159300

/-- Theorem: Percentage decrease in towel length
Given a towel that lost some percentage of its length and 20% of its breadth,
resulting in a 36% decrease in area, prove that the percentage decrease in length is 20%.
-/
theorem towel_length_decrease (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 0.8 * B →                         -- Breadth decreased by 20%
  L' * B' = 0.64 * (L * B) →             -- Area decreased by 36%
  L' = 0.8 * L                           -- Length decreased by 20%
  := by sorry

end NUMINAMATH_CALUDE_towel_length_decrease_l1593_159300


namespace NUMINAMATH_CALUDE_number_equation_l1593_159334

theorem number_equation (x : ℝ) : 38 + 2 * x = 124 ↔ x = 43 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1593_159334


namespace NUMINAMATH_CALUDE_temp_at_six_km_l1593_159364

/-- The temperature drop per kilometer of altitude increase -/
def temp_drop_per_km : ℝ := 5

/-- The temperature at ground level in Celsius -/
def ground_temp : ℝ := 25

/-- The height in kilometers at which we want to calculate the temperature -/
def target_height : ℝ := 6

/-- Calculates the temperature at a given height -/
def temp_at_height (h : ℝ) : ℝ := ground_temp - temp_drop_per_km * h

/-- Theorem stating that the temperature at 6 km height is -5°C -/
theorem temp_at_six_km : temp_at_height target_height = -5 := by sorry

end NUMINAMATH_CALUDE_temp_at_six_km_l1593_159364


namespace NUMINAMATH_CALUDE_partnership_capital_share_l1593_159350

theorem partnership_capital_share (T : ℝ) (x : ℝ) : 
  (x + 1/4 + 1/5 + (11/20 - x) = 1) →  -- Total shares add up to 1
  (810 / 2430 = x) →                   -- A's profit share equals capital share
  (x = 1/3) :=                         -- A's capital share is 1/3
by sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l1593_159350


namespace NUMINAMATH_CALUDE_honey_eaten_by_bears_l1593_159308

theorem honey_eaten_by_bears (initial_honey : Real) (remaining_honey : Real)
  (h1 : initial_honey = 0.36)
  (h2 : remaining_honey = 0.31) :
  initial_honey - remaining_honey = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_honey_eaten_by_bears_l1593_159308


namespace NUMINAMATH_CALUDE_part_one_part_two_l1593_159367

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x + a|

-- Part 1
theorem part_one :
  {x : ℝ | f 3 x ≤ 1/2} = {x : ℝ | x ≥ -11/4} := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  ({x : ℝ | f a x ≤ a} = Set.univ) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1593_159367


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l1593_159379

theorem square_area_error_percentage (x : ℝ) (h : x > 0) :
  let measured_side := 1.12 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 25.44 := by sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l1593_159379


namespace NUMINAMATH_CALUDE_max_min_kangaroo_weight_l1593_159373

theorem max_min_kangaroo_weight :
  ∀ (a b c : ℕ),
    a > 0 → b > 0 → c > 0 →
    a + b + c = 97 →
    min a (min b c) ≤ 32 ∧
    ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 97 ∧ min x (min y z) = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_min_kangaroo_weight_l1593_159373


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1593_159381

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → e = Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1593_159381


namespace NUMINAMATH_CALUDE_bakers_pastries_l1593_159325

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (total_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (sold_pastries : ℕ) 
  (remaining_pastries : ℕ) 
  (h1 : total_cakes = 124) 
  (h2 : sold_cakes = 104) 
  (h3 : sold_pastries = 29) 
  (h4 : remaining_pastries = 27) : 
  sold_pastries + remaining_pastries = 56 := by
  sorry

end NUMINAMATH_CALUDE_bakers_pastries_l1593_159325


namespace NUMINAMATH_CALUDE_symmetry_implies_m_equals_4_l1593_159331

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Checks if two points are symmetric with respect to the y-axis -/
def symmetric_y_axis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

theorem symmetry_implies_m_equals_4 (m : ℝ) :
  let a := Point2D.mk (-3) m
  let b := Point2D.mk 3 4
  symmetric_y_axis a b → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_equals_4_l1593_159331


namespace NUMINAMATH_CALUDE_log_base_4_properties_l1593_159380

noncomputable def y (x : ℝ) : ℝ := Real.log x / Real.log 4

theorem log_base_4_properties :
  (∀ x : ℝ, x = 1 → y x = 0) ∧
  (∀ x : ℝ, x = 4 → y x = 1) ∧
  (∀ x : ℝ, x = -4 → ¬ ∃ (r : ℝ), y x = r) ∧
  (∀ x : ℝ, 0 < x → x < 1 → y x < 0 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x', 0 < x' → x' < δ → y x' < -ε) :=
by sorry

end NUMINAMATH_CALUDE_log_base_4_properties_l1593_159380


namespace NUMINAMATH_CALUDE_incorrect_multiplication_result_l1593_159397

theorem incorrect_multiplication_result (x : ℕ) : 
  x * 153 = 109395 → x * 152 = 108680 := by
sorry

end NUMINAMATH_CALUDE_incorrect_multiplication_result_l1593_159397


namespace NUMINAMATH_CALUDE_hall_tables_l1593_159307

theorem hall_tables (total_chairs : ℕ) (tables_with_three : ℕ) : 
  total_chairs = 91 → tables_with_three = 5 →
  ∃ (total_tables : ℕ), 
    (total_tables / 2 : ℚ) * 2 + 
    (tables_with_three : ℚ) * 3 + 
    ((total_tables : ℚ) - (total_tables / 2 : ℚ) - (tables_with_three : ℚ)) * 4 = 
    total_chairs ∧ 
    total_tables = 32 := by
  sorry

end NUMINAMATH_CALUDE_hall_tables_l1593_159307


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1593_159384

theorem geometric_sequence_sixth_term
  (a : ℕ+) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 3)
  (h2 : a * r^3 = 243) -- fourth term condition
  : a * r^5 = 729 := by -- sixth term
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1593_159384


namespace NUMINAMATH_CALUDE_not_prime_if_two_square_sums_l1593_159398

theorem not_prime_if_two_square_sums (p a b x y : ℤ) 
  (sum1 : p = a^2 + b^2) 
  (sum2 : p = x^2 + y^2) 
  (diff_repr : (a, b) ≠ (x, y) ∧ (a, b) ≠ (y, x)) : 
  ¬ Nat.Prime p.natAbs := by
  sorry

end NUMINAMATH_CALUDE_not_prime_if_two_square_sums_l1593_159398


namespace NUMINAMATH_CALUDE_attic_junk_items_l1593_159347

theorem attic_junk_items (useful_percent : ℝ) (heirloom_percent : ℝ) (junk_percent : ℝ) (useful_count : ℕ) :
  useful_percent = 0.20 →
  heirloom_percent = 0.10 →
  junk_percent = 0.70 →
  useful_percent + heirloom_percent + junk_percent = 1 →
  useful_count = 8 →
  ⌊(useful_count / useful_percent) * junk_percent⌋ = 28 := by
sorry

end NUMINAMATH_CALUDE_attic_junk_items_l1593_159347


namespace NUMINAMATH_CALUDE_nancy_quarters_l1593_159305

/-- The number of quarters Nancy has -/
def number_of_quarters (total_amount : ℚ) (quarter_value : ℚ) : ℚ :=
  total_amount / quarter_value

theorem nancy_quarters : 
  let total_amount : ℚ := 3
  let quarter_value : ℚ := 1/4
  number_of_quarters total_amount quarter_value = 12 := by
sorry

end NUMINAMATH_CALUDE_nancy_quarters_l1593_159305


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1593_159346

theorem trig_expression_simplification (α : ℝ) :
  (Real.tan (2 * Real.pi + α)) / (Real.tan (α + Real.pi) - Real.cos (-α) + Real.sin (Real.pi / 2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l1593_159346


namespace NUMINAMATH_CALUDE_bus_speed_without_stoppages_l1593_159391

theorem bus_speed_without_stoppages 
  (speed_with_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stoppages = 45) 
  (h2 : stop_time = 10) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_speed_without_stoppages_l1593_159391


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1593_159356

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a = -2 → |a| = 2) ∧ 
  (∃ a, |a| = 2 ∧ a ≠ -2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1593_159356


namespace NUMINAMATH_CALUDE_unused_card_is_one_l1593_159357

def cards : Finset Nat := {1, 3, 4}

def largest_two_digit (a b : Nat) : Nat := 10 * max a b + min a b

def is_largest_two_digit (n : Nat) : Prop :=
  ∃ (a b : Nat), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧
  n = largest_two_digit a b ∧
  ∀ (x y : Nat), x ∈ cards → y ∈ cards → x ≠ y →
  largest_two_digit x y ≤ n

theorem unused_card_is_one :
  ∃ (n : Nat), is_largest_two_digit n ∧ (cards \ {n.div 10, n.mod 10}).toList = [1] := by
  sorry

end NUMINAMATH_CALUDE_unused_card_is_one_l1593_159357


namespace NUMINAMATH_CALUDE_zoe_calories_l1593_159368

/-- The number of calories Zoe ate from her snack -/
def total_calories (strawberry_count : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (calories_per_yogurt_ounce : ℕ) : ℕ :=
  strawberry_count * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

/-- Theorem stating that Zoe ate 150 calories -/
theorem zoe_calories : total_calories 12 6 4 17 = 150 := by
  sorry

end NUMINAMATH_CALUDE_zoe_calories_l1593_159368


namespace NUMINAMATH_CALUDE_min_red_tulips_for_arrangement_l1593_159378

/-- Represents the number of tulips in a bouquet -/
structure Bouquet where
  white : ℕ
  red : ℕ

/-- Represents the total number of tulips and bouquets -/
structure TulipArrangement where
  whiteTotal : ℕ
  redTotal : ℕ
  bouquetCount : ℕ

/-- Checks if a TulipArrangement is valid according to the problem constraints -/
def isValidArrangement (arr : TulipArrangement) : Prop :=
  ∃ (b : Bouquet),
    arr.whiteTotal = arr.bouquetCount * b.white ∧
    arr.redTotal = arr.bouquetCount * b.red

/-- The main theorem to prove -/
theorem min_red_tulips_for_arrangement :
  ∀ (arr : TulipArrangement),
    arr.whiteTotal = 21 →
    arr.bouquetCount = 7 →
    isValidArrangement arr →
    arr.redTotal ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_red_tulips_for_arrangement_l1593_159378


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_opposite_of_negative_2022_is_2022_l1593_159339

theorem opposite_of_negative_2022 : ℤ → Prop :=
  fun x => ((-2022 : ℤ) + x = 0) → x = 2022

-- The proof is omitted
theorem opposite_of_negative_2022_is_2022 : opposite_of_negative_2022 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_opposite_of_negative_2022_is_2022_l1593_159339


namespace NUMINAMATH_CALUDE_bicycle_distance_theorem_l1593_159302

/-- Represents a bicycle wheel -/
structure Wheel where
  perimeter : ℝ

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  backWheel : Wheel
  frontWheel : Wheel

/-- Calculates the distance traveled by a wheel given the number of revolutions -/
def distanceTraveled (wheel : Wheel) (revolutions : ℝ) : ℝ :=
  wheel.perimeter * revolutions

theorem bicycle_distance_theorem (bike : Bicycle) 
    (h1 : bike.backWheel.perimeter = 9)
    (h2 : bike.frontWheel.perimeter = 7)
    (h3 : ∃ (r : ℝ), distanceTraveled bike.backWheel r = distanceTraveled bike.frontWheel (r + 10)) :
  ∃ (d : ℝ), d = 315 ∧ ∃ (r : ℝ), d = distanceTraveled bike.backWheel r ∧ d = distanceTraveled bike.frontWheel (r + 10) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_theorem_l1593_159302


namespace NUMINAMATH_CALUDE_total_legs_sea_creatures_l1593_159337

/-- Calculate the total number of legs for sea creatures --/
theorem total_legs_sea_creatures :
  let num_octopuses : ℕ := 5
  let num_crabs : ℕ := 3
  let num_starfish : ℕ := 2
  let legs_per_octopus : ℕ := 8
  let legs_per_crab : ℕ := 10
  let legs_per_starfish : ℕ := 5
  num_octopuses * legs_per_octopus +
  num_crabs * legs_per_crab +
  num_starfish * legs_per_starfish = 80 :=
by sorry

end NUMINAMATH_CALUDE_total_legs_sea_creatures_l1593_159337


namespace NUMINAMATH_CALUDE_parabola_transformation_l1593_159318

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := λ x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := λ x => p.f x + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { f := λ x => 3 * x^2 }

/-- The final parabola after transformations -/
def final_parabola : Parabola :=
  { f := λ x => 3 * (x + 1)^2 - 2 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal original_parabola 1) (-2)).f = final_parabola.f :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1593_159318


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1593_159351

theorem algebraic_simplification (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x + 2) = -44 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1593_159351


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1593_159387

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^8
  let original_number : ℝ := 4947.66 * billion
  original_number = 4.94766 * 10^11 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1593_159387


namespace NUMINAMATH_CALUDE_mean_scores_equal_7_l1593_159327

def class1_scores : List Nat := [10, 9, 8, 7, 7, 7, 7, 5, 5, 5]
def class2_scores : List Nat := [9, 8, 8, 7, 7, 7, 7, 7, 5, 5]

def mean (scores : List Nat) : Rat :=
  (scores.sum : Rat) / scores.length

theorem mean_scores_equal_7 :
  mean class1_scores = 7 ∧ mean class2_scores = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_scores_equal_7_l1593_159327


namespace NUMINAMATH_CALUDE_dividend_calculation_l1593_159312

theorem dividend_calculation (dividend divisor : ℕ) : 
  dividend / divisor = 15 →
  dividend % divisor = 5 →
  dividend + divisor + 15 + 5 = 2169 →
  dividend = 2015 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1593_159312


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1593_159333

theorem sqrt_sum_inequality (x y α : ℝ) (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1593_159333


namespace NUMINAMATH_CALUDE_sum_of_squares_175_l1593_159399

theorem sum_of_squares_175 (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a^2 + b^2 + c^2 + d^2 = 175 →
  a + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_175_l1593_159399


namespace NUMINAMATH_CALUDE_sum_abc_l1593_159394

theorem sum_abc (a b c : ℝ) 
  (h1 : a - (b + c) = 16)
  (h2 : a^2 - (b + c)^2 = 1664) : 
  a + b + c = 104 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_l1593_159394


namespace NUMINAMATH_CALUDE_john_incentive_calculation_l1593_159344

/-- The incentive calculation for John's agency fees --/
theorem john_incentive_calculation 
  (commission : ℕ) 
  (advance_fees : ℕ) 
  (amount_given : ℕ) 
  (h1 : commission = 25000)
  (h2 : advance_fees = 8280)
  (h3 : amount_given = 18500) :
  amount_given - (commission - advance_fees) = 1780 :=
by sorry

end NUMINAMATH_CALUDE_john_incentive_calculation_l1593_159344


namespace NUMINAMATH_CALUDE_double_acute_angle_less_than_180_l1593_159341

theorem double_acute_angle_less_than_180 (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  2 * α < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_less_than_180_l1593_159341


namespace NUMINAMATH_CALUDE_rain_in_first_hour_l1593_159321

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by sorry

end NUMINAMATH_CALUDE_rain_in_first_hour_l1593_159321


namespace NUMINAMATH_CALUDE_circle_center_sum_l1593_159332

/-- Given a circle with equation x² + y² = 4x - 6y + 9, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1593_159332


namespace NUMINAMATH_CALUDE_A_intersect_B_l1593_159355

def A : Set ℕ := {0, 1, 2}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1593_159355


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosines_l1593_159338

theorem arithmetic_sequence_cosines (a : ℝ) : 
  (0 < a) ∧ (a < 2 * Real.pi) ∧ 
  (∃ d : ℝ, (Real.cos (2 * a) = Real.cos a + d) ∧ 
            (Real.cos (3 * a) = Real.cos (2 * a) + d)) ↔ 
  (a = Real.pi / 4) ∨ (a = 3 * Real.pi / 4) ∨ 
  (a = 5 * Real.pi / 4) ∨ (a = 7 * Real.pi / 4) :=
by sorry

#check arithmetic_sequence_cosines

end NUMINAMATH_CALUDE_arithmetic_sequence_cosines_l1593_159338


namespace NUMINAMATH_CALUDE_complement_union_M_N_l1593_159371

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l1593_159371


namespace NUMINAMATH_CALUDE_k_range_for_positive_f_l1593_159319

/-- Given a function f(x) = 32x - (k + 1)3^x + 2 that is always positive for real x,
    prove that k is in the range (-∞, 2^(-1)). -/
theorem k_range_for_positive_f (k : ℝ) :
  (∀ x : ℝ, 32 * x - (k + 1) * 3^x + 2 > 0) →
  k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_positive_f_l1593_159319


namespace NUMINAMATH_CALUDE_integer_solution_l1593_159328

theorem integer_solution (n : ℤ) : 
  n + 15 > 16 ∧ 4 * n < 20 ∧ |n - 2| ≤ 2 → n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l1593_159328


namespace NUMINAMATH_CALUDE_path_time_equality_implies_distance_ratio_l1593_159385

/-- Given two points A and B, and a point P between them, 
    if the time to go directly from P to B equals the time to go from P to A 
    and then from A to B at 6 times the speed, 
    then the ratio of PA to PB is 5/7 -/
theorem path_time_equality_implies_distance_ratio 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h_between : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is between A and B
  (speed : ℝ) -- walking speed
  (h_speed_pos : speed > 0) -- speed is positive
  : (dist P B / speed = dist P A / speed + (dist A B) / (6 * speed)) → 
    (dist P A / dist P B = 5 / 7) :=
by sorry

#check path_time_equality_implies_distance_ratio

end NUMINAMATH_CALUDE_path_time_equality_implies_distance_ratio_l1593_159385


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1593_159322

/-- Similarity transformation of a plane with coefficient k and center at the origin -/
def transform_plane (a b c d k : ℝ) : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ a * x + b * y + c * z + k * d = 0

/-- The point A -/
def A : ℝ × ℝ × ℝ := (-1, 2, 3)

/-- The original plane equation -/
def plane_a : ℝ → ℝ → ℝ → Prop :=
  fun x y z ↦ x - 3 * y + z + 2 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := 2.5

theorem point_not_on_transformed_plane :
  ¬ transform_plane 1 (-3) 1 2 k A.1 A.2.1 A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1593_159322


namespace NUMINAMATH_CALUDE_sequence_expression_l1593_159393

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^(n - 1)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l1593_159393


namespace NUMINAMATH_CALUDE_dividend_calculation_l1593_159374

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 9)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 160 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1593_159374
