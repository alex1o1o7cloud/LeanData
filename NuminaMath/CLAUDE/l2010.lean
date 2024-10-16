import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_proof_l2010_201063

def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def number_of_subjects : ℕ := 5

theorem average_marks_proof :
  (english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks) / number_of_subjects = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l2010_201063


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2010_201043

/-- Given a quadratic equation 3x² = -2x + 5, prove that it can be rewritten
    in the general form ax² + bx + c = 0 with specific coefficients. -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x^2 = -2 * x + 5) →
    (∀ x, a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = 2 ∧ c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2010_201043


namespace NUMINAMATH_CALUDE_number_puzzle_l2010_201020

theorem number_puzzle : ∃ N : ℚ, N = (3/8) * N + (1/4) * N + 15 ∧ N = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2010_201020


namespace NUMINAMATH_CALUDE_inverse_of_twelve_point_five_l2010_201009

theorem inverse_of_twelve_point_five (x : ℝ) : 1 / x = 12.5 → x = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_twelve_point_five_l2010_201009


namespace NUMINAMATH_CALUDE_divisibility_check_l2010_201077

theorem divisibility_check : 
  (5641713 % 29 ≠ 0) ∧ (1379235 % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_divisibility_check_l2010_201077


namespace NUMINAMATH_CALUDE_inequality_proof_l2010_201062

theorem inequality_proof (a b x : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hn : 2 ≤ n) 
  (h : x^n ≤ a*x + b) : 
  x < (2*a)^(1/(n-1 : ℝ)) + (2*b)^(1/n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2010_201062


namespace NUMINAMATH_CALUDE_paintings_not_in_both_collections_l2010_201086

theorem paintings_not_in_both_collections
  (andrew_total : ℕ)
  (shared : ℕ)
  (john_unique : ℕ)
  (h1 : andrew_total = 25)
  (h2 : shared = 15)
  (h3 : john_unique = 8) :
  andrew_total - shared + john_unique = 18 :=
by sorry

end NUMINAMATH_CALUDE_paintings_not_in_both_collections_l2010_201086


namespace NUMINAMATH_CALUDE_salesman_profit_is_442_l2010_201064

/-- Calculates the salesman's profit from selling backpacks -/
def salesmanProfit (totalBackpacks : ℕ) (totalCost : ℕ) 
  (firstBatchCount : ℕ) (firstBatchPrice : ℕ)
  (secondBatchCount : ℕ) (secondBatchPrice : ℕ)
  (remainingPrice : ℕ) : ℕ :=
  let remainingCount := totalBackpacks - firstBatchCount - secondBatchCount
  let totalSales := 
    firstBatchCount * firstBatchPrice + 
    secondBatchCount * secondBatchPrice + 
    remainingCount * remainingPrice
  totalSales - totalCost

/-- The salesman's profit is $442 given the specific conditions -/
theorem salesman_profit_is_442 : 
  salesmanProfit 48 576 17 18 10 25 22 = 442 := by
  sorry

end NUMINAMATH_CALUDE_salesman_profit_is_442_l2010_201064


namespace NUMINAMATH_CALUDE_ab_value_l2010_201095

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2010_201095


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2010_201056

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2010_201056


namespace NUMINAMATH_CALUDE_quadratic_roots_and_fraction_l2010_201051

theorem quadratic_roots_and_fraction (a b p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ (x = 2 + a*I ∨ x = b + I)) →
  (a = -1 ∧ b = 2 ∧ p = -4 ∧ q = 5) ∧
  (a + b*I) / (p + q*I) = 3/41 + 6/41*I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_fraction_l2010_201051


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2010_201036

theorem complex_modulus_equation : ∃ (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2010_201036


namespace NUMINAMATH_CALUDE_isabellas_travel_l2010_201087

/-- Proves that given the conditions of Isabella's travel and currency exchange, 
    the initial amount d is 120 U.S. dollars. -/
theorem isabellas_travel (d : ℚ) : 
  (8/5 * d - 72 = d) → d = 120 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_travel_l2010_201087


namespace NUMINAMATH_CALUDE_calculation_proof_l2010_201078

theorem calculation_proof : (35 / (8 + 3 - 5) - 2) * 4 = 46 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2010_201078


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2010_201046

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2010_201046


namespace NUMINAMATH_CALUDE_equation_system_properties_l2010_201033

/-- Represents a system of equations mx + ny² = 0 and mx² + ny² = 1 -/
structure EquationSystem where
  m : ℝ
  n : ℝ
  h_m_neg : m < 0
  h_n_pos : n > 0

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies both equations in the system -/
def satisfies_equations (sys : EquationSystem) (p : Point) : Prop :=
  sys.m * p.x + sys.n * p.y^2 = 0 ∧ sys.m * p.x^2 + sys.n * p.y^2 = 1

/-- States that the equation system represents a parabola -/
def is_parabola (sys : EquationSystem) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), sys.m * x + sys.n * y^2 = 0 ↔ y = a * x^2 + b * x + c

/-- States that the equation system represents a hyperbola -/
def is_hyperbola (sys : EquationSystem) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), sys.m * x^2 + sys.n * y^2 = 1 ↔ (x^2 / a^2) - (y^2 / b^2) = 1

theorem equation_system_properties (sys : EquationSystem) :
  is_parabola sys ∧ 
  is_hyperbola sys ∧ 
  satisfies_equations sys ⟨0, 0⟩ ∧ 
  satisfies_equations sys ⟨1, 0⟩ :=
sorry

end NUMINAMATH_CALUDE_equation_system_properties_l2010_201033


namespace NUMINAMATH_CALUDE_solve_equation_l2010_201016

theorem solve_equation (x : ℝ) (y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x + 1) (h3 : x ≠ 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2010_201016


namespace NUMINAMATH_CALUDE_crate_stack_probability_l2010_201013

-- Define the dimensions of a crate
def CrateDimensions : Fin 3 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 7

-- Define the number of crates
def NumCrates : ℕ := 15

-- Define the target height
def TargetHeight : ℕ := 50

-- Define the total number of possible arrangements
def TotalArrangements : ℕ := 3^NumCrates

-- Define the number of favorable arrangements
def FavorableArrangements : ℕ := 560

theorem crate_stack_probability :
  (FavorableArrangements : ℚ) / TotalArrangements = 560 / 14348907 := by
  sorry

#eval FavorableArrangements -- Should output 560

end NUMINAMATH_CALUDE_crate_stack_probability_l2010_201013


namespace NUMINAMATH_CALUDE_find_number_l2010_201041

theorem find_number : ∃ x : ℝ, 8 * x = 0.4 * 900 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2010_201041


namespace NUMINAMATH_CALUDE_rancher_lasso_probability_l2010_201025

/-- The probability of a rancher placing a lasso around a cow's neck in a single throw. -/
def single_throw_probability : ℚ := 1 / 2

/-- The number of attempts the rancher makes. -/
def number_of_attempts : ℕ := 3

/-- The probability of the rancher placing a lasso around a cow's neck at least once in the given number of attempts. -/
def success_probability : ℚ := 7 / 8

theorem rancher_lasso_probability :
  (1 : ℚ) - (1 - single_throw_probability) ^ number_of_attempts = success_probability :=
sorry

end NUMINAMATH_CALUDE_rancher_lasso_probability_l2010_201025


namespace NUMINAMATH_CALUDE_equal_sundays_tuesdays_count_l2010_201079

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a 30-day month -/
structure Month30 where
  firstDay : DayOfWeek

/-- Function to check if a 30-day month has equal Sundays and Tuesdays -/
def hasEqualSundaysAndTuesdays (m : Month30) : Prop :=
  -- Implementation details omitted
  sorry

/-- The number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
theorem equal_sundays_tuesdays_count :
  (∃ (days : Finset DayOfWeek),
    (∀ d : DayOfWeek, d ∈ days ↔ hasEqualSundaysAndTuesdays ⟨d⟩) ∧
    Finset.card days = 6) :=
  sorry

end NUMINAMATH_CALUDE_equal_sundays_tuesdays_count_l2010_201079


namespace NUMINAMATH_CALUDE_commercials_time_l2010_201023

/-- Given a total time and a ratio of music to commercials, 
    calculate the number of minutes of commercials played. -/
theorem commercials_time (total_time : ℕ) (music_ratio commercial_ratio : ℕ) 
  (h1 : total_time = 112)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  (total_time * commercial_ratio) / (music_ratio + commercial_ratio) = 40 := by
  sorry

#check commercials_time

end NUMINAMATH_CALUDE_commercials_time_l2010_201023


namespace NUMINAMATH_CALUDE_marble_probability_difference_l2010_201089

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 15 / 44985 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l2010_201089


namespace NUMINAMATH_CALUDE_problem_solution_l2010_201091

def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := x^2 + 2*x + 1 - m^4 ≤ 0

theorem problem_solution (m : ℝ) :
  (∀ x, q x m → p x) → (m ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ x, ¬(q x m) → ¬(p x)) → (m ≥ 3 ∨ m ≤ -3) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m))) → (m ≥ 3 ∨ m ≤ -3) :=
by sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l2010_201091


namespace NUMINAMATH_CALUDE_expression_evaluation_l2010_201065

theorem expression_evaluation : 
  (-1/2)⁻¹ + (π - 3)^0 + |1 - Real.sqrt 2| + Real.sin (45 * π / 180) * Real.sin (30 * π / 180) = 
  5 * Real.sqrt 2 / 4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2010_201065


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2010_201069

theorem sales_tax_difference (price : ℝ) (high_rate low_rate : ℝ) :
  price = 50 →
  high_rate = 0.0725 →
  low_rate = 0.0675 →
  price * high_rate - price * low_rate = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2010_201069


namespace NUMINAMATH_CALUDE_function_properties_l2010_201047

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| - |x - 25|

-- Define the theorem
theorem function_properties (a : ℝ) 
  (h : ∀ x, f x < 10 * a + 10) : 
  a > 1/2 ∧ ∃ (min_value : ℝ), min_value = 9 ∧ 
  ∀ a, a > 1/2 → 2 * a + 27 / (a^2) ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2010_201047


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_using_all_digits_l2010_201050

/-- A function that checks if a number uses each digit from 0 to 9 exactly once -/
def usesAllDigitsOnce (n : ℕ) : Prop := sorry

/-- A function that returns the largest number that can be formed using each digit from 0 to 9 exactly once and is a multiple of 12 -/
def largestMultipleOf12UsingAllDigits : ℕ := sorry

theorem largest_multiple_of_12_using_all_digits :
  largestMultipleOf12UsingAllDigits = 987654320 ∧
  usesAllDigitsOnce largestMultipleOf12UsingAllDigits ∧
  largestMultipleOf12UsingAllDigits % 12 = 0 ∧
  ∀ m : ℕ, usesAllDigitsOnce m ∧ m % 12 = 0 → m ≤ largestMultipleOf12UsingAllDigits :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_using_all_digits_l2010_201050


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2010_201001

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 = 3*x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3*x + 4)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2010_201001


namespace NUMINAMATH_CALUDE_temperature_average_bounds_l2010_201075

theorem temperature_average_bounds (temps : List ℝ) 
  (h_count : temps.length = 5)
  (h_min : temps.minimum? = some 42)
  (h_max : ∀ t ∈ temps, t ≤ 57) : 
  let avg := temps.sum / temps.length
  42 ≤ avg ∧ avg ≤ 57 := by sorry

end NUMINAMATH_CALUDE_temperature_average_bounds_l2010_201075


namespace NUMINAMATH_CALUDE_square_difference_of_constrained_integers_l2010_201026

theorem square_difference_of_constrained_integers (x y : ℕ+) 
  (h1 : 56 ≤ (x:ℝ) + y ∧ (x:ℝ) + y ≤ 59)
  (h2 : (0.9:ℝ) < (x:ℝ) / y ∧ (x:ℝ) / y < 0.91) :
  (y:ℤ)^2 - (x:ℤ)^2 = 177 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_constrained_integers_l2010_201026


namespace NUMINAMATH_CALUDE_definite_integral_proofs_l2010_201044

theorem definite_integral_proofs :
  (∫ x in (0:ℝ)..1, x^2 - x) = -1/6 ∧
  (∫ x in (1:ℝ)..3, |x - 2|) = 2 ∧
  (∫ x in (0:ℝ)..1, Real.sqrt (1 - x^2)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_proofs_l2010_201044


namespace NUMINAMATH_CALUDE_secret_reaches_2186_l2010_201055

def secret_spread (day : ℕ) : ℕ :=
  if day = 0 then 1
  else secret_spread (day - 1) + 3^day

theorem secret_reaches_2186 :
  ∃ d : ℕ, d ≤ 7 ∧ secret_spread d ≥ 2186 :=
by sorry

end NUMINAMATH_CALUDE_secret_reaches_2186_l2010_201055


namespace NUMINAMATH_CALUDE_largest_number_proof_l2010_201038

theorem largest_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (Nat.gcd a b = 42) → 
  (∃ k : ℕ, Nat.lcm a b = 42 * 11 * 12 * k) →
  (max a b = 504) := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l2010_201038


namespace NUMINAMATH_CALUDE_odd_function_properties_l2010_201097

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) (h : odd_function f) :
  (f 0 = 0) ∧
  (∀ a > 0, (∀ x > 0, f x ≥ a) → (∀ y < 0, f y ≤ -a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2010_201097


namespace NUMINAMATH_CALUDE_sin_period_scaled_l2010_201021

/-- The period of the function y = sin(x/3) is 6π -/
theorem sin_period_scaled (x : ℝ) : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), Real.sin (t / 3) = Real.sin ((t + p) / 3) ∧ p = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sin_period_scaled_l2010_201021


namespace NUMINAMATH_CALUDE_f_symmetry_f_max_min_on_interval_l2010_201000

def f (x : ℝ) : ℝ := x^3 - 27*x

theorem f_symmetry (x : ℝ) : f (-x) = -f x := by sorry

theorem f_max_min_on_interval :
  let a : ℝ := -4
  let b : ℝ := 5
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
  (∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f x ≤ f y) ∧
  (∃ x ∈ Set.Icc a b, f x = 54) ∧
  (∃ x ∈ Set.Icc a b, f x = -54) := by sorry

end NUMINAMATH_CALUDE_f_symmetry_f_max_min_on_interval_l2010_201000


namespace NUMINAMATH_CALUDE_seed_flower_probability_l2010_201081

theorem seed_flower_probability : ∀ (total_seeds small_seeds large_seeds : ℕ)
  (p_small_to_small p_large_to_large : ℝ),
  total_seeds = small_seeds + large_seeds →
  0 ≤ p_small_to_small ∧ p_small_to_small ≤ 1 →
  0 ≤ p_large_to_large ∧ p_large_to_large ≤ 1 →
  total_seeds = 10 →
  small_seeds = 6 →
  large_seeds = 4 →
  p_small_to_small = 0.9 →
  p_large_to_large = 0.8 →
  (small_seeds : ℝ) / (total_seeds : ℝ) * p_small_to_small +
  (large_seeds : ℝ) / (total_seeds : ℝ) * (1 - p_large_to_large) = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_seed_flower_probability_l2010_201081


namespace NUMINAMATH_CALUDE_shelter_animals_count_shelter_problem_l2010_201010

theorem shelter_animals_count : ℕ → ℕ
  | initial_cats =>
    let adopted_cats := initial_cats / 3
    let new_cats := adopted_cats * 2
    let current_cats := initial_cats - adopted_cats + new_cats
    let dogs := current_cats * 2
    current_cats + dogs

theorem shelter_problem (initial_cats : ℕ) (h : initial_cats = 15) :
  shelter_animals_count initial_cats = 60 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_shelter_problem_l2010_201010


namespace NUMINAMATH_CALUDE_inequality_condition_on_a_l2010_201084

theorem inequality_condition_on_a :
  ∀ a : ℝ, (∀ x : ℝ, (a - 3) * x^2 + 2 * (a - 3) * x - 4 < 0) ↔ a ∈ Set.Ioc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_on_a_l2010_201084


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l2010_201092

theorem x_plus_y_equals_negative_eight (x y : ℝ) 
  (h1 : (5 : ℝ)^x = 25^(y+2)) 
  (h2 : (16 : ℝ)^y = 4^(x+4)) : 
  x + y = -8 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_eight_l2010_201092


namespace NUMINAMATH_CALUDE_equidistant_point_x_coord_l2010_201014

/-- A point in the coordinate plane equally distant from the x-axis, y-axis, and the line x + y = 4 -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  dist_x_axis : |y| = |x|
  dist_y_axis : |x| = |y|
  dist_line : |x + y - 4| / Real.sqrt 2 = |x|

/-- The x-coordinate of an equidistant point is 2 -/
theorem equidistant_point_x_coord (p : EquidistantPoint) : p.x = 2 := by
  sorry

#check equidistant_point_x_coord

end NUMINAMATH_CALUDE_equidistant_point_x_coord_l2010_201014


namespace NUMINAMATH_CALUDE_investment_change_l2010_201058

/-- Proves that an investment of $200 with a 20% loss followed by a 25% gain results in 0% change --/
theorem investment_change (initial_investment : ℝ) (first_year_loss_percent : ℝ) (second_year_gain_percent : ℝ) :
  initial_investment = 200 →
  first_year_loss_percent = 20 →
  second_year_gain_percent = 25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let final_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  final_amount = initial_investment := by
  sorry

#check investment_change

end NUMINAMATH_CALUDE_investment_change_l2010_201058


namespace NUMINAMATH_CALUDE_smallest_number_l2010_201094

-- Define the numbers in their respective bases
def num_decimal : ℕ := 75
def num_binary : ℕ := 63  -- 111111₍₂₎ in decimal
def num_base_6 : ℕ := 2 * 6^2 + 1 * 6  -- 210₍₆₎
def num_base_9 : ℕ := 8 * 9 + 5  -- 85₍₉₎

-- Theorem statement
theorem smallest_number :
  num_binary < num_decimal ∧
  num_binary < num_base_6 ∧
  num_binary < num_base_9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2010_201094


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2010_201074

theorem positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2010_201074


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2010_201071

theorem isosceles_triangle_base_angle (base_angle : ℝ) (top_angle : ℝ) : 
  -- The triangle is isosceles
  -- The top angle is 20° more than twice the base angle
  top_angle = 2 * base_angle + 20 →
  -- The sum of angles in a triangle is 180°
  base_angle + base_angle + top_angle = 180 →
  -- The base angle is 40°
  base_angle = 40 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2010_201071


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2010_201073

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem intersection_of_M_and_N : M ∩ N = {5, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2010_201073


namespace NUMINAMATH_CALUDE_license_plate_count_l2010_201098

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special symbols available for the license plate. -/
def num_special_symbols : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_special_symbols

/-- Theorem stating that the total number of license plates is 48,000. -/
theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2010_201098


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2010_201057

theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 ∧ diagonal = 17 ∧ length^2 + width^2 = diagonal^2 →
  2 * (length + width) = 46 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2010_201057


namespace NUMINAMATH_CALUDE_employee_savings_l2010_201054

/-- Calculate the combined savings of three employees over a period of time. -/
def combined_savings (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (robby_save_ratio : ℚ) (jaylen_save_ratio : ℚ) (miranda_save_ratio : ℚ) 
  (num_weeks : ℕ) : ℚ :=
  let weekly_salary := hourly_wage * hours_per_day * days_per_week
  let robby_savings := robby_save_ratio * weekly_salary
  let jaylen_savings := jaylen_save_ratio * weekly_salary
  let miranda_savings := miranda_save_ratio * weekly_salary
  (robby_savings + jaylen_savings + miranda_savings) * num_weeks

/-- The combined savings of three employees after four weeks is $3000. -/
theorem employee_savings : 
  combined_savings 10 10 5 (2/5) (3/5) (1/2) 4 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_employee_savings_l2010_201054


namespace NUMINAMATH_CALUDE_city_graph_property_l2010_201066

/-- A graph representing cities and flights -/
structure CityGraph where
  V : Type*
  E : V → V → Prop
  N : Nat
  vertex_count : Fintype V
  city_count : Fintype.card V = N

/-- Path of length at most 2 between two vertices -/
def PathOfLength2 (G : CityGraph) (u v : G.V) : Prop :=
  G.E u v ∨ ∃ w, G.E u w ∧ G.E w v

/-- The main theorem -/
theorem city_graph_property (G : CityGraph) 
  (not_fully_connected : ∀ v : G.V, ∃ u : G.V, ¬G.E v u)
  (unique_path : ∀ u v : G.V, ∃! p : PathOfLength2 G u v, True) :
  ∃ k : Nat, G.N - 1 = k * k :=
sorry

end NUMINAMATH_CALUDE_city_graph_property_l2010_201066


namespace NUMINAMATH_CALUDE_square_root_meaningful_l2010_201037

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l2010_201037


namespace NUMINAMATH_CALUDE_five_dogs_not_eating_any_l2010_201093

/-- The number of dogs that do not eat any of the three foods (watermelon, salmon, chicken) -/
def dogs_not_eating_any (total dogs_watermelon dogs_salmon dogs_chicken dogs_watermelon_salmon dogs_chicken_salmon_not_watermelon : ℕ) : ℕ :=
  total - (dogs_watermelon + dogs_salmon + dogs_chicken - dogs_watermelon_salmon - dogs_chicken_salmon_not_watermelon)

/-- Theorem stating that 5 dogs do not eat any of the three foods -/
theorem five_dogs_not_eating_any :
  dogs_not_eating_any 75 15 54 20 12 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_dogs_not_eating_any_l2010_201093


namespace NUMINAMATH_CALUDE_jed_gives_away_two_cards_l2010_201017

/-- Represents the number of cards Jed gives away every two weeks -/
def cards_given_away : ℕ := 2

/-- Represents the initial number of cards Jed has -/
def initial_cards : ℕ := 20

/-- Represents the number of cards Jed gets every week -/
def weekly_cards : ℕ := 6

/-- Represents the number of weeks that have passed -/
def weeks_passed : ℕ := 4

/-- Represents the total number of cards Jed has after 4 weeks -/
def final_cards : ℕ := 40

/-- Theorem stating that Jed gives away 2 cards every two weeks -/
theorem jed_gives_away_two_cards : 
  initial_cards + weekly_cards * weeks_passed - cards_given_away * (weeks_passed / 2) = final_cards :=
sorry

end NUMINAMATH_CALUDE_jed_gives_away_two_cards_l2010_201017


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l2010_201019

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g 0| = 10 ∧ |g 1| = 10 ∧ |g 3| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 70 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l2010_201019


namespace NUMINAMATH_CALUDE_fathers_age_l2010_201004

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 10 = (1 / 2) * (father_age + 10) → 
  father_age = 50 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l2010_201004


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2010_201022

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2010_201022


namespace NUMINAMATH_CALUDE_like_terms_imply_exponents_l2010_201006

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, m1 x y ≠ 0 ∧ m2 x y ≠ 0 → (x = x ∧ y = y)

/-- The first monomial 2x^3y^4 -/
def m1 (x y : ℕ) : ℚ := 2 * (x^3 * y^4)

/-- The second monomial -2x^ay^(2b) -/
def m2 (a b x y : ℕ) : ℚ := -2 * (x^a * y^(2*b))

theorem like_terms_imply_exponents (a b : ℕ) :
  are_like_terms m1 (m2 a b) → a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponents_l2010_201006


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l2010_201045

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the 25-gingerbread trays -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the 20-gingerbread trays -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l2010_201045


namespace NUMINAMATH_CALUDE_dollar_square_sum_l2010_201070

/-- The dollar operation -/
def dollar (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x + y)²$(y + x)² -/
theorem dollar_square_sum (x y : ℝ) : 
  dollar ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_square_sum_l2010_201070


namespace NUMINAMATH_CALUDE_book_pages_calculation_l2010_201034

theorem book_pages_calculation (pages_read : ℕ) (fraction_read : ℚ) (h1 : pages_read = 16) (h2 : fraction_read = 0.4) : 
  (pages_read : ℚ) / fraction_read = 40 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l2010_201034


namespace NUMINAMATH_CALUDE_strawberries_left_l2010_201082

theorem strawberries_left (initial_strawberries eaten_strawberries : ℕ) :
  initial_strawberries = 35 →
  eaten_strawberries = 2 →
  initial_strawberries - eaten_strawberries = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l2010_201082


namespace NUMINAMATH_CALUDE_fraction_transformation_l2010_201042

theorem fraction_transformation (x : ℝ) (h : x ≠ 2) : 2 / (2 - x) = -(2 / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2010_201042


namespace NUMINAMATH_CALUDE_boxes_with_neither_markers_nor_crayons_l2010_201076

/-- The number of boxes containing neither markers nor crayons -/
def empty_boxes (total boxes_with_markers boxes_with_crayons boxes_with_both : ℕ) : ℕ :=
  total - (boxes_with_markers + boxes_with_crayons - boxes_with_both)

/-- Theorem: Given the conditions of the problem, there are 5 boxes with neither markers nor crayons -/
theorem boxes_with_neither_markers_nor_crayons :
  empty_boxes 15 9 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_markers_nor_crayons_l2010_201076


namespace NUMINAMATH_CALUDE_multiple_of_six_l2010_201085

theorem multiple_of_six (n : ℤ) (h : n ≥ 12) : ∃ k : ℤ, (n + 2) * (n + 1) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_l2010_201085


namespace NUMINAMATH_CALUDE_pedro_squares_difference_l2010_201024

theorem pedro_squares_difference (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_squares_difference_l2010_201024


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2010_201060

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) :
  flavors = 6 → toppings = 8 →
  flavors * (toppings.choose 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l2010_201060


namespace NUMINAMATH_CALUDE_team_total_score_l2010_201012

def team_score (connor_score amy_score jason_score emily_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score + emily_score

theorem team_total_score :
  ∀ (connor_score amy_score jason_score emily_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    emily_score = 3 * (connor_score + amy_score + jason_score) →
    team_score connor_score amy_score jason_score emily_score = 80 :=
by
  sorry

#check team_total_score

end NUMINAMATH_CALUDE_team_total_score_l2010_201012


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2010_201028

/-- A rectangular plot with length thrice its breadth and area 363 sq m has a breadth of 11 m -/
theorem rectangular_plot_breadth : 
  ∀ (breadth : ℝ),
  breadth > 0 →
  3 * breadth * breadth = 363 →
  breadth = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2010_201028


namespace NUMINAMATH_CALUDE_chips_ratio_l2010_201052

-- Define the total number of bags
def total_bags : ℕ := 3

-- Define the number of bags eaten for dinner
def dinner_bags : ℕ := 1

-- Define the number of bags eaten after dinner
def after_dinner_bags : ℕ := total_bags - dinner_bags

-- Theorem to prove
theorem chips_ratio :
  (after_dinner_bags : ℚ) / (dinner_bags : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_chips_ratio_l2010_201052


namespace NUMINAMATH_CALUDE_unique_configuration_l2010_201061

-- Define the type for statements
inductive Statement
| one_false : Statement
| two_false : Statement
| three_false : Statement
| four_false : Statement
| one_true : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate (s : Statement) (true_count : Nat) : Prop :=
  match s with
  | Statement.one_false => true_count = 4
  | Statement.two_false => true_count = 3
  | Statement.three_false => true_count = 2
  | Statement.four_false => true_count = 1
  | Statement.one_true => true_count = 1

-- Define the card as a list of statements
def card : List Statement := [
  Statement.one_false,
  Statement.two_false,
  Statement.three_false,
  Statement.four_false,
  Statement.one_true
]

-- Theorem: There exists a unique configuration with exactly one true statement
theorem unique_configuration :
  ∃! true_count : Nat,
    true_count ≤ 5 ∧
    true_count > 0 ∧
    (∀ s ∈ card, evaluate s true_count ↔ s = Statement.one_true) :=
by sorry

end NUMINAMATH_CALUDE_unique_configuration_l2010_201061


namespace NUMINAMATH_CALUDE_system_solutions_l2010_201027

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 - (y+z+y*z)*x + (y+z)*y*z = 0
def equation2 (x y z : ℝ) : Prop := y^2 - (z+x+z*x)*y + (z+x)*z*x = 0
def equation3 (x y z : ℝ) : Prop := z^2 - (x+y+x*y)*z + (x+y)*x*y = 0

-- Define the set of solutions
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0,0,0), (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1), (1, 1/2, 1/2), (1/2, 1, 1/2), (1/2, 1/2, 1)}

-- State the theorem
theorem system_solutions :
  ∀ x y z : ℝ, (equation1 x y z ∧ equation2 x y z ∧ equation3 x y z) ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2010_201027


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l2010_201011

theorem merchant_pricing_strategy (list_price : ℝ) (h : list_price > 0) :
  let cost_price := 0.7 * list_price
  let profit_ratio := 0.3
  let discount_ratio := 0.2
  let selling_price := list_price
  let marked_price := selling_price / (1 - discount_ratio)
  (marked_price - cost_price) / selling_price = profit_ratio →
  marked_price = 1.25 * list_price :=
by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l2010_201011


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2010_201015

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_arith_mean : (a + b) / 2 = 5 / 2) (h_geom_mean : Real.sqrt (a * b) = Real.sqrt 6) :
  let c := Real.sqrt (a^2 - b^2)
  (c / a) = Real.sqrt 13 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2010_201015


namespace NUMINAMATH_CALUDE_wall_length_calculation_l2010_201029

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  width : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.width

theorem wall_length_calculation
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℕ)
  (h1 : brick.length = 80)
  (h2 : brick.width = 11.25)
  (h3 : brick.height = 6)
  (h4 : wall.height = 600)
  (h5 : wall.width = 22.5)
  (h6 : num_bricks = 2000)
  (h7 : num_bricks * brickVolume brick = wallVolume wall) :
  wall.length = 800 := by
  sorry

#check wall_length_calculation

end NUMINAMATH_CALUDE_wall_length_calculation_l2010_201029


namespace NUMINAMATH_CALUDE_max_salary_is_260000_l2010_201068

/-- Represents the maximum possible salary for a single player on a minor league soccer team -/
def max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) : ℕ :=
  total_cap - (n - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player on the team -/
theorem max_salary_is_260000 :
  max_player_salary 18 20000 600000 = 260000 := by
  sorry

#eval max_player_salary 18 20000 600000

end NUMINAMATH_CALUDE_max_salary_is_260000_l2010_201068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2010_201059

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 14 = 2) :
  a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2010_201059


namespace NUMINAMATH_CALUDE_fourth_student_id_l2010_201005

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_id : ℕ

/-- Checks if a given ID is in the systematic sample. -/
def SystematicSample.contains (s : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.first_id + k * s.interval

/-- The theorem to be proved. -/
theorem fourth_student_id
  (s : SystematicSample)
  (h_class_size : s.class_size = 52)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_29 : s.contains 29)
  (h_contains_42 : s.contains 42) :
  s.contains 16 :=
sorry

end NUMINAMATH_CALUDE_fourth_student_id_l2010_201005


namespace NUMINAMATH_CALUDE_book_purchases_l2010_201008

/-- The number of people who purchased only book A -/
def v : ℕ := sorry

/-- The number of people who purchased only book B -/
def x : ℕ := sorry

/-- The number of people who purchased book B (both only and with book A) -/
def y : ℕ := sorry

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

theorem book_purchases : 
  (y = x + both) ∧ 
  (v = 2 * y) ∧ 
  (both = 2 * x) →
  v = 1500 := by sorry

end NUMINAMATH_CALUDE_book_purchases_l2010_201008


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l2010_201031

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2 + (a-b)^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l2010_201031


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l2010_201018

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) (h : i * i = -1) :
  Complex.im (i / (1 + i)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l2010_201018


namespace NUMINAMATH_CALUDE_x_values_l2010_201039

theorem x_values (x : ℝ) : x ∈ ({1, 2, x^2} : Set ℝ) → x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l2010_201039


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2010_201030

theorem inequality_equivalence (x : ℝ) : (x - 1) / (x - 3) ≥ 2 ↔ x ∈ Set.Ioo 3 5 ∪ {5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2010_201030


namespace NUMINAMATH_CALUDE_goals_scored_l2010_201048

def bruce_goals : ℕ := 4

def michael_goals : ℕ := 3 * bruce_goals

def total_goals : ℕ := bruce_goals + michael_goals

theorem goals_scored : total_goals = 16 := by
  sorry

end NUMINAMATH_CALUDE_goals_scored_l2010_201048


namespace NUMINAMATH_CALUDE_dog_walking_problem_l2010_201072

/-- Greg's dog walking business problem -/
theorem dog_walking_problem (x : ℕ) : 
  (20 + x) +                 -- Cost for one dog
  (2 * 20 + 2 * 7 * 1) +     -- Cost for two dogs for 7 minutes
  (3 * 20 + 3 * 9 * 1) = 171 -- Cost for three dogs for 9 minutes
  → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_walking_problem_l2010_201072


namespace NUMINAMATH_CALUDE_cost_of_apple_l2010_201053

/-- The cost of fruit problem -/
theorem cost_of_apple (banana_cost orange_cost : ℚ)
  (apple_count banana_count orange_count : ℕ)
  (average_cost : ℚ)
  (h1 : banana_cost = 1)
  (h2 : orange_cost = 3)
  (h3 : apple_count = 12)
  (h4 : banana_count = 4)
  (h5 : orange_count = 4)
  (h6 : average_cost = 2)
  (h7 : average_cost * (apple_count + banana_count + orange_count : ℚ) =
        apple_cost * apple_count + banana_cost * banana_count + orange_cost * orange_count) :
  apple_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_cost_of_apple_l2010_201053


namespace NUMINAMATH_CALUDE_silverware_reduction_l2010_201003

theorem silverware_reduction (initial_per_type : ℕ) (num_types : ℕ) (total_purchased : ℕ) :
  initial_per_type = 15 →
  num_types = 4 →
  total_purchased = 44 →
  (initial_per_type * num_types - total_purchased) / num_types = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_silverware_reduction_l2010_201003


namespace NUMINAMATH_CALUDE_min_value_expression_l2010_201096

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2010_201096


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2010_201083

/-- Given two planar vectors a and b, where a is parallel to b,
    prove that the magnitude of 3a + b is √5 -/
theorem parallel_vectors_magnitude (y : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (a 0 * b 1 = a 1 * b 0) →  -- Parallel condition
  Real.sqrt ((3 * a 0 + b 0)^2 + (3 * a 1 + b 1)^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2010_201083


namespace NUMINAMATH_CALUDE_symmetric_points_existence_l2010_201099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * Real.exp (-x) else Real.log (x / a)

theorem symmetric_points_existence (a : ℝ) (h : a > 0) :
  (∃ x₀ : ℝ, x₀ > 1 ∧ f a (-x₀) = f a x₀) ↔ 0 < a ∧ a < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_existence_l2010_201099


namespace NUMINAMATH_CALUDE_three_consecutive_heads_probability_l2010_201002

theorem three_consecutive_heads_probability (p : ℝ) :
  p = (1 : ℝ) / 2 →  -- probability of heads on a single flip
  p * p * p = (1 : ℝ) / 8 :=  -- probability of three consecutive heads
by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_heads_probability_l2010_201002


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2010_201035

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2010_201035


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2010_201067

theorem polygon_diagonals (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 150 → (n - 2) * 180 = n * interior_angle → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2010_201067


namespace NUMINAMATH_CALUDE_incorrect_quotient_calculation_l2010_201032

theorem incorrect_quotient_calculation (dividend : ℕ) (correct_divisor incorrect_divisor correct_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : incorrect_divisor = 12)
  (h4 : correct_quotient = 28) :
  dividend / incorrect_divisor = 49 := by
sorry

end NUMINAMATH_CALUDE_incorrect_quotient_calculation_l2010_201032


namespace NUMINAMATH_CALUDE_solve_cookies_problem_l2010_201040

def cookies_problem (total_baked : ℕ) (kristy_ate : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  let cookies_taken := kristy_ate + friend1_took + friend2_took + friend3_took
  let cookies_given_away := total_baked - cookies_left
  let brother_cookies := cookies_given_away - cookies_taken
  brother_cookies = 1

theorem solve_cookies_problem :
  cookies_problem 22 2 3 5 5 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cookies_problem_l2010_201040


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l2010_201088

/-- The work efficiency of a worker is defined as the fraction of the total work they can complete in one day -/
def work_efficiency (days : ℚ) : ℚ := 1 / days

theorem work_efficiency_ratio 
  (a_and_b_days : ℚ) 
  (b_alone_days : ℚ) 
  (h1 : a_and_b_days = 11) 
  (h2 : b_alone_days = 33) : 
  (work_efficiency a_and_b_days - work_efficiency b_alone_days) / work_efficiency b_alone_days = 2 := by
  sorry

#check work_efficiency_ratio

end NUMINAMATH_CALUDE_work_efficiency_ratio_l2010_201088


namespace NUMINAMATH_CALUDE_evaluate_expression_l2010_201049

theorem evaluate_expression : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2010_201049


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l2010_201007

theorem geometric_progression_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive real numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Geometric progression
  a * b * c = 64 →  -- Product is 64
  (a + b + c) / 3 = 14 / 3 →  -- Arithmetic mean is 14/3
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l2010_201007


namespace NUMINAMATH_CALUDE_rectangle_cutting_l2010_201080

/-- Represents a rectangle on a cartesian plane with sides parallel to coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h1 : x_min < x_max
  h2 : y_min < y_max

/-- Predicate to check if a vertical line intersects a rectangle -/
def vertical_intersects (x : ℝ) (r : Rectangle) : Prop :=
  r.x_min < x ∧ x < r.x_max

/-- Predicate to check if a horizontal line intersects a rectangle -/
def horizontal_intersects (y : ℝ) (r : Rectangle) : Prop :=
  r.y_min < y ∧ y < r.y_max

/-- Any two rectangles can be cut by a vertical or a horizontal line -/
axiom rectangle_separation (r1 r2 : Rectangle) :
  (∃ x : ℝ, vertical_intersects x r1 ∧ vertical_intersects x r2) ∨
  (∃ y : ℝ, horizontal_intersects y r1 ∧ horizontal_intersects y r2)

/-- The main theorem -/
theorem rectangle_cutting (rectangles : Set Rectangle) :
  ∃ (x y : ℝ), ∀ r ∈ rectangles, vertical_intersects x r ∨ horizontal_intersects y r :=
sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l2010_201080


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2010_201090

theorem inverse_proportion_percentage_change 
  (x y x' y' k q : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : x * y = k)
  (h_y_decrease : y' = y * (1 - q / 100))
  (h_constant : x' * y' = k) :
  (x' - x) / x * 100 = 100 * q / (100 - q) := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2010_201090
