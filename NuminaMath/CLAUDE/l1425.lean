import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l1425_142546

theorem expression_evaluation : 3^(0^(2^8)) + ((3^0)^2)^8 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1425_142546


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_1_l1425_142591

theorem quadratic_root_sqrt5_minus_1 :
  ∃ (a b c : ℚ), (a ≠ 0) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2*x - 6 = 0) ∧
  (Real.sqrt 5 - 1)^2 + 2*(Real.sqrt 5 - 1) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_1_l1425_142591


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l1425_142567

/-- The function that relates x and y --/
def f (x : ℕ) : ℕ := x^2 + x

/-- The set of data points from the table --/
def data_points : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

/-- Theorem stating that the function f satisfies all data points --/
theorem f_satisfies_data_points : ∀ (point : ℕ × ℕ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end NUMINAMATH_CALUDE_f_satisfies_data_points_l1425_142567


namespace NUMINAMATH_CALUDE_lcm_problem_l1425_142554

theorem lcm_problem (A B : ℕ+) (h1 : A * B = 45276) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2058 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1425_142554


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1425_142581

theorem tan_theta_in_terms_of_x (θ x : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.cos (θ/2) = Real.sqrt ((x - 2)/(2*x))) : 
  Real.tan θ = -1/2 * Real.sqrt (x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l1425_142581


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1425_142501

/-- The perimeter of a rectangle with length 6 cm and width 4 cm is 20 cm. -/
theorem rectangle_perimeter : 
  let length : ℝ := 6
  let width : ℝ := 4
  let perimeter := 2 * (length + width)
  perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1425_142501


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1425_142553

/-- Given two perpendicular vectors a and b in ℝ², where a = (3, x) and b = (y, 1),
    prove that x = -7/4 -/
theorem perpendicular_vectors_x_value (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3, x]
  let b : Fin 2 → ℝ := ![y, 1]
  (∀ i j, a i * b j = 0) → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1425_142553


namespace NUMINAMATH_CALUDE_inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l1425_142500

-- Part 1
theorem inverse_half_plus_sqrt_four (x y : ℝ) (h1 : x = 0.5) (h2 : y = 4) :
  x⁻¹ + y^(1/2) = 4 := by sorry

-- Part 2
theorem log_sum_minus_power (x y z : ℝ) (h1 : x = 2) (h2 : y = 5) (h3 : z = π / 23) :
  Real.log x / Real.log 10 + Real.log y / Real.log 10 - z^0 = 0 := by sorry

-- Part 3
theorem inverse_sum_with_sqrt_three (x : ℝ) (h : x = 3) :
  (2 - Real.sqrt x)⁻¹ + (2 + Real.sqrt x)⁻¹ = 4 := by sorry

end NUMINAMATH_CALUDE_inverse_half_plus_sqrt_four_log_sum_minus_power_inverse_sum_with_sqrt_three_l1425_142500


namespace NUMINAMATH_CALUDE_square_side_length_range_l1425_142519

theorem square_side_length_range (a : ℝ) : a^2 = 30 → 5.4 < a ∧ a < 5.5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l1425_142519


namespace NUMINAMATH_CALUDE_range_of_m_l1425_142572

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4/x + 1/y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 4/x + 1/y = 1 → x + y ≥ m^2 + m + 3) :
  -3 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1425_142572


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l1425_142523

theorem divisible_by_thirty (n : ℕ+) : ∃ k : ℤ, (n : ℤ)^19 - (n : ℤ)^7 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l1425_142523


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l1425_142528

theorem abs_inequality_solution (x : ℝ) : |x + 3| > x + 3 ↔ x < -3 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l1425_142528


namespace NUMINAMATH_CALUDE_claudia_weekend_earnings_l1425_142538

-- Define the price per class
def price_per_class : ℕ := 10

-- Define the number of kids attending Saturday's class
def saturday_attendance : ℕ := 20

-- Define the number of kids attending Sunday's class
def sunday_attendance : ℕ := saturday_attendance / 2

-- Calculate the total money made
def total_money : ℕ := price_per_class * (saturday_attendance + sunday_attendance)

-- Theorem to prove
theorem claudia_weekend_earnings : total_money = 300 := by
  sorry

end NUMINAMATH_CALUDE_claudia_weekend_earnings_l1425_142538


namespace NUMINAMATH_CALUDE_stating_min_weighings_to_determine_faulty_coin_l1425_142547

/-- Represents a pile of coins with one faulty coin. -/
structure CoinPile :=
  (total : ℕ)  -- Total number of coins
  (faulty : ℕ)  -- Index of the faulty coin (1-based)
  (is_lighter : Bool)  -- True if the faulty coin is lighter, False if heavier

/-- Represents a weighing on a balance scale. -/
inductive Weighing
  | Equal : Weighing  -- The scale is balanced
  | Left : Weighing   -- The left side is heavier
  | Right : Weighing  -- The right side is heavier

/-- Function to perform a weighing on a subset of coins. -/
def weigh (pile : CoinPile) (left : List ℕ) (right : List ℕ) : Weighing :=
  sorry  -- Implementation details omitted

/-- 
Theorem stating that the minimum number of weighings required to determine 
whether the faulty coin is lighter or heavier is 2.
-/
theorem min_weighings_to_determine_faulty_coin (pile : CoinPile) : 
  ∃ (strategy : List (List ℕ × List ℕ)), 
    (strategy.length = 2) ∧ 
    (∀ (outcome : List Weighing), 
      outcome.length = 2 → 
      (∃ (result : Bool), result = pile.is_lighter)) :=
sorry

end NUMINAMATH_CALUDE_stating_min_weighings_to_determine_faulty_coin_l1425_142547


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_9240_l1425_142550

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_perfect_square_factor_of_9240 :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_factor n 9240 ∧ 
             (∀ m : ℕ, is_perfect_square m → is_factor m 9240 → m ≤ n) ∧
             n = 36 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_9240_l1425_142550


namespace NUMINAMATH_CALUDE_option1_better_than_option2_l1425_142548

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.25, 0.10]
def option2_discounts : List ℝ := [0.25, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem option1_better_than_option2 :
  apply_successive_discounts initial_amount option1_discounts <
  apply_successive_discounts initial_amount option2_discounts :=
sorry

end NUMINAMATH_CALUDE_option1_better_than_option2_l1425_142548


namespace NUMINAMATH_CALUDE_couple_stock_purchase_l1425_142575

/-- Calculates the number of shares a couple can buy given their savings plan and stock price --/
def shares_to_buy (wife_weekly_savings : ℕ) (husband_monthly_savings : ℕ) (months : ℕ) (stock_price : ℕ) : ℕ :=
  let wife_monthly_savings := wife_weekly_savings * 4
  let total_monthly_savings := wife_monthly_savings + husband_monthly_savings
  let total_savings := total_monthly_savings * months
  let investment := total_savings / 2
  investment / stock_price

/-- Theorem stating that the couple can buy 25 shares given their specific savings plan --/
theorem couple_stock_purchase :
  shares_to_buy 100 225 4 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_couple_stock_purchase_l1425_142575


namespace NUMINAMATH_CALUDE_aeroplane_distance_l1425_142502

theorem aeroplane_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) 
  (h1 : speed1 = 590)
  (h2 : time1 = 8)
  (h3 : speed2 = 1716.3636363636363)
  (h4 : time2 = 2.75)
  (h5 : speed1 * time1 = speed2 * time2) : 
  speed1 * time1 = 4720 := by
  sorry

end NUMINAMATH_CALUDE_aeroplane_distance_l1425_142502


namespace NUMINAMATH_CALUDE_incorrect_calculation_correction_l1425_142534

theorem incorrect_calculation_correction (x : ℝ) (h : x * 7 = 115.15) : 
  115.15 / 49 = 2.35 := by
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_correction_l1425_142534


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l1425_142510

/-- Given two functions f and g, where f(x) = |x-3| and g(x) = -|x-7| + m,
    if f(x) > g(x) for all real x, then m < 4 -/
theorem function_inequality_implies_m_bound
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = |x - 3|)
  (hg : ∀ x, g x = -|x - 7| + m)
  (h_above : ∀ x, f x > g x) :
  m < 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l1425_142510


namespace NUMINAMATH_CALUDE_colored_pencils_total_l1425_142509

theorem colored_pencils_total (madeline_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : ∃ cheryl_pencils : ℕ, cheryl_pencils = 2 * madeline_pencils)
  (h3 : ∃ cyrus_pencils : ℕ, 3 * cyrus_pencils = cheryl_pencils) :
  ∃ total_pencils : ℕ, total_pencils = madeline_pencils + cheryl_pencils + cyrus_pencils ∧ total_pencils = 231 :=
by sorry

end NUMINAMATH_CALUDE_colored_pencils_total_l1425_142509


namespace NUMINAMATH_CALUDE_sugar_solution_concentration_increases_l1425_142549

theorem sugar_solution_concentration_increases 
  (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) : 
  (b + m) / (a + m) > b / a := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_concentration_increases_l1425_142549


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l1425_142588

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧ f c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l1425_142588


namespace NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l1425_142559

-- Define the type for algorithms
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | SolveQuadraticInequality
  | AreaOfTrapezoid

-- Define a function to check if an algorithm requires a conditional branch
def requiresConditionalBranch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.SolveQuadraticInequality => True
  | _ => False

-- State the theorem
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm),
    requiresConditionalBranch a ↔ a = Algorithm.SolveQuadraticInequality :=
by sorry

#check quadratic_inequality_requires_conditional_branch

end NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l1425_142559


namespace NUMINAMATH_CALUDE_max_value_expression_l1425_142597

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + y^2 = 9) :
  x^2 - 2*x*y + y^2 ≤ 9/4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*a*b + b^2 = 9 ∧ a^2 - 2*a*b + b^2 = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1425_142597


namespace NUMINAMATH_CALUDE_percentage_five_digit_numbers_with_repeated_digits_l1425_142522

theorem percentage_five_digit_numbers_with_repeated_digits :
  let total_five_digit_numbers : ℕ := 90000
  let five_digit_numbers_without_repeats : ℕ := 27216
  let five_digit_numbers_with_repeats : ℕ := total_five_digit_numbers - five_digit_numbers_without_repeats
  let percentage : ℚ := (five_digit_numbers_with_repeats : ℚ) / (total_five_digit_numbers : ℚ) * 100
  ∃ (ε : ℚ), abs (percentage - 69.8) < ε ∧ ε ≤ 0.05 :=
by sorry

end NUMINAMATH_CALUDE_percentage_five_digit_numbers_with_repeated_digits_l1425_142522


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l1425_142582

open Real

theorem indefinite_integral_proof (x : ℝ) (C : ℝ) (h : x ≠ -2 ∧ x ≠ -1) :
  deriv (λ y => 2 * log (abs (y + 2)) - 1 / (2 * (y + 1)^2) + C) x =
  (2 * x^3 + 6 * x^2 + 7 * x + 4) / ((x + 2) * (x + 1)^3) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l1425_142582


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1425_142598

theorem fraction_equivalence : (15 : ℝ) / (4 * 63) = 1.5 / (0.4 * 63) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1425_142598


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_bound_l1425_142540

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1)*x + 2

-- State the theorem
theorem monotonic_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y → y ≤ 4 → f a y < f a x) → a ≤ -7 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_bound_l1425_142540


namespace NUMINAMATH_CALUDE_fraction_equality_l1425_142578

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 9) :
  m / q = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1425_142578


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1425_142518

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℝ) (rate_paise : ℝ) (time_months : ℝ) :
  interest = 23 * (rate_paise / 100) * time_months →
  interest = 3.45 ∧ rate_paise = 5 ∧ time_months = 3 →
  23 = interest / ((rate_paise / 100) * time_months) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1425_142518


namespace NUMINAMATH_CALUDE_smallest_number_l1425_142507

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, -4, 5}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1425_142507


namespace NUMINAMATH_CALUDE_safe_plucking_percentage_is_correct_l1425_142589

/-- The number of tail feathers each flamingo has -/
def feathers_per_flamingo : ℕ := 20

/-- The number of boas Milly needs to make -/
def number_of_boas : ℕ := 12

/-- The number of feathers needed for each boa -/
def feathers_per_boa : ℕ := 200

/-- The number of flamingoes Milly needs to harvest -/
def flamingoes_to_harvest : ℕ := 480

/-- The percentage of tail feathers Milly can safely pluck from each flamingo -/
def safe_plucking_percentage : ℚ := 25 / 100

theorem safe_plucking_percentage_is_correct :
  safe_plucking_percentage = 
    (number_of_boas * feathers_per_boa) / 
    (flamingoes_to_harvest * feathers_per_flamingo) := by
  sorry

end NUMINAMATH_CALUDE_safe_plucking_percentage_is_correct_l1425_142589


namespace NUMINAMATH_CALUDE_prime_sum_problem_l1425_142560

theorem prime_sum_problem (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  1 < p → p < q → q < s →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l1425_142560


namespace NUMINAMATH_CALUDE_negative_one_squared_equals_negative_one_l1425_142531

theorem negative_one_squared_equals_negative_one : -1^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_squared_equals_negative_one_l1425_142531


namespace NUMINAMATH_CALUDE_cos_product_from_sum_relations_l1425_142580

theorem cos_product_from_sum_relations (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 0.6) 
  (h2 : Real.cos x + Real.cos y = 0.8) : 
  Real.cos x * Real.cos y = -11/100 := by
sorry

end NUMINAMATH_CALUDE_cos_product_from_sum_relations_l1425_142580


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1425_142558

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1425_142558


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1425_142596

theorem sum_of_fractions_equals_one 
  {x y z : ℝ} (h : x * y * z = 1) : 
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1425_142596


namespace NUMINAMATH_CALUDE_davids_biology_marks_l1425_142526

/-- Calculates the marks in Biology given the marks in other subjects and the average -/
def marks_in_biology (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

/-- Theorem stating that David's marks in Biology are 85 -/
theorem davids_biology_marks :
  marks_in_biology 81 65 82 67 76 = 85 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l1425_142526


namespace NUMINAMATH_CALUDE_non_officers_count_l1425_142556

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := 450

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Average salary of all employees in Rs/month -/
def avg_salary_all : ℚ := 120

/-- Average salary of officers in Rs/month -/
def avg_salary_officers : ℚ := 420

/-- Average salary of non-officers in Rs/month -/
def avg_salary_non_officers : ℚ := 110

/-- Theorem stating that the number of non-officers is 450 given the conditions -/
theorem non_officers_count :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / 
  (num_officers + num_non_officers : ℚ) = avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_non_officers_count_l1425_142556


namespace NUMINAMATH_CALUDE_last_digits_of_11_power_l1425_142584

theorem last_digits_of_11_power (n : ℕ) (h : n ≥ 1) :
  11^(10^n) ≡ 6 * 10^(n+1) + 1 [MOD 10^(n+2)] := by
sorry

end NUMINAMATH_CALUDE_last_digits_of_11_power_l1425_142584


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1425_142587

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A regular dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1425_142587


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l1425_142539

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_energy_conversion :
  base5_to_base10 [0, 2, 3] = 85 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l1425_142539


namespace NUMINAMATH_CALUDE_lemonade_ratio_l1425_142595

/-- The number of glasses of lemonade that can be made -/
def num_glasses : ℕ := 9

/-- The total number of lemons used -/
def total_lemons : ℚ := 18

/-- The number of lemons needed per glass -/
def lemons_per_glass : ℚ := total_lemons / num_glasses

theorem lemonade_ratio : lemons_per_glass = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_ratio_l1425_142595


namespace NUMINAMATH_CALUDE_student_count_l1425_142579

theorem student_count (ratio : ℝ) (teachers : ℕ) (h1 : ratio = 27.5) (h2 : teachers = 42) :
  ↑teachers * ratio = 1155 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1425_142579


namespace NUMINAMATH_CALUDE_work_completion_time_l1425_142537

/-- The time taken for A to complete the work alone -/
def time_A : ℝ := 10

/-- The time taken for A and B to complete the work together -/
def time_AB : ℝ := 4.444444444444445

/-- The time taken for B to complete the work alone -/
def time_B : ℝ := 8

/-- Theorem stating that given the time for A alone and A and B together, 
    the time for B alone is 8 days -/
theorem work_completion_time : 
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1425_142537


namespace NUMINAMATH_CALUDE_remainder_of_B_divided_by_9_l1425_142571

theorem remainder_of_B_divided_by_9 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_divided_by_9_l1425_142571


namespace NUMINAMATH_CALUDE_S_is_infinite_l1425_142515

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + 2 * p.2 ≤ 10}

-- Theorem stating that the set S is infinite
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l1425_142515


namespace NUMINAMATH_CALUDE_midpoint_square_sum_l1425_142524

/-- Given that C = (5, 3) is the midpoint of line segment AB, where A = (3, -3) and B = (x, y),
    prove that x^2 + y^2 = 130. -/
theorem midpoint_square_sum (x y : ℝ) : 
  (5 : ℝ) = (3 + x) / 2 ∧ (3 : ℝ) = (-3 + y) / 2 → x^2 + y^2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_square_sum_l1425_142524


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1425_142542

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the range of a when A ∩ C is non-empty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1425_142542


namespace NUMINAMATH_CALUDE_prime_power_digit_repetition_l1425_142583

theorem prime_power_digit_repetition (p n : ℕ) : 
  Prime p → p > 3 → (10^19 ≤ p^n ∧ p^n < 10^20) → 
  ∃ (d : ℕ) (i j k : ℕ), i < j ∧ j < k ∧ i < 20 ∧ j < 20 ∧ k < 20 ∧
  d < 10 ∧ (p^n / 10^i) % 10 = d ∧ (p^n / 10^j) % 10 = d ∧ (p^n / 10^k) % 10 = d :=
by sorry

end NUMINAMATH_CALUDE_prime_power_digit_repetition_l1425_142583


namespace NUMINAMATH_CALUDE_trig_expression_value_l1425_142514

theorem trig_expression_value (α : Real) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  2 + 2/3 * (Real.sin α)^2 + 1/4 * (Real.cos α)^2 = 21/8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l1425_142514


namespace NUMINAMATH_CALUDE_average_age_of_fourteen_students_l1425_142504

theorem average_age_of_fourteen_students
  (total_students : Nat)
  (total_average_age : ℚ)
  (ten_students_average : ℚ)
  (twenty_fifth_student_age : ℚ)
  (h1 : total_students = 25)
  (h2 : total_average_age = 25)
  (h3 : ten_students_average = 22)
  (h4 : twenty_fifth_student_age = 13) :
  (total_students * total_average_age - 10 * ten_students_average - twenty_fifth_student_age) / 14 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_fourteen_students_l1425_142504


namespace NUMINAMATH_CALUDE_perfect_square_swap_l1425_142536

theorem perfect_square_swap (a b : ℕ) (ha : a > b) (hb : b > 0) 
  (hA : ∃ k : ℕ, a^2 + 4*b + 1 = k^2) 
  (hB : ∃ m : ℕ, b^2 + 4*a + 1 = m^2) : 
  a = 8 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_perfect_square_swap_l1425_142536


namespace NUMINAMATH_CALUDE_lateral_edges_coplanar_iff_height_eq_edge_l1425_142585

/-- A cube with regular 4-sided pyramids on each face -/
structure PyramidCube where
  -- Edge length of the cube
  a : ℝ
  -- Height of the pyramids
  h : ℝ
  -- Assumption that a and h are positive
  a_pos : 0 < a
  h_pos : 0 < h

/-- The condition for lateral edges to lie in the same plane -/
def lateral_edges_coplanar (cube : PyramidCube) : Prop :=
  cube.h = cube.a

/-- Theorem stating the condition for lateral edges to be coplanar -/
theorem lateral_edges_coplanar_iff_height_eq_edge (cube : PyramidCube) :
  lateral_edges_coplanar cube ↔ cube.h = cube.a :=
sorry


end NUMINAMATH_CALUDE_lateral_edges_coplanar_iff_height_eq_edge_l1425_142585


namespace NUMINAMATH_CALUDE_figure_36_to_square_cut_and_rearrange_to_square_l1425_142563

/-- Represents a figure made up of small squares --/
structure Figure where
  squares : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to check if a figure can be rearranged into a square --/
def can_form_square (f : Figure) : Prop :=
  ∃ (s : Square), s.side_length * s.side_length = f.squares

/-- Theorem stating that a figure with 36 squares can form a square --/
theorem figure_36_to_square :
  ∀ (f : Figure), f.squares = 36 → can_form_square f :=
by
  sorry

/-- Theorem stating that a figure with 36 squares can be cut into two pieces
    and rearranged to form a square --/
theorem cut_and_rearrange_to_square :
  ∀ (f : Figure), f.squares = 36 →
  ∃ (piece1 piece2 : Figure),
    piece1.squares + piece2.squares = f.squares ∧
    can_form_square (Figure.mk (piece1.squares + piece2.squares)) :=
by
  sorry

end NUMINAMATH_CALUDE_figure_36_to_square_cut_and_rearrange_to_square_l1425_142563


namespace NUMINAMATH_CALUDE_blue_paint_cans_l1425_142564

/-- Given a paint mixture with a blue to green ratio of 4:3 and a total of 35 cans,
    prove that 20 cans of blue paint are needed. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : 
  total_cans = 35 → 
  blue_ratio = 4 → 
  green_ratio = 3 → 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l1425_142564


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1425_142592

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 35)
  (eq2 : 3 * u + 5 * v = -10) :
  u + v = -40 / 43 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1425_142592


namespace NUMINAMATH_CALUDE_parallelepiped_with_rectangular_opposite_faces_is_right_l1425_142568

/-- A parallelepiped is a three-dimensional figure with six faces, 
    where each pair of opposite faces are parallel parallelograms. -/
structure Parallelepiped

/-- A right parallelepiped is a parallelepiped where the lateral edges 
    are perpendicular to the base. -/
structure RightParallelepiped extends Parallelepiped

/-- A face of a parallelepiped -/
structure Face (P : Parallelepiped)

/-- Predicate to check if a face is rectangular -/
def is_rectangular (F : Face P) : Prop := sorry

/-- Predicate to check if two faces are opposite -/
def are_opposite (F1 F2 : Face P) : Prop := sorry

theorem parallelepiped_with_rectangular_opposite_faces_is_right 
  (P : Parallelepiped) 
  (F1 F2 : Face P) 
  (h1 : is_rectangular F1) 
  (h2 : is_rectangular F2) 
  (h3 : are_opposite F1 F2) : 
  RightParallelepiped := sorry

end NUMINAMATH_CALUDE_parallelepiped_with_rectangular_opposite_faces_is_right_l1425_142568


namespace NUMINAMATH_CALUDE_second_number_proof_l1425_142593

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 150)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 5 / 7) :
  y = 1000 / 21 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l1425_142593


namespace NUMINAMATH_CALUDE_p_shape_points_count_l1425_142533

/-- Represents the "П" shape formed from a square --/
structure PShape :=
  (side_length : ℕ)

/-- Calculates the number of points along the "П" shape --/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

theorem p_shape_points_count :
  ∀ (p : PShape), p.side_length = 10 → count_points p = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_p_shape_points_count_l1425_142533


namespace NUMINAMATH_CALUDE_total_credit_hours_l1425_142513

/-- Represents the number of credit hours for a course -/
structure CreditHours where
  hours : ℕ

/-- Represents a college course -/
structure Course where
  credits : CreditHours

def standard_course : Course :=
  { credits := { hours := 3 } }

def advanced_course : Course :=
  { credits := { hours := 4 } }

def max_courses : ℕ := 40
def max_semesters : ℕ := 4
def max_courses_per_semester : ℕ := 5
def max_advanced_courses : ℕ := 2

def sid_courses : ℕ := 4 * max_courses
def sid_advanced_courses : ℕ := 2 * max_advanced_courses

theorem total_credit_hours : 
  (max_courses - max_advanced_courses) * standard_course.credits.hours +
  max_advanced_courses * advanced_course.credits.hours +
  (sid_courses - sid_advanced_courses) * standard_course.credits.hours +
  sid_advanced_courses * advanced_course.credits.hours = 606 := by
  sorry


end NUMINAMATH_CALUDE_total_credit_hours_l1425_142513


namespace NUMINAMATH_CALUDE_set_union_problem_l1425_142545

theorem set_union_problem (x y : ℝ) :
  let A : Set ℝ := {x, y}
  let B : Set ℝ := {x + 1, 5}
  A ∩ B = {2} →
  A ∪ B = {1, 2, 5} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1425_142545


namespace NUMINAMATH_CALUDE_function_composition_l1425_142505

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_composition (h : ∀ x, f (3*x + 2) = 9*x + 8) : 
  ∀ x, f x = 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l1425_142505


namespace NUMINAMATH_CALUDE_line_perp_plane_contained_in_plane_implies_planes_perp_l1425_142599

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_contained_in_plane_implies_planes_perp
  (a : Line) (M N : Plane) :
  perpendicular a M → contained_in a N → planes_perpendicular M N :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_contained_in_plane_implies_planes_perp_l1425_142599


namespace NUMINAMATH_CALUDE_darcie_family_ratio_l1425_142557

/-- Represents the ages and relationships in Darcie's family -/
structure Family where
  darcie_age : ℕ
  father_age : ℕ
  mother_age_ratio : ℚ
  darcie_mother_ratio : ℚ

/-- Calculates the ratio of mother's age to father's age -/
def mother_father_ratio (f : Family) : ℚ :=
  (f.darcie_age : ℚ) * f.mother_age_ratio / f.father_age

/-- Theorem stating the ratio of Darcie's mother's age to her father's age -/
theorem darcie_family_ratio (f : Family) 
  (h1 : f.darcie_age = 4)
  (h2 : f.father_age = 30)
  (h3 : f.darcie_mother_ratio = 1/6)
  (h4 : f.mother_age_ratio = 1 / f.darcie_mother_ratio) :
  mother_father_ratio f = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_darcie_family_ratio_l1425_142557


namespace NUMINAMATH_CALUDE_melanie_breadcrumbs_count_l1425_142516

/-- Represents the number of pieces a bread slice is divided into -/
structure BreadDivision where
  firstHalf : Nat
  secondHalf : Nat

/-- Calculates the total number of pieces for a bread slice -/
def totalPieces (division : BreadDivision) : Nat :=
  division.firstHalf + division.secondHalf

/-- Represents Melanie's bread slicing method -/
def melanieBreadSlicing : List BreadDivision :=
  [{ firstHalf := 3, secondHalf := 4 },  -- First slice
   { firstHalf := 2, secondHalf := 10 }] -- Second slice

/-- Theorem: Melanie's bread slicing method results in 19 total pieces -/
theorem melanie_breadcrumbs_count :
  (melanieBreadSlicing.map totalPieces).sum = 19 := by
  sorry

#eval (melanieBreadSlicing.map totalPieces).sum

end NUMINAMATH_CALUDE_melanie_breadcrumbs_count_l1425_142516


namespace NUMINAMATH_CALUDE_concert_songs_count_l1425_142544

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- Calculates the total number of songs sung by the trios -/
def total_songs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The theorem to be proved -/
theorem concert_songs_count :
  ∀ (sc : SongCount),
    sc.mary = 3 →
    sc.alina = 5 →
    sc.hanna = 6 →
    sc.mary < sc.tina →
    sc.tina < sc.hanna →
    total_songs sc = 6 := by
  sorry


end NUMINAMATH_CALUDE_concert_songs_count_l1425_142544


namespace NUMINAMATH_CALUDE_cindys_calculation_l1425_142503

theorem cindys_calculation (x : ℚ) : 4 * (x / 2 - 6) = 24 → (2 * x - 4) / 6 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1425_142503


namespace NUMINAMATH_CALUDE_smallest_number_l1425_142521

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def octal_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := octal_to_decimal 101
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1425_142521


namespace NUMINAMATH_CALUDE_circle_center_l1425_142586

/-- The center of a circle with diameter endpoints (3, 3) and (9, -3) is (6, 0) -/
theorem circle_center (K : Set (ℝ × ℝ)) (p₁ p₂ : ℝ × ℝ) : 
  p₁ = (3, 3) → p₂ = (9, -3) → 
  (∀ x ∈ K, ∃ y ∈ K, (x.1 - y.1)^2 + (x.2 - y.2)^2 = (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) →
  (∃ c : ℝ × ℝ, c = (6, 0) ∧ ∀ x ∈ K, (x.1 - c.1)^2 + (x.2 - c.2)^2 = ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1425_142586


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1425_142565

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * Real.sqrt 2 * x

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 1

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), asymptote x y ∧ (∃ (k : ℝ), y = (b/a) * x + k)) 
  (h4 : ∃ (x y : ℝ), hyperbola a b x y ∧ directrix x ∧ parabola x y) :
  a^2 = 2 ∧ b^2 = 6 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1425_142565


namespace NUMINAMATH_CALUDE_population_is_all_scores_l1425_142517

/-- Represents a math exam with participants and their scores -/
structure MathExam where
  participants : ℕ
  scores : Finset ℝ

/-- Represents a statistical analysis of a math exam -/
structure StatisticalAnalysis where
  exam : MathExam
  sample_size : ℕ

/-- The definition of population in the context of this statistical analysis -/
def population (analysis : StatisticalAnalysis) : Finset ℝ :=
  analysis.exam.scores

/-- Theorem stating that the population in this statistical analysis
    is the set of all participants' scores -/
theorem population_is_all_scores
  (exam : MathExam)
  (analysis : StatisticalAnalysis)
  (h1 : exam.participants = 40000)
  (h2 : analysis.sample_size = 400)
  (h3 : analysis.exam = exam)
  (h4 : exam.scores.card = exam.participants) :
  population analysis = exam.scores :=
sorry

end NUMINAMATH_CALUDE_population_is_all_scores_l1425_142517


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1425_142543

theorem sum_of_squares_of_roots (x y : ℝ) : 
  (3 * x^2 - 7 * x + 5 = 0) → 
  (3 * y^2 - 7 * y + 5 = 0) → 
  (x^2 + y^2 = 19/9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1425_142543


namespace NUMINAMATH_CALUDE_school_survey_sample_size_l1425_142552

/-- Represents a survey conducted in a school -/
structure SchoolSurvey where
  total_students : ℕ
  selected_students : ℕ

/-- The sample size of a school survey is the number of selected students -/
def sample_size (survey : SchoolSurvey) : ℕ := survey.selected_students

/-- Theorem: For a school with 3600 students and 200 randomly selected for a survey,
    the sample size is 200 -/
theorem school_survey_sample_size :
  let survey := SchoolSurvey.mk 3600 200
  sample_size survey = 200 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_sample_size_l1425_142552


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l1425_142535

theorem unique_two_digit_integer (s : ℕ) : 
  (10 ≤ s ∧ s < 100) ∧ (13 * s) % 100 = 52 ↔ s = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l1425_142535


namespace NUMINAMATH_CALUDE_three_eighths_divided_by_one_fourth_l1425_142529

theorem three_eighths_divided_by_one_fourth : (3 : ℚ) / 8 / ((1 : ℚ) / 4) = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_divided_by_one_fourth_l1425_142529


namespace NUMINAMATH_CALUDE_factorization_proof_l1425_142527

theorem factorization_proof (x : ℝ) : -8*x^2 + 8*x - 2 = -2*(2*x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1425_142527


namespace NUMINAMATH_CALUDE_multiply_special_form_l1425_142511

theorem multiply_special_form (x : ℝ) : 
  (x^4 + 18*x^2 + 324) * (x^2 - 18) = x^6 - 5832 := by
  sorry

end NUMINAMATH_CALUDE_multiply_special_form_l1425_142511


namespace NUMINAMATH_CALUDE_constant_function_from_functional_equation_l1425_142594

/-- A continuous function f satisfying f(x) + f(x^2) = 2 for all real x is constant and equal to 1. -/
theorem constant_function_from_functional_equation (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (h : ∀ x : ℝ, f x + f (x^2) = 2) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_from_functional_equation_l1425_142594


namespace NUMINAMATH_CALUDE_simon_red_stamps_count_l1425_142574

/-- The number of red stamps Simon has -/
def simon_red_stamps : ℕ := 34

/-- The number of white stamps Peter has -/
def peter_white_stamps : ℕ := 80

/-- The selling price of a red stamp in dollars -/
def red_stamp_price : ℚ := 1/2

/-- The selling price of a white stamp in dollars -/
def white_stamp_price : ℚ := 1/5

/-- The difference in the amount of money they make in dollars -/
def money_difference : ℚ := 1

theorem simon_red_stamps_count : 
  (simon_red_stamps : ℚ) * red_stamp_price - (peter_white_stamps : ℚ) * white_stamp_price = money_difference :=
by sorry

end NUMINAMATH_CALUDE_simon_red_stamps_count_l1425_142574


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_l1425_142561

theorem bernardo_silvia_game (M : ℕ) : 
  (M ≤ 1999) →
  (32 * M + 1600 < 2000) →
  (32 * M + 1700 ≥ 2000) →
  (∀ N : ℕ, N < M → (32 * N + 1600 < 2000 → 32 * N + 1700 < 2000)) →
  (M = 10 ∧ (M / 10 + M % 10 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_l1425_142561


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1425_142566

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (m+3, m+1) -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m + 1 }

/-- Theorem: If P(m+3, m+1) lies on the x-axis, then its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (P m).y = 0 → P m = { x := 2, y := 0 } := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1425_142566


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1425_142555

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1425_142555


namespace NUMINAMATH_CALUDE_hundred_digit_number_theorem_l1425_142569

def is_valid_number (N : ℕ) : Prop :=
  ∃ (b : ℕ), b ∈ ({1, 2, 3} : Set ℕ) ∧ N = 325 * b * (10 ^ 97)

theorem hundred_digit_number_theorem (N : ℕ) :
  (∃ (k : ℕ) (a : ℕ), 
    a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    N ≥ 10^99 ∧ N < 10^100 ∧
    ∃ (N' : ℕ), (N' = N - a * 10^k ∨ (k = 99 ∧ N' = N - a * 10^99)) ∧ N = 13 * N') →
  is_valid_number N :=
sorry

end NUMINAMATH_CALUDE_hundred_digit_number_theorem_l1425_142569


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1425_142520

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The point symmetric to P(1, -2) with respect to the x-axis is (1, 2) -/
theorem symmetric_point_x_axis :
  let P : Point := { x := 1, y := -2 }
  symmetricXAxis P = { x := 1, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1425_142520


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1425_142506

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧ 
  10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1425_142506


namespace NUMINAMATH_CALUDE_total_amount_is_200_l1425_142512

/-- Represents the distribution of money among four individuals -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount of money distributed -/
def total_amount (d : MoneyDistribution) : ℝ :=
  d.w + d.x + d.y + d.z

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_200 (d : MoneyDistribution) 
  (h1 : d.x = 0.75 * d.w)
  (h2 : d.y = 0.45 * d.w)
  (h3 : d.z = 0.30 * d.w)
  (h4 : d.y = 36) :
  total_amount d = 200 := by
  sorry

#check total_amount_is_200

end NUMINAMATH_CALUDE_total_amount_is_200_l1425_142512


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1425_142525

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 for some real number a. -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, p x = (x + a)^2

/-- Given that x^2 + (m-2)x + 9 is a perfect square trinomial, prove that m = 8 or m = -4. -/
theorem perfect_square_trinomial_m_values (m : ℝ) :
  IsPerfectSquareTrinomial (fun x ↦ x^2 + (m-2)*x + 9) → m = 8 ∨ m = -4 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l1425_142525


namespace NUMINAMATH_CALUDE_difference_of_squares_625_575_l1425_142532

theorem difference_of_squares_625_575 : 625^2 - 575^2 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_625_575_l1425_142532


namespace NUMINAMATH_CALUDE_magnitude_e1_minus_sqrt3_e2_l1425_142551

-- Define the vector space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- State the theorem
theorem magnitude_e1_minus_sqrt3_e2 
  (h1 : ‖e₁‖ = 1) 
  (h2 : ‖e₂‖ = 1) 
  (h3 : inner e₁ e₂ = Real.sqrt 3 / 2) : 
  ‖e₁ - Real.sqrt 3 • e₂‖ = 1 := by sorry

end NUMINAMATH_CALUDE_magnitude_e1_minus_sqrt3_e2_l1425_142551


namespace NUMINAMATH_CALUDE_num_non_officers_calculation_l1425_142562

-- Define the problem parameters
def avg_salary_all : ℝ := 120
def avg_salary_officers : ℝ := 420
def avg_salary_non_officers : ℝ := 110
def num_officers : ℕ := 15

-- Define the theorem
theorem num_non_officers_calculation :
  ∃ (num_non_officers : ℕ),
    (num_officers : ℝ) * avg_salary_officers + (num_non_officers : ℝ) * avg_salary_non_officers =
    ((num_officers : ℝ) + (num_non_officers : ℝ)) * avg_salary_all ∧
    num_non_officers = 450 := by
  sorry

end NUMINAMATH_CALUDE_num_non_officers_calculation_l1425_142562


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1425_142508

/-- A quadratic function f(x) = x^2 + ax + b satisfying certain conditions -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that f(x + 1/x) = f(x) + f(1/x) for all nonzero real x -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f (x + 1/x) = f x + f (1/x)

/-- The property that the roots of f(x) = 0 are integers -/
def HasIntegerRoots (f : ℝ → ℝ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℝ, f x = 0 ↔ x = p ∨ x = q

theorem quadratic_function_property (a b : ℝ) :
  SatisfiesProperty (QuadraticFunction a b) →
  HasIntegerRoots (QuadraticFunction a b) →
  a^2 + b^2 = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1425_142508


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1425_142576

/-- Given that Mike found a total of 6 seashells and 4 of them were broken,
    prove that the number of unbroken seashells is 2. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 6) (h2 : broken = 4) :
  total - broken = 2 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1425_142576


namespace NUMINAMATH_CALUDE_fred_cards_l1425_142570

theorem fred_cards (initial_cards torn_cards bought_cards total_cards : ℕ) : 
  initial_cards = 18 →
  torn_cards = 8 →
  bought_cards = 40 →
  total_cards = 84 →
  total_cards = initial_cards - torn_cards + bought_cards + (total_cards - (initial_cards - torn_cards + bought_cards)) →
  total_cards - (initial_cards - torn_cards + bought_cards) = 34 := by
  sorry

end NUMINAMATH_CALUDE_fred_cards_l1425_142570


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l1425_142541

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 := by
  sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l1425_142541


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l1425_142573

theorem complex_roots_theorem (a b c : ℂ) : 
  a + b + c = 1 ∧ 
  a * b + a * c + b * c = 1 ∧ 
  a * b * c = -1 → 
  ({a, b, c} : Set ℂ) = {1, Complex.I, -Complex.I} :=
sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l1425_142573


namespace NUMINAMATH_CALUDE_cards_given_by_jeff_l1425_142530

theorem cards_given_by_jeff (initial_cards final_cards : ℝ) 
  (h1 : initial_cards = 304.0)
  (h2 : final_cards = 580) :
  final_cards - initial_cards = 276 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_by_jeff_l1425_142530


namespace NUMINAMATH_CALUDE_lcm_problem_l1425_142590

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) :
  Nat.lcm b c = 540 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1425_142590


namespace NUMINAMATH_CALUDE_jessica_calculation_l1425_142577

theorem jessica_calculation (y : ℝ) : (y - 8) / 4 = 22 → (y - 4) / 8 = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_calculation_l1425_142577
