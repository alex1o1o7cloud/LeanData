import Mathlib

namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l3750_375081

theorem range_of_k_for_inequality (k : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → k ∈ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l3750_375081


namespace NUMINAMATH_CALUDE_intersection_equality_l3750_375094

def M : Set ℝ := {x : ℝ | x < 2012}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_equality : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l3750_375094


namespace NUMINAMATH_CALUDE_triangle_problem_l3750_375017

theorem triangle_problem (a b c A B C : Real) 
  (h1 : 2 * Real.sqrt 3 * a * b * Real.sin C = a^2 + b^2 - c^2)
  (h2 : a * Real.sin B = b * Real.cos A)
  (h3 : a = 2) :
  C = π/6 ∧ (1/2 * a * c * Real.sin B = (Real.sqrt 3 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3750_375017


namespace NUMINAMATH_CALUDE_f_of_three_equals_nine_sevenths_l3750_375098

/-- Given f(x) = (2x + 3) / (4x - 5), prove that f(3) = 9/7 -/
theorem f_of_three_equals_nine_sevenths :
  let f : ℝ → ℝ := λ x ↦ (2*x + 3) / (4*x - 5)
  f 3 = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_three_equals_nine_sevenths_l3750_375098


namespace NUMINAMATH_CALUDE_pigeon_problem_l3750_375072

/-- The number of pigeons in a group with the following properties:
  1. When each pigeonhole houses 6 pigeons, 3 pigeons are left without a pigeonhole.
  2. When 5 more pigeons arrive, each pigeonhole fits exactly 8 pigeons. -/
def original_pigeons : ℕ := 27

/-- The number of pigeonholes available. -/
def pigeonholes : ℕ := 3

theorem pigeon_problem :
  (6 * pigeonholes + 3 = original_pigeons) ∧
  (8 * pigeonholes = original_pigeons + 5) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_problem_l3750_375072


namespace NUMINAMATH_CALUDE_exams_fourth_year_l3750_375075

theorem exams_fourth_year 
  (a b c d e : ℕ) 
  (h_sum : a + b + c + d + e = 31)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_fifth : e = 3 * a)
  : d = 8 := by
  sorry

end NUMINAMATH_CALUDE_exams_fourth_year_l3750_375075


namespace NUMINAMATH_CALUDE_smallest_total_hits_is_twelve_l3750_375001

/-- Represents a baseball player's batting statistics -/
structure BattingStats where
  initialHits : ℕ
  initialAtBats : ℕ
  newHits : ℕ
  newAtBats : ℕ
  initialAverage : ℚ
  newAverage : ℚ

/-- Calculates the smallest number of total hits given initial and new batting averages -/
def smallestTotalHits (stats : BattingStats) : ℕ :=
  stats.initialHits + stats.newHits

/-- Theorem: The smallest number of total hits is 12 given the specified conditions -/
theorem smallest_total_hits_is_twelve :
  ∃ (stats : BattingStats),
    stats.initialAverage = 360 / 1000 ∧
    stats.newAverage = 400 / 1000 ∧
    stats.newAtBats = stats.initialAtBats + 5 ∧
    smallestTotalHits stats = 12 ∧
    ∀ (otherStats : BattingStats),
      otherStats.initialAverage = 360 / 1000 ∧
      otherStats.newAverage = 400 / 1000 ∧
      otherStats.newAtBats = otherStats.initialAtBats + 5 →
      smallestTotalHits otherStats ≥ 12 :=
by sorry


end NUMINAMATH_CALUDE_smallest_total_hits_is_twelve_l3750_375001


namespace NUMINAMATH_CALUDE_largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l3750_375099

theorem largest_n_for_negative_quadratic : 
  ∀ n : ℤ, n^2 - 9*n + 18 < 0 → n ≤ 5 :=
by sorry

theorem five_satisfies_condition : 
  (5 : ℤ)^2 - 9*5 + 18 < 0 :=
by sorry

theorem six_does_not_satisfy : 
  ¬((6 : ℤ)^2 - 9*6 + 18 < 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_negative_quadratic_five_satisfies_condition_six_does_not_satisfy_l3750_375099


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l3750_375054

-- Define the properties of the first triangle
def first_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a = 15 ∧ c = 34

-- Define the similarity ratio between the two triangles
def similarity_ratio (r : ℝ) : Prop :=
  r = 102 / 34

-- Define the shortest side of the second triangle
def shortest_side (x : ℝ) : Prop :=
  x = 3 * Real.sqrt 931

-- Theorem statement
theorem triangle_similarity_theorem :
  ∀ a b c r x : ℝ,
  first_triangle a b c →
  similarity_ratio r →
  shortest_side x →
  x = r * a :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l3750_375054


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3750_375041

def original_expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squared_coefficients :
  ∃ a b c d : ℝ, 
    (∀ x : ℝ, original_expression x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 2395 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3750_375041


namespace NUMINAMATH_CALUDE_time_until_sunset_l3750_375076

-- Define the initial sunset time in minutes past midnight
def initial_sunset : ℕ := 18 * 60

-- Define the daily sunset delay in minutes
def daily_delay : ℚ := 1.2

-- Define the number of days since March 1st
def days_passed : ℕ := 40

-- Define the current time in minutes past midnight
def current_time : ℕ := 18 * 60 + 10

-- Theorem statement
theorem time_until_sunset :
  let total_delay : ℚ := daily_delay * days_passed
  let new_sunset : ℚ := initial_sunset + total_delay
  ⌊new_sunset⌋ - current_time = 38 := by sorry

end NUMINAMATH_CALUDE_time_until_sunset_l3750_375076


namespace NUMINAMATH_CALUDE_radical_product_equals_64_l3750_375093

theorem radical_product_equals_64 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_64_l3750_375093


namespace NUMINAMATH_CALUDE_discriminant_positive_roots_when_k_zero_l3750_375071

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 1

-- Define the discriminant of the quadratic equation f(x) = 0
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*1*(-1)

-- Theorem 1: The discriminant is always positive
theorem discriminant_positive (k : ℝ) : discriminant k > 0 := by
  sorry

-- Theorem 2: When k = 0, the roots are 1 and -1
theorem roots_when_k_zero :
  ∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = -1 ∧ f 0 x1 = 0 ∧ f 0 x2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_positive_roots_when_k_zero_l3750_375071


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_18_16_l3750_375078

theorem half_abs_diff_squares_18_16 : (1 / 2 : ℝ) * |18^2 - 16^2| = 34 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_18_16_l3750_375078


namespace NUMINAMATH_CALUDE_x_varies_as_z_power_l3750_375079

-- Define the relationships between x, y, and z
def varies_as (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ t, f t = k * g t

-- State the theorem
theorem x_varies_as_z_power (x y z : ℝ → ℝ) :
  varies_as x (λ t => (y t)^4) →
  varies_as y (λ t => (z t)^(1/3)) →
  varies_as x (λ t => (z t)^(4/3)) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_z_power_l3750_375079


namespace NUMINAMATH_CALUDE_number_of_operations_indicates_quality_l3750_375005

-- Define a type for algorithms
structure Algorithm : Type where
  name : String

-- Define a measure for the number of operations
def numberOfOperations (a : Algorithm) : ℕ := sorry

-- Define a measure for algorithm quality
def algorithmQuality (a : Algorithm) : ℝ := sorry

-- Define a measure for computer speed
def computerSpeed : ℝ := sorry

-- State the theorem
theorem number_of_operations_indicates_quality (a : Algorithm) :
  computerSpeed > 0 →
  algorithmQuality a = (1 / numberOfOperations a) * computerSpeed :=
sorry

end NUMINAMATH_CALUDE_number_of_operations_indicates_quality_l3750_375005


namespace NUMINAMATH_CALUDE_principal_calculation_l3750_375042

theorem principal_calculation (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 720)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 600 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l3750_375042


namespace NUMINAMATH_CALUDE_games_that_didnt_work_l3750_375058

/-- The number of games that didn't work, given Edward's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_games good_games : ℕ) : 
  friend_games = 41 → garage_games = 14 → good_games = 24 → 
  friend_games + garage_games - good_games = 31 := by
  sorry

end NUMINAMATH_CALUDE_games_that_didnt_work_l3750_375058


namespace NUMINAMATH_CALUDE_find_A_l3750_375057

theorem find_A : ∃ A : ℕ, A % 5 = 4 ∧ A / 5 = 6 ∧ A = 34 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3750_375057


namespace NUMINAMATH_CALUDE_product_and_sum_of_squares_l3750_375070

theorem product_and_sum_of_squares (x y : ℝ) : 
  x * y = 120 → x^2 + y^2 = 289 → x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_squares_l3750_375070


namespace NUMINAMATH_CALUDE_smallest_with_sum_2011_has_224_digits_l3750_375040

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def numberOfDigits (n : ℕ) : ℕ := sorry

/-- The smallest natural number with a given sum of digits -/
def smallestWithSumOfDigits (s : ℕ) : ℕ := sorry

theorem smallest_with_sum_2011_has_224_digits :
  numberOfDigits (smallestWithSumOfDigits 2011) = 224 := by sorry

end NUMINAMATH_CALUDE_smallest_with_sum_2011_has_224_digits_l3750_375040


namespace NUMINAMATH_CALUDE_rulers_in_drawer_l3750_375087

theorem rulers_in_drawer (initial_rulers : ℕ) (added_rulers : ℕ) : 
  initial_rulers = 46 → added_rulers = 25 → initial_rulers + added_rulers = 71 := by
  sorry

end NUMINAMATH_CALUDE_rulers_in_drawer_l3750_375087


namespace NUMINAMATH_CALUDE_discount_difference_l3750_375082

theorem discount_difference (bill : ℝ) (single_discount : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : 
  bill = 12000 ∧ 
  single_discount = 0.35 ∧ 
  discount1 = 0.25 ∧ 
  discount2 = 0.08 ∧ 
  discount3 = 0.02 → 
  bill * (1 - (1 - discount1) * (1 - discount2) * (1 - discount3)) - 
  bill * single_discount = 314.40 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3750_375082


namespace NUMINAMATH_CALUDE_modulo_six_equality_l3750_375033

theorem modulo_six_equality : 47^1860 - 25^1860 ≡ 0 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_modulo_six_equality_l3750_375033


namespace NUMINAMATH_CALUDE_downstream_distance_l3750_375048

/-- Calculates the distance traveled downstream given boat speed, stream rate, and time. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_rate : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_rate = 5)
  (h3 : time = 6) :
  boat_speed + stream_rate * time = 126 :=
by sorry

end NUMINAMATH_CALUDE_downstream_distance_l3750_375048


namespace NUMINAMATH_CALUDE_power_bank_sales_theorem_l3750_375073

/-- Represents the sales scenario of mobile power banks -/
structure PowerBankSales where
  m : ℝ  -- Wholesale price per power bank
  n : ℝ  -- Markup per power bank
  total_count : ℕ := 100  -- Total number of power banks
  full_price_sold : ℕ := 60  -- Number of power banks sold at full price
  discount_rate : ℝ := 0.2  -- Discount rate for remaining power banks

/-- Calculates the total selling price of all power banks -/
def total_selling_price (s : PowerBankSales) : ℝ :=
  s.total_count * (s.m + s.n)

/-- Calculates the actual total revenue -/
def actual_revenue (s : PowerBankSales) : ℝ :=
  s.full_price_sold * (s.m + s.n) + 
  (s.total_count - s.full_price_sold) * (1 - s.discount_rate) * (s.m + s.n)

/-- Calculates the additional profit without discount -/
def additional_profit (s : PowerBankSales) : ℝ :=
  s.total_count * s.n - (actual_revenue s - s.total_count * s.m)

theorem power_bank_sales_theorem (s : PowerBankSales) :
  total_selling_price s = 100 * (s.m + s.n) ∧
  actual_revenue s = 92 * (s.m + s.n) ∧
  additional_profit s = 8 * (s.m + s.n) := by
  sorry

#check power_bank_sales_theorem

end NUMINAMATH_CALUDE_power_bank_sales_theorem_l3750_375073


namespace NUMINAMATH_CALUDE_stream_rate_proof_l3750_375060

/-- The speed of the man rowing in still water -/
def still_water_speed : ℝ := 24

/-- The rate of the stream -/
def stream_rate : ℝ := 12

/-- The ratio of time taken to row upstream vs downstream -/
def time_ratio : ℝ := 3

theorem stream_rate_proof :
  (1 / (still_water_speed - stream_rate) = time_ratio * (1 / (still_water_speed + stream_rate))) →
  stream_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_stream_rate_proof_l3750_375060


namespace NUMINAMATH_CALUDE_divisible_by_19_l3750_375012

theorem divisible_by_19 (n : ℕ) : 
  19 ∣ (12000 + 3 * 10^n + 8) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_19_l3750_375012


namespace NUMINAMATH_CALUDE_kermit_final_positions_l3750_375089

/-- The number of integer coordinate pairs (x, y) satisfying |x| + |y| = n -/
def count_coordinate_pairs (n : ℕ) : ℕ :=
  2 * (n + 1) * (n + 1) - 2 * n * (n + 1) + 1

/-- Kermit's energy in Joules -/
def kermit_energy : ℕ := 100

theorem kermit_final_positions : 
  count_coordinate_pairs kermit_energy = 10201 :=
sorry

end NUMINAMATH_CALUDE_kermit_final_positions_l3750_375089


namespace NUMINAMATH_CALUDE_delta_five_three_l3750_375066

def delta (a b : ℤ) : ℤ := 4 * a - 6 * b

theorem delta_five_three : delta 5 3 = 2 := by sorry

end NUMINAMATH_CALUDE_delta_five_three_l3750_375066


namespace NUMINAMATH_CALUDE_solution_set_l3750_375018

theorem solution_set (x : ℝ) : 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 7 ↔ 49 / 20 < x ∧ x ≤ 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3750_375018


namespace NUMINAMATH_CALUDE_min_swaps_100_l3750_375049

/-- The type representing a permutation of the first 100 natural numbers. -/
def Perm100 := Fin 100 → Fin 100

/-- The identity permutation. -/
def id_perm : Perm100 := fun i => i

/-- The target permutation we want to achieve. -/
def target_perm : Perm100 := fun i =>
  if i = 99 then 0 else i + 1

/-- A swap operation on a permutation. -/
def swap (p : Perm100) (i j : Fin 100) : Perm100 := fun k =>
  if k = i then p j
  else if k = j then p i
  else p k

/-- The number of swaps needed to transform one permutation into another. -/
def num_swaps (p q : Perm100) : ℕ := sorry

theorem min_swaps_100 :
  num_swaps id_perm target_perm = 99 := by sorry

end NUMINAMATH_CALUDE_min_swaps_100_l3750_375049


namespace NUMINAMATH_CALUDE_brick_width_calculation_l3750_375045

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ brick_width : ℝ,
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = brick_length * brick_width * total_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l3750_375045


namespace NUMINAMATH_CALUDE_basic_computer_price_l3750_375061

/-- Given the price of a basic computer and printer, prove the price of the basic computer. -/
theorem basic_computer_price (basic_price printer_price enhanced_price : ℝ) : 
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  enhanced_price + printer_price = 6 * printer_price →
  basic_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_basic_computer_price_l3750_375061


namespace NUMINAMATH_CALUDE_profit_increase_approx_l3750_375095

/-- Represents the monthly profit changes as factors -/
def march_to_april : ℝ := 1.35
def april_to_may : ℝ := 0.80
def may_to_june : ℝ := 1.50
def june_to_july : ℝ := 0.75
def july_to_august : ℝ := 1.45

/-- The overall factor of profit change from March to August -/
def overall_factor : ℝ :=
  march_to_april * april_to_may * may_to_june * june_to_july * july_to_august

/-- The overall percentage increase from March to August -/
def overall_percentage_increase : ℝ := (overall_factor - 1) * 100

/-- Theorem stating the overall percentage increase is approximately 21.95% -/
theorem profit_increase_approx :
  ∃ ε > 0, abs (overall_percentage_increase - 21.95) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_increase_approx_l3750_375095


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l3750_375062

theorem sum_of_factors_72 : (Finset.filter (· ∣ 72) (Finset.range 73)).sum id = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l3750_375062


namespace NUMINAMATH_CALUDE_smallest_x_cos_equality_l3750_375022

theorem smallest_x_cos_equality : ∃ x : ℝ, 
  x > 30 ∧ 
  Real.cos (x * Real.pi / 180) = Real.cos ((2 * x + 10) * Real.pi / 180) ∧
  x < 117 ∧
  ∀ y : ℝ, y > 30 ∧ 
    Real.cos (y * Real.pi / 180) = Real.cos ((2 * y + 10) * Real.pi / 180) → 
    y ≥ x ∧
  ⌈x⌉ = 117 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_cos_equality_l3750_375022


namespace NUMINAMATH_CALUDE_wednesday_tips_calculation_l3750_375080

/-- Represents Hallie's work data for a day -/
structure WorkDay where
  hours : ℕ
  tips : ℕ

/-- Calculates the total earnings for a given work day with an hourly rate -/
def dailyEarnings (day : WorkDay) (hourlyRate : ℕ) : ℕ :=
  day.hours * hourlyRate + day.tips

theorem wednesday_tips_calculation (hourlyRate : ℕ) (monday tuesday wednesday : WorkDay) 
    (totalEarnings : ℕ) : 
    hourlyRate = 10 →
    monday.hours = 7 →
    monday.tips = 18 →
    tuesday.hours = 5 →
    tuesday.tips = 12 →
    wednesday.hours = 7 →
    totalEarnings = 240 →
    totalEarnings = dailyEarnings monday hourlyRate + 
                    dailyEarnings tuesday hourlyRate + 
                    dailyEarnings wednesday hourlyRate →
    wednesday.tips = 20 := by
  sorry

#check wednesday_tips_calculation

end NUMINAMATH_CALUDE_wednesday_tips_calculation_l3750_375080


namespace NUMINAMATH_CALUDE_initial_gasohol_volume_l3750_375044

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ
  gasoline : ℝ
  valid : ethanol + gasoline = 1

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ
  mixture : FuelMixture

def initial_mixture : FuelMixture := {
  ethanol := 0.05,
  gasoline := 0.95,
  valid := by norm_num
}

def desired_mixture : FuelMixture := {
  ethanol := 0.1,
  gasoline := 0.9,
  valid := by norm_num
}

def ethanol_added : ℝ := 2

theorem initial_gasohol_volume (initial : FuelTank) :
  initial.mixture = initial_mixture →
  (∃ (final : FuelTank), 
    final.volume = initial.volume + ethanol_added ∧
    final.mixture = desired_mixture) →
  initial.volume = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasohol_volume_l3750_375044


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l3750_375059

theorem sqrt_sum_equals_eight :
  Real.sqrt (18 - 8 * Real.sqrt 2) + Real.sqrt (18 + 8 * Real.sqrt 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l3750_375059


namespace NUMINAMATH_CALUDE_two_tangent_or_parallel_lines_l3750_375051

/-- A parabola in the x-y plane defined by y^2 = -8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -8 * p.1}

/-- The point P through which the lines must pass -/
def P : ℝ × ℝ := (-2, -4)

/-- A line that passes through point P and has only one common point with the parabola -/
def TangentOrParallelLine (l : Set (ℝ × ℝ)) : Prop :=
  P ∈ l ∧ (∃! p, p ∈ l ∩ Parabola)

/-- There are exactly two lines that pass through P and have only one common point with the parabola -/
theorem two_tangent_or_parallel_lines : 
  ∃! (l1 l2 : Set (ℝ × ℝ)), l1 ≠ l2 ∧ TangentOrParallelLine l1 ∧ TangentOrParallelLine l2 ∧ 
  (∀ l, TangentOrParallelLine l → l = l1 ∨ l = l2) :=
sorry

end NUMINAMATH_CALUDE_two_tangent_or_parallel_lines_l3750_375051


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l3750_375030

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Given a circle with center (1, 2) and one endpoint of a diameter at (4, 6),
    the other endpoint of the diameter is at (-2, -2) -/
theorem circle_diameter_endpoint (P : Circle) (d : Diameter) :
  P.center = (1, 2) →
  d.endpoint1 = (4, 6) →
  d.endpoint2 = (-2, -2) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l3750_375030


namespace NUMINAMATH_CALUDE_heart_diamond_inequality_l3750_375085

def heart (x y : ℝ) : ℝ := |x - y|

def diamond (z w : ℝ) : ℝ := (z + w)^2

theorem heart_diamond_inequality : ∃ x y : ℝ, (heart x y)^2 ≠ diamond x y := by sorry

end NUMINAMATH_CALUDE_heart_diamond_inequality_l3750_375085


namespace NUMINAMATH_CALUDE_tv_price_change_l3750_375013

theorem tv_price_change (original_price : ℝ) (h : original_price > 0) :
  let price_after_decrease := original_price * (1 - 0.2)
  let final_price := price_after_decrease * (1 + 0.4)
  let net_change := (final_price - original_price) / original_price
  net_change = 0.12 := by
sorry

end NUMINAMATH_CALUDE_tv_price_change_l3750_375013


namespace NUMINAMATH_CALUDE_f_symmetry_l3750_375034

/-- Given a function f(x) = x^5 + ax^3 + bx, if f(-2) = 10, then f(2) = -10 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by sorry

end NUMINAMATH_CALUDE_f_symmetry_l3750_375034


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3750_375020

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3750_375020


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l3750_375039

theorem sum_of_possible_values (x y : ℝ) 
  (h : 2 * x * y - 2 * x / (y^2) - 2 * y / (x^2) = 4) : 
  ∃ (v₁ v₂ : ℝ), (x - 2) * (y - 2) = v₁ ∨ (x - 2) * (y - 2) = v₂ ∧ v₁ + v₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l3750_375039


namespace NUMINAMATH_CALUDE_smaller_box_length_l3750_375011

/-- Given a larger box and smaller boxes with specified dimensions, 
    proves that the length of the smaller box is 60 cm when 1000 boxes fit. -/
theorem smaller_box_length 
  (large_box_length : ℕ) 
  (large_box_width : ℕ) 
  (large_box_height : ℕ)
  (small_box_width : ℕ) 
  (small_box_height : ℕ)
  (max_small_boxes : ℕ)
  (h1 : large_box_length = 600)
  (h2 : large_box_width = 500)
  (h3 : large_box_height = 400)
  (h4 : small_box_width = 50)
  (h5 : small_box_height = 40)
  (h6 : max_small_boxes = 1000) :
  ∃ (small_box_length : ℕ), 
    small_box_length = 60 ∧ 
    (small_box_length * small_box_width * small_box_height) * max_small_boxes ≤ 
      large_box_length * large_box_width * large_box_height :=
by sorry

end NUMINAMATH_CALUDE_smaller_box_length_l3750_375011


namespace NUMINAMATH_CALUDE_train_speed_l3750_375026

theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3750_375026


namespace NUMINAMATH_CALUDE_square_root_problem_l3750_375043

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 3) = 3) 
  (h2 : (3 * b - 2) ^ (1/3 : ℝ) = 2) : 
  Real.sqrt (a + 3*b) = 6 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l3750_375043


namespace NUMINAMATH_CALUDE_percentage_to_pass_l3750_375084

/-- Given a test with maximum marks, a student's score, and the amount by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_by : ℕ) :
  max_marks = 300 →
  student_score = 80 →
  fail_by = 100 →
  (((student_score + fail_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l3750_375084


namespace NUMINAMATH_CALUDE_find_number_l3750_375021

theorem find_number : ∃! x : ℝ, ((x / 12 - 32) * 3 - 45) = 159 := by sorry

end NUMINAMATH_CALUDE_find_number_l3750_375021


namespace NUMINAMATH_CALUDE_michael_has_52_robots_l3750_375037

/-- The number of flying robots Tom has -/
def tom_robots : ℕ := 12

/-- The ratio of Michael's robots to Tom's robots -/
def michael_to_tom_ratio : ℕ := 4

/-- The number of robots Tom gives away for every group of robots he has -/
def tom_giveaway_ratio : ℕ := 1

/-- The size of the group of robots Tom considers when giving away -/
def tom_group_size : ℕ := 3

/-- Calculates the number of flying robots Michael has in total -/
def michael_total_robots : ℕ :=
  (michael_to_tom_ratio * tom_robots) + (tom_robots / tom_group_size)

/-- Theorem stating that Michael has 52 flying robots in total -/
theorem michael_has_52_robots : michael_total_robots = 52 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_52_robots_l3750_375037


namespace NUMINAMATH_CALUDE_image_difference_l3750_375032

/-- Define the mapping f -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.1 + p.2)

/-- Theorem statement -/
theorem image_difference (m n : ℝ) (h : (m, n) = f (2, 1)) :
  m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_image_difference_l3750_375032


namespace NUMINAMATH_CALUDE_opposite_face_is_t_l3750_375023

-- Define the faces of the cube
inductive Face : Type
  | p | q | r | s | t | u

-- Define the cube structure
structure Cube where
  top : Face
  right : Face
  left : Face
  bottom : Face
  front : Face
  back : Face

-- Define the conditions of the problem
def problem_cube : Cube :=
  { top := Face.p
  , right := Face.q
  , left := Face.r
  , bottom := Face.t  -- We'll prove this is correct
  , front := Face.s   -- Arbitrary assignment for remaining faces
  , back := Face.u }  -- Arbitrary assignment for remaining faces

-- Theorem statement
theorem opposite_face_is_t (c : Cube) :
  c.top = Face.p → c.right = Face.q → c.left = Face.r → c.bottom = Face.t :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_t_l3750_375023


namespace NUMINAMATH_CALUDE_diagonal_sum_is_384_l3750_375004

/-- A cyclic hexagon with five sides of length 81 and one side of length 31 -/
structure CyclicHexagon where
  -- Five sides have length 81
  side_length : ℝ
  side_length_eq : side_length = 81
  -- One side (AB) has length 31
  AB_length : ℝ
  AB_length_eq : AB_length = 31

/-- The sum of the lengths of the three diagonals drawn from one vertex in the hexagon -/
def diagonal_sum (h : CyclicHexagon) : ℝ := sorry

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex is 384 -/
theorem diagonal_sum_is_384 (h : CyclicHexagon) : diagonal_sum h = 384 := by sorry

end NUMINAMATH_CALUDE_diagonal_sum_is_384_l3750_375004


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3750_375074

theorem min_value_of_sum_of_fractions (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) (hy : -1 < y ∧ y < 0) (hz : -1 < z ∧ z < 0) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (w : ℝ), w = 1/((1-x)*(1-y)*(1-z)) + 1/((1+x)*(1+y)*(1+z)) → m ≤ w :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3750_375074


namespace NUMINAMATH_CALUDE_quadratic_sum_l3750_375036

/-- Given a quadratic polynomial 20x^2 + 160x + 800, when expressed in the form a(x+b)^2 + c,
    the sum a + b + c equals 504. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧
  (a + b + c = 504) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3750_375036


namespace NUMINAMATH_CALUDE_nancy_bills_denomination_l3750_375008

/-- Given Nancy has 9 bills of equal denomination and a total of 45 dollars, 
    the denomination of each bill is $5. -/
theorem nancy_bills_denomination (num_bills : ℕ) (total_amount : ℕ) (denomination : ℕ) :
  num_bills = 9 →
  total_amount = 45 →
  num_bills * denomination = total_amount →
  denomination = 5 := by
sorry

end NUMINAMATH_CALUDE_nancy_bills_denomination_l3750_375008


namespace NUMINAMATH_CALUDE_foldPointSetArea_l3750_375029

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of a right triangle ABC with AB = 45, AC = 90 -/
def rightTriangle : Triangle :=
  { A := { x := 0, y := 0 }
  , B := { x := 45, y := 0 }
  , C := { x := 0, y := 90 }
  }

/-- A point P is a fold point if creases formed when A, B, and C are folded onto P do not intersect inside the triangle -/
def isFoldPoint (P : Point) (T : Triangle) : Prop := sorry

/-- The set of all fold points for a given triangle -/
def foldPointSet (T : Triangle) : Set Point :=
  {P | isFoldPoint P T}

/-- The area of a set of points -/
def areaOfSet (S : Set Point) : ℝ := sorry

/-- Theorem: The area of the fold point set for the right triangle is 506.25π - 607.5√3 -/
theorem foldPointSetArea :
  areaOfSet (foldPointSet rightTriangle) = 506.25 * Real.pi - 607.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_foldPointSetArea_l3750_375029


namespace NUMINAMATH_CALUDE_order_of_even_increasing_function_l3750_375002

-- Define an even function f on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem order_of_even_increasing_function (f : ℝ → ℝ) 
  (h_even : even_function f) (h_incr : increasing_on_nonneg f) :
  f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry


end NUMINAMATH_CALUDE_order_of_even_increasing_function_l3750_375002


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l3750_375047

theorem cubic_polynomial_property (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) :
  x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l3750_375047


namespace NUMINAMATH_CALUDE_apple_pie_count_l3750_375038

/-- Represents the number of pies of each type --/
structure PieOrder where
  peach : ℕ
  apple : ℕ
  blueberry : ℕ

/-- Represents the cost of fruit per pound for each type of pie --/
structure FruitCosts where
  peach : ℚ
  apple : ℚ
  blueberry : ℚ

/-- Calculates the total cost of fruit for a given pie order --/
def totalCost (order : PieOrder) (costs : FruitCosts) (poundsPerPie : ℕ) : ℚ :=
  (order.peach * costs.peach + order.apple * costs.apple + order.blueberry * costs.blueberry) * poundsPerPie

theorem apple_pie_count (order : PieOrder) (costs : FruitCosts) (poundsPerPie totalSpent : ℕ) :
  order.peach = 5 →
  order.blueberry = 3 →
  poundsPerPie = 3 →
  costs.peach = 2 →
  costs.apple = 1 →
  costs.blueberry = 1 →
  totalCost order costs poundsPerPie = totalSpent →
  order.apple = 4 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l3750_375038


namespace NUMINAMATH_CALUDE_range_of_t_l3750_375000

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def cubic_for_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^3

theorem range_of_t (f : ℝ → ℝ) (t : ℝ) :
  is_even_function f →
  cubic_for_nonneg f →
  (∀ x ∈ Set.Icc (2*t - 1) (2*t + 3), f (3*x - t) ≥ 8 * f x) →
  t ∈ Set.Iic (-3) ∪ {0} ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_l3750_375000


namespace NUMINAMATH_CALUDE_mentorship_arrangements_count_l3750_375046

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of permutations of k items from n items --/
def permutations (n k : ℕ) : ℕ := sorry

/-- Calculates the number of mentorship arrangements for 5 students and 3 teachers --/
def mentorshipArrangements : ℕ :=
  let studentGroups := choose 5 2 * choose 3 2 * choose 1 1 / 2
  studentGroups * permutations 3 3

theorem mentorship_arrangements_count :
  mentorshipArrangements = 90 := by sorry

end NUMINAMATH_CALUDE_mentorship_arrangements_count_l3750_375046


namespace NUMINAMATH_CALUDE_boys_passed_exam_l3750_375014

theorem boys_passed_exam (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 36 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), passed = 105 ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_boys_passed_exam_l3750_375014


namespace NUMINAMATH_CALUDE_empty_board_prob_2013_l3750_375063

/-- Represents the state of the blackboard -/
inductive BoardState
| Empty : BoardState
| NonEmpty : Nat → BoardState

/-- The rules for updating the blackboard based on a coin flip -/
def updateBoard (state : BoardState) (n : Nat) (isHeads : Bool) : BoardState :=
  match state, isHeads with
  | BoardState.Empty, true => BoardState.NonEmpty n
  | BoardState.NonEmpty m, true => 
      if (m^2 + 2*n^2) % 3 = 0 then BoardState.Empty else BoardState.NonEmpty n
  | _, false => state

/-- The probability of an empty blackboard after n flips -/
def emptyBoardProb (n : Nat) : ℚ :=
  sorry  -- Definition omitted for brevity

theorem empty_board_prob_2013 :
  ∃ (u v : ℕ), emptyBoardProb 2013 = (2 * u + 1) / (2^1336 * (2 * v + 1)) :=
sorry

#check empty_board_prob_2013

end NUMINAMATH_CALUDE_empty_board_prob_2013_l3750_375063


namespace NUMINAMATH_CALUDE_optimal_candy_purchase_l3750_375009

/-- Represents the number of candies in a purchase strategy -/
structure CandyPurchase where
  singles : ℕ
  packs : ℕ
  bulks : ℕ

/-- Calculates the total cost of a purchase strategy -/
def totalCost (p : CandyPurchase) : ℕ :=
  p.singles + 3 * p.packs + 4 * p.bulks

/-- Calculates the total number of candies in a purchase strategy -/
def totalCandies (p : CandyPurchase) : ℕ :=
  p.singles + 4 * p.packs + 7 * p.bulks

/-- Represents a valid purchase strategy within the $10 budget -/
def ValidPurchase (p : CandyPurchase) : Prop :=
  totalCost p ≤ 10

/-- The maximum number of candies that can be purchased with $10 -/
def maxCandies : ℕ := 16

theorem optimal_candy_purchase :
  ∀ p : CandyPurchase, ValidPurchase p → totalCandies p ≤ maxCandies ∧
  ∃ q : CandyPurchase, ValidPurchase q ∧ totalCandies q = maxCandies :=
by sorry

end NUMINAMATH_CALUDE_optimal_candy_purchase_l3750_375009


namespace NUMINAMATH_CALUDE_parabola_segment_sum_l3750_375050

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12*y

-- Define the focus F (we don't know its exact coordinates, so we leave it abstract)
variable (F : ℝ × ℝ)

-- Define points A, B, and P
variable (A B : ℝ × ℝ)
def P : ℝ × ℝ := (2, 1)

-- State that A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- State that P is the midpoint of AB
axiom P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- State the theorem
theorem parabola_segment_sum : 
  ∀ (F A B : ℝ × ℝ), 
  parabola A.1 A.2 → 
  parabola B.1 B.2 → 
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  dist A F + dist B F = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_segment_sum_l3750_375050


namespace NUMINAMATH_CALUDE_doll_cost_is_15_l3750_375025

/-- Represents the cost of gifts for each sister -/
def gift_cost : ℕ := 60

/-- Represents the number of dolls bought for the younger sister -/
def num_dolls : ℕ := 4

/-- Represents the number of Lego sets bought for the older sister -/
def num_lego_sets : ℕ := 3

/-- Represents the cost of each Lego set -/
def lego_set_cost : ℕ := 20

/-- Theorem stating that the cost of each doll is $15 -/
theorem doll_cost_is_15 : 
  gift_cost = num_lego_sets * lego_set_cost ∧ 
  gift_cost = num_dolls * 15 := by
  sorry

end NUMINAMATH_CALUDE_doll_cost_is_15_l3750_375025


namespace NUMINAMATH_CALUDE_total_cats_sum_l3750_375019

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℝ := 15.5

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℝ := 11.6

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℝ := 24.2

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℝ := 18.3

/-- The total number of cats owned by all four people -/
def total_cats : ℝ := thompson_cats + sheridan_cats + garrett_cats + ravi_cats

theorem total_cats_sum :
  total_cats = 69.6 := by sorry

end NUMINAMATH_CALUDE_total_cats_sum_l3750_375019


namespace NUMINAMATH_CALUDE_inverse_sum_zero_l3750_375065

theorem inverse_sum_zero (a b : ℝ) (h : a * b = 1) :
  a^2015 * b^2016 + a^2016 * b^2017 + a^2017 * b^2016 + a^2016 * b^2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_zero_l3750_375065


namespace NUMINAMATH_CALUDE_no_consecutive_product_l3750_375069

theorem no_consecutive_product (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 7*n + 8 = k * (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_product_l3750_375069


namespace NUMINAMATH_CALUDE_lotto_ticket_cost_l3750_375088

/-- Proves that the cost per ticket is $2 given the lottery conditions --/
theorem lotto_ticket_cost (total_tickets : ℕ) (winning_percentage : ℚ)
  (five_dollar_winners_percentage : ℚ) (grand_prize_tickets : ℕ)
  (grand_prize_amount : ℕ) (other_winners_average : ℕ) (total_profit : ℕ) :
  total_tickets = 200 →
  winning_percentage = 1/5 →
  five_dollar_winners_percentage = 4/5 →
  grand_prize_tickets = 1 →
  grand_prize_amount = 5000 →
  other_winners_average = 10 →
  total_profit = 4830 →
  ∃ (cost_per_ticket : ℚ), cost_per_ticket = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_lotto_ticket_cost_l3750_375088


namespace NUMINAMATH_CALUDE_nine_people_four_consecutive_l3750_375096

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def consecutive_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

def valid_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  total_arrangements n - consecutive_arrangements n k

theorem nine_people_four_consecutive (n : ℕ) (k : ℕ) :
  n = 9 ∧ k = 4 → valid_arrangements n k = 345600 := by
  sorry

end NUMINAMATH_CALUDE_nine_people_four_consecutive_l3750_375096


namespace NUMINAMATH_CALUDE_sqrt_a_squared_plus_a_equals_two_thirds_l3750_375068

theorem sqrt_a_squared_plus_a_equals_two_thirds (a : ℝ) :
  a > 0 ∧ Real.sqrt (a^2 + a) = 2/3 ↔ a = 1/3 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_squared_plus_a_equals_two_thirds_l3750_375068


namespace NUMINAMATH_CALUDE_oil_price_reduction_oil_price_reduction_result_l3750_375067

/-- Calculates the percentage reduction in oil price given the conditions -/
theorem oil_price_reduction (additional_oil : ℝ) (total_cost : ℝ) (reduced_price : ℝ) : ℝ :=
  let original_amount := (total_cost / reduced_price) - additional_oil
  let original_price := total_cost / original_amount
  let price_difference := original_price - reduced_price
  (price_difference / original_price) * 100

/-- The percentage reduction in oil price is approximately 24.99% -/
theorem oil_price_reduction_result : 
  ∃ ε > 0, |oil_price_reduction 5 500 25 - 24.99| < ε :=
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_oil_price_reduction_result_l3750_375067


namespace NUMINAMATH_CALUDE_part_one_part_two_l3750_375052

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  {x : ℝ | f 1 x ≥ 4 - |x - 1|} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  ({x : ℝ | f ((1/m) + 1/(2*n)) x ≤ 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) →
  (∀ k l : ℝ, k > 0 → l > 0 → k * l ≥ m * n) →
  m * n = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3750_375052


namespace NUMINAMATH_CALUDE_second_day_visitors_count_l3750_375083

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  first_day_visitors : ℕ
  first_day_cans_per_person : ℕ
  first_restock : ℕ
  second_day_cans_per_person : ℕ
  second_restock : ℕ
  second_day_cans_given : ℕ

/-- Calculates the number of people who showed up on the second day --/
def second_day_visitors (fb : FoodBank) : ℕ :=
  fb.second_day_cans_given / fb.second_day_cans_per_person

/-- Theorem stating that given the conditions, 1250 people showed up on the second day --/
theorem second_day_visitors_count (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.first_day_visitors = 500)
  (h3 : fb.first_day_cans_per_person = 1)
  (h4 : fb.first_restock = 1500)
  (h5 : fb.second_day_cans_per_person = 2)
  (h6 : fb.second_restock = 3000)
  (h7 : fb.second_day_cans_given = 2500) :
  second_day_visitors fb = 1250 := by
  sorry

#eval second_day_visitors {
  initial_stock := 2000,
  first_day_visitors := 500,
  first_day_cans_per_person := 1,
  first_restock := 1500,
  second_day_cans_per_person := 2,
  second_restock := 3000,
  second_day_cans_given := 2500
}

end NUMINAMATH_CALUDE_second_day_visitors_count_l3750_375083


namespace NUMINAMATH_CALUDE_total_matches_l3750_375007

def dozen : ℕ := 12

def boxes : ℕ := 5 * dozen

def matches_per_box : ℕ := 20

theorem total_matches : boxes * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_l3750_375007


namespace NUMINAMATH_CALUDE_function_with_two_zeros_l3750_375015

theorem function_with_two_zeros 
  (f : ℝ → ℝ) 
  (hcont : ContinuousOn f (Set.Icc 1 3))
  (h1 : f 1 * f 2 < 0)
  (h2 : f 2 * f 3 < 0) :
  ∃ (x y : ℝ), x ∈ Set.Ioo 1 3 ∧ y ∈ Set.Ioo 1 3 ∧ x ≠ y ∧ f x = 0 ∧ f y = 0 :=
sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_l3750_375015


namespace NUMINAMATH_CALUDE_survey_result_l3750_375077

theorem survey_result (U : Finset Int) (A B : Finset Int) 
  (h1 : Finset.card U = 70)
  (h2 : Finset.card A = 37)
  (h3 : Finset.card B = 49)
  (h4 : Finset.card (A ∩ B) = 20) :
  Finset.card (U \ (A ∪ B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l3750_375077


namespace NUMINAMATH_CALUDE_prob_at_least_two_females_l3750_375016

/-- The probability of selecting at least two females when choosing three finalists
    from a group of eight contestants consisting of five females and three males. -/
theorem prob_at_least_two_females (total : ℕ) (females : ℕ) (males : ℕ) (finalists : ℕ) :
  total = 8 →
  females = 5 →
  males = 3 →
  finalists = 3 →
  (Nat.choose females 2 * Nat.choose males 1 + Nat.choose females 3) / Nat.choose total finalists = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_females_l3750_375016


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l3750_375006

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x_0,
    prove that x_0 = -1 / ∛6 -/
theorem perpendicular_tangents_intersection (x_0 : ℝ) :
  (2 * x_0) * (3 * x_0^2) = -1 →
  x_0 = -1 / Real.rpow 6 (1/3) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l3750_375006


namespace NUMINAMATH_CALUDE_proposition_relationship_l3750_375064

theorem proposition_relationship (x y : ℤ) :
  (∀ x y, x + y ≠ 2010 → (x ≠ 1010 ∨ y ≠ 1000)) ∧
  (∃ x y, (x ≠ 1010 ∨ y ≠ 1000) ∧ x + y = 2010) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l3750_375064


namespace NUMINAMATH_CALUDE_salary_problem_l3750_375092

/-- Proves that given the conditions of the problem, A's salary is Rs. 3000 --/
theorem salary_problem (total : ℝ) (a_salary : ℝ) (b_salary : ℝ) 
  (h1 : total = 4000)
  (h2 : a_salary + b_salary = total)
  (h3 : 0.05 * a_salary = 0.15 * b_salary) :
  a_salary = 3000 := by
  sorry

#check salary_problem

end NUMINAMATH_CALUDE_salary_problem_l3750_375092


namespace NUMINAMATH_CALUDE_lines_equivalence_l3750_375090

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A cylinder in 3D space -/
structure Cylinder3D where
  axis : Line3D
  radius : ℝ

/-- The set of lines passing through a point and at a given distance from another line -/
def linesAtDistanceFromLine (M : Point3D) (d : ℝ) (AB : Line3D) : Set Line3D :=
  sorry

/-- The set of lines lying in two planes tangent to a cylinder passing through a point -/
def linesInTangentPlanes (M : Point3D) (cylinder : Cylinder3D) : Set Line3D :=
  sorry

/-- Theorem stating the equivalence of the two sets of lines -/
theorem lines_equivalence (M : Point3D) (d : ℝ) (AB : Line3D) :
  let cylinder := Cylinder3D.mk AB d
  linesAtDistanceFromLine M d AB = linesInTangentPlanes M cylinder :=
sorry

end NUMINAMATH_CALUDE_lines_equivalence_l3750_375090


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l3750_375055

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- 
Given a man's upstream speed and still water speed, 
calculates and proves his downstream speed
-/
theorem downstream_speed_calculation (speed : RowingSpeed) 
  (h1 : speed.upstream = 30)
  (h2 : speed.stillWater = 45) :
  speed.downstream = 60 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l3750_375055


namespace NUMINAMATH_CALUDE_present_age_of_A_l3750_375056

/-- Given two people A and B, their ages, and future age ratios, 
    prove that A's present age is 15 years. -/
theorem present_age_of_A (a b : ℕ) : 
  a * 3 = b * 5 →  -- Present age ratio
  (a + 6) * 5 = (b + 6) * 7 →  -- Future age ratio
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_present_age_of_A_l3750_375056


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3750_375035

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((246835 + y) % 169 = 0 ∧ (246835 + y) % 289 = 0)) ∧ 
  ((246835 + x) % 169 = 0 ∧ (246835 + x) % 289 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3750_375035


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l3750_375086

theorem max_leftover_grapes (n : ℕ) : ∃ k : ℕ, n = 7 * k + (n % 7) ∧ n % 7 ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l3750_375086


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3750_375027

theorem quadratic_factorization (b c d e f : ℤ) : 
  (∀ x : ℚ, 24 * x^2 + b * x + 24 = (c * x + d) * (e * x + f)) →
  c + d = 10 →
  c * e = 24 →
  d * f = 24 →
  b = 52 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3750_375027


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l3750_375053

/-- The quadratic function g(x) = x^2 + 3x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- 3 is not in the range of g(x) if and only if c > 21/4 -/
theorem three_not_in_range_of_g (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c > 21/4 := by sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l3750_375053


namespace NUMINAMATH_CALUDE_binomial_10_0_l3750_375024

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_0_l3750_375024


namespace NUMINAMATH_CALUDE_wario_field_goals_l3750_375031

/-- Given the conditions of Wario's field goal attempts, prove the number of wide right misses. -/
theorem wario_field_goals (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right_ratio : ℚ) 
  (h1 : total_attempts = 60)
  (h2 : miss_ratio = 1 / 4)
  (h3 : wide_right_ratio = 1 / 5) : 
  ⌊(total_attempts : ℚ) * miss_ratio * wide_right_ratio⌋ = 3 := by
  sorry

#check wario_field_goals

end NUMINAMATH_CALUDE_wario_field_goals_l3750_375031


namespace NUMINAMATH_CALUDE_remainder_thirteen_plus_x_l3750_375097

theorem remainder_thirteen_plus_x (x : ℕ+) (h : 8 * x.val ≡ 1 [MOD 29]) :
  (13 + x.val) % 29 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_plus_x_l3750_375097


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3750_375028

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  3 -- Each dimension contributes 2 pairs, so 2 + 2 + 2 = 6

/-- Theorem stating that a rectangular prism with dimensions 8, 4, and 2 has 6 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges :
  let prism : RectangularPrism := { length := 8, width := 4, height := 2 }
  parallel_edge_pairs prism = 6 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3750_375028


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3750_375003

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 10
  h3 : Even side3
  h4 : side1 + side2 > side3
  h5 : side1 + side3 > side2
  h6 : side2 + side3 > side1

/-- The perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of the given triangle is 21 --/
theorem smallest_perimeter :
  ∃ (t : Triangle), ∀ (t' : Triangle), perimeter t ≤ perimeter t' ∧ perimeter t = 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3750_375003


namespace NUMINAMATH_CALUDE_g_max_value_f_upper_bound_l3750_375010

noncomputable def f (x : ℝ) := Real.log (x + 1)

noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem g_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 2 * Real.log 2 - 7 / 4 := by sorry

theorem f_upper_bound (x : ℝ) (hx : x > 0) :
  f x < (Real.exp x - 1) / x^2 := by sorry

end NUMINAMATH_CALUDE_g_max_value_f_upper_bound_l3750_375010


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3750_375091

/-- Calculates the total water used by Ginger for drinking and watering plants -/
def total_water_used (work_hours : ℕ) (bottle_capacity : ℚ) 
  (first_hour_drink : ℚ) (second_hour_drink : ℚ) (third_hour_drink : ℚ) 
  (hourly_increase : ℚ) (plant_type1_water : ℚ) (plant_type2_water : ℚ) 
  (plant_type3_water : ℚ) (plant_type1_count : ℕ) (plant_type2_count : ℕ) 
  (plant_type3_count : ℕ) : ℚ :=
  sorry

theorem ginger_water_usage :
  total_water_used 8 2 1 (3/2) 2 (1/2) 3 4 5 2 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l3750_375091
