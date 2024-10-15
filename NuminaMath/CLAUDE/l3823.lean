import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_statement_C_l3823_382338

theorem incorrect_statement_C : 
  (∀ x : ℝ, x > 0 → (∃ y : ℝ, y > 0 ∧ y^2 = x) ∧ (∃! y : ℝ, y > 0 ∧ y^2 = x)) ∧ 
  (∀ x : ℝ, x^3 < 0 → ∃ y : ℝ, y < 0 ∧ y^3 = x) ∧
  (∃ x : ℝ, x^(1/3) = x^(1/2)) ∧
  ¬(∃ y : ℝ, y ≠ 0 ∧ (y^2 = 81 → (y = 9 ∨ y = -9))) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_C_l3823_382338


namespace NUMINAMATH_CALUDE_largest_prime_divisor_xyxyxy_l3823_382306

/-- The largest prime divisor of a number in the form xyxyxy -/
theorem largest_prime_divisor_xyxyxy (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) → q ≤ 97 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_xyxyxy_l3823_382306


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3823_382311

theorem quadratic_equation_problem (a : ℝ) : 2 * (5 - a) * (6 + a) = 100 → a^2 + a + 1 = -19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3823_382311


namespace NUMINAMATH_CALUDE_town_population_l3823_382301

theorem town_population (growth_rate : ℝ) (future_population : ℕ) (present_population : ℕ) :
  growth_rate = 0.1 →
  future_population = 264 →
  present_population * (1 + growth_rate) = future_population →
  present_population = 240 := by
sorry

end NUMINAMATH_CALUDE_town_population_l3823_382301


namespace NUMINAMATH_CALUDE_no_solution_for_a_l3823_382335

theorem no_solution_for_a (x a : ℝ) : x = 4 → 1 / (x + a) + 1 / (x - a) ≠ 1 / (x - a) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_a_l3823_382335


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3823_382322

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {a_n} where a_1 + a_3 = 1 and a_2 + a_4 = 2, 
    the sum of the 5th, 6th, 7th, and 8th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : is_geometric_sequence a)
    (h_sum1 : a 1 + a 3 = 1)
    (h_sum2 : a 2 + a 4 = 2) :
    a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3823_382322


namespace NUMINAMATH_CALUDE_divisibility_by_three_l3823_382303

theorem divisibility_by_three (B : Nat) : 
  B < 10 → (514 * 10 + B) % 3 = 0 ↔ B = 2 ∨ B = 5 ∨ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l3823_382303


namespace NUMINAMATH_CALUDE_janes_age_problem_l3823_382327

theorem janes_age_problem :
  ∃ n : ℕ+, 
    (∃ x : ℕ+, n - 1 = x^3) ∧ 
    (∃ y : ℕ+, n + 4 = y^2) ∧ 
    n = 1332 := by
  sorry

end NUMINAMATH_CALUDE_janes_age_problem_l3823_382327


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3823_382398

def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem complement_A_intersect_B : (U \ A) ∩ B = {6} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3823_382398


namespace NUMINAMATH_CALUDE_white_white_overlapping_pairs_l3823_382354

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of overlapping pairs of each type when the figure is folded -/
structure OverlappingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ

/-- The main theorem stating the number of white-white overlapping pairs -/
theorem white_white_overlapping_pairs
  (counts : TriangleCounts)
  (overlaps : OverlappingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : overlaps.redRed = 3)
  (h5 : overlaps.blueBlue = 4)
  (h6 : overlaps.redWhite = 3) :
  counts.white - overlaps.redWhite = 6 :=
sorry

end NUMINAMATH_CALUDE_white_white_overlapping_pairs_l3823_382354


namespace NUMINAMATH_CALUDE_a_6_equals_12_l3823_382319

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_6_equals_12 
  (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 9 = (1/2) * a 12 + 6) : 
  a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_a_6_equals_12_l3823_382319


namespace NUMINAMATH_CALUDE_union_complement_problem_l3823_382309

open Set

theorem union_complement_problem (A B : Set ℝ) 
  (hA : A = {x : ℝ | -2 ≤ x ∧ x ≤ 3})
  (hB : B = {x : ℝ | x < -1 ∨ 4 < x}) :
  A ∪ (univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l3823_382309


namespace NUMINAMATH_CALUDE_peach_apple_pear_pricing_l3823_382359

theorem peach_apple_pear_pricing (x y z : ℝ) 
  (h1 : 7 * x = y + 2 * z)
  (h2 : 7 * y = 10 * z + x) :
  12 * y = 18 * z := by sorry

end NUMINAMATH_CALUDE_peach_apple_pear_pricing_l3823_382359


namespace NUMINAMATH_CALUDE_johns_hourly_rate_is_10_l3823_382329

/-- Calculates John's hourly rate when earning the performance bonus -/
def johnsHourlyRateWithBonus (basePay dayHours bonusPay bonusHours : ℚ) : ℚ :=
  (basePay + bonusPay) / (dayHours + bonusHours)

/-- Theorem: John's hourly rate with bonus is $10 per hour -/
theorem johns_hourly_rate_is_10 :
  johnsHourlyRateWithBonus 80 8 20 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_johns_hourly_rate_is_10_l3823_382329


namespace NUMINAMATH_CALUDE_total_practice_hours_l3823_382341

def monday_hours : ℕ := 6
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := 5
def thursday_hours : ℕ := 7
def friday_hours : ℕ := 3

def total_scheduled_hours : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def practice_days : ℕ := 5

def player_a_missed_hours : ℕ := 2
def player_b_missed_hours : ℕ := 3

def rainy_day_hours : ℕ := total_scheduled_hours / practice_days

theorem total_practice_hours :
  total_scheduled_hours - (rainy_day_hours + player_a_missed_hours + player_b_missed_hours) = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_hours_l3823_382341


namespace NUMINAMATH_CALUDE_vertex_is_correct_l3823_382378

/-- The quadratic function f(x) = 3(x+5)^2 - 2 -/
def f (x : ℝ) : ℝ := 3 * (x + 5)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-5, -2)

theorem vertex_is_correct : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_is_correct_l3823_382378


namespace NUMINAMATH_CALUDE_ann_oatmeal_raisin_cookies_l3823_382387

/-- The number of dozens of oatmeal raisin cookies Ann baked -/
def oatmeal_raisin_dozens : ℝ := sorry

/-- The number of dozens of sugar cookies Ann baked -/
def sugar_dozens : ℝ := 2

/-- The number of dozens of chocolate chip cookies Ann baked -/
def chocolate_chip_dozens : ℝ := 4

/-- The number of dozens of oatmeal raisin cookies Ann gave away -/
def oatmeal_raisin_given : ℝ := 2

/-- The number of dozens of sugar cookies Ann gave away -/
def sugar_given : ℝ := 1.5

/-- The number of dozens of chocolate chip cookies Ann gave away -/
def chocolate_chip_given : ℝ := 2.5

/-- The number of dozens of cookies Ann kept -/
def kept_dozens : ℝ := 3

theorem ann_oatmeal_raisin_cookies :
  oatmeal_raisin_dozens = 3 :=
by sorry

end NUMINAMATH_CALUDE_ann_oatmeal_raisin_cookies_l3823_382387


namespace NUMINAMATH_CALUDE_percentage_sold_first_day_l3823_382375

-- Define the initial number of watermelons
def initial_watermelons : ℕ := 10 * 12

-- Define the number of watermelons left after two days of selling
def remaining_watermelons : ℕ := 54

-- Define the percentage sold on the second day
def second_day_percentage : ℚ := 1 / 4

-- Theorem to prove
theorem percentage_sold_first_day :
  ∃ (p : ℚ), 0 ≤ p ∧ p ≤ 1 ∧
  (1 - second_day_percentage) * ((1 - p) * initial_watermelons) = remaining_watermelons ∧
  p = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sold_first_day_l3823_382375


namespace NUMINAMATH_CALUDE_expression_value_l3823_382372

theorem expression_value : (5^2 - 5 - 12) / (5 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3823_382372


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3823_382355

theorem gain_percent_calculation (cost_price selling_price : ℝ) :
  50 * cost_price = 45 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 11.11 :=
by sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3823_382355


namespace NUMINAMATH_CALUDE_binomial_7_2_l3823_382362

theorem binomial_7_2 : (7 : ℕ).choose 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l3823_382362


namespace NUMINAMATH_CALUDE_problem_solution_l3823_382334

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (Set.Iio 1 = {a : ℝ | ∀ x ≥ 0, f a x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3823_382334


namespace NUMINAMATH_CALUDE_probability_at_least_six_sevens_l3823_382370

-- Define the number of sides on the die
def die_sides : ℕ := 8

-- Define the number of rolls
def num_rolls : ℕ := 7

-- Define the minimum number of successful rolls required
def min_successes : ℕ := 6

-- Define the minimum value considered a success
def success_value : ℕ := 7

-- Function to calculate the probability of a single success
def single_success_prob : ℚ := (die_sides - success_value + 1) / die_sides

-- Function to calculate the probability of exactly k successes in n rolls
def exact_success_prob (n k : ℕ) : ℚ :=
  Nat.choose n k * single_success_prob^k * (1 - single_success_prob)^(n - k)

-- Theorem statement
theorem probability_at_least_six_sevens :
  (exact_success_prob num_rolls min_successes + exact_success_prob num_rolls (num_rolls)) = 11 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_six_sevens_l3823_382370


namespace NUMINAMATH_CALUDE_chef_fries_problem_l3823_382333

/-- Given a chef making fries, prove the number of fries needed. -/
theorem chef_fries_problem (fries_per_potato : ℕ) (total_potatoes : ℕ) (leftover_potatoes : ℕ) :
  fries_per_potato = 25 →
  total_potatoes = 15 →
  leftover_potatoes = 7 →
  fries_per_potato * (total_potatoes - leftover_potatoes) = 200 := by
  sorry

end NUMINAMATH_CALUDE_chef_fries_problem_l3823_382333


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3823_382313

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / (x - 5) = 1) ∧ (x = 6) := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3823_382313


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3823_382337

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 5 - a 1) * (a 17 - a 5) = (a 5 - a 1)^2) : 
  a 5 / a 1 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3823_382337


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l3823_382318

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 5 20 = 97 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l3823_382318


namespace NUMINAMATH_CALUDE_rakesh_salary_rakesh_salary_proof_l3823_382388

theorem rakesh_salary : ℝ → Prop :=
  fun salary =>
    let fixed_deposit := 0.15 * salary
    let remaining_after_deposit := salary - fixed_deposit
    let groceries := 0.30 * remaining_after_deposit
    let cash_in_hand := remaining_after_deposit - groceries
    cash_in_hand = 2380 → salary = 4000

-- Proof
theorem rakesh_salary_proof : rakesh_salary 4000 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_salary_rakesh_salary_proof_l3823_382388


namespace NUMINAMATH_CALUDE_sin_squared_sum_l3823_382377

theorem sin_squared_sum (α β : ℝ) 
  (h : Real.arcsin (Real.sin α + Real.sin β) + Real.arcsin (Real.sin α - Real.sin β) = π / 2) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l3823_382377


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l3823_382357

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ l > 0) 
  (h_diagonal : a^2 + b^2 + c^2 = l^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l3823_382357


namespace NUMINAMATH_CALUDE_class_size_l3823_382366

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyGerman : ℕ
  germanTotal : ℕ

/-- Theorem stating the total number of students in the class -/
theorem class_size (c : ClassEnrollment) 
  (h1 : c.bothSubjects = 12)
  (h2 : c.germanTotal = 22)
  (h3 : c.onlyEnglish = 18)
  (h4 : c.germanTotal = c.bothSubjects + c.onlyGerman)
  (h5 : c.total = c.onlyEnglish + c.onlyGerman + c.bothSubjects) :
  c.total = 40 := by
  sorry


end NUMINAMATH_CALUDE_class_size_l3823_382366


namespace NUMINAMATH_CALUDE_remainder_difference_l3823_382340

theorem remainder_difference (d : ℕ) (r : ℕ) (h1 : d > 1) : 
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) → d - r = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_difference_l3823_382340


namespace NUMINAMATH_CALUDE_cos_two_pi_third_plus_two_alpha_l3823_382320

theorem cos_two_pi_third_plus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_third_plus_two_alpha_l3823_382320


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3823_382312

def purchase_price : ℕ := 12000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 27000
def profit_percentage : ℚ := 1/2

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l3823_382312


namespace NUMINAMATH_CALUDE_triangle_similarity_l3823_382360

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the perpendicular foot point
def perp_foot (P : Point) (line : Point × Point) : Point := sorry

-- Define the similarity relation between triangles
def similar (T1 T2 : Triangle) : Prop := sorry

-- Define the construction process
def construct_next_triangle (T : Triangle) (P : Point) : Triangle :=
  let B1 := perp_foot P (T.B, T.C)
  let B2 := perp_foot P (T.C, T.A)
  let B3 := perp_foot P (T.A, T.B)
  Triangle.mk B1 B2 B3

-- Theorem statement
theorem triangle_similarity 
  (A : Triangle) 
  (P : Point) 
  (h_interior : sorry) -- Assumption that P is interior to A
  : 
  let B := construct_next_triangle A P
  let C := construct_next_triangle B P
  let D := construct_next_triangle C P
  similar A D := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l3823_382360


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3823_382381

theorem sqrt_inequality : Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3823_382381


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3823_382346

theorem sqrt_50_between_consecutive_integers_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (b : ℝ) → a * b = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3823_382346


namespace NUMINAMATH_CALUDE_white_balls_count_l3823_382323

def total_balls : ℕ := 40
def red_frequency : ℚ := 15 / 100
def black_frequency : ℚ := 45 / 100

theorem white_balls_count :
  ∃ (white_balls : ℕ),
    white_balls = total_balls * (1 - red_frequency - black_frequency) := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3823_382323


namespace NUMINAMATH_CALUDE_problem_proof_l3823_382392

theorem problem_proof : (-1)^2023 - (-1/4)^0 + 2 * Real.cos (π/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3823_382392


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_max_min_values_even_function_l3823_382389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 3

-- Part 1
theorem monotonic_increasing_interval (a : ℝ) (h : a = 2) :
  ∀ x y, x ≤ y ∧ y ≤ 1 → f a x ≤ f a y :=
sorry

-- Part 2
theorem max_min_values_even_function (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f a x₀ = 3 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧
  (∃ x₁ ∈ Set.Icc (-1) 3, f a x₁ = -6 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≥ -6) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_max_min_values_even_function_l3823_382389


namespace NUMINAMATH_CALUDE_gasohol_calculation_l3823_382379

/-- The amount of gasohol initially in the tank (in liters) -/
def initial_gasohol : ℝ := 27

/-- The fraction of ethanol in the initial mixture -/
def initial_ethanol_fraction : ℝ := 0.05

/-- The fraction of ethanol in the desired mixture -/
def desired_ethanol_fraction : ℝ := 0.10

/-- The amount of pure ethanol added (in liters) -/
def added_ethanol : ℝ := 1.5

theorem gasohol_calculation :
  initial_gasohol * initial_ethanol_fraction + added_ethanol =
  desired_ethanol_fraction * (initial_gasohol + added_ethanol) := by
  sorry

end NUMINAMATH_CALUDE_gasohol_calculation_l3823_382379


namespace NUMINAMATH_CALUDE_paper_length_wrapped_around_cylinder_l3823_382384

/-- Calculates the length of paper wrapped around a cylindrical tube. -/
theorem paper_length_wrapped_around_cylinder
  (initial_diameter : ℝ)
  (paper_width : ℝ)
  (num_wraps : ℕ)
  (final_diameter : ℝ)
  (h1 : initial_diameter = 3)
  (h2 : paper_width = 4)
  (h3 : num_wraps = 400)
  (h4 : final_diameter = 11) :
  ∃ (paper_length : ℝ), paper_length = 28 * π ∧ paper_length * 100 = π * (num_wraps * (initial_diameter + final_diameter)) :=
by sorry

end NUMINAMATH_CALUDE_paper_length_wrapped_around_cylinder_l3823_382384


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3823_382386

/-- Given a sequence {a_n} defined by a_n = 2n + 5, prove it is an arithmetic sequence with common difference 2 -/
theorem arithmetic_sequence_proof (n : ℕ) : ∃ (d : ℝ), d = 2 ∧ ∀ k, (2 * (k + 1) + 5) - (2 * k + 5) = d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3823_382386


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3823_382336

theorem arithmetic_mean_problem (a b c d : ℝ) 
  (h1 : (a + d) / 2 = 40)
  (h2 : (b + d) / 2 = 60)
  (h3 : (a + b) / 2 = 50)
  (h4 : (b + c) / 2 = 70) :
  c - a = 40 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3823_382336


namespace NUMINAMATH_CALUDE_symmetric_lines_l3823_382302

/-- Given two lines symmetric about y = x, if one line is y = 2x - 3, 
    then the other line is y = (1/2)x + (3/2) -/
theorem symmetric_lines (x y : ℝ) : 
  (y = 2 * x - 3) ↔ 
  (∃ (x' y' : ℝ), y' = (1/2) * x' + (3/2) ∧ 
    (x + x') / 2 = (y + y') / 2 ∧
    y = x) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_lines_l3823_382302


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_dimensions_l3823_382324

def is_valid_dimension (x y z : ℕ) : Prop :=
  2 * (x * y + y * z + z * x) = x * y * z

def valid_dimensions : List (ℕ × ℕ × ℕ) :=
  [(6,6,6), (5,5,10), (4,8,8), (3,12,12), (3,7,42), (3,8,24), (3,9,18), (3,10,15), (4,5,20), (4,6,12)]

theorem rectangular_parallelepiped_dimensions (x y z : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 → is_valid_dimension x y z → (x, y, z) ∈ valid_dimensions := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_dimensions_l3823_382324


namespace NUMINAMATH_CALUDE_complex_power_simplification_l3823_382315

theorem complex_power_simplification :
  (Complex.exp (Complex.I * (123 * π / 180)))^25 = 
  -Complex.cos (15 * π / 180) - Complex.I * Complex.sin (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l3823_382315


namespace NUMINAMATH_CALUDE_manganese_percentage_after_iron_addition_l3823_382310

theorem manganese_percentage_after_iron_addition 
  (initial_mixture_mass : ℝ)
  (initial_manganese_percentage : ℝ)
  (added_iron_mass : ℝ)
  (h1 : initial_mixture_mass = 1)
  (h2 : initial_manganese_percentage = 20)
  (h3 : added_iron_mass = 1)
  : (initial_manganese_percentage / 100 * initial_mixture_mass) / 
    (initial_mixture_mass + added_iron_mass) * 100 = 10 := by
  sorry

#check manganese_percentage_after_iron_addition

end NUMINAMATH_CALUDE_manganese_percentage_after_iron_addition_l3823_382310


namespace NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l3823_382347

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l3823_382347


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3823_382345

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3823_382345


namespace NUMINAMATH_CALUDE_equation_solution_l3823_382363

theorem equation_solution : ∃ x : ℝ, 
  x = 625 ∧ 
  Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3823_382363


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_area_l3823_382391

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- Assumption that the center of the circle lies on the longer base -/
  centerOnBase : Bool

/-- 
Given a trapezoid ABCD with an inscribed circle of radius 6,
where the center of the circle lies on the base AD,
and BC = 4, prove that the area of the trapezoid is 24√2.
-/
theorem inscribed_trapezoid_area 
  (t : InscribedTrapezoid) 
  (h1 : t.radius = 6) 
  (h2 : t.shorterBase = 4) 
  (h3 : t.centerOnBase = true) : 
  ∃ (area : ℝ), area = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_area_l3823_382391


namespace NUMINAMATH_CALUDE_jills_work_days_month3_l3823_382371

def daily_rate_month1 : ℕ := 10
def daily_rate_month2 : ℕ := 2 * daily_rate_month1
def daily_rate_month3 : ℕ := daily_rate_month2
def days_per_month : ℕ := 30
def total_earnings : ℕ := 1200

def earnings_month1 : ℕ := daily_rate_month1 * days_per_month
def earnings_month2 : ℕ := daily_rate_month2 * days_per_month

theorem jills_work_days_month3 :
  (total_earnings - earnings_month1 - earnings_month2) / daily_rate_month3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_jills_work_days_month3_l3823_382371


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l3823_382300

/-- The minimum number of additional coins needed for distribution --/
def min_additional_coins (n : ℕ) (initial_coins : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution --/
theorem alex_coin_distribution :
  min_additional_coins 20 192 = 18 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l3823_382300


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3823_382365

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h1 : a = 24) 
  (h2 : c = 25) 
  (h3 : a^2 + b^2 = c^2) 
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h5 : a ≤ b) 
  (h_hypotenuse : ℝ) 
  (h6 : h_hypotenuse = 100) :
  ∃ (x : ℝ), x = 28 ∧ x = (a * h_hypotenuse) / c :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3823_382365


namespace NUMINAMATH_CALUDE_system_solutions_system_no_solutions_l3823_382330

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  3 * x - 4 * y = 9 ∧ 6 * x - 8 * y = k

-- Theorem statement
theorem system_solutions (k : ℝ) :
  (∃ x y, system x y k) ↔ k = 18 :=
by sorry

-- Corollary for no solutions
theorem system_no_solutions (k : ℝ) :
  (¬ ∃ x y, system x y k) ↔ k ≠ 18 :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_system_no_solutions_l3823_382330


namespace NUMINAMATH_CALUDE_C₁_intersects_C₂_max_value_on_C₂_l3823_382399

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y - 1 = 0
def C₂ (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y + Real.sqrt 2 / 2)^2 = 1

-- Define a point M on C₂
structure PointOnC₂ where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y

-- Theorem 1: C₁ intersects C₂
theorem C₁_intersects_C₂ : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

-- Theorem 2: Maximum value of 2x + y for points on C₂
theorem max_value_on_C₂ :
  ∃ (max : ℝ), max = Real.sqrt 2 / 2 + Real.sqrt 5 ∧
  ∀ (M : PointOnC₂), 2 * M.x + M.y ≤ max :=
sorry

end NUMINAMATH_CALUDE_C₁_intersects_C₂_max_value_on_C₂_l3823_382399


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3823_382314

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 360 ∧ 
  train_speed_kmh = 90 ∧ 
  time_to_pass = 20 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3823_382314


namespace NUMINAMATH_CALUDE_coefficient_is_40_l3823_382331

/-- The coefficient of x^3y^2 in the expansion of (x-2y)^5 -/
def coefficient : ℤ := 
  (Nat.choose 5 2) * (-2)^2

/-- Theorem stating that the coefficient of x^3y^2 in (x-2y)^5 is 40 -/
theorem coefficient_is_40 : coefficient = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_is_40_l3823_382331


namespace NUMINAMATH_CALUDE_group_size_l3823_382390

theorem group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 2.5 ∧
  old_weight = 40 ∧
  new_weight = 60 →
  (new_weight - old_weight) / average_increase = 8 := by
sorry

end NUMINAMATH_CALUDE_group_size_l3823_382390


namespace NUMINAMATH_CALUDE_object_speed_approximation_l3823_382369

/-- Given an object traveling 80 feet in 4 seconds, prove that its speed is approximately 13.64 miles per hour, given that 1 mile equals 5280 feet. -/
theorem object_speed_approximation : 
  let distance_feet : ℝ := 80
  let time_seconds : ℝ := 4
  let feet_per_mile : ℝ := 5280
  let seconds_per_hour : ℝ := 3600
  let speed_mph := (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour)
  ∃ ε > 0, |speed_mph - 13.64| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_object_speed_approximation_l3823_382369


namespace NUMINAMATH_CALUDE_point_N_coordinates_l3823_382361

def M : ℝ × ℝ := (0, -1)

def N : ℝ × ℝ → Prop := fun p => p.1 - p.2 + 1 = 0

def perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (s.1 - r.1) + (q.2 - p.2) * (s.2 - r.2) = 0

def line_x_plus_2y_minus_3 (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 - 3 = 0

theorem point_N_coordinates :
  ∃ n : ℝ × ℝ, N n ∧ perpendicular M n (0, 0) (1, -2) ∧ n = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l3823_382361


namespace NUMINAMATH_CALUDE_range_of_linear_function_l3823_382367

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c < 0) :
  ∀ y ∈ Set.range (g c d),
    ∃ x ∈ Set.Icc (-1 : ℝ) 1,
      y = g c d x ∧ c + d ≤ y ∧ y ≤ -c + d :=
by sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l3823_382367


namespace NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l3823_382326

/-- The number of bacteria on a pair of unwashed hands -/
def bacteria_count : ℕ := 750000

/-- Scientific notation representation of the bacteria count -/
def scientific_notation : ℝ := 7.5 * (10 ^ 5)

theorem bacteria_count_scientific_notation : 
  (bacteria_count : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_scientific_notation_l3823_382326


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3823_382376

theorem arithmetic_equality : 3 * (7 - 5) - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3823_382376


namespace NUMINAMATH_CALUDE_vacuum_savings_theorem_l3823_382373

/-- Calculate the number of weeks needed to save for a vacuum cleaner. -/
def weeks_to_save (initial_amount : ℕ) (weekly_savings : ℕ) (vacuum_cost : ℕ) : ℕ :=
  ((vacuum_cost - initial_amount) + weekly_savings - 1) / weekly_savings

/-- Theorem: It takes 10 weeks to save for the vacuum cleaner. -/
theorem vacuum_savings_theorem :
  weeks_to_save 20 10 120 = 10 := by
  sorry

#eval weeks_to_save 20 10 120

end NUMINAMATH_CALUDE_vacuum_savings_theorem_l3823_382373


namespace NUMINAMATH_CALUDE_angle_sum_proof_l3823_382395

theorem angle_sum_proof (x : ℝ) : 
  (6*x + 3*x + 4*x + 2*x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l3823_382395


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l3823_382368

/-- Represents the ages of two people A and B -/
structure Ages where
  a : ℕ
  b : ℕ

/-- The ratio between A's and B's present ages is 6:3 -/
def present_ratio (ages : Ages) : Prop :=
  2 * ages.b = ages.a

/-- The ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
def past_future_ratio (ages : Ages) : Prop :=
  ages.a - 4 = ages.b + 4

/-- The ratio between A's age 4 years hence and B's age 4 years ago is 5:1 -/
def future_past_ratio (ages : Ages) : Prop :=
  5 * (ages.b - 4) = ages.a + 4

/-- Theorem stating the relationship between the given conditions and the result -/
theorem age_ratio_theorem (ages : Ages) :
  present_ratio ages → past_future_ratio ages → future_past_ratio ages :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l3823_382368


namespace NUMINAMATH_CALUDE_divisibility_property_l3823_382321

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ n : ℕ, n ≥ 1 → (a ^ n : ℕ) ∣ (b ^ (n + 1) : ℕ)) : 
  (a : ℕ) ∣ (b : ℕ) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3823_382321


namespace NUMINAMATH_CALUDE_carnival_total_cost_l3823_382328

def carnival_cost (bumper_rides_mara : ℕ) (space_rides_riley : ℕ) (ferris_rides_each : ℕ)
  (bumper_cost : ℕ) (space_cost : ℕ) (ferris_cost : ℕ) : ℕ :=
  bumper_rides_mara * bumper_cost +
  space_rides_riley * space_cost +
  2 * ferris_rides_each * ferris_cost

theorem carnival_total_cost :
  carnival_cost 2 4 3 2 4 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_carnival_total_cost_l3823_382328


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3823_382304

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : sin α = sqrt 5 / 5) : 
  sin (α + π/4) = 3 * sqrt 10 / 10 ∧ tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3823_382304


namespace NUMINAMATH_CALUDE_cubic_sum_equals_36_l3823_382385

theorem cubic_sum_equals_36 (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^3 + b^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_36_l3823_382385


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3823_382332

theorem fraction_equivalence : 
  ∃ n : ℤ, (2 + n : ℚ) / (7 + n) = 3 / 4 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3823_382332


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3823_382317

/-- Given a rectangle with width 5 inches and area 100 square inches, 
    prove that the ratio of length to width is 4:1 -/
theorem rectangle_ratio (width : ℝ) (length : ℝ) (area : ℝ) :
  width = 5 →
  area = 100 →
  area = length * width →
  length / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3823_382317


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l3823_382352

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 43) 
  (h2 : sweaters = 2) 
  (h3 : loads = 9) : 
  (shirts + sweaters) / loads = 5 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l3823_382352


namespace NUMINAMATH_CALUDE_multiples_count_l3823_382364

def count_multiples (n : ℕ) : ℕ := 
  (n.div 3 + n.div 4 - n.div 12) - (n.div 15 + n.div 20 - n.div 60)

theorem multiples_count : count_multiples 2010 = 804 := by sorry

end NUMINAMATH_CALUDE_multiples_count_l3823_382364


namespace NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l3823_382308

theorem no_solution_implies_m_geq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_geq_two_l3823_382308


namespace NUMINAMATH_CALUDE_cubic_minus_linear_at_five_l3823_382305

theorem cubic_minus_linear_at_five : 
  let x : ℝ := 5
  (x^3 - 3*x) = 110 := by sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_at_five_l3823_382305


namespace NUMINAMATH_CALUDE_unique_solution_l3823_382343

-- Define the functions f and p
def f (x : ℝ) : ℝ := |x + 1|

def p (x a : ℝ) : ℝ := |2*x + 5| + a

-- Define the set of values for 'a'
def A : Set ℝ := {-6.5, -5, 1.5}

-- State the theorem
theorem unique_solution (a : ℝ) :
  a ∈ A ↔ ∃! x : ℝ, x ≠ 1 ∧ x ≠ 2.5 ∧ f x = p x a :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3823_382343


namespace NUMINAMATH_CALUDE_balls_placement_count_l3823_382374

-- Define the number of balls and boxes
def num_balls : ℕ := 4
def num_boxes : ℕ := 4

-- Define the function to calculate the number of ways to place the balls
def place_balls : ℕ := sorry

-- Theorem statement
theorem balls_placement_count :
  place_balls = 144 := by sorry

end NUMINAMATH_CALUDE_balls_placement_count_l3823_382374


namespace NUMINAMATH_CALUDE_zero_product_theorem_l3823_382348

theorem zero_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (sum_condition : x₁ + x₂ + x₃ + x₄ = 0)
  (power_sum_condition : x₁^7 + x₂^7 + x₃^7 + x₄^7 = 0) :
  x₄ * (x₄ + x₁) * (x₄ + x₂) * (x₄ + x₃) = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_product_theorem_l3823_382348


namespace NUMINAMATH_CALUDE_two_digit_number_product_l3823_382397

theorem two_digit_number_product (n : ℕ) (tens units : ℕ) : 
  n = tens * 10 + units →
  n = 24 →
  units = tens + 2 →
  n * (tens + units) = 144 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_product_l3823_382397


namespace NUMINAMATH_CALUDE_train_ride_total_time_l3823_382396

def train_ride_duration (reading_time eating_time movie_time nap_time : ℕ) : ℕ :=
  reading_time + eating_time + movie_time + nap_time

theorem train_ride_total_time :
  let reading_time : ℕ := 2
  let eating_time : ℕ := 1
  let movie_time : ℕ := 3
  let nap_time : ℕ := 3
  train_ride_duration reading_time eating_time movie_time nap_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_ride_total_time_l3823_382396


namespace NUMINAMATH_CALUDE_power_of_negative_product_l3823_382382

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l3823_382382


namespace NUMINAMATH_CALUDE_seventh_term_value_l3823_382356

/-- The general term of the series at position n -/
def seriesTerm (n : ℕ) (a : ℝ) : ℝ := (-2)^n * a^(2*n - 1)

/-- The 7th term of the series -/
def seventhTerm (a : ℝ) : ℝ := seriesTerm 7 a

theorem seventh_term_value (a : ℝ) : seventhTerm a = -128 * a^13 := by sorry

end NUMINAMATH_CALUDE_seventh_term_value_l3823_382356


namespace NUMINAMATH_CALUDE_cube_cutting_theorem_l3823_382380

/-- Represents a cube with an integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes -/
def CubeSet := List Cube

/-- Calculates the total volume of a set of cubes -/
def totalVolume (cubes : CubeSet) : ℕ :=
  cubes.map (fun c => c.edge ^ 3) |>.sum

/-- Represents a cutting and reassembly solution -/
structure Solution where
  pieces : ℕ
  isValid : Bool

/-- Theorem stating the existence of a valid solution -/
theorem cube_cutting_theorem (original : CubeSet) (target : CubeSet) : 
  (original = [Cube.mk 14, Cube.mk 10] ∧ 
   target = [Cube.mk 13, Cube.mk 11, Cube.mk 6] ∧
   totalVolume original = totalVolume target) →
  ∃ (sol : Solution), sol.pieces = 11 ∧ sol.isValid = true := by
  sorry

#check cube_cutting_theorem

end NUMINAMATH_CALUDE_cube_cutting_theorem_l3823_382380


namespace NUMINAMATH_CALUDE_opposite_reciprocal_calc_l3823_382349

theorem opposite_reciprocal_calc 
  (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) 
  (h4 : m < 0) : 
  m^3 + c*d + (a+b)/m = -26 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_calc_l3823_382349


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l3823_382353

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (1, 1)
  f2 : ℝ × ℝ := (1, 5)
  -- Point on the ellipse
  p : ℝ × ℝ := (-4, 3)
  -- Constants in the equation (x-h)^2/a^2 + (y-k)^2/b^2 = 1
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- Ensure a and b are positive
  ha : a > 0
  hb : b > 0
  -- Ensure the point p satisfies the equation
  heq : (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) : e.a + e.k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l3823_382353


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3823_382344

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if b = 2, C = π/3, and c = √3, then B = π/2 -/
theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  b = 2 → C = π/3 → c = Real.sqrt 3 → B = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3823_382344


namespace NUMINAMATH_CALUDE_min_packages_for_scooter_l3823_382339

/-- The minimum number of packages to recover the cost of a scooter -/
def min_packages (scooter_cost : ℕ) (earning_per_package : ℕ) (fuel_cost : ℕ) : ℕ :=
  (scooter_cost + (earning_per_package - fuel_cost - 1)) / (earning_per_package - fuel_cost)

/-- Theorem stating the minimum number of packages needed to recover the scooter cost -/
theorem min_packages_for_scooter :
  min_packages 3200 15 4 = 291 :=
by sorry

end NUMINAMATH_CALUDE_min_packages_for_scooter_l3823_382339


namespace NUMINAMATH_CALUDE_min_waves_to_21_l3823_382325

/-- Represents the direction of a wand wave -/
inductive WaveDirection
  | Up
  | Down

/-- Calculates the number of open flowers after a single wave -/
def wave (n : ℕ) (d : WaveDirection) : ℕ :=
  match d with
  | WaveDirection.Up => if n > 0 then n - 1 else 0
  | WaveDirection.Down => 2 * n

/-- Calculates the number of open flowers after a sequence of waves -/
def waveSequence (initial : ℕ) (waves : List WaveDirection) : ℕ :=
  waves.foldl wave initial

/-- Checks if a sequence of waves results in the target number of flowers -/
def isValidSequence (initial target : ℕ) (waves : List WaveDirection) : Prop :=
  waveSequence initial waves = target

/-- Theorem: The minimum number of waves to reach 21 flowers from 3 flowers is 6 -/
theorem min_waves_to_21 :
  ∃ (waves : List WaveDirection),
    waves.length = 6 ∧
    isValidSequence 3 21 waves ∧
    ∀ (other : List WaveDirection),
      isValidSequence 3 21 other → waves.length ≤ other.length :=
by sorry

end NUMINAMATH_CALUDE_min_waves_to_21_l3823_382325


namespace NUMINAMATH_CALUDE_expression_evaluation_l3823_382307

theorem expression_evaluation :
  let y : ℚ := 1/2
  (y + 1) * (y - 1) + (2*y - 1)^2 - 2*y*(2*y - 1) = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3823_382307


namespace NUMINAMATH_CALUDE_ratio_difference_bound_l3823_382350

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, a i > 0) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_bound_l3823_382350


namespace NUMINAMATH_CALUDE_age_ratio_12_years_ago_l3823_382394

-- Define Neha's current age and her mother's current age
def neha_age : ℕ := sorry
def mother_age : ℕ := 60

-- Define the relationship between their ages 12 years ago
axiom past_relation : ∃ x : ℚ, mother_age - 12 = x * (neha_age - 12)

-- Define the relationship between their ages 12 years from now
axiom future_relation : mother_age + 12 = 2 * (neha_age + 12)

-- Theorem to prove
theorem age_ratio_12_years_ago : 
  (mother_age - 12) / (neha_age - 12) = 4 := by sorry

end NUMINAMATH_CALUDE_age_ratio_12_years_ago_l3823_382394


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l3823_382342

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.5

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 945

theorem blue_pill_cost_proof :
  blue_pill_cost = 23.5 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 945 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l3823_382342


namespace NUMINAMATH_CALUDE_expected_occurrences_100_rolls_l3823_382358

/-- The expected number of times a specific face appears when rolling a fair die multiple times -/
def expected_occurrences (num_rolls : ℕ) : ℚ :=
  num_rolls * (1 : ℚ) / 6

/-- Theorem: The expected number of times a specific face appears when rolling a fair die 100 times is 50/3 -/
theorem expected_occurrences_100_rolls :
  expected_occurrences 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_occurrences_100_rolls_l3823_382358


namespace NUMINAMATH_CALUDE_adult_ticket_cost_is_correct_l3823_382351

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The total number of tickets -/
def total_tickets : ℕ := 5

/-- The number of adult tickets -/
def adult_tickets : ℕ := 2

/-- The number of child tickets -/
def child_tickets : ℕ := 3

/-- The total cost of all tickets in dollars -/
def total_cost : ℕ := 77

theorem adult_ticket_cost_is_correct : 
  adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_is_correct_l3823_382351


namespace NUMINAMATH_CALUDE_books_movies_difference_l3823_382316

theorem books_movies_difference : 
  ∀ (total_books total_movies : ℕ),
    total_books = 10 →
    total_movies = 6 →
    total_books - total_movies = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_books_movies_difference_l3823_382316


namespace NUMINAMATH_CALUDE_max_diagonals_same_length_l3823_382393

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- Theorem: The maximum number of diagonals that can be selected in a regular 1000-gon 
    such that among any three of the chosen diagonals, at least two have the same length is 2000 -/
theorem max_diagonals_same_length : 
  ∃ (k : ℕ), k = 2000 ∧ 
  k ≤ total_diagonals n ∧
  k = 2 * diagonals_per_length n ∧
  ∀ (m : ℕ), m > k → ¬(∀ (a b c : ℕ), a < m ∧ b < m ∧ c < m → a = b ∨ b = c ∨ a = c) :=
sorry

end NUMINAMATH_CALUDE_max_diagonals_same_length_l3823_382393


namespace NUMINAMATH_CALUDE_envelope_weight_l3823_382383

/-- Given 800 envelopes with a total weight of 6.8 kg, prove that one envelope weighs 8.5 grams. -/
theorem envelope_weight (num_envelopes : ℕ) (total_weight_kg : ℝ) :
  num_envelopes = 800 →
  total_weight_kg = 6.8 →
  (total_weight_kg * 1000) / num_envelopes = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_envelope_weight_l3823_382383
