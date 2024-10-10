import Mathlib

namespace fifth_month_sales_l623_62338

def sales_1 : ℕ := 4000
def sales_2 : ℕ := 6524
def sales_3 : ℕ := 5689
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 12557
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 6000 := by
  sorry

end fifth_month_sales_l623_62338


namespace prob_at_least_two_fruits_l623_62389

/-- The number of fruit types available -/
def num_fruit_types : ℕ := 4

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
theorem prob_at_least_two_fruits : 
  1 - (num_fruit_types : ℚ) * prob_same_fruit = 15 / 16 := by sorry

end prob_at_least_two_fruits_l623_62389


namespace arrangement_theorem_l623_62393

/-- The number of ways to arrange 3 male and 2 female students in a row with females not at ends -/
def arrangement_count : ℕ := sorry

/-- There are 3 male students -/
def male_count : ℕ := 3

/-- There are 2 female students -/
def female_count : ℕ := 2

/-- Total number of students -/
def total_students : ℕ := male_count + female_count

/-- Number of positions where female students can stand (not at ends) -/
def female_positions : ℕ := total_students - 2

theorem arrangement_theorem : arrangement_count = 36 := by sorry

end arrangement_theorem_l623_62393


namespace fraction_sum_product_equality_l623_62377

theorem fraction_sum_product_equality (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : a / b + c / d = (a / b) * (c / d)) : 
  b / a + d / c = 1 := by
  sorry

end fraction_sum_product_equality_l623_62377


namespace direct_inverse_variation_l623_62378

theorem direct_inverse_variation (c R₁ R₂ S₁ S₂ T₁ T₂ : ℚ) 
  (h1 : R₁ = c * (S₁ / T₁))
  (h2 : R₂ = c * (S₂ / T₂))
  (h3 : R₁ = 2)
  (h4 : T₁ = 1/2)
  (h5 : S₁ = 8)
  (h6 : R₂ = 16)
  (h7 : T₂ = 1/4) :
  S₂ = 32 := by
sorry

end direct_inverse_variation_l623_62378


namespace factoring_expression_l623_62302

theorem factoring_expression (x : ℝ) :
  (12 * x^6 + 40 * x^4 - 6) - (2 * x^6 - 6 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 23) := by
  sorry

end factoring_expression_l623_62302


namespace fraction_simplification_l623_62358

theorem fraction_simplification : (180 : ℚ) / 16200 = 1 / 90 := by
  sorry

end fraction_simplification_l623_62358


namespace tangent_line_constant_l623_62329

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The tangent line function -/
def tangent_line (x b : ℝ) : ℝ := -3*x + b

/-- Theorem stating that if the line y = -3x + b is tangent to the curve y = x^3 - 3x^2, then b = 1 -/
theorem tangent_line_constant (b : ℝ) : 
  (∃ x : ℝ, f x = tangent_line x b ∧ 
    (∀ y : ℝ, y ≠ x → f y ≠ tangent_line y b)) → 
  b = 1 := by sorry

end tangent_line_constant_l623_62329


namespace minimum_distance_problem_l623_62320

open Real

theorem minimum_distance_problem (a : ℝ) : 
  (∃ x₀ : ℝ, (x₀ - a)^2 + (log (3 * x₀) - 3 * a)^2 ≤ 1/10) → a = 1/30 := by
  sorry

end minimum_distance_problem_l623_62320


namespace election_outcomes_count_l623_62364

def boys : ℕ := 28
def girls : ℕ := 22
def total_students : ℕ := boys + girls
def committee_size : ℕ := 5

theorem election_outcomes_count :
  (Nat.descFactorial total_students committee_size) -
  (Nat.descFactorial boys committee_size) -
  (Nat.descFactorial girls committee_size) = 239297520 :=
by sorry

end election_outcomes_count_l623_62364


namespace negation_of_universal_statement_l623_62343

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end negation_of_universal_statement_l623_62343


namespace special_number_fraction_l623_62306

theorem special_number_fraction (numbers : List ℝ) (n : ℝ) :
  numbers.length = 21 ∧
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end special_number_fraction_l623_62306


namespace b_initial_investment_l623_62371

/-- Represents the business investment problem --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_advance : ℕ  -- Amount B advances after 8 months

/-- Calculates B's initial investment given the business conditions --/
def calculate_b_initial (bi : BusinessInvestment) : ℕ :=
  sorry

/-- Theorem stating that B's initial investment is 4000 given the problem conditions --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.total_profit = 630)
  (h3 : bi.a_profit = 240)
  (h4 : bi.a_withdraw = 1000)
  (h5 : bi.b_advance = 1000) :
  calculate_b_initial bi = 4000 := by
  sorry

end b_initial_investment_l623_62371


namespace ordering_abc_l623_62303

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define a, b, and c
noncomputable def a : ℝ := log2 9 - log2 (Real.sqrt 3)
noncomputable def b : ℝ := 1 + log2 (Real.sqrt 7)
noncomputable def c : ℝ := 1/2 + log2 (Real.sqrt 13)

-- Theorem statement
theorem ordering_abc : b > a ∧ a > c := by sorry

end ordering_abc_l623_62303


namespace student_arrangement_count_l623_62373

/-- The number of ways to arrange students from three grades in a row --/
def arrange_students (grade1 : ℕ) (grade2 : ℕ) (grade3 : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial grade2) * (Nat.factorial grade3)

/-- Theorem stating the number of arrangements for the specific case --/
theorem student_arrangement_count :
  arrange_students 1 2 3 = 72 :=
by
  sorry

end student_arrangement_count_l623_62373


namespace smallest_integer_with_remainders_l623_62346

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 3 = 2) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 11 = 10) ∧
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧ n % 11 = 10 → n ≥ M) :=
by
  sorry

#eval Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 11)))) - 1

end smallest_integer_with_remainders_l623_62346


namespace brianas_yield_percentage_l623_62315

theorem brianas_yield_percentage (emma_investment briana_investment : ℝ)
                                 (emma_yield : ℝ)
                                 (investment_period : ℕ)
                                 (roi_difference : ℝ) :
  emma_investment = 300 →
  briana_investment = 500 →
  emma_yield = 0.15 →
  investment_period = 2 →
  roi_difference = 10 →
  briana_investment * (investment_period : ℝ) * (briana_yield / 100) -
  emma_investment * (investment_period : ℝ) * emma_yield = roi_difference →
  briana_yield = 10 :=
by
  sorry

#check brianas_yield_percentage

end brianas_yield_percentage_l623_62315


namespace age_difference_proof_l623_62327

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition that C is 10 years younger than A
def age_difference : Prop := C = A - 10

-- Define the difference in total ages
def total_age_difference : ℕ := (A + B) - (B + C)

-- Theorem to prove
theorem age_difference_proof (h : age_difference A C) : total_age_difference A B C = 10 := by
  sorry

end age_difference_proof_l623_62327


namespace inequality_proof_l623_62344

theorem inequality_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_proof_l623_62344


namespace compound_interest_principal_l623_62394

/-- Given a future value, time, annual interest rate, and compounding frequency,
    calculate the principal amount using the compound interest formula. -/
theorem compound_interest_principal
  (A : ℝ) -- Future value
  (t : ℝ) -- Time in years
  (r : ℝ) -- Annual interest rate (as a decimal)
  (n : ℝ) -- Number of times interest is compounded per year
  (h1 : A = 1000000)
  (h2 : t = 5)
  (h3 : r = 0.08)
  (h4 : n = 4)
  : ∃ P : ℝ, A = P * (1 + r/n)^(n*t) :=
by sorry

end compound_interest_principal_l623_62394


namespace salary_grade_increase_amount_l623_62351

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on salary grade and base increase -/
def hourlyWage (s : SalaryGrade) (x : ℝ) : ℝ :=
  7.50 + x * (s.val - 1)

/-- States that the difference in hourly wage between grade 5 and grade 1 is $1.25 -/
def wageDifference (x : ℝ) : Prop :=
  hourlyWage ⟨5, by norm_num⟩ x - hourlyWage ⟨1, by norm_num⟩ x = 1.25

theorem salary_grade_increase_amount :
  ∃ x : ℝ, wageDifference x ∧ x = 0.3125 := by sorry

end salary_grade_increase_amount_l623_62351


namespace sunglasses_and_hats_probability_l623_62392

theorem sunglasses_and_hats_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (prob_sunglasses_given_hat : ℚ) :
  total_sunglasses = 75 →
  total_hats = 50 →
  prob_sunglasses_given_hat = 1 / 5 →
  (total_hats * prob_sunglasses_given_hat : ℚ) / total_sunglasses = 2 / 15 :=
by sorry

end sunglasses_and_hats_probability_l623_62392


namespace cut_piece_weight_for_equal_copper_percent_l623_62332

/-- Represents an alloy with a given weight and copper percentage -/
structure Alloy where
  weight : ℝ
  copper_percent : ℝ

/-- Theorem stating the weight of the cut piece that equalizes copper percentages -/
theorem cut_piece_weight_for_equal_copper_percent 
  (alloy1 alloy2 : Alloy) 
  (h1 : alloy1.weight = 10)
  (h2 : alloy2.weight = 15)
  (h3 : alloy1.copper_percent ≠ alloy2.copper_percent) :
  ∃ x : ℝ, 
    x > 0 ∧ 
    x < min alloy1.weight alloy2.weight ∧
    ((alloy1.weight - x) * alloy1.copper_percent + x * alloy2.copper_percent) / alloy1.weight = 
    ((alloy2.weight - x) * alloy2.copper_percent + x * alloy1.copper_percent) / alloy2.weight → 
    x = 6 := by
  sorry

#check cut_piece_weight_for_equal_copper_percent

end cut_piece_weight_for_equal_copper_percent_l623_62332


namespace kids_stayed_home_l623_62366

theorem kids_stayed_home (camp_kids : ℕ) (additional_home_kids : ℕ) 
  (h1 : camp_kids = 202958)
  (h2 : additional_home_kids = 574664) :
  camp_kids + additional_home_kids = 777622 := by
  sorry

end kids_stayed_home_l623_62366


namespace point_on_y_axis_l623_62304

/-- A point on the y-axis has an x-coordinate of 0 -/
axiom y_axis_x_zero (x y : ℝ) : (x, y) ∈ Set.range (λ t : ℝ => (0, t)) ↔ x = 0

/-- The point A with coordinates (2-a, -3a+1) lies on the y-axis -/
def A_on_y_axis (a : ℝ) : Prop := (2 - a, -3 * a + 1) ∈ Set.range (λ t : ℝ => (0, t))

theorem point_on_y_axis (a : ℝ) (h : A_on_y_axis a) : a = 2 := by
  sorry

end point_on_y_axis_l623_62304


namespace complex_magnitude_problem_l623_62328

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l623_62328


namespace george_sticker_count_l623_62339

/-- Given the following sticker counts:
  * Dan has 2 times as many stickers as Tom
  * Tom has 3 times as many stickers as Bob
  * George has 5 times as many stickers as Dan
  * Bob has 12 stickers
  Prove that George has 360 stickers -/
theorem george_sticker_count :
  ∀ (bob tom dan george : ℕ),
    dan = 2 * tom →
    tom = 3 * bob →
    george = 5 * dan →
    bob = 12 →
    george = 360 := by
  sorry

end george_sticker_count_l623_62339


namespace lucas_purchase_problem_l623_62310

theorem lucas_purchase_problem :
  ∀ (a b c : ℕ),
    a + b + c = 50 →
    50 * a + 400 * b + 500 * c = 10000 →
    a = 30 :=
by
  sorry

end lucas_purchase_problem_l623_62310


namespace boat_speed_in_still_water_l623_62354

/-- Proves that the speed of a boat in still water is 24 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 84
  let downstream_time : ℝ := 3
  let boat_speed : ℝ := (downstream_distance / downstream_time) - stream_speed
  boat_speed = 24 := by
  sorry

end boat_speed_in_still_water_l623_62354


namespace isosceles_triangle_perimeter_l623_62384

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ
  is_positive : 0 < a

/-- Checks if the given side lengths can form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  3 + t.a > 6 ∧ 3 + 6 > t.a ∧ t.a + 6 > 3

/-- Calculates the perimeter of the triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  3 + t.a + 6

/-- Theorem: If a valid isosceles triangle can be formed with side lengths 3, a, and 6,
    then its perimeter is 15 -/
theorem isosceles_triangle_perimeter
  (t : IsoscelesTriangle)
  (h : is_valid_triangle t) :
  perimeter t = 15 :=
sorry

end isosceles_triangle_perimeter_l623_62384


namespace gcd_1113_1897_l623_62369

theorem gcd_1113_1897 : Nat.gcd 1113 1897 = 7 := by
  sorry

end gcd_1113_1897_l623_62369


namespace race_distance_l623_62381

/-- The race problem -/
theorem race_distance (a_time b_time : ℝ) (lead_distance : ℝ) (race_distance : ℝ) : 
  a_time = 36 →
  b_time = 45 →
  lead_distance = 30 →
  (race_distance / a_time) * b_time = race_distance + lead_distance →
  race_distance = 120 := by
sorry

end race_distance_l623_62381


namespace shaded_area_l623_62365

/-- The shaded area in a geometric configuration --/
theorem shaded_area (AB BC : ℝ) (h1 : AB = Real.sqrt ((8 + Real.sqrt (64 - π^2)) / π))
  (h2 : BC = Real.sqrt ((8 - Real.sqrt (64 - π^2)) / π)) :
  (π / 4) * (AB^2 + BC^2) - AB * BC = 3 := by
  sorry

end shaded_area_l623_62365


namespace height_difference_l623_62322

theorem height_difference (amy_height helen_height angela_height : ℕ) : 
  helen_height = amy_height + 3 →
  amy_height = 150 →
  angela_height = 157 →
  angela_height - helen_height = 4 := by
sorry

end height_difference_l623_62322


namespace moon_speed_km_per_hour_l623_62367

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 103/100

/-- Converts speed from km/s to km/h -/
def convert_km_per_sec_to_km_per_hour (speed_km_per_sec : ℚ) : ℚ :=
  speed_km_per_sec * seconds_per_hour

theorem moon_speed_km_per_hour :
  convert_km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3708 := by
  sorry

end moon_speed_km_per_hour_l623_62367


namespace population_increase_birth_rate_l623_62398

/-- Calculates the percentage increase in population due to birth over a given time period. -/
def population_increase_percentage (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (emigration_rate : ℕ) (immigration_rate : ℕ) : ℚ :=
  let net_migration := (immigration_rate - emigration_rate) * years
  let total_increase := final_population - initial_population - net_migration
  (total_increase : ℚ) / (initial_population : ℚ) * 100

/-- The percentage increase in population due to birth over 10 years is 55%. -/
theorem population_increase_birth_rate : 
  population_increase_percentage 100000 165000 10 2000 2500 = 55 := by
  sorry

end population_increase_birth_rate_l623_62398


namespace validSquaresCount_l623_62376

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 7 black squares -/
def hasAtLeast7BlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  sorry

/-- Theorem stating the correct number of valid squares -/
theorem validSquaresCount :
  countValidSquares = 140 := by sorry

end validSquaresCount_l623_62376


namespace game_cost_l623_62308

/-- The cost of a new game given initial money, birthday gift, and remaining money -/
theorem game_cost (initial : ℕ) (gift : ℕ) (remaining : ℕ) : 
  initial = 16 → gift = 28 → remaining = 19 → initial + gift - remaining = 25 := by
  sorry

end game_cost_l623_62308


namespace distance_to_x_axis_l623_62399

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-4, 1)) : 
  |P.2| = 1 := by sorry

end distance_to_x_axis_l623_62399


namespace money_distribution_l623_62355

/-- Given a distribution of money in the ratio 3:5:7 among three people,
    where the second person's share is 1500,
    prove that the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (share1 share2 share3 : ℕ) :
  share1 + share2 + share3 = total →
  3 * share2 = 5 * share1 →
  7 * share1 = 3 * share3 →
  share2 = 1500 →
  share3 - share1 = 1200 := by
sorry

end money_distribution_l623_62355


namespace quadratic_equation_roots_two_distinct_real_roots_l623_62372

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ↔ 
    (discriminant > 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧
    (discriminant = 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ∧
    (discriminant < 0 → ¬∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by sorry

theorem two_distinct_real_roots :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 2*x1 - 1 = 0 ∧ x2^2 - 2*x2 - 1 = 0 :=
by sorry

end quadratic_equation_roots_two_distinct_real_roots_l623_62372


namespace fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l623_62395

-- Define the quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define what it means to be a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop :=
  quadratic m n t x = x

-- Part 1: Fixed points of y = x^2 - x - 3
theorem fixed_points_of_specific_quadratic :
  {x : ℝ | is_fixed_point 1 (-1) (-3) x} = {-1, 3} := by sorry

-- Part 2: Minimum value of x1/x2 + x2/x1
theorem min_value_of_ratio_sum :
  ∀ a x₁ x₂ : ℝ,
    x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    is_fixed_point 2 (-(2+a)) (a-1) x₁ →
    is_fixed_point 2 (-(2+a)) (a-1) x₂ →
    (x₁ / x₂ + x₂ / x₁) ≥ 6 := by sorry

-- The minimum value is achieved when a = 5
theorem min_value_achieved :
  ∃ a x₁ x₂ : ℝ,
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₁ ∧
    is_fixed_point 2 (-(2+a)) (a-1) x₂ ∧
    x₁ / x₂ + x₂ / x₁ = 6 := by sorry

end fixed_points_of_specific_quadratic_min_value_of_ratio_sum_min_value_achieved_l623_62395


namespace equality_from_cubic_equation_l623_62331

theorem equality_from_cubic_equation (a b : ℕ) 
  (h : a^3 + a + 4*b^2 = 4*a*b + b + b*a^2) : a = b := by
  sorry

end equality_from_cubic_equation_l623_62331


namespace negative_double_inequality_l623_62345

theorem negative_double_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end negative_double_inequality_l623_62345


namespace quadratic_even_iff_b_eq_zero_l623_62359

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

/-- Theorem: f(x) = x^2 + bx + c is an even function if and only if b = 0 -/
theorem quadratic_even_iff_b_eq_zero (b c : ℝ) :
  IsEven (f b c) ↔ b = 0 := by
  sorry

end quadratic_even_iff_b_eq_zero_l623_62359


namespace cos_150_degrees_l623_62300

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l623_62300


namespace number_of_students_l623_62318

theorem number_of_students (n : ℕ) : 
  (n : ℝ) * 15 = 7 * 14 + 7 * 16 + 15 → n = 15 := by
  sorry

end number_of_students_l623_62318


namespace cubic_factorization_l623_62317

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end cubic_factorization_l623_62317


namespace triangle_area_change_l623_62353

theorem triangle_area_change (h b : ℝ) (h_pos : h > 0) (b_pos : b > 0) :
  let new_height := 0.6 * h
  let new_base := b * (1 + 40 / 100)
  let original_area := (1 / 2) * b * h
  let new_area := (1 / 2) * new_base * new_height
  new_area = 0.84 * original_area :=
by sorry

end triangle_area_change_l623_62353


namespace closest_integer_to_expression_l623_62319

theorem closest_integer_to_expression : 
  let expr := (8^1500 + 8^1502) / (8^1501 + 8^1501)
  expr = 65/16 ∧ 
  ∀ n : ℤ, |expr - 4| ≤ |expr - n| := by
  sorry

end closest_integer_to_expression_l623_62319


namespace polynomial_divisibility_l623_62307

theorem polynomial_divisibility (W : ℕ → ℤ) :
  (∀ n : ℕ, (2^n - 1) % W n = 0) →
  (∀ n : ℕ, W n = 1 ∨ W n = -1 ∨ W n = 2*n - 1 ∨ W n = -2*n + 1) :=
by sorry

end polynomial_divisibility_l623_62307


namespace equal_integers_in_table_l623_62374

theorem equal_integers_in_table (t : Fin 10 → Fin 10 → ℤ) 
  (h : ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) → |t i j - t i' j'| ≤ 5) :
  ∃ i j i' j', (i, j) ≠ (i', j') ∧ t i j = t i' j' :=
sorry

end equal_integers_in_table_l623_62374


namespace canal_bottom_width_l623_62330

/-- Given a trapezoidal canal cross-section with the following properties:
  - Top width: 12 meters
  - Depth: 84 meters
  - Area: 840 square meters
  Prove that the bottom width is 8 meters. -/
theorem canal_bottom_width (top_width : ℝ) (depth : ℝ) (area : ℝ) (bottom_width : ℝ) :
  top_width = 12 →
  depth = 84 →
  area = 840 →
  area = (1/2) * (top_width + bottom_width) * depth →
  bottom_width = 8 := by
sorry

end canal_bottom_width_l623_62330


namespace deepak_age_l623_62383

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
    (h1 : rahul_ratio = 4)
    (h2 : deepak_ratio = 3)
    (h3 : rahul_future_age = 26)
    (h4 : rahul_ratio * (rahul_future_age - 10) = deepak_ratio * deepak_present_age) :
  deepak_present_age = 12 := by
  sorry

end deepak_age_l623_62383


namespace sum_of_squares_of_solutions_l623_62350

theorem sum_of_squares_of_solutions : ∃ (a b c d : ℝ),
  (|a^2 - 2*a + 1/1004| = 1/502) ∧
  (|b^2 - 2*b + 1/1004| = 1/502) ∧
  (|c^2 - 2*c + 1/1004| = 1/502) ∧
  (|d^2 - 2*d + 1/1004| = 1/502) ∧
  (a^2 + b^2 + c^2 + d^2 = 8050/1008) :=
by sorry

end sum_of_squares_of_solutions_l623_62350


namespace min_value_expression_l623_62313

theorem min_value_expression (x y : ℝ) (hx : x ≥ 4) (hy : y ≥ -3) :
  x^2 + y^2 - 8*x + 6*y + 20 ≥ -5 ∧
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 4 ∧ y₀ ≥ -3 ∧ x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 20 = -5 :=
sorry

end min_value_expression_l623_62313


namespace no_intersection_in_S_l623_62396

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | mul {f} : S f → S (λ x => x * f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Theorem statement
theorem no_intersection_in_S (f g : ℝ → ℝ) (hf : S f) (hg : S g) (h_neq : f ≠ g) :
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
sorry

end no_intersection_in_S_l623_62396


namespace line_intercept_form_l623_62305

/-- A line passing through a point with a given direction vector has a specific intercept form -/
theorem line_intercept_form (P : ℝ × ℝ) (v : ℝ × ℝ) :
  P = (2, 3) →
  v = (2, -6) →
  ∃ (f : ℝ × ℝ → ℝ), f = (λ (x, y) => x / 3 + y / 9) ∧
    (∀ (Q : ℝ × ℝ), (∃ t : ℝ, Q = (P.1 + t * v.1, P.2 + t * v.2)) ↔ f Q = 1) :=
by sorry

end line_intercept_form_l623_62305


namespace fraction_power_equality_l623_62348

theorem fraction_power_equality : (81000 ^ 5) / (27000 ^ 5) = 243 := by
  sorry

end fraction_power_equality_l623_62348


namespace radius_of_circle_Q_l623_62356

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Circle with center and radius -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Tangency between a circle and a line segment -/
def IsTangent (c : Circle) (p q : ℝ × ℝ) : Prop := sorry

/-- External tangency between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Circle lies inside a triangle -/
def CircleInsideTriangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem radius_of_circle_Q (t : Triangle) (p q : Circle) :
  t.AB = 144 ∧ t.AC = 144 ∧ t.BC = 80 ∧
  p.radius = 24 ∧
  IsTangent p (0, 0) (t.AC, 0) ∧
  IsTangent p (t.BC, 0) (0, 0) ∧
  IsExternallyTangent p q ∧
  IsTangent q (0, 0) (t.AB, 0) ∧
  IsTangent q (t.BC, 0) (0, 0) ∧
  CircleInsideTriangle q t →
  q.radius = 64 - 12 * Real.sqrt 21 := by
  sorry

end radius_of_circle_Q_l623_62356


namespace circle_sum_l623_62360

theorem circle_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end circle_sum_l623_62360


namespace unique_congruence_in_range_l623_62324

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end unique_congruence_in_range_l623_62324


namespace polar_to_cartesian_equivalence_l623_62349

theorem polar_to_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ),
    ρ = 2 * Real.sin θ + 4 * Real.cos θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    (x - 8)^2 + (y - 2)^2 = 68 :=
by sorry

end polar_to_cartesian_equivalence_l623_62349


namespace function_characterization_l623_62352

theorem function_characterization
  (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 →
       f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 :=
by sorry

end function_characterization_l623_62352


namespace pages_per_day_l623_62357

theorem pages_per_day (total_pages : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (h1 : total_pages = 2100)
  (h2 : weeks = 7)
  (h3 : days_per_week = 3) :
  total_pages / (weeks * days_per_week) = 100 := by
  sorry

end pages_per_day_l623_62357


namespace convex_polygon_equal_division_l623_62333

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A straight line that divides a polygon -/
structure DividingLine where
  -- Add necessary fields for a dividing line

/-- A smaller polygon resulting from division -/
structure SmallerPolygon where
  perimeter : ℝ
  longest_side : ℝ

/-- Function to divide a convex polygon with a dividing line -/
def divide_polygon (p : ConvexPolygon) (l : DividingLine) : (SmallerPolygon × SmallerPolygon) :=
  sorry

/-- Theorem stating that any convex polygon can be divided into two smaller polygons
    with equal perimeters and equal longest sides -/
theorem convex_polygon_equal_division (p : ConvexPolygon) :
  ∃ (l : DividingLine),
    let (p1, p2) := divide_polygon p l
    p1.perimeter = p2.perimeter ∧ p1.longest_side = p2.longest_side :=
  sorry

end convex_polygon_equal_division_l623_62333


namespace rhombus_area_l623_62370

/-- A rhombus with specific properties. -/
structure Rhombus where
  /-- The side length of the rhombus. -/
  side_length : ℝ
  /-- The length of half of the shorter diagonal. -/
  half_shorter_diagonal : ℝ
  /-- The difference between the diagonals. -/
  diagonal_difference : ℝ
  /-- The side length is √109. -/
  side_length_eq : side_length = Real.sqrt 109
  /-- The diagonal difference is 12. -/
  diagonal_difference_eq : diagonal_difference = 12
  /-- The Pythagorean theorem holds for the right triangle formed by half of each diagonal and the side. -/
  pythagorean_theorem : half_shorter_diagonal ^ 2 + (half_shorter_diagonal + diagonal_difference / 2) ^ 2 = side_length ^ 2

/-- The area of a rhombus with the given properties is 364 square units. -/
theorem rhombus_area (r : Rhombus) : r.half_shorter_diagonal * (r.half_shorter_diagonal + r.diagonal_difference / 2) * 2 = 364 := by
  sorry

end rhombus_area_l623_62370


namespace fill_tank_times_l623_62382

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Represents the dimensions of the tank -/
def tank_length : ℝ := 30
def tank_width : ℝ := 20
def tank_height : ℝ := 5

/-- Represents the dimensions of the bowl -/
def bowl_length : ℝ := 6
def bowl_width : ℝ := 4
def bowl_height : ℝ := 1

/-- Theorem stating the number of times needed to fill the tank -/
theorem fill_tank_times : 
  (cuboid_volume tank_length tank_width tank_height) / 
  (cuboid_volume bowl_length bowl_width bowl_height) = 125 := by
  sorry

end fill_tank_times_l623_62382


namespace function_symmetry_l623_62379

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

theorem function_symmetry 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (h_sym : ∀ x : ℝ, f a b (π/4 + x) = f a b (π/4 - x)) :
  let y := fun x => f a b (3*π/4 - x)
  (∀ x : ℝ, y (-x) = -y x) ∧ 
  (∀ x : ℝ, y (2*π - x) = y x) := by
sorry

end function_symmetry_l623_62379


namespace minimum_value_implies_a_l623_62335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 → f a x ≥ 3) ∧ 
  (∃ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 ∧ f a x = 3) → 
  a = Real.exp 2 := by
sorry

end minimum_value_implies_a_l623_62335


namespace floor_2a_eq_floor_a_plus_floor_a_half_l623_62341

theorem floor_2a_eq_floor_a_plus_floor_a_half (a : ℝ) (h : a > 0) :
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ := by sorry

end floor_2a_eq_floor_a_plus_floor_a_half_l623_62341


namespace sin_300_degrees_l623_62375

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l623_62375


namespace geometric_sequence_first_term_l623_62363

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 3) -- third term is 3
  (h2 : a * r^4 = 27) -- fifth term is 27
  : a = 1/3 := by
  sorry

end geometric_sequence_first_term_l623_62363


namespace min_sum_p_q_l623_62380

theorem min_sum_p_q (p q : ℝ) : 
  0 < p → 0 < q → 
  (∃ x : ℝ, x^2 + p*x + 2*q = 0) → 
  (∃ x : ℝ, x^2 + 2*q*x + p = 0) → 
  6 ≤ p + q ∧ ∃ p₀ q₀ : ℝ, 0 < p₀ ∧ 0 < q₀ ∧ p₀ + q₀ = 6 ∧ 
    (∃ x : ℝ, x^2 + p₀*x + 2*q₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 2*q₀*x + p₀ = 0) :=
by sorry

end min_sum_p_q_l623_62380


namespace cindy_added_pens_l623_62361

/-- Proves the number of pens Cindy added given the initial conditions and final result --/
theorem cindy_added_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 7)
  (h2 : mike_gives = 22)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 39) :
  final_pens = initial_pens + mike_gives - sharon_receives + 29 := by
  sorry

#check cindy_added_pens

end cindy_added_pens_l623_62361


namespace hoseok_calculation_l623_62334

theorem hoseok_calculation (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end hoseok_calculation_l623_62334


namespace binary_101011_eq_43_l623_62301

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101011_eq_43 : 
  binary_to_decimal [true, true, false, true, false, true] = 43 := by
sorry

end binary_101011_eq_43_l623_62301


namespace smallest_n_for_unique_k_l623_62385

theorem smallest_n_for_unique_k : ∃ (n : ℕ),
  n > 0 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 10/19) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 10/19) ∧
  n = 10 :=
by sorry

end smallest_n_for_unique_k_l623_62385


namespace complex_equation_solution_l623_62362

/-- Given a real number a such that (2+ai)/(1+i) = 3+i, prove that a = 4 -/
theorem complex_equation_solution (a : ℝ) : (2 + a * Complex.I) / (1 + Complex.I) = 3 + Complex.I → a = 4 := by
  sorry

end complex_equation_solution_l623_62362


namespace factor_theorem_l623_62312

theorem factor_theorem (h k : ℝ) : 
  (∃ c : ℝ, 3 * x^3 - h * x + k = c * (x + 3) * (x - 2)) →
  |3 * h - 2 * k| = 27 := by
sorry

end factor_theorem_l623_62312


namespace quadratic_inequality_solution_l623_62340

theorem quadratic_inequality_solution (b c : ℝ) : 
  (∀ x, x^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) → c + b = -3 := by
sorry

end quadratic_inequality_solution_l623_62340


namespace different_color_probability_l623_62314

theorem different_color_probability (blue_chips yellow_chips : ℕ) 
  (h_blue : blue_chips = 5) (h_yellow : yellow_chips = 7) :
  let total_chips := blue_chips + yellow_chips
  let p_blue := blue_chips / total_chips
  let p_yellow := yellow_chips / total_chips
  p_blue * p_yellow + p_yellow * p_blue = 35 / 72 := by
  sorry

end different_color_probability_l623_62314


namespace power_function_through_point_is_odd_l623_62387

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_through_point_is_odd
  (f : ℝ → ℝ)
  (h1 : is_power_function f)
  (h2 : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  is_odd_function f :=
sorry

end power_function_through_point_is_odd_l623_62387


namespace investment_return_correct_l623_62336

def investment_return (n : ℕ+) : ℚ :=
  2^(n.val - 2)

theorem investment_return_correct :
  ∀ (n : ℕ+),
  (n = 1 → investment_return n = (1/2)) ∧
  (∀ (k : ℕ+), investment_return (k + 1) = 2 * investment_return k) :=
by sorry

end investment_return_correct_l623_62336


namespace pages_read_l623_62388

/-- Given that Tom read a certain number of chapters in a book with a fixed number of pages per chapter,
    prove that the total number of pages read is equal to the product of chapters and pages per chapter. -/
theorem pages_read (chapters : ℕ) (pages_per_chapter : ℕ) (h1 : chapters = 20) (h2 : pages_per_chapter = 15) :
  chapters * pages_per_chapter = 300 := by
  sorry

end pages_read_l623_62388


namespace division_remainder_problem_l623_62311

theorem division_remainder_problem : ∃ r, 0 ≤ r ∧ r < 9 ∧ 83 = 9 * 9 + r := by
  sorry

end division_remainder_problem_l623_62311


namespace square_sum_given_difference_and_product_l623_62386

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end square_sum_given_difference_and_product_l623_62386


namespace chloe_profit_l623_62397

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def strawberry_profit (cost_per_dozen : ℚ) (price_per_half_dozen : ℚ) (dozens_sold : ℕ) : ℚ :=
  let profit_per_half_dozen := price_per_half_dozen - (cost_per_dozen / 2)
  let total_half_dozens := dozens_sold * 2
  profit_per_half_dozen * total_half_dozens

/-- Theorem: Chloe's profit from selling chocolate-dipped strawberries is $500 -/
theorem chloe_profit :
  strawberry_profit 50 30 50 = 500 := by
  sorry

end chloe_profit_l623_62397


namespace intersection_point_on_both_lines_unique_intersection_point_l623_62321

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-36, 26)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -4y = 3x + 4 -/
def line2 (x y : ℝ) : Prop := -4 * y = 3 * x + 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_on_both_lines_unique_intersection_point_l623_62321


namespace ticket_price_increase_l623_62337

theorem ticket_price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  original_price = 85 → 
  increase_percentage = 20 → 
  original_price * (1 + increase_percentage / 100) = 102 := by
sorry

end ticket_price_increase_l623_62337


namespace charles_picked_50_pears_l623_62368

/-- The number of pears Charles picked -/
def pears_picked : ℕ := sorry

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := sorry

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

theorem charles_picked_50_pears :
  (dishes_washed = bananas_cooked + 10) ∧
  (bananas_cooked = 3 * pears_picked) →
  pears_picked = 50 := by sorry

end charles_picked_50_pears_l623_62368


namespace imaginary_part_of_complex_fraction_l623_62323

theorem imaginary_part_of_complex_fraction (m : ℝ) : 
  (Complex.im ((2 - Complex.I) * (m + Complex.I)) = 0) → 
  (Complex.im (m * Complex.I / (1 - Complex.I)) = 1) := by
sorry

end imaginary_part_of_complex_fraction_l623_62323


namespace even_function_derivative_zero_l623_62391

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Theorem: If f is an even function and its derivative exists, then f'(0) = 0 -/
theorem even_function_derivative_zero (f : ℝ → ℝ) (hf : IsEven f) (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 := by
  sorry

end even_function_derivative_zero_l623_62391


namespace base_conversion_subtraction_l623_62342

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_subtraction :
  base8_to_base10 52143 - base9_to_base10 3456 = 19041 := by sorry

end base_conversion_subtraction_l623_62342


namespace exterior_angle_square_octagon_exterior_angle_is_135_l623_62347

/-- The exterior angle of a square and a regular octagon sharing a common side is 135°. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  fun angle =>
    let square_angle := 90
    let octagon_interior_angle := 135
    let exterior_angle := 360 - square_angle - octagon_interior_angle
    exterior_angle = angle

/-- The theorem statement -/
theorem exterior_angle_is_135 : exterior_angle_square_octagon 135 := by
  sorry

end exterior_angle_square_octagon_exterior_angle_is_135_l623_62347


namespace sum_even_10_mod_6_l623_62316

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that the remainder of the sum of the first 10 even numbers divided by 6 is 2 -/
theorem sum_even_10_mod_6 : sum_even 10 % 6 = 2 := by
  sorry

end sum_even_10_mod_6_l623_62316


namespace correct_statements_l623_62325

-- Define the types of relationships
inductive Relationship
| Function
| Correlation

-- Define the types of analysis methods
inductive AnalysisMethod
| Regression

-- Define the properties of relationships
def isDeterministic (r : Relationship) : Prop :=
  match r with
  | Relationship.Function => True
  | Relationship.Correlation => False

-- Define the properties of analysis methods
def isCommonlyUsedFor (m : AnalysisMethod) (r : Relationship) : Prop :=
  match m, r with
  | AnalysisMethod.Regression, Relationship.Correlation => True
  | _, _ => False

-- Theorem to prove
theorem correct_statements :
  isDeterministic Relationship.Function ∧
  ¬isDeterministic Relationship.Correlation ∧
  isCommonlyUsedFor AnalysisMethod.Regression Relationship.Correlation :=
by sorry


end correct_statements_l623_62325


namespace max_regions_circle_rectangle_triangle_l623_62326

/-- Represents a shape in the plane -/
inductive Shape
  | Circle
  | Rectangle
  | Triangle

/-- The number of regions created by intersecting shapes in the plane -/
def num_regions (shapes : List Shape) : ℕ :=
  sorry

/-- The maximum number of regions created by intersecting a circle, rectangle, and triangle -/
theorem max_regions_circle_rectangle_triangle :
  num_regions [Shape.Circle, Shape.Rectangle, Shape.Triangle] = 21 :=
by sorry

end max_regions_circle_rectangle_triangle_l623_62326


namespace worm_length_difference_l623_62390

theorem worm_length_difference : 
  let longer_worm : ℝ := 0.8
  let shorter_worm : ℝ := 0.1
  longer_worm - shorter_worm = 0.7 := by
  sorry

end worm_length_difference_l623_62390


namespace zero_smallest_natural_l623_62309

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end zero_smallest_natural_l623_62309
