import Mathlib

namespace NUMINAMATH_GPT_red_balls_in_bag_l1298_129833

/-- Given the conditions of the ball distribution in the bag,
we need to prove the number of red balls is 9. -/
theorem red_balls_in_bag (total_balls white_balls green_balls yellow_balls purple_balls : ℕ)
  (prob_neither_red_nor_purple : ℝ) (h_total : total_balls = 100)
  (h_white : white_balls = 50) (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 8) (h_purple : purple_balls = 3)
  (h_prob : prob_neither_red_nor_purple = 0.88) :
  ∃ R : ℕ, (total_balls = white_balls + green_balls + yellow_balls + purple_balls + R) ∧ R = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_red_balls_in_bag_l1298_129833


namespace NUMINAMATH_GPT_ak_not_perfect_square_l1298_129894

theorem ak_not_perfect_square (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k1 k2, a k1 = 1988 ∧ b k2 = 1988) :
  ∀ k, ¬ ∃ n, a k = n * n :=
by
  sorry

end NUMINAMATH_GPT_ak_not_perfect_square_l1298_129894


namespace NUMINAMATH_GPT_find_a_plus_b_l1298_129882

theorem find_a_plus_b (a b : ℝ) (h₁ : ∀ x, x - b < 0 → x < b) 
  (h₂ : ∀ x, x + a > 0 → x > -a) 
  (h₃ : ∀ x, 2 < x ∧ x < 3 → -a < x ∧ x < b) : 
  a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1298_129882


namespace NUMINAMATH_GPT_gross_profit_value_l1298_129827

theorem gross_profit_value (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 54) 
    (h2 : gross_profit = 1.25 * cost) 
    (h3 : sales_price = cost + gross_profit): gross_profit = 30 := 
  sorry

end NUMINAMATH_GPT_gross_profit_value_l1298_129827


namespace NUMINAMATH_GPT_find_t_l1298_129811

theorem find_t (t : ℚ) : 
  ((t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5) → t = 5 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_t_l1298_129811


namespace NUMINAMATH_GPT_evaluate_expression_l1298_129829

theorem evaluate_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1298_129829


namespace NUMINAMATH_GPT_algebraic_expression_value_l1298_129802

theorem algebraic_expression_value (x y : ℝ) (h : x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7) :
(x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨ (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1298_129802


namespace NUMINAMATH_GPT_period_and_symmetry_of_function_l1298_129808

-- Given conditions
variables (f : ℝ → ℝ)
variable (hf_odd : ∀ x, f (-x) = -f x)
variable (hf_cond : ∀ x, f (-2 * x + 4) = -f (2 * x))
variable (hf_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1)

-- Prove that 4 is a period and x=1 is a line of symmetry for the graph of f(x)
theorem period_and_symmetry_of_function :
  (∀ x, f (x + 4) = f x) ∧ (∀ x, f (x) + f (4 - x) = 0) :=
by sorry

end NUMINAMATH_GPT_period_and_symmetry_of_function_l1298_129808


namespace NUMINAMATH_GPT_problem1_problem2_l1298_129824

-- Define the required conditions
variables {a b : ℤ}
-- Conditions
axiom h1 : a ≥ 1
axiom h2 : b ≥ 1

-- Proof statement for question 1
theorem problem1 : ¬ (a ∣ b^2 ↔ a ∣ b) := by
  sorry

-- Proof statement for question 2
theorem problem2 : (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1298_129824


namespace NUMINAMATH_GPT_team_A_has_more_uniform_heights_l1298_129870

-- Definitions of the conditions
def avg_height_team_A : ℝ := 1.65
def avg_height_team_B : ℝ := 1.65

def variance_team_A : ℝ := 1.5
def variance_team_B : ℝ := 2.4

-- Theorem stating the problem solution
theorem team_A_has_more_uniform_heights :
  variance_team_A < variance_team_B :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_team_A_has_more_uniform_heights_l1298_129870


namespace NUMINAMATH_GPT_unique_real_solution_N_l1298_129864

theorem unique_real_solution_N (N : ℝ) :
  (∃! (x y : ℝ), 2 * x^2 + 4 * x * y + 7 * y^2 - 12 * x - 2 * y + N = 0) ↔ N = 23 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_N_l1298_129864


namespace NUMINAMATH_GPT_theo_needs_84_eggs_l1298_129847

def customers_hour1 := 5
def customers_hour2 := 7
def customers_hour3 := 3
def customers_hour4 := 8

def eggs_per_omelette_3 := 3
def eggs_per_omelette_4 := 4

def total_eggs_needed : Nat :=
  (customers_hour1 * eggs_per_omelette_3) +
  (customers_hour2 * eggs_per_omelette_4) +
  (customers_hour3 * eggs_per_omelette_3) +
  (customers_hour4 * eggs_per_omelette_4)

theorem theo_needs_84_eggs : total_eggs_needed = 84 :=
by
  sorry

end NUMINAMATH_GPT_theo_needs_84_eggs_l1298_129847


namespace NUMINAMATH_GPT_divisibility_by_3_l1298_129822

theorem divisibility_by_3 (a b c : ℤ) (h1 : c ≠ b)
    (h2 : ∃ x : ℂ, (a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0)) :
    3 ∣ (a + b + 2 * c) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_3_l1298_129822


namespace NUMINAMATH_GPT_house_total_volume_l1298_129815

def room_volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

def bathroom_volume := room_volume 4 2 7
def bedroom_volume := room_volume 12 10 8
def livingroom_volume := room_volume 15 12 9

def total_volume := bathroom_volume + bedroom_volume + livingroom_volume

theorem house_total_volume : total_volume = 2636 := by
  sorry

end NUMINAMATH_GPT_house_total_volume_l1298_129815


namespace NUMINAMATH_GPT_a3_5a6_value_l1298_129831

variable {a : ℕ → ℤ}

-- Conditions: The sequence {a_n} is an arithmetic sequence, and a_4 + a_7 = 19
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

axiom a_seq_arithmetic : is_arithmetic_sequence a
axiom a4_a7_sum : a 4 + a 7 = 19

-- Problem statement: Prove that a_3 + 5a_6 = 57
theorem a3_5a6_value : a 3 + 5 * a 6 = 57 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_a3_5a6_value_l1298_129831


namespace NUMINAMATH_GPT_conditions_for_right_triangle_l1298_129807

universe u

variables {A B C : Type u}
variables [OrderedAddCommGroup A] [OrderedAddCommGroup B] [OrderedAddCommGroup C]

noncomputable def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem conditions_for_right_triangle :
  (∀ (A B C : ℝ), A + B = C → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), ( A / C = 1 / 6 ) → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), A = 90 - B → is_right_triangle A B C) ∧
  (∀ (A B C : ℝ), (A = B → B = C / 2) → is_right_triangle A B C) ∧
  ∀ (A B C : ℝ), ¬ ((A = 2 * B) ∧ B = 3 * C) 
:=
sorry

end NUMINAMATH_GPT_conditions_for_right_triangle_l1298_129807


namespace NUMINAMATH_GPT_tiffany_bags_l1298_129845

/-!
## Problem Statement
Tiffany was collecting cans for recycling. On Monday she had some bags of cans. 
She found 3 bags of cans on the next day and 7 bags of cans the day after that. 
She had altogether 20 bags of cans. Prove that the number of bags of cans she had on Monday is 10.
-/

theorem tiffany_bags (M : ℕ) (h1 : M + 3 + 7 = 20) : M = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_tiffany_bags_l1298_129845


namespace NUMINAMATH_GPT_symmetric_point_reflection_l1298_129888

theorem symmetric_point_reflection (x y : ℝ) : (2, -(-5)) = (2, 5) := by
  sorry

end NUMINAMATH_GPT_symmetric_point_reflection_l1298_129888


namespace NUMINAMATH_GPT_intercepts_sum_eq_seven_l1298_129892

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end NUMINAMATH_GPT_intercepts_sum_eq_seven_l1298_129892


namespace NUMINAMATH_GPT_mutually_exclusive_not_contradictory_l1298_129830

namespace BallProbability
  -- Definitions of events based on the conditions
  def at_least_two_white (outcome : Multiset (String)) : Prop := 
    Multiset.count "white" outcome ≥ 2

  def all_red (outcome : Multiset (String)) : Prop := 
    Multiset.count "red" outcome = 3

  -- Problem statement
  theorem mutually_exclusive_not_contradictory :
    ∀ outcome : Multiset (String),
    Multiset.card outcome = 3 →
    (at_least_two_white outcome → ¬all_red outcome) ∧
    ¬(∀ outcome, at_least_two_white outcome ↔ ¬all_red outcome) := 
  by
    intros
    sorry
end BallProbability

end NUMINAMATH_GPT_mutually_exclusive_not_contradictory_l1298_129830


namespace NUMINAMATH_GPT_intersection_M_N_l1298_129879

def M : Set ℝ :=
  {x | |x| ≤ 2}

def N : Set ℝ :=
  {x | Real.exp x ≥ 1}

theorem intersection_M_N :
  (M ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1298_129879


namespace NUMINAMATH_GPT_lana_total_pages_l1298_129823

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end NUMINAMATH_GPT_lana_total_pages_l1298_129823


namespace NUMINAMATH_GPT_max_d_value_l1298_129899

theorem max_d_value : ∀ (d e : ℕ), (d < 10) → (e < 10) → (5 * 10^5 + d * 10^4 + 5 * 10^3 + 2 * 10^2 + 2 * 10 + e ≡ 0 [MOD 22]) → (e % 2 = 0) → (d + e = 10) → d ≤ 8 :=
by
  intros d e h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_max_d_value_l1298_129899


namespace NUMINAMATH_GPT_savings_percentage_l1298_129883

theorem savings_percentage (I S : ℝ) (h1 : I > 0) (h2 : S > 0) (h3 : S ≤ I) 
  (h4 : 1.25 * I - 2 * S + I - S = 2 * (I - S)) :
  (S / I) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_savings_percentage_l1298_129883


namespace NUMINAMATH_GPT_minimum_workers_needed_l1298_129841

theorem minimum_workers_needed 
  (total_days : ℕ)
  (completed_days : ℕ)
  (initial_workers : ℕ)
  (fraction_completed : ℚ)
  (remaining_fraction : ℚ)
  (remaining_days : ℕ)
  (rate_completed_per_day : ℚ)
  (required_rate_per_day : ℚ)
  (equal_productivity : Prop) 
  : initial_workers = 10 :=
by
  -- Definitions
  let total_days := 40
  let completed_days := 10
  let initial_workers := 10
  let fraction_completed := 1 / 4
  let remaining_fraction := 1 - fraction_completed
  let remaining_days := total_days - completed_days
  let rate_completed_per_day := fraction_completed / completed_days
  let required_rate_per_day := remaining_fraction / remaining_days
  let equal_productivity := true

  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_minimum_workers_needed_l1298_129841


namespace NUMINAMATH_GPT_largest_integer_less_than_120_with_remainder_5_div_8_l1298_129818

theorem largest_integer_less_than_120_with_remainder_5_div_8 :
  ∃ n : ℤ, n < 120 ∧ n % 8 = 5 ∧ ∀ m : ℤ, m < 120 → m % 8 = 5 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_120_with_remainder_5_div_8_l1298_129818


namespace NUMINAMATH_GPT_exponent_multiplication_identity_l1298_129859

theorem exponent_multiplication_identity : 2^4 * 3^2 * 5^2 * 7 = 6300 := sorry

end NUMINAMATH_GPT_exponent_multiplication_identity_l1298_129859


namespace NUMINAMATH_GPT_distance_between_points_l1298_129881

noncomputable def distance (x1 y1 x2 y2 : ℝ) := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : 
  distance (-3) (1/2) 4 (-7) = Real.sqrt 105.25 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_points_l1298_129881


namespace NUMINAMATH_GPT_forest_coverage_2009_min_annual_growth_rate_l1298_129886

variables (a : ℝ)

-- Conditions
def initially_forest_coverage (a : ℝ) := a
def annual_natural_growth_rate := 0.02

-- Questions reformulated:
-- Part 1: Prove the forest coverage at the end of 2009
theorem forest_coverage_2009 : (∃ a : ℝ, (y : ℝ) = a * (1 + 0.02)^5 ∧ y = 1.104 * a) :=
by sorry

-- Part 2: Prove the minimum annual average growth rate by 2014
theorem min_annual_growth_rate : (∀ p : ℝ, (a : ℝ) * (1 + p)^10 ≥ 2 * a → p ≥ 0.072) :=
by sorry

end NUMINAMATH_GPT_forest_coverage_2009_min_annual_growth_rate_l1298_129886


namespace NUMINAMATH_GPT_exists_strictly_increasing_sequence_l1298_129887

theorem exists_strictly_increasing_sequence 
  (N : ℕ) : 
  (∃ (t : ℕ), t^2 ≤ N ∧ N < t^2 + t) →
  (∃ (s : ℕ → ℕ), (∀ n : ℕ, s n < s (n + 1)) ∧ 
   (∃ k : ℕ, ∀ n : ℕ, s (n + 1) - s n = k) ∧
   (∀ n : ℕ, s (s n) - s (s (n - 1)) ≤ N 
      ∧ N < s (1 + s n) - s (s (n - 1)))) :=
by
  sorry

end NUMINAMATH_GPT_exists_strictly_increasing_sequence_l1298_129887


namespace NUMINAMATH_GPT_pills_per_week_l1298_129896

theorem pills_per_week (hours_per_pill : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) 
(h1: hours_per_pill = 6) (h2: hours_per_day = 24) (h3: days_per_week = 7) :
(hours_per_day / hours_per_pill) * days_per_week = 28 :=
by
  sorry

end NUMINAMATH_GPT_pills_per_week_l1298_129896


namespace NUMINAMATH_GPT_new_volume_of_cylinder_l1298_129838

theorem new_volume_of_cylinder
  (r h : ℝ) -- original radius and height
  (V : ℝ) -- original volume
  (h_volume : V = π * r^2 * h) -- volume formula for the original cylinder
  (new_radius : ℝ := 3 * r) -- new radius is three times the original radius
  (new_volume : ℝ) -- new volume to be determined
  (h_original_volume : V = 10) -- original volume equals 10 cubic feet
  : new_volume = 9 * V := -- new volume should be 9 times the original volume
by
  sorry

end NUMINAMATH_GPT_new_volume_of_cylinder_l1298_129838


namespace NUMINAMATH_GPT_gcd_282_470_l1298_129868

theorem gcd_282_470 : Int.gcd 282 470 = 94 :=
by
  sorry

end NUMINAMATH_GPT_gcd_282_470_l1298_129868


namespace NUMINAMATH_GPT_largest_non_representable_as_sum_of_composites_l1298_129873

-- Define what a composite number is
def is_composite (n : ℕ) : Prop := 
  ∃ k m : ℕ, 1 < k ∧ 1 < m ∧ k * m = n

-- Statement: Prove that the largest natural number that cannot be represented
-- as the sum of two composite numbers is 11.
theorem largest_non_representable_as_sum_of_composites : 
  ∀ n : ℕ, n ≤ 11 ↔ ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) := 
sorry

end NUMINAMATH_GPT_largest_non_representable_as_sum_of_composites_l1298_129873


namespace NUMINAMATH_GPT_even_function_value_of_a_l1298_129854

theorem even_function_value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x * (Real.exp x + a * Real.exp (-x))) (h_even : ∀ x : ℝ, f x = f (-x)) : a = -1 := 
by
  sorry

end NUMINAMATH_GPT_even_function_value_of_a_l1298_129854


namespace NUMINAMATH_GPT_tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l1298_129893

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem tangent_line_at_x_equals_1 (a : ℝ) (x : ℝ) (h₀ : a = 2) (h₁ : x = 1) : 
  3 * x - (f a 1) - 1 = 0 := 
sorry

theorem monotonic_intervals (a x : ℝ) (h₀ : x > 0) :
  ((a >= 0 ∧ ∀ (x : ℝ), x > 0 → (f a x) > (f a (x - 1))) ∨ 
  (a < 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < -1/a → (f a x) > (f a (x - 1)) ∧ ∀ (x : ℝ), x > -1/a → (f a x) < (f a (x - 1)))) :=
sorry

theorem range_of_a (a x : ℝ) (h₀ : 0 < x) (h₁ : f a x < 2) : a < -1 / Real.exp (3) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l1298_129893


namespace NUMINAMATH_GPT_school_adding_seats_l1298_129836

theorem school_adding_seats (row_seats : ℕ) (seat_cost : ℕ) (discount_rate : ℝ) (total_cost : ℕ) (n : ℕ) 
                         (total_seats : ℕ) (discounted_seat_cost : ℕ)
                         (total_groups : ℕ) (rows : ℕ) :
  row_seats = 8 →
  seat_cost = 30 →
  discount_rate = 0.10 →
  total_cost = 1080 →
  discounted_seat_cost = seat_cost * (1 - discount_rate) →
  total_seats = total_cost / discounted_seat_cost →
  total_groups = total_seats / 10 →
  rows = total_seats / row_seats →
  rows = 5 :=
by
  intros hrowseats hseatcost hdiscountrate htotalcost hdiscountedseatcost htotalseats htotalgroups hrows
  sorry

end NUMINAMATH_GPT_school_adding_seats_l1298_129836


namespace NUMINAMATH_GPT_root_diff_condition_l1298_129816

noncomputable def g (x : ℝ) : ℝ := 4^x + 2*x - 2
noncomputable def f (x : ℝ) : ℝ := 4*x - 1

theorem root_diff_condition :
  ∃ x₀, g x₀ = 0 ∧ |x₀ - 1/4| ≤ 1/4 ∧ ∃ y₀, f y₀ = 0 ∧ |y₀ - x₀| ≤ 0.25 :=
sorry

end NUMINAMATH_GPT_root_diff_condition_l1298_129816


namespace NUMINAMATH_GPT_roots_absolute_value_l1298_129877

noncomputable def quadratic_roots_property (p : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 ≠ r2 ∧
  r1 + r2 = -p ∧
  r1 * r2 = 16 ∧
  ∃ r : ℝ, r = r1 ∨ r = r2 ∧ abs r > 4

theorem roots_absolute_value (p : ℝ) (r1 r2 : ℝ) :
  quadratic_roots_property p r1 r2 → ∃ r : ℝ, (r = r1 ∨ r = r2) ∧ abs r > 4 :=
sorry

end NUMINAMATH_GPT_roots_absolute_value_l1298_129877


namespace NUMINAMATH_GPT_find_salary_J_l1298_129897

variables (J F M A May : ℝ)

def avg_salary_J_F_M_A (J F M A : ℝ) : Prop :=
  (J + F + M + A) / 4 = 8000

def avg_salary_F_M_A_May (F M A May : ℝ) : Prop :=
  (F + M + A + May) / 4 = 8700

def salary_May (May : ℝ) : Prop :=
  May = 6500

theorem find_salary_J (h1 : avg_salary_J_F_M_A J F M A) (h2 : avg_salary_F_M_A_May F M A May) (h3 : salary_May May) :
  J = 3700 :=
sorry

end NUMINAMATH_GPT_find_salary_J_l1298_129897


namespace NUMINAMATH_GPT_eval_expression_l1298_129809

theorem eval_expression :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1298_129809


namespace NUMINAMATH_GPT_least_number_with_remainder_l1298_129813

theorem least_number_with_remainder (n : ℕ) (d₁ d₂ d₃ d₄ r : ℕ) 
  (h₁ : d₁ = 5) (h₂ : d₂ = 6) (h₃ : d₃ = 9) (h₄ : d₄ = 12) (hr : r = 184) :
  (∀ d, d ∈ [d₁, d₂, d₃, d₄] → n % d = r % d) → n = 364 := 
sorry

end NUMINAMATH_GPT_least_number_with_remainder_l1298_129813


namespace NUMINAMATH_GPT_probability_A_not_losing_l1298_129849

variable (P_A_wins : ℝ)
variable (P_draw : ℝ)
variable (P_A_not_losing : ℝ)

theorem probability_A_not_losing 
  (h1 : P_A_wins = 0.3) 
  (h2 : P_draw = 0.5) 
  (h3 : P_A_not_losing = P_A_wins + P_draw) :
  P_A_not_losing = 0.8 :=
sorry

end NUMINAMATH_GPT_probability_A_not_losing_l1298_129849


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l1298_129821

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x + 3) * (x - 4) = 18 → (∃ a b : ℝ, x ^ 2 + a * x + b = 0) ∧ (a = -1) ∧ (b = -30) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l1298_129821


namespace NUMINAMATH_GPT_f_8_plus_f_9_l1298_129878

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_even_transformed : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 1

theorem f_8_plus_f_9 : f 8 + f 9 = 1 :=
sorry

end NUMINAMATH_GPT_f_8_plus_f_9_l1298_129878


namespace NUMINAMATH_GPT_minimize_expression_l1298_129812

variables {x y : ℝ}

theorem minimize_expression : ∃ (x y : ℝ), 2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 = -2 :=
by sorry

end NUMINAMATH_GPT_minimize_expression_l1298_129812


namespace NUMINAMATH_GPT_answered_both_correctly_l1298_129834

variable (A B : Prop)
variable (P_A P_B P_not_A_and_not_B P_A_and_B : ℝ)

axiom P_A_eq : P_A = 0.75
axiom P_B_eq : P_B = 0.35
axiom P_not_A_and_not_B_eq : P_not_A_and_not_B = 0.20

theorem answered_both_correctly (h1 : P_A = 0.75) (h2 : P_B = 0.35) (h3 : P_not_A_and_not_B = 0.20) : 
  P_A_and_B = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_answered_both_correctly_l1298_129834


namespace NUMINAMATH_GPT_proof_of_greatest_sum_quotient_remainder_l1298_129835

def greatest_sum_quotient_remainder : Prop :=
  ∃ q r : ℕ, 1051 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ q + r = 61

theorem proof_of_greatest_sum_quotient_remainder : greatest_sum_quotient_remainder := 
sorry

end NUMINAMATH_GPT_proof_of_greatest_sum_quotient_remainder_l1298_129835


namespace NUMINAMATH_GPT_parallel_vectors_l1298_129891

variable {k m : ℝ}

theorem parallel_vectors (h₁ : (2 : ℝ) = k * m) (h₂ : m = 2 * k) : m = 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l1298_129891


namespace NUMINAMATH_GPT_polynomial_solution_l1298_129851

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x, (x + 2019) * (P.eval x) = x * (P.eval (x + 1))) :
  ∃ C : ℝ, P = Polynomial.C C * Polynomial.X * (Polynomial.X + 2018) :=
sorry

end NUMINAMATH_GPT_polynomial_solution_l1298_129851


namespace NUMINAMATH_GPT_line_equation_is_correct_l1298_129872

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

theorem line_equation_is_correct (x y t : ℝ)
  (h1: x = 3 * t + 6)
  (h2: y = 5 * t - 7) :
  y = (5 / 3) * x - 17 :=
sorry

end NUMINAMATH_GPT_line_equation_is_correct_l1298_129872


namespace NUMINAMATH_GPT_contractor_engaged_days_l1298_129804

theorem contractor_engaged_days (x y : ℕ) (earnings_per_day : ℕ) (fine_per_day : ℝ) 
    (total_earnings : ℝ) (absent_days : ℕ) 
    (h1 : earnings_per_day = 25) 
    (h2 : fine_per_day = 7.50) 
    (h3 : total_earnings = 555) 
    (h4 : absent_days = 6) 
    (h5 : total_earnings = (earnings_per_day * x : ℝ) - fine_per_day * y) 
    (h6 : y = absent_days) : 
    x = 24 := 
by
  sorry

end NUMINAMATH_GPT_contractor_engaged_days_l1298_129804


namespace NUMINAMATH_GPT_ratio_of_jars_to_pots_l1298_129844

theorem ratio_of_jars_to_pots 
  (jars : ℕ)
  (pots : ℕ)
  (k : ℕ)
  (marbles_total : ℕ)
  (h1 : jars = 16)
  (h2 : jars = k * pots)
  (h3 : ∀ j, j = 5)
  (h4 : ∀ p, p = 15)
  (h5 : marbles_total = 200) :
  (jars / pots = 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_jars_to_pots_l1298_129844


namespace NUMINAMATH_GPT_max_halls_visitable_max_triangles_in_chain_l1298_129885

-- Definition of the problem conditions
def castle_side_length : ℝ := 100
def num_halls : ℕ := 100
def hall_side_length : ℝ := 10
def max_visitable_halls : ℕ := 91

-- Theorem statements
theorem max_halls_visitable (S : ℝ) (n : ℕ) (H : ℝ) :
  S = 100 ∧ n = 100 ∧ H = 10 → max_visitable_halls = 91 :=
by sorry

-- Definitions for subdividing an equilateral triangle and the chain of triangles
def side_divisions (k : ℕ) : ℕ := k
def total_smaller_triangles (k : ℕ) : ℕ := k^2
def max_chain_length (k : ℕ) : ℕ := k^2 - k + 1

-- Theorem statements
theorem max_triangles_in_chain (k : ℕ) :
  max_chain_length k = k^2 - k + 1 :=
by sorry

end NUMINAMATH_GPT_max_halls_visitable_max_triangles_in_chain_l1298_129885


namespace NUMINAMATH_GPT_tunnel_length_correct_l1298_129862

noncomputable def length_of_tunnel
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time_s := crossing_time_min * 60
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem tunnel_length_correct :
  length_of_tunnel 800 78 1 = 500.2 :=
by
  -- The proof will be filled later.
  sorry

end NUMINAMATH_GPT_tunnel_length_correct_l1298_129862


namespace NUMINAMATH_GPT_right_triangle_ineq_l1298_129817

-- Definitions based on conditions in (a)
variables {a b c m f : ℝ}
variable (h_a : a ≥ 0)
variable (h_b : b ≥ 0)
variable (h_c : c > 0)
variable (h_a_b : a ≤ b)
variable (h_triangle : c = Real.sqrt (a^2 + b^2))
variable (h_m : m = a * b / c)
variable (h_f : f = (Real.sqrt 2 * a * b) / (a + b))

-- Proof goal based on the problem in (c)
theorem right_triangle_ineq : m + f ≤ c :=
sorry

end NUMINAMATH_GPT_right_triangle_ineq_l1298_129817


namespace NUMINAMATH_GPT_find_a_plus_b_l1298_129820

theorem find_a_plus_b (a b : ℝ) 
  (h_a : a^3 - 3 * a^2 + 5 * a = 1) 
  (h_b : b^3 - 3 * b^2 + 5 * b = 5) : 
  a + b = 2 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1298_129820


namespace NUMINAMATH_GPT_shortest_travel_time_to_sunny_town_l1298_129800

-- Definitions based on the given conditions
def highway_length : ℕ := 12

def railway_crossing_closed (t : ℕ) : Prop :=
  ∃ k : ℕ, t = 6 * k + 0 ∨ t = 6 * k + 1 ∨ t = 6 * k + 2

def traffic_light_red (t : ℕ) : Prop :=
  ∃ k1 : ℕ, t = 5 * k1 + 0 ∨ t = 5 * k1 + 1

def initial_conditions (t : ℕ) : Prop :=
  railway_crossing_closed 0 ∧ traffic_light_red 0

def shortest_time_to_sunny_town (time : ℕ) : Prop := 
  time = 24

-- The proof statement
theorem shortest_travel_time_to_sunny_town :
  ∃ time : ℕ, shortest_time_to_sunny_town time ∧
  (∀ t : ℕ, 0 ≤ t → t ≤ time → ¬railway_crossing_closed t ∧ ¬traffic_light_red t) :=
sorry

end NUMINAMATH_GPT_shortest_travel_time_to_sunny_town_l1298_129800


namespace NUMINAMATH_GPT_cos_B_in_triangle_l1298_129880

theorem cos_B_in_triangle (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = Real.pi) : 
  Real.cos B = 1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_B_in_triangle_l1298_129880


namespace NUMINAMATH_GPT_percentage_for_overnight_stays_l1298_129856

noncomputable def total_bill : ℝ := 5000
noncomputable def medication_percentage : ℝ := 0.50
noncomputable def food_cost : ℝ := 175
noncomputable def ambulance_cost : ℝ := 1700

theorem percentage_for_overnight_stays :
  let medication_cost := medication_percentage * total_bill
  let remaining_bill := total_bill - medication_cost
  let cost_for_overnight_stays := remaining_bill - food_cost - ambulance_cost
  (cost_for_overnight_stays / remaining_bill) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_for_overnight_stays_l1298_129856


namespace NUMINAMATH_GPT_quotient_of_larger_divided_by_smaller_l1298_129828

theorem quotient_of_larger_divided_by_smaller
  (x y : ℕ)
  (h1 : x * y = 9375)
  (h2 : x + y = 400)
  (h3 : x > y) :
  x / y = 15 :=
sorry

end NUMINAMATH_GPT_quotient_of_larger_divided_by_smaller_l1298_129828


namespace NUMINAMATH_GPT_point_on_y_axis_m_value_l1298_129850

theorem point_on_y_axis_m_value (m : ℝ) (h : 6 - 2 * m = 0) : m = 3 := by
  sorry

end NUMINAMATH_GPT_point_on_y_axis_m_value_l1298_129850


namespace NUMINAMATH_GPT_inequality_solution_l1298_129846

theorem inequality_solution (x : ℝ) :
  4 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 → x ∈ Set.Ioc (5 / 2 : ℝ) (20 / 7 : ℝ) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1298_129846


namespace NUMINAMATH_GPT_retail_price_l1298_129875

theorem retail_price (W M : ℝ) (hW : W = 20) (hM : M = 80) : W + (M / 100) * W = 36 := by
  sorry

end NUMINAMATH_GPT_retail_price_l1298_129875


namespace NUMINAMATH_GPT_simplest_form_fraction_l1298_129889

theorem simplest_form_fraction 
  (m n a : ℤ) (h_f1 : (2 * m) / (10 * m * n) = 1 / (5 * n))
  (h_f2 : (m^2 - n^2) / (m + n) = (m - n))
  (h_f3 : (2 * a) / (a^2) = 2 / a) : 
  ∀ (f : ℤ), f = (m^2 + n^2) / (m + n) → 
    (∀ (k : ℤ), k ≠ 1 → (m^2 + n^2) / (m + n) ≠ k * f) :=
by
  intros f h_eq k h_kneq1
  sorry

end NUMINAMATH_GPT_simplest_form_fraction_l1298_129889


namespace NUMINAMATH_GPT_maximum_p_value_l1298_129871

noncomputable def max_p_value (a b c : ℝ) : ℝ :=
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1)

theorem maximum_p_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a + c = b) :
  ∃ p_max, p_max = 10 / 3 ∧ ∀ p, p = max_p_value a b c → p ≤ p_max :=
sorry

end NUMINAMATH_GPT_maximum_p_value_l1298_129871


namespace NUMINAMATH_GPT_total_notebooks_l1298_129837

-- Define the problem conditions
theorem total_notebooks (x : ℕ) (hx : x*x + 20 = (x+1)*(x+1) - 9) : x*x + 20 = 216 :=
by
  have h1 : x*x + 20 = 216 := sorry
  exact h1

end NUMINAMATH_GPT_total_notebooks_l1298_129837


namespace NUMINAMATH_GPT_dot_product_is_six_l1298_129803

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_is_six : (a.1 * b.1 + a.2 * b.2) = 6 := 
by 
  -- definition and proof logic follows
  sorry

end NUMINAMATH_GPT_dot_product_is_six_l1298_129803


namespace NUMINAMATH_GPT_xyz_inequality_l1298_129805

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z + x * y + y * z + z * x = 4) : x + y + z ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l1298_129805


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_l1298_129874

theorem rhombus_longer_diagonal (a b d : ℝ) (h : a = 65) (h_d : d = 56) :
  ∃ l, l = 118 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_l1298_129874


namespace NUMINAMATH_GPT_range_of_a_l1298_129876

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 5^x = a + 3) → a > -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1298_129876


namespace NUMINAMATH_GPT_triple_supplementary_angle_l1298_129865

theorem triple_supplementary_angle (x : ℝ) (hx : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end NUMINAMATH_GPT_triple_supplementary_angle_l1298_129865


namespace NUMINAMATH_GPT_total_wait_time_difference_l1298_129898

theorem total_wait_time_difference :
  let kids_swings := 6
  let kids_slide := 4 * kids_swings
  let wait_time_swings := [210, 420, 840] -- in seconds
  let total_wait_time_swings := wait_time_swings.sum
  let wait_time_slide := [45, 90, 180] -- in seconds
  let total_wait_time_slide := wait_time_slide.sum
  let total_wait_time_all_kids_swings := kids_swings * total_wait_time_swings
  let total_wait_time_all_kids_slide := kids_slide * total_wait_time_slide
  let difference := total_wait_time_all_kids_swings - total_wait_time_all_kids_slide
  difference = 1260 := sorry

end NUMINAMATH_GPT_total_wait_time_difference_l1298_129898


namespace NUMINAMATH_GPT_children_being_catered_l1298_129866

-- Define the total meal units available
def meal_units_for_adults : ℕ := 70
def meal_units_for_children : ℕ := 90
def meals_eaten_by_adults : ℕ := 14
def remaining_meal_units : ℕ := meal_units_for_adults - meals_eaten_by_adults

theorem children_being_catered :
  (remaining_meal_units * meal_units_for_children) / meal_units_for_adults = 72 := by
{
  sorry
}

end NUMINAMATH_GPT_children_being_catered_l1298_129866


namespace NUMINAMATH_GPT_masha_lives_on_seventh_floor_l1298_129890

/-- Masha lives in apartment No. 290, which is in the 4th entrance of a 17-story building.
The number of apartments is the same in all entrances of the building on all 17 floors; apartment numbers start from 1.
We need to prove that Masha lives on the 7th floor. -/
theorem masha_lives_on_seventh_floor 
  (n_apartments_per_floor : ℕ) 
  (total_floors : ℕ := 17) 
  (entrances : ℕ := 4) 
  (masha_apartment : ℕ := 290) 
  (start_apartment : ℕ := 1) 
  (h1 : (masha_apartment - start_apartment + 1) > 0) 
  (h2 : masha_apartment ≤ entrances * total_floors * n_apartments_per_floor)
  (h4 : masha_apartment > (entrances - 1) * total_floors * n_apartments_per_floor)  
   : ((masha_apartment - ((entrances - 1) * total_floors * n_apartments_per_floor) - 1) / n_apartments_per_floor) + 1 = 7 := 
by
  sorry

end NUMINAMATH_GPT_masha_lives_on_seventh_floor_l1298_129890


namespace NUMINAMATH_GPT_abigail_fence_building_l1298_129848

theorem abigail_fence_building :
  ∀ (initial_fences : Nat) (time_per_fence : Nat) (hours_building : Nat) (minutes_per_hour : Nat),
    initial_fences = 10 →
    time_per_fence = 30 →
    hours_building = 8 →
    minutes_per_hour = 60 →
    initial_fences + (minutes_per_hour / time_per_fence) * hours_building = 26 :=
by
  intros initial_fences time_per_fence hours_building minutes_per_hour
  sorry

end NUMINAMATH_GPT_abigail_fence_building_l1298_129848


namespace NUMINAMATH_GPT_find_function_f_l1298_129895

-- The function f maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- The statement to be proved
theorem find_function_f (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, (f m)^2 + f n ∣ (m^2 + n)^2) : ∀ n : ℕ+, f n = n :=
sorry

end NUMINAMATH_GPT_find_function_f_l1298_129895


namespace NUMINAMATH_GPT_sachin_rahul_age_ratio_l1298_129819

theorem sachin_rahul_age_ratio :
  ∀ (Sachin_age Rahul_age: ℕ),
    Sachin_age = 49 →
    Rahul_age = Sachin_age + 14 →
    Nat.gcd Sachin_age Rahul_age = 7 →
    (Sachin_age / Nat.gcd Sachin_age Rahul_age) = 7 ∧ (Rahul_age / Nat.gcd Sachin_age Rahul_age) = 9 :=
by
  intros Sachin_age Rahul_age h1 h2 h3
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_sachin_rahul_age_ratio_l1298_129819


namespace NUMINAMATH_GPT_min_area_triangle_ABC_l1298_129857

def point (α : Type*) := (α × α)

def area_of_triangle (A B C : point ℤ) : ℚ :=
  (1/2 : ℚ) * abs (36 * (C.snd) - 15 * (C.fst))

theorem min_area_triangle_ABC :
  ∃ (C : point ℤ), area_of_triangle (0, 0) (36, 15) C = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_area_triangle_ABC_l1298_129857


namespace NUMINAMATH_GPT_smallest_w_factor_l1298_129860

theorem smallest_w_factor:
  ∃ w : ℕ, (∃ n : ℕ, n = 936 * w ∧ 
              2 ^ 5 ∣ n ∧ 
              3 ^ 3 ∣ n ∧ 
              14 ^ 2 ∣ n) ∧ 
              w = 1764 :=
sorry

end NUMINAMATH_GPT_smallest_w_factor_l1298_129860


namespace NUMINAMATH_GPT_gifts_receiving_ribbon_l1298_129843

def total_ribbon := 18
def ribbon_per_gift := 2
def remaining_ribbon := 6

theorem gifts_receiving_ribbon : (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 := by
  sorry

end NUMINAMATH_GPT_gifts_receiving_ribbon_l1298_129843


namespace NUMINAMATH_GPT_sequence_property_l1298_129826

theorem sequence_property (a : ℕ+ → ℚ)
  (h1 : ∀ p q : ℕ+, a p + a q = a (p + q))
  (h2 : a 1 = 1 / 9) :
  a 36 = 4 :=
sorry

end NUMINAMATH_GPT_sequence_property_l1298_129826


namespace NUMINAMATH_GPT_find_range_of_a_l1298_129825

noncomputable def range_of_a (a : ℝ) (n : ℕ) : Prop :=
  1 + 1 / (n : ℝ) ≤ a ∧ a < 1 + 1 / ((n - 1) : ℝ)

theorem find_range_of_a (a : ℝ) (n : ℕ) (h1 : 1 < a) (h2 : 2 ≤ n) :
  (∃ x : ℕ, ∀ x₀ < x, (⌊a * (x₀ : ℝ)⌋ : ℝ) = x₀) ↔ range_of_a a n := by
  sorry

end NUMINAMATH_GPT_find_range_of_a_l1298_129825


namespace NUMINAMATH_GPT_max_area_of_rectangle_with_perimeter_40_l1298_129814

theorem max_area_of_rectangle_with_perimeter_40 :
  ∃ (A : ℝ), (A = 100) ∧ (∀ (length width : ℝ), 2 * (length + width) = 40 → length * width ≤ A) :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangle_with_perimeter_40_l1298_129814


namespace NUMINAMATH_GPT_vertex_of_parabola_find_shift_m_l1298_129858

-- Problem 1: Vertex of the given parabola
theorem vertex_of_parabola : 
  ∃ x y: ℝ, (y = 2 * x^2 + 4 * x - 6) ∧ (x, y) = (-1, -8) := 
by
  -- Proof goes here
  sorry

-- Problem 2: Finding the shift m
theorem find_shift_m (m : ℝ) (h : m > 0) : 
  (∀ x (hx : (x = (x + m)) ∧ (2 * x^2 + 4 * x - 6 = 0)), x = 1 ∨ x = -3) ∧ 
  ((-3 + m) = 0) → m = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_find_shift_m_l1298_129858


namespace NUMINAMATH_GPT_angle_between_AD_and_BC_l1298_129832

variables {a b c : ℝ} 
variables {θ : ℝ}
variables {α β γ δ ε ζ : ℝ} -- representing the angles

-- Conditions of the problem
def conditions (a b c : ℝ) (α β γ δ ε ζ : ℝ) : Prop :=
  (α + β + γ = 180) ∧ (δ + ε + ζ = 180) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Definition of the theorem to prove the angle between AD and BC
theorem angle_between_AD_and_BC
  (a b c : ℝ) (α β γ δ ε ζ : ℝ)
  (h : conditions a b c α β γ δ ε ζ) :
  θ = Real.arccos ((|b^2 - c^2|) / a^2) :=
sorry

end NUMINAMATH_GPT_angle_between_AD_and_BC_l1298_129832


namespace NUMINAMATH_GPT_arithmetic_geometric_seq_l1298_129840

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_seq_l1298_129840


namespace NUMINAMATH_GPT_number_of_machines_l1298_129861

theorem number_of_machines (X : ℕ)
  (h1 : 20 = (10 : ℝ) * X * 0.4) :
  X = 5 := sorry

end NUMINAMATH_GPT_number_of_machines_l1298_129861


namespace NUMINAMATH_GPT_cos_double_angle_l1298_129869

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l1298_129869


namespace NUMINAMATH_GPT_stone_travel_distance_l1298_129852

/-- Define the radii --/
def radius_fountain := 15
def radius_stone := 3

/-- Prove the distance the stone needs to travel along the fountain's edge --/
theorem stone_travel_distance :
  let circumference_fountain := 2 * Real.pi * ↑radius_fountain
  let circumference_stone := 2 * Real.pi * ↑radius_stone
  let distance_traveled := circumference_stone
  distance_traveled = 6 * Real.pi := by
  -- Placeholder for proof, based on conditions given
  sorry

end NUMINAMATH_GPT_stone_travel_distance_l1298_129852


namespace NUMINAMATH_GPT_inequality_solution_set_l1298_129839

theorem inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - 5 * x + b > 0) :
  ∀ x : ℝ, x < -1/3 ∨ x > 1/2 ↔ b * x^2 - 5 * x + a > 0 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1298_129839


namespace NUMINAMATH_GPT_arithmetic_expression_value_l1298_129810

theorem arithmetic_expression_value :
  2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_expression_value_l1298_129810


namespace NUMINAMATH_GPT_marcy_sip_amount_l1298_129806

theorem marcy_sip_amount (liters : ℕ) (ml_per_liter : ℕ) (total_minutes : ℕ) (interval_minutes : ℕ) (total_ml : ℕ) (total_sips : ℕ) (ml_per_sip : ℕ) 
  (h1 : liters = 2) 
  (h2 : ml_per_liter = 1000)
  (h3 : total_minutes = 250) 
  (h4 : interval_minutes = 5)
  (h5 : total_ml = liters * ml_per_liter)
  (h6 : total_sips = total_minutes / interval_minutes)
  (h7 : ml_per_sip = total_ml / total_sips) : 
  ml_per_sip = 40 := 
by
  sorry

end NUMINAMATH_GPT_marcy_sip_amount_l1298_129806


namespace NUMINAMATH_GPT_average_of_first_12_l1298_129884

theorem average_of_first_12 (avg25 : ℝ) (avg12 : ℝ) (avg_last12 : ℝ) (result_13th : ℝ) : 
  (avg25 = 18) → (avg_last12 = 17) → (result_13th = 78) → 
  25 * avg25 = (12 * avg12) + result_13th + (12 * avg_last12) → avg12 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_first_12_l1298_129884


namespace NUMINAMATH_GPT_cuboid_volume_l1298_129863

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 14) (h_height : height = 13) : base_area * height = 182 := by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1298_129863


namespace NUMINAMATH_GPT_time_to_chop_an_onion_is_4_minutes_l1298_129855

noncomputable def time_to_chop_pepper := 3
noncomputable def time_to_grate_cheese_per_omelet := 1
noncomputable def time_to_cook_omelet := 5
noncomputable def peppers_needed := 4
noncomputable def onions_needed := 2
noncomputable def omelets_needed := 5
noncomputable def total_time := 50

theorem time_to_chop_an_onion_is_4_minutes : 
  (total_time - (peppers_needed * time_to_chop_pepper + omelets_needed * time_to_grate_cheese_per_omelet + omelets_needed * time_to_cook_omelet)) / onions_needed = 4 := by sorry

end NUMINAMATH_GPT_time_to_chop_an_onion_is_4_minutes_l1298_129855


namespace NUMINAMATH_GPT_lap_time_improvement_l1298_129853

theorem lap_time_improvement (initial_laps : ℕ) (initial_time : ℕ) (current_laps : ℕ) (current_time : ℕ)
  (h1 : initial_laps = 15) (h2 : initial_time = 45) (h3 : current_laps = 18) (h4 : current_time = 42) :
  (45 / 15 - 42 / 18 : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_lap_time_improvement_l1298_129853


namespace NUMINAMATH_GPT_ca_co3_to_ca_cl2_l1298_129801

theorem ca_co3_to_ca_cl2 (caCO3 HCl : ℕ) (main_reaction : caCO3 = 1 ∧ HCl = 2) : ∃ CaCl2, CaCl2 = 1 :=
by
  -- The proof of the theorem will go here.
  sorry

end NUMINAMATH_GPT_ca_co3_to_ca_cl2_l1298_129801


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1298_129842

noncomputable def y (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem monotonic_decreasing_interval :
  {x : ℝ | (∃ y', y' = 3 * x^2 - 3 ∧ y' < 0)} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1298_129842


namespace NUMINAMATH_GPT_werewolf_knight_is_A_l1298_129867

structure Person :=
  (isKnight : Prop)
  (isLiar : Prop)
  (isWerewolf : Prop)

variables (A B C : Person)

-- A's statement: "At least one of us is a liar."
def statementA (A B C : Person) : Prop := A.isLiar ∨ B.isLiar ∨ C.isLiar

-- B's statement: "C is a knight."
def statementB (C : Person) : Prop := C.isKnight

theorem werewolf_knight_is_A (A B C : Person) 
  (hA : statementA A B C)
  (hB : statementB C)
  (hWerewolfKnight : ∃ x : Person, x.isWerewolf ∧ x.isKnight ∧ ¬ (A ≠ x ∧ B ≠ x ∧ C ≠ x))
  : A.isWerewolf ∧ A.isKnight :=
sorry

end NUMINAMATH_GPT_werewolf_knight_is_A_l1298_129867
