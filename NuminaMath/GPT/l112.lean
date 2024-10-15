import Mathlib

namespace NUMINAMATH_GPT_exists_no_zero_digits_divisible_by_2_pow_100_l112_11267

theorem exists_no_zero_digits_divisible_by_2_pow_100 :
  ∃ (N : ℕ), (2^100 ∣ N) ∧ (∀ d ∈ (N.digits 10), d ≠ 0) := sorry

end NUMINAMATH_GPT_exists_no_zero_digits_divisible_by_2_pow_100_l112_11267


namespace NUMINAMATH_GPT_negation_of_proposition_l112_11266

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l112_11266


namespace NUMINAMATH_GPT_subtraction_example_l112_11297

theorem subtraction_example : 6102 - 2016 = 4086 := by
  sorry

end NUMINAMATH_GPT_subtraction_example_l112_11297


namespace NUMINAMATH_GPT_cricket_bat_profit_percentage_l112_11278

-- Definitions for the problem conditions
def selling_price : ℝ := 850
def profit : ℝ := 255
def cost_price : ℝ := selling_price - profit
def expected_profit_percentage : ℝ := 42.86

-- The theorem to be proven
theorem cricket_bat_profit_percentage : 
  (profit / cost_price) * 100 = expected_profit_percentage :=
by 
  sorry

end NUMINAMATH_GPT_cricket_bat_profit_percentage_l112_11278


namespace NUMINAMATH_GPT_carlotta_total_time_l112_11290

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end NUMINAMATH_GPT_carlotta_total_time_l112_11290


namespace NUMINAMATH_GPT_darcy_folded_shorts_l112_11203

-- Define the conditions
def total_shirts : Nat := 20
def total_shorts : Nat := 8
def folded_shirts : Nat := 12
def remaining_pieces : Nat := 11

-- Expected result to prove
def folded_shorts : Nat := 5

-- The statement to prove
theorem darcy_folded_shorts : total_shorts - (remaining_pieces - (total_shirts - folded_shirts)) = folded_shorts :=
by
  sorry

end NUMINAMATH_GPT_darcy_folded_shorts_l112_11203


namespace NUMINAMATH_GPT_f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l112_11233

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (k x : ℝ) : ℝ := x^2 + k * x
noncomputable def a (x1 x2 : ℝ) : ℝ := (f x1 - f x2) / (x1 - x2)
noncomputable def b (z1 z2 k : ℝ) : ℝ := (g k z1 - g k z2) / (z1 - z2)

theorem f_is_increasing (x1 x2 : ℝ) (h : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) : a x1 x2 > 0 := by
  sorry

theorem exists_ratio_two (k : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = 2 * a x1 x2 := by
  sorry

theorem range_k_for_negative_two_ratio (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = -2 * a x1 x2) → k < -4 := by
  sorry

end NUMINAMATH_GPT_f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l112_11233


namespace NUMINAMATH_GPT_find_b1_b7_b10_value_l112_11271

open Classical

theorem find_b1_b7_b10_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith_seq : ∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2))
  (h_geom_seq : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r)
  (a3_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (b6_a6_eq : b 6 = a 6)
  (non_zero_seq : ∀ n : ℕ, a n ≠ 0) :
  b 1 * b 7 * b 10 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_b1_b7_b10_value_l112_11271


namespace NUMINAMATH_GPT_garden_least_cost_l112_11277

-- Define the costs per flower type
def cost_sunflower : ℝ := 0.75
def cost_tulip : ℝ := 2
def cost_marigold : ℝ := 1.25
def cost_orchid : ℝ := 4
def cost_violet : ℝ := 3.5

-- Define the areas of each section
def area_top_left : ℝ := 5 * 2
def area_bottom_left : ℝ := 5 * 5
def area_top_right : ℝ := 3 * 5
def area_bottom_right : ℝ := 3 * 4
def area_central_right : ℝ := 5 * 3

-- Calculate the total costs after assigning the most cost-effective layout
def total_cost : ℝ :=
  (area_top_left * cost_orchid) +
  (area_bottom_right * cost_violet) +
  (area_central_right * cost_tulip) +
  (area_bottom_left * cost_marigold) +
  (area_top_right * cost_sunflower)

-- Prove that the total cost is $154.50
theorem garden_least_cost : total_cost = 154.50 :=
by sorry

end NUMINAMATH_GPT_garden_least_cost_l112_11277


namespace NUMINAMATH_GPT_tan_30_degrees_correct_l112_11257

noncomputable def tan_30_degrees : ℝ := Real.tan (Real.pi / 6)

theorem tan_30_degrees_correct : tan_30_degrees = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_30_degrees_correct_l112_11257


namespace NUMINAMATH_GPT_probability_intersection_three_elements_l112_11262

theorem probability_intersection_three_elements (U : Finset ℕ) (hU : U = {1, 2, 3, 4, 5}) : 
  ∃ (p : ℚ), p = 5 / 62 :=
by
  sorry

end NUMINAMATH_GPT_probability_intersection_three_elements_l112_11262


namespace NUMINAMATH_GPT_min_product_of_three_l112_11227

theorem min_product_of_three :
  ∀ (list : List Int), 
    list = [-9, -7, -1, 2, 4, 6, 8] →
    ∃ (a b c : Int), a ∈ list ∧ b ∈ list ∧ c ∈ list ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ (x y z : Int), x ∈ list → y ∈ list → z ∈ list → x ≠ y → y ≠ z → x ≠ z → x * y * z ≥ a * b * c) ∧
    a * b * c = -432 :=
by
  sorry

end NUMINAMATH_GPT_min_product_of_three_l112_11227


namespace NUMINAMATH_GPT_mother_returns_home_at_8_05_l112_11272

noncomputable
def xiaoMing_home_time : Nat := 7 * 60 -- 7:00 AM in minutes
def xiaoMing_speed : Nat := 40 -- in meters per minute
def mother_home_time : Nat := 7 * 60 + 20 -- 7:20 AM in minutes
def meet_point : Nat := 1600 -- in meters
def stay_time : Nat := 5 -- in minutes
def return_duration_by_bike : Nat := 20 -- in minutes

theorem mother_returns_home_at_8_05 :
    (xiaoMing_home_time + (meet_point / xiaoMing_speed) + stay_time + return_duration_by_bike) = (8 * 60 + 5) :=
by
    sorry

end NUMINAMATH_GPT_mother_returns_home_at_8_05_l112_11272


namespace NUMINAMATH_GPT_log_neg_inequality_l112_11211

theorem log_neg_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  Real.log (-a) > Real.log (-b) := 
sorry

end NUMINAMATH_GPT_log_neg_inequality_l112_11211


namespace NUMINAMATH_GPT_tens_digit_of_sum_l112_11229

theorem tens_digit_of_sum (a b c : ℕ) (h : a = c + 3) (ha : 0 ≤ a ∧ a < 10) (hb : 0 ≤ b ∧ b < 10) (hc : 0 ≤ c ∧ c < 10) :
    ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ (202 * c + 20 * b + 303) % 100 = t ∧ t / 10 = 1 :=
by
  use (20 * b + 3)
  sorry

end NUMINAMATH_GPT_tens_digit_of_sum_l112_11229


namespace NUMINAMATH_GPT_xenia_weekly_earnings_l112_11268

theorem xenia_weekly_earnings
  (hours_week_1 : ℕ)
  (hours_week_2 : ℕ)
  (week2_additional_earnings : ℕ)
  (hours_week_3 : ℕ)
  (bonus_week_3 : ℕ)
  (hourly_wage : ℚ)
  (earnings_week_1 : ℚ)
  (earnings_week_2 : ℚ)
  (earnings_week_3 : ℚ)
  (total_earnings : ℚ) :
  hours_week_1 = 18 →
  hours_week_2 = 25 →
  week2_additional_earnings = 60 →
  hours_week_3 = 28 →
  bonus_week_3 = 30 →
  hourly_wage = (60 : ℚ) / (25 - 18) →
  earnings_week_1 = hours_week_1 * hourly_wage →
  earnings_week_2 = hours_week_2 * hourly_wage →
  earnings_week_2 = earnings_week_1 + 60 →
  earnings_week_3 = hours_week_3 * hourly_wage + 30 →
  total_earnings = earnings_week_1 + earnings_week_2 + earnings_week_3 →
  hourly_wage = (857 : ℚ) / 1000 ∧
  total_earnings = (63947 : ℚ) / 100
:= by
  intros h1 h2 h3 h4 h5 hw he1 he2 he2_60 he3 hte
  sorry

end NUMINAMATH_GPT_xenia_weekly_earnings_l112_11268


namespace NUMINAMATH_GPT_total_cookies_l112_11292

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1: num_people = 4) (h2: cookies_per_person = 22) : total_cookies = 88 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_l112_11292


namespace NUMINAMATH_GPT_negation_of_existential_l112_11212

theorem negation_of_existential :
  ¬ (∃ x : ℝ, x^2 - 2 * x - 3 < 0) ↔ ∀ x : ℝ, x^2 - 2 * x - 3 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_existential_l112_11212


namespace NUMINAMATH_GPT_max_students_with_equal_distribution_l112_11293

theorem max_students_with_equal_distribution (pens pencils : ℕ) (h_pens : pens = 3540) (h_pencils : pencils = 2860) :
  gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_max_students_with_equal_distribution_l112_11293


namespace NUMINAMATH_GPT_hyperbola_asymptotes_equation_l112_11279

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Now we assert the theorem that states this
theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y :=
by
  intros x y
  unfold hyperbola_asymptotes
  -- proof here
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_equation_l112_11279


namespace NUMINAMATH_GPT_fraction_simplification_l112_11246

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_simplification_l112_11246


namespace NUMINAMATH_GPT_total_students_is_48_l112_11206

-- Definitions according to the given conditions
def boys'_row := 24
def girls'_row := 24

-- Theorem based on the question and the correct answer
theorem total_students_is_48 :
  boys'_row + girls'_row = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_students_is_48_l112_11206


namespace NUMINAMATH_GPT_ninas_money_l112_11242

theorem ninas_money (C M : ℝ) (h1 : 6 * C = M) (h2 : 8 * (C - 1.15) = M) : M = 27.6 := 
by
  sorry

end NUMINAMATH_GPT_ninas_money_l112_11242


namespace NUMINAMATH_GPT_triangle_angle_solution_exists_l112_11220

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (A = 120 ∨ B = 120 ∨ C = 120) ∧
  (
    ((A = 40 ∧ B = 20) ∨ (A = 20 ∧ B = 40)) ∨
    ((A = 45 ∧ B = 15) ∨ (A = 15 ∧ B = 45))
  )
  
theorem triangle_angle_solution_exists :
  ∃ A B C : ℝ, possible_angles A B C :=
sorry

end NUMINAMATH_GPT_triangle_angle_solution_exists_l112_11220


namespace NUMINAMATH_GPT_inequality_selection_l112_11234

theorem inequality_selection (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) 
  (h₃ : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 ∧ (∀ x, |x + a| + |x - b| + c = 4 → x = (a - b)/2) ∧ (a = 8 / 7 ∧ b = 18 / 7 ∧ c = 2 / 7) :=
by
  sorry

end NUMINAMATH_GPT_inequality_selection_l112_11234


namespace NUMINAMATH_GPT_two_digit_numbers_count_l112_11245

theorem two_digit_numbers_count : 
  ∃ (count : ℕ), (
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b = 2 * a → 
      (10 * b + a = 7 / 4 * (10 * a + b))) 
      ∧ count = 4
  ) :=
sorry

end NUMINAMATH_GPT_two_digit_numbers_count_l112_11245


namespace NUMINAMATH_GPT_combined_weight_of_three_l112_11201

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end NUMINAMATH_GPT_combined_weight_of_three_l112_11201


namespace NUMINAMATH_GPT_total_bowling_balls_l112_11258

theorem total_bowling_balls (red_balls : ℕ) (green_balls : ℕ) (h1 : red_balls = 30) (h2 : green_balls = red_balls + 6) : red_balls + green_balls = 66 :=
by
  sorry

end NUMINAMATH_GPT_total_bowling_balls_l112_11258


namespace NUMINAMATH_GPT_sin_cos_identity_l112_11264

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l112_11264


namespace NUMINAMATH_GPT_number_of_books_in_library_l112_11291

def number_of_bookcases : ℕ := 28
def shelves_per_bookcase : ℕ := 6
def books_per_shelf : ℕ := 19

theorem number_of_books_in_library : number_of_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_in_library_l112_11291


namespace NUMINAMATH_GPT_machine_worked_minutes_l112_11236

theorem machine_worked_minutes
  (shirts_today : ℕ)
  (rate : ℕ)
  (h1 : shirts_today = 8)
  (h2 : rate = 2) :
  (shirts_today / rate) = 4 :=
by
  sorry

end NUMINAMATH_GPT_machine_worked_minutes_l112_11236


namespace NUMINAMATH_GPT_sum_formula_l112_11223

open Nat

/-- The sequence a_n defined as (-1)^n * (2 * n - 1) -/
def a_n (n : ℕ) : ℤ :=
  (-1) ^ n * (2 * n - 1)

/-- The partial sum S_n of the first n terms of the sequence a_n -/
def S_n : ℕ → ℤ
| 0     => 0
| (n+1) => S_n n + a_n (n + 1)

/-- The main theorem: For all n in natural numbers, S_n = (-1)^n * n -/
theorem sum_formula (n : ℕ) : S_n n = (-1) ^ n * n := by
  sorry

end NUMINAMATH_GPT_sum_formula_l112_11223


namespace NUMINAMATH_GPT_find_W_from_conditions_l112_11222

theorem find_W_from_conditions :
  ∀ (x y : ℝ), (y = 1 / x ∧ y = |x| + 1) → (x + y = Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_find_W_from_conditions_l112_11222


namespace NUMINAMATH_GPT_albania_inequality_l112_11208

variable (a b c r R s : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)
variable (h4 : r > 0)
variable (h5 : R > 0)
variable (h6 : s = (a + b + c) / 2)

theorem albania_inequality :
    1 / (a + b) + 1 / (a + c) + 1 / (b + c) ≤ r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s) :=
sorry

end NUMINAMATH_GPT_albania_inequality_l112_11208


namespace NUMINAMATH_GPT_amount_of_bill_correct_l112_11251

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 421.7142857142857
noncomputable def computeFV (TD BD : ℝ) := (TD * BD) / (BD - TD)

theorem amount_of_bill_correct :
  computeFV TD BD = 2460 := 
sorry

end NUMINAMATH_GPT_amount_of_bill_correct_l112_11251


namespace NUMINAMATH_GPT_years_later_l112_11294

variables (R F Y : ℕ)

-- Conditions
def condition1 := F = 4 * R
def condition2 := F + Y = 5 * (R + Y) / 2
def condition3 := F + Y + 8 = 2 * (R + Y + 8)

-- The result to be proved
theorem years_later (R F Y : ℕ) (h1 : condition1 R F) (h2 : condition2 R F Y) (h3 : condition3 R F Y) : 
  Y = 8 := by
  sorry

end NUMINAMATH_GPT_years_later_l112_11294


namespace NUMINAMATH_GPT_first_half_speed_l112_11265

noncomputable def speed_first_half : ℝ := 21

theorem first_half_speed (total_distance first_half_distance second_half_distance second_half_speed total_time : ℝ)
  (h1 : total_distance = 224)
  (h2 : first_half_distance = total_distance / 2)
  (h3 : second_half_distance = total_distance / 2)
  (h4 : second_half_speed = 24)
  (h5 : total_time = 10)
  (h6 : total_time = first_half_distance / speed_first_half + second_half_distance / second_half_speed) :
  speed_first_half = 21 :=
sorry

end NUMINAMATH_GPT_first_half_speed_l112_11265


namespace NUMINAMATH_GPT_max_distance_traveled_l112_11207

theorem max_distance_traveled (fare: ℝ) (x: ℝ) :
  fare = 17.2 → 
  x > 3 →
  1.4 * (x - 3) + 6 ≤ fare → 
  x ≤ 11 := by
  sorry

end NUMINAMATH_GPT_max_distance_traveled_l112_11207


namespace NUMINAMATH_GPT_initial_order_cogs_l112_11231

theorem initial_order_cogs (x : ℕ) (h : (x + 60 : ℚ) / (x / 36 + 1) = 45) : x = 60 := 
sorry

end NUMINAMATH_GPT_initial_order_cogs_l112_11231


namespace NUMINAMATH_GPT_find_values_l112_11249

noncomputable def equation_satisfaction (x y : ℝ) : Prop :=
  x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3

theorem find_values (x y : ℝ) :
  equation_satisfaction x y → x = 1 / 3 ∧ y = 2 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_values_l112_11249


namespace NUMINAMATH_GPT_eval_f_at_10_l112_11295

def f (x : ℚ) : ℚ := (6 * x + 3) / (x - 2)

theorem eval_f_at_10 : f 10 = 63 / 8 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_at_10_l112_11295


namespace NUMINAMATH_GPT_henry_finishes_on_thursday_l112_11219

theorem henry_finishes_on_thursday :
  let total_days := 210
  let start_day := 4  -- Assume Thursday is 4th day of the week in 0-indexed (0=Sunday, 1=Monday, ..., 6=Saturday)
  (start_day + total_days) % 7 = start_day :=
by
  sorry

end NUMINAMATH_GPT_henry_finishes_on_thursday_l112_11219


namespace NUMINAMATH_GPT_verify_triangle_inequality_l112_11224

-- Conditions of the problem
variables (L : ℕ → ℕ)
-- The rods lengths are arranged in increasing order
axiom rods_in_order : ∀ i : ℕ, L i ≤ L (i + 1)

-- Define the critical check
def critical_check : Prop :=
  L 98 + L 99 > L 100

-- Prove that verifying the critical_check is sufficient
theorem verify_triangle_inequality (h : critical_check L) :
  ∀ i j k : ℕ, 1 ≤ i → i < j → j < k → k ≤ 100 → L i + L j > L k :=
by
  sorry

end NUMINAMATH_GPT_verify_triangle_inequality_l112_11224


namespace NUMINAMATH_GPT_prob_of_caps_given_sunglasses_l112_11286

theorem prob_of_caps_given_sunglasses (n_sunglasses n_caps n_both : ℕ) (P_sunglasses_given_caps : ℚ) 
  (h_nsunglasses : n_sunglasses = 80) (h_ncaps : n_caps = 45)
  (h_Psunglasses_given_caps : P_sunglasses_given_caps = 3/8)
  (h_nboth : n_both = P_sunglasses_given_caps * n_sunglasses) :
  (n_both / n_caps) = 2/3 := 
by
  sorry

end NUMINAMATH_GPT_prob_of_caps_given_sunglasses_l112_11286


namespace NUMINAMATH_GPT_range_for_a_l112_11200

theorem range_for_a (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  sorry

end NUMINAMATH_GPT_range_for_a_l112_11200


namespace NUMINAMATH_GPT_no_integer_solutions_for_trapezoid_bases_l112_11235

theorem no_integer_solutions_for_trapezoid_bases :
  ∃ (A h : ℤ) (b1_b2 : ℤ → Prop),
    A = 2800 ∧ h = 80 ∧
    (∀ m n : ℤ, b1_b2 (12 * m) ∧ b1_b2 (12 * n) → (12 * m + 12 * n = 70) → false) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_for_trapezoid_bases_l112_11235


namespace NUMINAMATH_GPT_joe_initial_cars_l112_11221

theorem joe_initial_cars (x : ℕ) (h : x + 12 = 62) : x = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_joe_initial_cars_l112_11221


namespace NUMINAMATH_GPT_remove_toothpicks_l112_11274

-- Definitions based on problem conditions
def toothpicks := 40
def triangles := 40
def initial_triangulation := True
def additional_condition := True

-- Statement to be proved
theorem remove_toothpicks :
  initial_triangulation ∧ additional_condition ∧ (triangles > 40) → ∃ (t: ℕ), t = 15 :=
by
  sorry

end NUMINAMATH_GPT_remove_toothpicks_l112_11274


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l112_11285

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ y : ℝ, y = a^0 + 3 ∧ (0, y) = (0, 4)) :=
by
  use 4
  have h : a^0 = 1 := by simp
  rw [h]
  simp
  sorry

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l112_11285


namespace NUMINAMATH_GPT_mediant_fraction_of_6_11_and_5_9_minimized_is_31_l112_11259

theorem mediant_fraction_of_6_11_and_5_9_minimized_is_31 
  (p q : ℕ) (h_pos : 0 < p ∧ 0 < q)
  (h_bounds : (6 : ℝ) / 11 < p / q ∧ p / q < 5 / 9)
  (h_min_q : ∀ r s : ℕ, (6 : ℝ) / 11 < r / s ∧ r / s < 5 / 9 → s ≥ q) :
  p + q = 31 :=
sorry

end NUMINAMATH_GPT_mediant_fraction_of_6_11_and_5_9_minimized_is_31_l112_11259


namespace NUMINAMATH_GPT_uncle_welly_roses_l112_11282

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end NUMINAMATH_GPT_uncle_welly_roses_l112_11282


namespace NUMINAMATH_GPT_sin_cos_special_l112_11213

def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem sin_cos_special (x : ℝ) : 
  special_operation (Real.sin (x / 12)) (Real.cos (x / 12)) = -(1 + 2 * Real.sqrt 3) / 4 :=
  sorry

end NUMINAMATH_GPT_sin_cos_special_l112_11213


namespace NUMINAMATH_GPT_square_side_lengths_l112_11252

theorem square_side_lengths (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 120) :
  (x = 13 ∧ y = 7) ∨ (x = 7 ∧ y = 13) :=
by {
  -- skip proof
  sorry
}

end NUMINAMATH_GPT_square_side_lengths_l112_11252


namespace NUMINAMATH_GPT_solve_equation_l112_11269

theorem solve_equation (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) → x = -0.5 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l112_11269


namespace NUMINAMATH_GPT_rope_segments_l112_11210

theorem rope_segments (total_length : ℝ) (n : ℕ) (h1 : total_length = 3) (h2 : n = 7) :
  (∃ segment_fraction : ℝ, segment_fraction = 1 / n ∧
   ∃ segment_length : ℝ, segment_length = total_length / n) :=
sorry

end NUMINAMATH_GPT_rope_segments_l112_11210


namespace NUMINAMATH_GPT_imo_hosting_arrangements_l112_11241

structure IMOCompetition where
  countries : Finset String
  continents : Finset String
  assignments : Finset (String × String)
  constraints : String → String
  assignments_must_be_unique : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                 (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                 constraints c1 ≠ constraints c2 → c1 ≠ c2
  no_consecutive_same_continent : ∀ {c1 c2 : String} {cnt1 cnt2 : String},
                                   (c1, cnt1) ∈ assignments → (c2, cnt2) ∈ assignments → 
                                   (c1, cnt1) ≠ (c2, cnt2) →
                                   constraints c1 ≠ constraints c2

def number_of_valid_arrangements (comp: IMOCompetition) : Nat := 240

theorem imo_hosting_arrangements (comp : IMOCompetition) :
  number_of_valid_arrangements comp = 240 := by
  sorry

end NUMINAMATH_GPT_imo_hosting_arrangements_l112_11241


namespace NUMINAMATH_GPT_miles_from_second_friend_to_work_l112_11254
variable (distance_to_first_friend := 8)
variable (distance_to_second_friend := distance_to_first_friend / 2)
variable (total_distance_to_second_friend := distance_to_first_friend + distance_to_second_friend)
variable (distance_to_work := 3 * total_distance_to_second_friend)

theorem miles_from_second_friend_to_work :
  distance_to_work = 36 := 
by
  sorry

end NUMINAMATH_GPT_miles_from_second_friend_to_work_l112_11254


namespace NUMINAMATH_GPT_exists_nonneg_poly_div_l112_11281

theorem exists_nonneg_poly_div (P : Polynomial ℝ) 
  (hP_pos : ∀ x : ℝ, x > 0 → P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ n, Q.coeff n ≥ 0) ∧ (∀ n, R.coeff n ≥ 0) ∧ (P = Q / R) := 
sorry

end NUMINAMATH_GPT_exists_nonneg_poly_div_l112_11281


namespace NUMINAMATH_GPT_highway_length_on_map_l112_11261

theorem highway_length_on_map (total_length_km : ℕ) (scale : ℚ) (length_on_map_cm : ℚ) 
  (h1 : total_length_km = 155) (h2 : scale = 1 / 500000) :
  length_on_map_cm = 31 :=
by
  sorry

end NUMINAMATH_GPT_highway_length_on_map_l112_11261


namespace NUMINAMATH_GPT_percentage_increase_l112_11209

theorem percentage_increase (original new : ℝ) (h₁ : original = 50) (h₂ : new = 80) :
  ((new - original) / original) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l112_11209


namespace NUMINAMATH_GPT_quadratic_coefficient_c_l112_11232

theorem quadratic_coefficient_c (b c: ℝ) 
  (h_sum: 12 = b) (h_prod: 20 = c) : 
  c = 20 := 
by sorry

end NUMINAMATH_GPT_quadratic_coefficient_c_l112_11232


namespace NUMINAMATH_GPT_minimum_tangent_length_l112_11248

theorem minimum_tangent_length
  (a b : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0)
  (h_symmetry : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x + b * y + 6 = 0) :
  ∃ t : ℝ, t = 2 :=
by sorry

end NUMINAMATH_GPT_minimum_tangent_length_l112_11248


namespace NUMINAMATH_GPT_jon_toaster_total_cost_l112_11228

def total_cost_toaster (MSRP : ℝ) (std_ins_pct : ℝ) (premium_upgrade_cost : ℝ) (state_tax_pct : ℝ) (environmental_fee : ℝ) : ℝ :=
  let std_ins_cost := std_ins_pct * MSRP
  let premium_ins_cost := std_ins_cost + premium_upgrade_cost
  let subtotal_before_tax := MSRP + premium_ins_cost
  let state_tax := state_tax_pct * subtotal_before_tax
  let total_before_env_fee := subtotal_before_tax + state_tax
  total_before_env_fee + environmental_fee

theorem jon_toaster_total_cost :
  total_cost_toaster 30 0.2 7 0.5 5 = 69.5 :=
by
  sorry

end NUMINAMATH_GPT_jon_toaster_total_cost_l112_11228


namespace NUMINAMATH_GPT_test_point_third_l112_11270

def interval := (1000, 2000)
def phi := 0.618
def x1 := 1000 + phi * (2000 - 1000)
def x2 := 1000 + 2000 - x1

-- By definition and given the conditions, x3 is computed in a specific manner
def x3 := x2 + 2000 - x1

theorem test_point_third : x3 = 1764 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_test_point_third_l112_11270


namespace NUMINAMATH_GPT_conditional_probability_l112_11214

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end NUMINAMATH_GPT_conditional_probability_l112_11214


namespace NUMINAMATH_GPT_arithmetic_sequences_sum_l112_11255

theorem arithmetic_sequences_sum
  (a b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d1)
  (h2 : ∀ n, b (n + 1) = b n + d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequences_sum_l112_11255


namespace NUMINAMATH_GPT_jack_total_plates_after_smashing_and_buying_l112_11217

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1
def new_polka_dotted_plates : ℕ := initial_checked_plates * initial_checked_plates

theorem jack_total_plates_after_smashing_and_buying : 
  initial_flower_plates - smashed_flower_plates
  + initial_checked_plates
  + initial_striped_plates - smashed_striped_plates
  + new_polka_dotted_plates = 96 := 
by {
  -- calculation proof here
  sorry
}

end NUMINAMATH_GPT_jack_total_plates_after_smashing_and_buying_l112_11217


namespace NUMINAMATH_GPT_prob_2_lt_X_le_4_l112_11280

-- Define the PMF of the random variable X
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if h : k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the probability that X lies in the range (2, 4]
noncomputable def P_2_lt_X_le_4 : ℝ :=
  pmf_X 3 + pmf_X 4

-- Theorem stating the probability of x lying in (2, 4) is 3/16.
theorem prob_2_lt_X_le_4 : P_2_lt_X_le_4 = 3 / 16 := 
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_prob_2_lt_X_le_4_l112_11280


namespace NUMINAMATH_GPT_three_people_on_staircase_l112_11202

theorem three_people_on_staircase (A B C : Type) (steps : Finset ℕ) (h1 : steps.card = 7) 
  (h2 : ∀ step ∈ steps, step ≤ 2) : 
  ∃ (total_ways : ℕ), total_ways = 336 :=
by {
  sorry
}

end NUMINAMATH_GPT_three_people_on_staircase_l112_11202


namespace NUMINAMATH_GPT_deepak_present_age_l112_11230

def present_age_rahul (x : ℕ) : ℕ := 4 * x
def present_age_deepak (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age : ∀ (x : ℕ), 
  (present_age_rahul x + 22 = 26) →
  present_age_deepak x = 3 := 
by
  intros x h
  sorry

end NUMINAMATH_GPT_deepak_present_age_l112_11230


namespace NUMINAMATH_GPT_difference_in_students_specific_case_diff_l112_11244

-- Define the variables and conditions
variables (a b : ℕ)

-- Condition: a > b
axiom h1 : a > b

-- Definition of eighth grade students
def eighth_grade_students := (3 * a + b) * (2 * a + 2 * b)

-- Definition of seventh grade students
def seventh_grade_students := (2 * (a + b)) ^ 2

-- Theorem for the difference in the number of students
theorem difference_in_students : (eighth_grade_students a b) - (seventh_grade_students a b) = 2 * a^2 - 2 * b^2 :=
sorry

-- Theorem for the specific example when a = 10 and b = 2
theorem specific_case_diff : eighth_grade_students 10 2 - seventh_grade_students 10 2 = 192 :=
sorry

end NUMINAMATH_GPT_difference_in_students_specific_case_diff_l112_11244


namespace NUMINAMATH_GPT_find_x_l112_11260

theorem find_x (x y z : ℕ) (h_pos : 0 < x) (h_pos : 0 < y) (h_pos : 0 < z) (h_eq1 : x + y + z = 37) (h_eq2 : 5 * y = 6 * z) : x = 21 :=
sorry

end NUMINAMATH_GPT_find_x_l112_11260


namespace NUMINAMATH_GPT_compare_M_N_l112_11289

theorem compare_M_N (a b c : ℝ) (h1 : a > 0) (h2 : b < -2 * a) : 
  (|a - b + c| + |2 * a + b|) < (|a + b + c| + |2 * a - b|) :=
by
  sorry

end NUMINAMATH_GPT_compare_M_N_l112_11289


namespace NUMINAMATH_GPT_goblin_treasure_l112_11296

theorem goblin_treasure : 
  (∃ d : ℕ, 8000 + 300 * d = 5000 + 500 * d) ↔ ∃ (d : ℕ), d = 15 :=
by
  sorry

end NUMINAMATH_GPT_goblin_treasure_l112_11296


namespace NUMINAMATH_GPT_find_a8_l112_11240

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) = a n / n) (h2 : a 5 = 15) : a 8 = 24 :=
sorry

end NUMINAMATH_GPT_find_a8_l112_11240


namespace NUMINAMATH_GPT_imaginary_part_inv_z_l112_11256

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_inv_z : Complex.im (1 / z) = 2 / 5 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_imaginary_part_inv_z_l112_11256


namespace NUMINAMATH_GPT_find_eq_thirteen_l112_11243

open Real

theorem find_eq_thirteen
  (a x b y c z : ℝ)
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 6) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := 
sorry

end NUMINAMATH_GPT_find_eq_thirteen_l112_11243


namespace NUMINAMATH_GPT_inequality_of_f_log2015_l112_11287

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_of_f_log2015 :
  (∀ x : ℝ, deriv f x > f x) →
  f (Real.log 2015) > 2015 * f 0 :=
by sorry

end NUMINAMATH_GPT_inequality_of_f_log2015_l112_11287


namespace NUMINAMATH_GPT_simplify_fraction_l112_11215

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end NUMINAMATH_GPT_simplify_fraction_l112_11215


namespace NUMINAMATH_GPT_badminton_members_count_l112_11299

def total_members := 30
def neither_members := 2
def both_members := 6

def members_play_badminton_and_tennis (B T : ℕ) : Prop :=
  B + T - both_members = total_members - neither_members

theorem badminton_members_count (B T : ℕ) (hbt : B = T) :
  members_play_badminton_and_tennis B T → B = 17 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_badminton_members_count_l112_11299


namespace NUMINAMATH_GPT_conic_sections_ab_value_l112_11283

theorem conic_sections_ab_value
  (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
by
  -- Proof will be filled in later
  sorry

end NUMINAMATH_GPT_conic_sections_ab_value_l112_11283


namespace NUMINAMATH_GPT_ellipse_equation_parabola_equation_l112_11276

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  a = 6 → b = 2 * Real.sqrt 5 → c = 4 → 
  ((∀ x y : ℝ, (y^2 / 36) + (x^2 / 20) = 1))

noncomputable def parabola_standard_equation (focus_x focus_y : ℝ) : Prop :=
  focus_x = 3 → focus_y = 0 → 
  (∀ x y : ℝ, y^2 = 12 * x)

theorem ellipse_equation : ellipse_standard_equation 6 (2 * Real.sqrt 5) 4 := by
  sorry

theorem parabola_equation : parabola_standard_equation 3 0 := by
  sorry

end NUMINAMATH_GPT_ellipse_equation_parabola_equation_l112_11276


namespace NUMINAMATH_GPT_range_of_a_l112_11298

theorem range_of_a (a x : ℝ) (h_eq : 2 * x - 1 = x + a) (h_pos : x > 0) : a > -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l112_11298


namespace NUMINAMATH_GPT_distance_down_correct_l112_11247

-- Conditions
def rate_up : ℕ := 5  -- rate on the way up (miles per day)
def time_up : ℕ := 2  -- time to travel up (days)
def rate_factor : ℕ := 3 / 2  -- factor for the rate on the way down
def time_down := time_up  -- time to travel down is the same

-- Formula for computation
def distance_up : ℕ := rate_up * time_up
def rate_down : ℕ := rate_up * rate_factor
def distance_down : ℕ := rate_down * time_down

-- Theorem to be proved
theorem distance_down_correct : distance_down = 15 := by
  sorry

end NUMINAMATH_GPT_distance_down_correct_l112_11247


namespace NUMINAMATH_GPT_isosceles_triangle_area_l112_11225

noncomputable def area_of_isosceles_triangle (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20) : ℝ :=
  1 / 2 * (2 * b) * 10

theorem isosceles_triangle_area (b s : ℝ) (h1 : 10 * 10 + b * b = s * s) (h2 : s + b = 20)
  (h3 : 2 * s + 2 * b = 40) : area_of_isosceles_triangle b s h1 h2 = 75 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l112_11225


namespace NUMINAMATH_GPT_heptagon_labeling_impossible_l112_11273

/-- 
  Let a heptagon be given with vertices labeled by integers a₁, a₂, a₃, a₄, a₅, a₆, a₇.
  The following two conditions are imposed:
  1. For every pair of consecutive vertices (aᵢ, aᵢ₊₁) (with indices mod 7), 
     at least one of aᵢ and aᵢ₊₁ divides the other.
  2. For every pair of non-consecutive vertices (aᵢ, aⱼ) where i ≠ j ± 1 mod 7, 
     neither aᵢ divides aⱼ nor aⱼ divides aᵢ. 

  Prove that such a labeling is impossible.
-/
theorem heptagon_labeling_impossible :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) ∧
    (∀ {i j : Fin 7}, (i ≠ j + 1 % 7) → (i ≠ j + 6 % 7) → ¬ (a i ∣ a j) ∧ ¬ (a j ∣ a i)) :=
sorry

end NUMINAMATH_GPT_heptagon_labeling_impossible_l112_11273


namespace NUMINAMATH_GPT_valid_range_of_x_l112_11216

theorem valid_range_of_x (x : ℝ) (h1 : 2 - x ≥ 0) (h2 : x + 1 ≠ 0) : x ≤ 2 ∧ x ≠ -1 :=
sorry

end NUMINAMATH_GPT_valid_range_of_x_l112_11216


namespace NUMINAMATH_GPT_set_listing_method_l112_11237

theorem set_listing_method :
  {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 5} = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_listing_method_l112_11237


namespace NUMINAMATH_GPT_range_of_a_minus_b_l112_11205

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : 1 < b) (h₄ : b < 3) : 
  -4 < a - b ∧ a - b < 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l112_11205


namespace NUMINAMATH_GPT_mean_equality_l112_11218

theorem mean_equality (y : ℝ) : 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_mean_equality_l112_11218


namespace NUMINAMATH_GPT_find_positive_x_l112_11204

theorem find_positive_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z) (h3 : x * z = 40 - 5 * x - 3 * z) :
  x = 3 :=
by sorry

end NUMINAMATH_GPT_find_positive_x_l112_11204


namespace NUMINAMATH_GPT_domain_ln_l112_11238

theorem domain_ln (x : ℝ) (h : x - 1 > 0) : x > 1 := 
sorry

end NUMINAMATH_GPT_domain_ln_l112_11238


namespace NUMINAMATH_GPT_Marty_combinations_l112_11253

theorem Marty_combinations : 
  let colors := 4
  let decorations := 3
  colors * decorations = 12 :=
by
  sorry

end NUMINAMATH_GPT_Marty_combinations_l112_11253


namespace NUMINAMATH_GPT_equal_intercepts_line_l112_11288

theorem equal_intercepts_line (x y : ℝ)
  (h1 : x + 2*y - 6 = 0) 
  (h2 : x - 2*y + 2 = 0) 
  (hx : x = 2) 
  (hy : y = 2) :
  (y = x) ∨ (x + y = 4) :=
sorry

end NUMINAMATH_GPT_equal_intercepts_line_l112_11288


namespace NUMINAMATH_GPT_hydropump_output_l112_11226

theorem hydropump_output :
  ∀ (rate : ℕ) (time_hours : ℚ), 
    rate = 600 → 
    time_hours = 1.5 → 
    rate * time_hours = 900 :=
by
  intros rate time_hours rate_cond time_cond 
  sorry

end NUMINAMATH_GPT_hydropump_output_l112_11226


namespace NUMINAMATH_GPT_pool_drain_rate_l112_11263

-- Define the dimensions and other conditions
def poolLength : ℝ := 150
def poolWidth : ℝ := 40
def poolDepth : ℝ := 10
def poolCapacityPercent : ℝ := 0.80
def drainTime : ℕ := 800

-- Define the problem statement
theorem pool_drain_rate :
  let fullVolume := poolLength * poolWidth * poolDepth
  let volumeAt80Percent := fullVolume * poolCapacityPercent
  let drainRate := volumeAt80Percent / drainTime
  drainRate = 60 :=
by
  sorry

end NUMINAMATH_GPT_pool_drain_rate_l112_11263


namespace NUMINAMATH_GPT_find_cost_price_l112_11239

/-- Statement: Given Mohit sold an article for $18000 and 
if he offered a discount of 10% on the selling price, he would have earned a profit of 8%, 
prove that the cost price (CP) of the article is $15000. -/

def discounted_price (sp : ℝ) := sp - (0.10 * sp)
def profit_price (cp : ℝ) := cp * 1.08

theorem find_cost_price (sp : ℝ) (discount: sp = 18000) (profit_discount: profit_price (discounted_price sp) = discounted_price sp):
    ∃ (cp : ℝ), cp = 15000 :=
by
    sorry

end NUMINAMATH_GPT_find_cost_price_l112_11239


namespace NUMINAMATH_GPT_increasing_function_range_a_l112_11250

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem increasing_function_range_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 1 < a ∧ a ≤ 5 / 3 :=
sorry

end NUMINAMATH_GPT_increasing_function_range_a_l112_11250


namespace NUMINAMATH_GPT_gcd_180_450_l112_11275

theorem gcd_180_450 : Nat.gcd 180 450 = 90 :=
by
  sorry

end NUMINAMATH_GPT_gcd_180_450_l112_11275


namespace NUMINAMATH_GPT_no_valid_n_l112_11284

theorem no_valid_n (n : ℕ) (h₁ : 100 ≤ n / 4) (h₂ : n / 4 ≤ 999) (h₃ : 100 ≤ 4 * n) (h₄ : 4 * n ≤ 999) : false := by
  sorry

end NUMINAMATH_GPT_no_valid_n_l112_11284
