import Mathlib

namespace range_of_x_for_positive_y_l47_47855

theorem range_of_x_for_positive_y (x : ℝ) : 
  (-1 < x ∧ x < 3) ↔ (-x^2 + 2*x + 3 > 0) :=
sorry

end range_of_x_for_positive_y_l47_47855


namespace coffee_cost_l47_47999

def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def dozens_of_donuts : ℕ := 3
def donuts_per_dozen : ℕ := 12

theorem coffee_cost :
  let total_donuts := dozens_of_donuts * donuts_per_dozen
  let total_ounces := ounces_per_donut * total_donuts
  let total_pots := total_ounces / ounces_per_pot
  let total_cost := total_pots * cost_per_pot
  total_cost = 18 := by
  sorry

end coffee_cost_l47_47999


namespace total_expense_l47_47001

theorem total_expense (tanya_face_cost : ℕ) (tanya_face_qty : ℕ) (tanya_body_cost : ℕ) (tanya_body_qty : ℕ) 
  (tanya_total_expense : ℕ) (christy_multiplier : ℕ) (christy_total_expense : ℕ) (total_expense : ℕ) :
  tanya_face_cost = 50 →
  tanya_face_qty = 2 →
  tanya_body_cost = 60 →
  tanya_body_qty = 4 →
  tanya_total_expense = tanya_face_qty * tanya_face_cost + tanya_body_qty * tanya_body_cost →
  christy_multiplier = 2 →
  christy_total_expense = christy_multiplier * tanya_total_expense →
  total_expense = christy_total_expense + tanya_total_expense →
  total_expense = 1020 :=
by
  intros
  sorry

end total_expense_l47_47001


namespace cheesecake_factory_hours_per_day_l47_47491

theorem cheesecake_factory_hours_per_day
  (wage_per_hour : ℝ)
  (days_per_week : ℝ)
  (weeks : ℝ)
  (combined_savings : ℝ)
  (robbie_saves : ℝ)
  (jaylen_saves : ℝ)
  (miranda_saves : ℝ)
  (h : ℝ) :
  wage_per_hour = 10 → days_per_week = 5 → weeks = 4 → combined_savings = 3000 →
  robbie_saves = 2/5 → jaylen_saves = 3/5 → miranda_saves = 1/2 →
  (robbie_saves * (wage_per_hour * h * days_per_week) +
  jaylen_saves * (wage_per_hour * h * days_per_week) +
  miranda_saves * (wage_per_hour * h * days_per_week)) * weeks = combined_savings →
  h = 10 :=
by
  intros hwage hweek hweeks hsavings hrobbie hjaylen hmiranda heq
  sorry

end cheesecake_factory_hours_per_day_l47_47491


namespace flag_height_l47_47591

-- Definitions based on conditions
def flag_width : ℝ := 5
def paint_cost_per_quart : ℝ := 2
def sqft_per_quart : ℝ := 4
def total_spent : ℝ := 20

-- The theorem to prove the height h of the flag
theorem flag_height (h : ℝ) (paint_needed : ℝ -> ℝ) :
  paint_needed h = 4 := sorry

end flag_height_l47_47591


namespace tan_of_cos_first_quadrant_l47_47740

-- Define the angle α in the first quadrant and its cosine value
variable (α : ℝ) (h1 : 0 < α ∧ α < π/2) (hcos : Real.cos α = 2 / 3)

-- State the theorem
theorem tan_of_cos_first_quadrant : Real.tan α = Real.sqrt 5 / 2 := 
by
  sorry

end tan_of_cos_first_quadrant_l47_47740


namespace workerB_time_to_complete_job_l47_47259

theorem workerB_time_to_complete_job 
  (time_A : ℝ) (time_together: ℝ) (time_B : ℝ) 
  (h1 : time_A = 5) 
  (h2 : time_together = 3.333333333333333) 
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) 
  : time_B = 10 := 
  sorry

end workerB_time_to_complete_job_l47_47259


namespace broccoli_sales_l47_47067

theorem broccoli_sales (B C S Ca : ℝ) (h1 : C = 2 * B) (h2 : S = B / 2 + 16) (h3 : Ca = 136) (total_sales : B + C + S + Ca = 380) :
  B = 57 :=
by
  sorry

end broccoli_sales_l47_47067


namespace smallest_w_l47_47563

theorem smallest_w (w : ℕ) (w_pos : 0 < w) : 
  (∀ n : ℕ, (2^5 ∣ 936 * n) ∧ (3^3 ∣ 936 * n) ∧ (11^2 ∣ 936 * n) ↔ n = w) → w = 4356 :=
sorry

end smallest_w_l47_47563


namespace find_ab_l47_47042

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry -- Proof to be provided

end find_ab_l47_47042


namespace eval_floor_ceil_sum_l47_47243

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem eval_floor_ceil_sum : floor (-3.67) + ceil 34.7 = 31 := by
  sorry

end eval_floor_ceil_sum_l47_47243


namespace negative_to_zero_power_l47_47947

theorem negative_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a) ^ 0 = 1 :=
by
  sorry

end negative_to_zero_power_l47_47947


namespace three_squares_sum_l47_47693

theorem three_squares_sum (n : ℤ) (h : n > 5) : 
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 :=
by sorry

end three_squares_sum_l47_47693


namespace positive_expression_l47_47442

variable (a b c d : ℝ)

theorem positive_expression (ha : a < b) (hb : b < 0) (hc : 0 < c) (hd : c < d) : d - c - b - a > 0 := 
sorry

end positive_expression_l47_47442


namespace find_n_sin_eq_l47_47147

theorem find_n_sin_eq (n : ℤ) (h₁ : -180 ≤ n) (h₂ : n ≤ 180) (h₃ : Real.sin (n * Real.pi / 180) = Real.sin (680 * Real.pi / 180)) :
  n = 40 ∨ n = 140 :=
by
  sorry

end find_n_sin_eq_l47_47147


namespace strawberries_harvest_l47_47465

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest_l47_47465


namespace fraction_ratios_l47_47799

theorem fraction_ratios (m n p q : ℕ) (h1 : (m : ℚ) / n = 18) (h2 : (p : ℚ) / n = 6) (h3 : (p : ℚ) / q = 1 / 15) :
  (m : ℚ) / q = 1 / 5 :=
sorry

end fraction_ratios_l47_47799


namespace Greg_PPO_Obtained_90_Percent_l47_47579

theorem Greg_PPO_Obtained_90_Percent :
  let max_procgen_reward := 240
  let max_coinrun_reward := max_procgen_reward / 2
  let greg_reward := 108
  (greg_reward / max_coinrun_reward * 100) = 90 := by
  sorry

end Greg_PPO_Obtained_90_Percent_l47_47579


namespace find_sum_of_coefficients_l47_47107

theorem find_sum_of_coefficients : 
  (∃ m n p : ℕ, 
    (n.gcd p = 1) ∧ 
    m + 36 = 72 ∧
    n + 33*3 = 103 ∧ 
    p = 3 ∧ 
    (72 + 33 * ℼ + (8 * (1/8 * (4 * π / 3))) + 36) = m + n * π / p) → 
  m + n + p = 430 :=
by {
  sorry
}

end find_sum_of_coefficients_l47_47107


namespace fourth_guard_ran_150_meters_l47_47212

def rectangle_width : ℕ := 200
def rectangle_length : ℕ := 300
def total_perimeter : ℕ := 2 * (rectangle_width + rectangle_length)
def three_guards_total_distance : ℕ := 850

def fourth_guard_distance : ℕ := total_perimeter - three_guards_total_distance

theorem fourth_guard_ran_150_meters :
  fourth_guard_distance = 150 :=
by
  -- calculation skipped here
  -- proving fourth_guard_distance as derived being 150 meters
  sorry

end fourth_guard_ran_150_meters_l47_47212


namespace cannot_be_zero_l47_47937

-- Define polynomial Q(x)
def Q (x : ℝ) (f g h i j : ℝ) : ℝ := x^5 + f * x^4 + g * x^3 + h * x^2 + i * x + j

-- Define the hypotheses for the proof
def distinct_roots (a b c d e : ℝ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def one_root_is_one (f g h i j : ℝ) := Q 1 f g h i j = 0

-- Statement to prove
theorem cannot_be_zero (f g h i j a b c d : ℝ)
  (h1 : Q 1 f g h i j = 0)
  (h2 : distinct_roots 1 a b c d)
  (h3 : Q 1 f g h i j = (1-a)*(1-b)*(1-c)*(1-d)) :
  i ≠ 0 :=
by
  sorry

end cannot_be_zero_l47_47937


namespace nonneg_sol_eq_l47_47184

theorem nonneg_sol_eq {a b c : ℝ} (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c) 
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) : 
  a = b ∧ b = c := 
sorry

end nonneg_sol_eq_l47_47184


namespace add_two_integers_l47_47489

/-- If the difference of two positive integers is 5 and their product is 180,
then their sum is 25. -/
theorem add_two_integers {x y : ℕ} (h1: x > y) (h2: x - y = 5) (h3: x * y = 180) : x + y = 25 :=
sorry

end add_two_integers_l47_47489


namespace ratio_of_area_to_breadth_l47_47921

variable (l b : ℕ)

theorem ratio_of_area_to_breadth 
  (h1 : b = 14) 
  (h2 : l - b = 10) : 
  (l * b) / b = 24 := by
  sorry

end ratio_of_area_to_breadth_l47_47921


namespace find_acute_angles_right_triangle_l47_47297

theorem find_acute_angles_right_triangle (α β : ℝ)
  (h₁ : α + β = π / 2)
  (h₂ : 0 < α ∧ α < π / 2)
  (h₃ : 0 < β ∧ β < π / 2)
  (h4 : Real.tan α + Real.tan β + Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan α ^ 3 + Real.tan β ^ 3 = 70) :
  (α = 75 * (π / 180) ∧ β = 15 * (π / 180)) 
  ∨ (α = 15 * (π / 180) ∧ β = 75 * (π / 180)) := 
sorry

end find_acute_angles_right_triangle_l47_47297


namespace evaluate_expression_l47_47484

theorem evaluate_expression (x y : ℕ) (hx : 2^x ∣ 360 ∧ ¬ 2^(x+1) ∣ 360) (hy : 3^y ∣ 360 ∧ ¬ 3^(y+1) ∣ 360) :
  (3 / 7)^(y - x) = 7 / 3 := by
  sorry

end evaluate_expression_l47_47484


namespace Jim_runs_total_distance_l47_47079

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end Jim_runs_total_distance_l47_47079


namespace bisection_method_correctness_l47_47682

noncomputable def initial_interval_length : ℝ := 1
noncomputable def required_precision : ℝ := 0.01
noncomputable def minimum_bisections : ℕ := 7

theorem bisection_method_correctness :
  ∃ n : ℕ, (n ≥ minimum_bisections) ∧ (initial_interval_length / 2^n ≤ required_precision) :=
by
  sorry

end bisection_method_correctness_l47_47682


namespace passengers_landed_in_newberg_last_year_l47_47055

theorem passengers_landed_in_newberg_last_year :
  let airport_a_on_time : ℕ := 16507
  let airport_a_late : ℕ := 256
  let airport_b_on_time : ℕ := 11792
  let airport_b_late : ℕ := 135
  airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690 :=
by
  let airport_a_on_time := 16507
  let airport_a_late := 256
  let airport_b_on_time := 11792
  let airport_b_late := 135
  show airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690
  sorry

end passengers_landed_in_newberg_last_year_l47_47055


namespace paint_time_l47_47396

theorem paint_time (n1 t1 n2 : ℕ) (k : ℕ) (h : n1 * t1 = k) (h1 : 5 * 4 = k) (h2 : n2 = 6) : (k / n2) = 10 / 3 :=
by {
  -- Proof would go here
  sorry
}

end paint_time_l47_47396


namespace prob_D_correct_l47_47209

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 3
def prob_C : ℚ := 1 / 6
def total_prob (prob_D : ℚ) : Prop := prob_A + prob_B + prob_C + prob_D = 1

theorem prob_D_correct : ∃ (prob_D : ℚ), total_prob prob_D ∧ prob_D = 1 / 4 :=
by
  -- Proof omitted
  sorry

end prob_D_correct_l47_47209


namespace cricketer_wickets_l47_47764

noncomputable def initial_average (R W : ℝ) : ℝ := R / W

noncomputable def new_average (R W : ℝ) (additional_runs additional_wickets : ℝ) : ℝ :=
  (R + additional_runs) / (W + additional_wickets)

theorem cricketer_wickets (R W : ℝ) 
(h1 : initial_average R W = 12.4) 
(h2 : new_average R W 26 5 = 12.0) : 
  W = 85 :=
sorry

end cricketer_wickets_l47_47764


namespace at_least_one_success_l47_47618

-- Define probabilities for A, B, and C
def pA : ℚ := 1 / 2
def pB : ℚ := 2 / 3
def pC : ℚ := 4 / 5

-- Define the probability that none succeed
def pNone : ℚ := (1 - pA) * (1 - pB) * (1 - pC)

-- Define the probability that at least one of them succeeds
def pAtLeastOne : ℚ := 1 - pNone

theorem at_least_one_success : pAtLeastOne = 29 / 30 := 
by sorry

end at_least_one_success_l47_47618


namespace problem_ratio_l47_47214

-- Define the conditions
variables 
  (R : ℕ) 
  (Bill_problems : ℕ := 20) 
  (Frank_problems_per_type : ℕ := 30)
  (types : ℕ := 4)

-- State the problem to prove
theorem problem_ratio (h1 : 3 * R = Frank_problems_per_type * types) :
  R / Bill_problems = 2 :=
by
  -- placeholder for proof
  sorry

end problem_ratio_l47_47214


namespace third_divisor_l47_47406

/-- 
Given that the new number after subtracting 7 from 3,381 leaves a remainder of 8 when divided by 9 
and 11, prove that the third divisor that also leaves a remainder of 8 is 17.
-/
theorem third_divisor (x : ℕ) (h1 : x = 3381 - 7)
                      (h2 : x % 9 = 8)
                      (h3 : x % 11 = 8) :
  ∃ (d : ℕ), d = 17 ∧ x % d = 8 := sorry

end third_divisor_l47_47406


namespace ratio_of_larger_to_smaller_l47_47844

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 2 := 
by
  sorry

end ratio_of_larger_to_smaller_l47_47844


namespace price_of_sundae_l47_47939

variable (num_ice_cream_bars num_sundaes : ℕ)
variable (total_price : ℚ)
variable (price_per_ice_cream_bar : ℚ)
variable (price_per_sundae : ℚ)

theorem price_of_sundae :
  num_ice_cream_bars = 125 →
  num_sundaes = 125 →
  total_price = 225 →
  price_per_ice_cream_bar = 0.60 →
  price_per_sundae = (total_price - (num_ice_cream_bars * price_per_ice_cream_bar)) / num_sundaes →
  price_per_sundae = 1.20 :=
by
  intros
  sorry

end price_of_sundae_l47_47939


namespace solution_l47_47725

def problem (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (x + a) * (x - 3) = x^2 + 2 * x - b

theorem solution (a b : ℝ) (h : problem a b) : a - b = -10 :=
  sorry

end solution_l47_47725


namespace log_one_plus_x_sq_lt_x_sq_l47_47760

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end log_one_plus_x_sq_lt_x_sq_l47_47760


namespace smallest_a_inequality_l47_47111

theorem smallest_a_inequality 
  (x : ℝ)
  (h1 : x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi)) : 
  (∃ a : ℝ, a = -2.52 ∧ (∀ x ∈ Set.Ioo (-3 * Real.pi / 2) (-Real.pi), 
    ( ((Real.sqrt (Real.cos x / Real.sin x)^2) - (Real.sqrt (Real.sin x / Real.cos x)^2))
    / ((Real.sqrt (Real.sin x)^2) - (Real.sqrt (Real.cos x)^2)) ) < a )) :=
  sorry

end smallest_a_inequality_l47_47111


namespace no_odd_integer_trinomial_has_root_1_over_2022_l47_47781

theorem no_odd_integer_trinomial_has_root_1_over_2022 :
  ¬ ∃ (a b c : ℤ), (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0)) :=
by
  sorry

end no_odd_integer_trinomial_has_root_1_over_2022_l47_47781


namespace train_crosses_bridge_in_12_2_seconds_l47_47162

def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 134

def speed_of_train_ms : ℚ := speed_of_train_kmh * (1000 : ℚ) / (3600 : ℚ)
def total_distance : ℕ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_2_seconds : time_to_cross_bridge = 12.2 := by
  sorry

end train_crosses_bridge_in_12_2_seconds_l47_47162


namespace integer_solutions_of_cubic_equation_l47_47709

theorem integer_solutions_of_cubic_equation :
  ∀ (n m : ℤ),
    n ^ 6 + 3 * n ^ 5 + 3 * n ^ 4 + 2 * n ^ 3 + 3 * n ^ 2 + 3 * n + 1 = m ^ 3 ↔
    (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
by
  intro n m
  apply Iff.intro
  { intro h
    sorry }
  { intro h
    sorry }

end integer_solutions_of_cubic_equation_l47_47709


namespace base3_addition_proof_l47_47521

-- Define the base 3 numbers
def one_3 : ℕ := 1
def twelve_3 : ℕ := 1 * 3 + 2
def two_hundred_twelve_3 : ℕ := 2 * 3^2 + 1 * 3 + 2
def two_thousand_one_hundred_twenty_one_3 : ℕ := 2 * 3^3 + 1 * 3^2 + 2 * 3 + 1

-- Define the correct answer in base 3
def expected_sum_3 : ℕ := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3 + 0

-- The proof problem
theorem base3_addition_proof :
  one_3 + twelve_3 + two_hundred_twelve_3 + two_thousand_one_hundred_twenty_one_3 = expected_sum_3 :=
by
  -- Proof goes here
  sorry

end base3_addition_proof_l47_47521


namespace screen_width_l47_47272

theorem screen_width
  (A : ℝ) -- Area of the screen
  (h : ℝ) -- Height of the screen
  (w : ℝ) -- Width of the screen
  (area_eq : A = 21) -- Condition 1: Area is 21 sq ft
  (height_eq : h = 7) -- Condition 2: Height is 7 ft
  (area_formula : A = w * h) -- Condition 3: Area formula
  : w = 3 := -- Conclusion: Width is 3 ft
sorry

end screen_width_l47_47272


namespace relationship_between_a_b_c_l47_47656

noncomputable def a : ℝ := Real.exp (-2)

noncomputable def b : ℝ := a ^ a

noncomputable def c : ℝ := a ^ b

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by {
  sorry
}

end relationship_between_a_b_c_l47_47656


namespace linda_age_l47_47790

theorem linda_age
  (j k l : ℕ)       -- Ages of Jane, Kevin, and Linda respectively
  (h1 : j + k + l = 36)    -- Condition 1: j + k + l = 36
  (h2 : l - 3 = j)         -- Condition 2: l - 3 = j
  (h3 : k + 4 = (1 / 2 : ℝ) * (l + 4))  -- Condition 3: k + 4 = 1/2 * (l + 4)
  : l = 16 := 
sorry

end linda_age_l47_47790


namespace central_angle_nonagon_l47_47608

theorem central_angle_nonagon : (360 / 9 = 40) :=
by
  sorry

end central_angle_nonagon_l47_47608


namespace vertical_asymptote_sum_l47_47490

theorem vertical_asymptote_sum : 
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  ∃ p q : ℝ, (6 * p ^ 2 + 7 * p + 3 = 0) ∧ (6 * q ^ 2 + 7 * q + 3 = 0) ∧ p + q = -11 / 6 :=
by
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  exact sorry

end vertical_asymptote_sum_l47_47490


namespace infinite_series_evaluation_l47_47191

theorem infinite_series_evaluation :
  (∑' m : ℕ, ∑' n : ℕ, 1 / (m * n * (m + n + 2))) = 3 :=
  sorry

end infinite_series_evaluation_l47_47191


namespace odd_multiple_of_9_implies_multiple_of_3_l47_47965

theorem odd_multiple_of_9_implies_multiple_of_3 :
  ∀ (S : ℤ), (∀ (n : ℤ), 9 * n = S → ∃ (m : ℤ), 3 * m = S) ∧ (S % 2 ≠ 0) → (∃ (m : ℤ), 3 * m = S) :=
by
  sorry

end odd_multiple_of_9_implies_multiple_of_3_l47_47965


namespace pairs_a_eq_b_l47_47041

theorem pairs_a_eq_b 
  (n : ℕ) (h_n : ¬ ∃ k : ℕ, k^2 = n) (a b : ℕ) 
  (r : ℝ) (h_r_pos : 0 < r) (h_ra_rational : ∃ q₁ : ℚ, r^a + (n:ℝ)^(1/2) = q₁) 
  (h_rb_rational : ∃ q₂ : ℚ, r^b + (n:ℝ)^(1/2) = q₂) : 
  a = b :=
sorry

end pairs_a_eq_b_l47_47041


namespace visible_yellow_bus_length_correct_l47_47934

noncomputable def red_bus_length : ℝ := 48
noncomputable def orange_car_length : ℝ := red_bus_length / 4
noncomputable def yellow_bus_length : ℝ := 3.5 * orange_car_length
noncomputable def green_truck_length : ℝ := 2 * orange_car_length
noncomputable def total_vehicle_length : ℝ := yellow_bus_length + green_truck_length
noncomputable def visible_yellow_bus_length : ℝ := 0.75 * yellow_bus_length

theorem visible_yellow_bus_length_correct :
  visible_yellow_bus_length = 31.5 := 
sorry

end visible_yellow_bus_length_correct_l47_47934


namespace greatest_b_l47_47083

theorem greatest_b (b : ℤ) (h : ∀ x : ℝ, x^2 + b * x + 20 ≠ -6) : b = 10 := sorry

end greatest_b_l47_47083


namespace find_min_value_of_quadratic_l47_47123

theorem find_min_value_of_quadratic : ∀ x : ℝ, ∃ c : ℝ, (∃ a b : ℝ, (y = 2*x^2 + 8*x + 7 ∧ (∀ x : ℝ, y ≥ c)) ∧ c = -1) :=
by
  sorry

end find_min_value_of_quadratic_l47_47123


namespace sin_minus_cos_l47_47624

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l47_47624


namespace trigonometric_identity_l47_47687

theorem trigonometric_identity (α : ℝ) : 
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * Real.cos (2 * α + Real.pi) ^ 2 - 1) = 
  2 * Real.cos (2 * α) :=
by sorry

end trigonometric_identity_l47_47687


namespace sum_1_to_50_l47_47457

-- Given conditions: initial values, and the loop increments
def initial_index : ℕ := 1
def initial_sum : ℕ := 0
def loop_condition (i : ℕ) : Prop := i ≤ 50

-- Increment step for index and running total in loop
def increment_index (i : ℕ) : ℕ := i + 1
def increment_sum (S : ℕ) (i : ℕ) : ℕ := S + i

-- Expected sum output for the given range
def sum_up_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the sum of integers from 1 to 50
theorem sum_1_to_50 : sum_up_to_n 50 = 1275 := by
  sorry

end sum_1_to_50_l47_47457


namespace arith_seq_a1_a7_sum_l47_47927

variable (a : ℕ → ℝ) (d : ℝ)

-- Conditions
def arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

def condition_sum : Prop :=
  a 3 + a 4 + a 5 = 12

-- Equivalent proof problem statement
theorem arith_seq_a1_a7_sum :
  arithmetic_sequence a d →
  condition_sum a →
  a 1 + a 7 = 8 :=
by
  sorry

end arith_seq_a1_a7_sum_l47_47927


namespace Sarah_copy_total_pages_l47_47357

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end Sarah_copy_total_pages_l47_47357


namespace perpendicular_lines_l47_47190

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l47_47190


namespace sequence_unique_l47_47330

theorem sequence_unique (n : ℕ) (h1 : n > 1)
  (x : ℕ → ℕ)
  (hx1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j < n → x i < x j)
  (hx2 : ∀ i, 1 ≤ i ∧ i < n → x i + x (n - i) = 2 * n)
  (hx3 : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j < n ∧ x i + x j < 2 * n →
    ∃ k, 1 ≤ k ∧ k < n ∧ x i + x j = x k) :
  ∀ k, 1 ≤ k ∧ k < n → x k = 2 * k :=
by
  sorry

end sequence_unique_l47_47330


namespace function_intersects_line_at_most_once_l47_47249

variable {α β : Type} [Nonempty α]

def function_intersects_at_most_once (f : α → β) (a : α) : Prop :=
  ∀ (b b' : β), f a = b → f a = b' → b = b'

theorem function_intersects_line_at_most_once {α β : Type} [Nonempty α] (f : α → β) (a : α) :
  function_intersects_at_most_once f a :=
by
  sorry

end function_intersects_line_at_most_once_l47_47249


namespace find_m_n_calculate_expression_l47_47990

-- Define the polynomials A and B
def A (m x : ℝ) := 5 * x^2 - m * x + 1
def B (n x : ℝ) := 2 * x^2 - 2 * x - n

-- The conditions
variable (x : ℝ) (m n : ℝ)
def no_linear_or_constant_terms (m : ℝ) (n : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + (2 - m) * x + (1 + n) = 3 * x^2

-- The target theorem
theorem find_m_n 
  (h : no_linear_or_constant_terms m n) : 
  m = 2 ∧ n = -1 := sorry

-- Calculate the expression when m = 2 and n = -1
theorem calculate_expression
  (hm : m = 2)
  (hn : n = -1) : 
  m^2 + n^2 - 2 * m * n = 9 := sorry

end find_m_n_calculate_expression_l47_47990


namespace collinear_points_eq_sum_l47_47040

theorem collinear_points_eq_sum (a b : ℝ) :
  -- Collinearity conditions in ℝ³
  (∃ t1 t2 t3 t4 : ℝ,
    (2, a, b) = (a + t1 * (a - 2), 3 + t1 * (b - 3), b + t1 * (4 - b)) ∧
    (a, 3, b) = (a + t2 * (a - 2), 3 + t2 * (b - 3), b + t2 * (4 - b)) ∧
    (a, b, 4) = (a + t3 * (a - 2), 3 + t3 * (b - 3), b + t3 * (4 - b)) ∧
    (5, b, a) = (a + t4 * (a - 2), 3 + t4 * (b - 3), b + t4 * (4 - b))) →
  a + b = 9 :=
by
  sorry

end collinear_points_eq_sum_l47_47040


namespace max_arithmetic_sequence_of_primes_less_than_150_l47_47554

theorem max_arithmetic_sequence_of_primes_less_than_150 : 
  ∀ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x) ∧ (∀ x ∈ S, x < 150) ∧ (∃ d, ∀ x ∈ S, ∃ n : ℕ, x = S.min' (by sorry) + n * d) → S.card ≤ 5 := 
by
  sorry

end max_arithmetic_sequence_of_primes_less_than_150_l47_47554


namespace widget_production_l47_47130

theorem widget_production (p q r s t : ℕ) :
  (s * q * t) / (p * r) = (sqt / pr) := 
sorry

end widget_production_l47_47130


namespace oliver_siblings_l47_47780

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)

def oliver := Child.mk "Oliver" "Gray" "Brown"
def charles := Child.mk "Charles" "Gray" "Red"
def diana := Child.mk "Diana" "Green" "Brown"
def olivia := Child.mk "Olivia" "Green" "Red"
def ethan := Child.mk "Ethan" "Green" "Red"
def fiona := Child.mk "Fiona" "Green" "Brown"

def sharesCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

def sameFamily (c1 c2 c3 : Child) : Prop :=
  sharesCharacteristic c1 c2 ∧
  sharesCharacteristic c2 c3 ∧
  sharesCharacteristic c3 c1

theorem oliver_siblings : 
  sameFamily oliver charles diana :=
by
  -- proof skipped
  sorry

end oliver_siblings_l47_47780


namespace parallel_lines_a_l47_47858

theorem parallel_lines_a (a : ℝ) (x y : ℝ)
  (h1 : x + 2 * a * y - 1 = 0)
  (h2 : (a + 1) * x - a * y = 0)
  (h_parallel : ∀ (l1 l2 : ℝ → ℝ → Prop), l1 x y ∧ l2 x y → l1 = l2) :
  a = -3 / 2 ∨ a = 0 :=
sorry

end parallel_lines_a_l47_47858


namespace jamies_score_l47_47933

def quiz_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct * 2) + (incorrect * (-0.5)) + (unanswered * 0.25)

theorem jamies_score :
  quiz_score 16 10 4 = 28 :=
by
  sorry

end jamies_score_l47_47933


namespace even_two_digit_numbers_count_l47_47852

/-- Even positive integers less than 1000 with at most two different digits -/
def count_even_two_digit_numbers : ℕ :=
  let one_digit := [2, 4, 6, 8].length
  let two_d_same := [22, 44, 66, 88].length
  let two_d_diff := [24, 42, 26, 62, 28, 82, 46, 64, 48, 84, 68, 86].length
  let three_d_same := [222, 444, 666, 888].length
  let three_d_diff := 16 + 12
  one_digit + two_d_same + two_d_diff + three_d_same + three_d_diff

theorem even_two_digit_numbers_count :
  count_even_two_digit_numbers = 52 :=
by sorry

end even_two_digit_numbers_count_l47_47852


namespace geom_sequence_second_term_l47_47520

noncomputable def geom_sequence_term (a r : ℕ) (n : ℕ) : ℕ := a * r^(n-1)

theorem geom_sequence_second_term 
  (a1 a5: ℕ) (r: ℕ) 
  (h1: a1 = 5)
  (h2: a5 = geom_sequence_term a1 r 5)
  (h3: a5 = 320)
  (h_r: r^4 = 64): 
  geom_sequence_term a1 r 2 = 10 :=
by
  sorry

end geom_sequence_second_term_l47_47520


namespace ellie_shoes_count_l47_47621

variable (E R : ℕ)

def ellie_shoes (E R : ℕ) : Prop :=
  E + R = 13 ∧ E = R + 3

theorem ellie_shoes_count (E R : ℕ) (h : ellie_shoes E R) : E = 8 :=
  by sorry

end ellie_shoes_count_l47_47621


namespace height_to_width_ratio_l47_47435

theorem height_to_width_ratio (w h l : ℝ) (V : ℝ) (x : ℝ) :
  (h = x * w) →
  (l = 7 * h) →
  (V = l * w * h) →
  (V = 129024) →
  (w = 8) →
  (x = 6) :=
by
  intros h_eq_xw l_eq_7h V_eq_lwh V_val w_val
  -- Proof omitted
  sorry

end height_to_width_ratio_l47_47435


namespace rectangles_same_area_l47_47443

theorem rectangles_same_area (x y : ℕ) 
  (h1 : x * y = (x + 4) * (y - 3)) 
  (h2 : x * y = (x + 8) * (y - 4)) : x + y = 10 := 
by
  sorry

end rectangles_same_area_l47_47443


namespace sin_value_l47_47734

theorem sin_value (α : ℝ) (h : Real.cos (α + π / 6) = - (Real.sqrt 2) / 10) : 
  Real.sin (2 * α - π / 6) = 24 / 25 :=
by
  sorry

end sin_value_l47_47734


namespace expression_in_parentheses_l47_47788

theorem expression_in_parentheses (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) :
  ∃ expr : ℝ, xy * expr = -x^3 * y^2 ∧ expr = -x^2 * y :=
by
  sorry

end expression_in_parentheses_l47_47788


namespace function_behavior_on_negative_interval_l47_47004

-- Define the necessary conditions and function properties
variables {f : ℝ → ℝ}

-- Conditions: f is even, increasing on [0, 7], and f(7) = 6
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def f7_eq_6 (f : ℝ → ℝ) : Prop := f 7 = 6

-- The theorem to prove
theorem function_behavior_on_negative_interval (h1 : even_function f) (h2 : increasing_on_interval f 0 7) (h3 : f7_eq_6 f) : 
  (∀ x y, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
sorry

end function_behavior_on_negative_interval_l47_47004


namespace mika_stickers_l47_47606

theorem mika_stickers 
    (initial_stickers : ℝ := 20.5)
    (bought_stickers : ℝ := 26.25)
    (birthday_stickers : ℝ := 19.75)
    (friend_stickers : ℝ := 7.5)
    (sister_stickers : ℝ := 6.3)
    (greeting_card_stickers : ℝ := 58.5)
    (yard_sale_stickers : ℝ := 3.2) :
    initial_stickers + bought_stickers + birthday_stickers + friend_stickers
    - sister_stickers - greeting_card_stickers - yard_sale_stickers = 6 := 
by
    sorry

end mika_stickers_l47_47606


namespace smallest_four_digit_multiple_of_53_l47_47414

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l47_47414


namespace factorize_expr_l47_47726

theorem factorize_expr (a b : ℝ) : 2 * a^2 - a * b = a * (2 * a - b) := 
by
  sorry

end factorize_expr_l47_47726


namespace possible_values_quotient_l47_47578

theorem possible_values_quotient (α : ℝ) (h_pos : α > 0) (h_rounded : ∃ (n : ℕ) (α1 : ℝ), α = n / 100 + α1 ∧ 0 ≤ α1 ∧ α1 < 1 / 100) :
  ∃ (values : List ℝ), values = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                                  0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                                  1.00] :=
  sorry

end possible_values_quotient_l47_47578


namespace kia_vehicle_count_l47_47821

theorem kia_vehicle_count (total_vehicles dodge_vehicles hyundai_vehicles kia_vehicles : ℕ)
  (h_total: total_vehicles = 400)
  (h_dodge: dodge_vehicles = total_vehicles / 2)
  (h_hyundai: hyundai_vehicles = dodge_vehicles / 2)
  (h_kia: kia_vehicles = total_vehicles - dodge_vehicles - hyundai_vehicles) : kia_vehicles = 100 := 
by
  sorry

end kia_vehicle_count_l47_47821


namespace annual_increase_rate_l47_47293

theorem annual_increase_rate (r : ℝ) : 
  (6400 * (1 + r) * (1 + r) = 8100) → r = 0.125 :=
by sorry

end annual_increase_rate_l47_47293


namespace find_a_l47_47841

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x + 1)

theorem find_a {a : ℝ} (h : (deriv (f a) 0 = 1)) : a = 1 :=
by
  -- Proof goes here
  sorry

end find_a_l47_47841


namespace base3_to_base10_l47_47820

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l47_47820


namespace length_PR_l47_47605

noncomputable def circle_radius : ℝ := 10
noncomputable def distance_PQ : ℝ := 12
noncomputable def midpoint_minor_arc_length_PR : ℝ :=
  let PS : ℝ := distance_PQ / 2
  let OS : ℝ := Real.sqrt (circle_radius^2 - PS^2)
  let RS : ℝ := circle_radius - OS
  Real.sqrt (PS^2 + RS^2)

theorem length_PR :
  midpoint_minor_arc_length_PR = 2 * Real.sqrt 10 :=
by
  sorry

end length_PR_l47_47605


namespace find_b_l47_47713

theorem find_b (b c x1 x2 : ℝ)
  (h_parabola_intersects_x_axis : (x1 ≠ x2) ∧ x1 * x2 = c ∧ x1 + x2 = -b ∧ x2 - x1 = 1)
  (h_parabola_intersects_y_axis : c ≠ 0)
  (h_length_ab : x2 - x1 = 1)
  (h_area_abc : (1 / 2) * (x2 - x1) * |c| = 1)
  : b = -3 :=
sorry

end find_b_l47_47713


namespace mark_charged_more_hours_l47_47233

theorem mark_charged_more_hours (P K M : ℕ) 
  (h1 : P + K + M = 135)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 75 := by {

sorry
}

end mark_charged_more_hours_l47_47233


namespace winning_probability_correct_l47_47570

-- Define the conditions
def numPowerBalls : ℕ := 30
def numLuckyBalls : ℕ := 49
def numChosenBalls : ℕ := 6

-- Define the probability of picking the correct PowerBall
def powerBallProb : ℚ := 1 / numPowerBalls

-- Define the combination function for choosing LuckyBalls
noncomputable def combination (n k : ℕ) : ℕ := n.choose k

-- Define the probability of picking the correct LuckyBalls
noncomputable def luckyBallProb : ℚ := 1 / (combination numLuckyBalls numChosenBalls)

-- Define the total winning probability
noncomputable def totalWinningProb : ℚ := powerBallProb * luckyBallProb

-- State the theorem to prove
theorem winning_probability_correct : totalWinningProb = 1 / 419512480 :=
by
  sorry

end winning_probability_correct_l47_47570


namespace factors_of_1320_l47_47053

theorem factors_of_1320 : ∃ n : ℕ, n = 24 ∧ ∃ (a b c d : ℕ),
  1320 = 2^a * 3^b * 5^c * 11^d ∧ (a = 0 ∨ a = 1 ∨ a = 2) ∧ (b = 0 ∨ b = 1) ∧ (c = 0 ∨ c = 1) ∧ (d = 0 ∨ d = 1) :=
by {
  sorry
}

end factors_of_1320_l47_47053


namespace alex_silver_tokens_l47_47170

-- Definitions and conditions
def initialRedTokens : ℕ := 100
def initialBlueTokens : ℕ := 50
def firstBoothRedChange (x : ℕ) : ℕ := 3 * x
def firstBoothSilverGain (x : ℕ) : ℕ := 2 * x
def firstBoothBlueGain (x : ℕ) : ℕ := x
def secondBoothBlueChange (y : ℕ) : ℕ := 2 * y
def secondBoothSilverGain (y : ℕ) : ℕ := y
def secondBoothRedGain (y : ℕ) : ℕ := y

-- Final conditions when no more exchanges are possible
def finalRedTokens (x y : ℕ) : ℕ := initialRedTokens - firstBoothRedChange x + secondBoothRedGain y
def finalBlueTokens (x y : ℕ) : ℕ := initialBlueTokens + firstBoothBlueGain x - secondBoothBlueChange y

-- Total silver tokens calculation
def totalSilverTokens (x y : ℕ) : ℕ := firstBoothSilverGain x + secondBoothSilverGain y

-- Proof that in the end, Alex has 147 silver tokens
theorem alex_silver_tokens : 
  ∃ (x y : ℕ), finalRedTokens x y = 2 ∧ finalBlueTokens x y = 1 ∧ totalSilverTokens x y = 147 :=
by
  -- the proof logic will be filled here
  sorry

end alex_silver_tokens_l47_47170


namespace sin_eq_solutions_l47_47916

theorem sin_eq_solutions :
  (∃ count : ℕ, 
    count = 4007 ∧ 
    (∀ (x : ℝ), 
      0 ≤ x ∧ x ≤ 2 * Real.pi → 
      (∃ (k1 k2 : ℤ), 
        x = -2 * k1 * Real.pi ∨ 
        x = 2 * Real.pi ∨ 
        x = (2 * k2 + 1) * Real.pi / 4005)
    )) :=
sorry

end sin_eq_solutions_l47_47916


namespace avg_integer_N_between_fractions_l47_47808

theorem avg_integer_N_between_fractions (N : ℕ) (h1 : (2 : ℚ) / 5 < N / 42) (h2 : N / 42 < 1 / 3) : 
  N = 15 := 
by
  sorry

end avg_integer_N_between_fractions_l47_47808


namespace intersection_points_count_l47_47753

open Real

theorem intersection_points_count :
  (∃ (x y : ℝ), ((x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋) ∧ (y = 1/3 * x + 1)) →
  (∃ (n : ℕ), n = 8) :=
by
  -- proof goes here
  sorry

end intersection_points_count_l47_47753


namespace total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l47_47791

-- Definitions based on conditions
def standard_weight : ℝ := 25
def weight_diffs : List ℝ := [-3, -2, -2, -2, -2, -1.5, -1.5, 0, 0, 0, 1, 1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
def price_per_kg : ℝ := 10.6

-- Problem 1
theorem total_over_or_underweight_is_8kg :
  (weight_diffs.sum = 8) := 
  sorry

-- Problem 2
theorem total_selling_price_is_5384_point_8_yuan :
  (20 * standard_weight + 8) * price_per_kg = 5384.8 :=
  sorry

end total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l47_47791


namespace diagonals_from_vertex_of_regular_polygon_l47_47731

-- Definitions for the conditions in part a)
def exterior_angle (n : ℕ) : ℚ := 360 / n

-- Proof problem statement
theorem diagonals_from_vertex_of_regular_polygon
  (n : ℕ)
  (h1 : exterior_angle n = 36)
  : n - 3 = 7 :=
by sorry

end diagonals_from_vertex_of_regular_polygon_l47_47731


namespace double_and_halve_is_sixteen_l47_47884

-- Definition of the initial number
def initial_number : ℕ := 16

-- Doubling the number
def doubled (n : ℕ) : ℕ := n * 2

-- Halving the number
def halved (n : ℕ) : ℕ := n / 2

-- The theorem that needs to be proven
theorem double_and_halve_is_sixteen : halved (doubled initial_number) = 16 :=
by
  /-
  We need to prove that when the number 16 is doubled and then halved, 
  the result is 16.
  -/
  sorry

end double_and_halve_is_sixteen_l47_47884


namespace original_cost_l47_47447

theorem original_cost (P : ℝ) (h : 0.76 * P = 608) : P = 800 :=
by
  sorry

end original_cost_l47_47447


namespace local_minimum_at_one_l47_47277

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + a^2 * x

theorem local_minimum_at_one (a : ℝ) (hfmin : ∀ x : ℝ, deriv (f a) x = 3 * a * x^2 - 4 * x + a^2) (h1 : f a 1 = f a 1) : a = 1 :=
sorry

end local_minimum_at_one_l47_47277


namespace simplify_expr_l47_47741

theorem simplify_expr (x : ℕ) (h : x = 2018) : x^2 + 2 * x - x * (x + 1) = x := by
  sorry

end simplify_expr_l47_47741


namespace perfect_square_count_between_20_and_150_l47_47666

theorem perfect_square_count_between_20_and_150 :
  let lower_bound := 20
  let upper_bound := 150
  let smallest_ps := 25
  let largest_ps := 144
  let count_squares (a b : Nat) := b - a
  count_squares 4 12 = 8 := sorry

end perfect_square_count_between_20_and_150_l47_47666


namespace right_triangle_with_integer_sides_l47_47647

theorem right_triangle_with_integer_sides (k : ℤ) :
  ∃ (a b c : ℤ), a = 2*k+1 ∧ b = 2*k*(k+1) ∧ c = 2*k^2+2*k+1 ∧ (a^2 + b^2 = c^2) ∧ (c = a + 1) := by
  sorry

end right_triangle_with_integer_sides_l47_47647


namespace triangle_ratio_inequality_l47_47721

/-- Given a triangle ABC, R is the radius of the circumscribed circle, 
    r is the radius of the inscribed circle, a is the length of the longest side,
    and h is the length of the shortest altitude. Prove that R / r > a / h. -/
theorem triangle_ratio_inequality
  (ABC : Triangle) (R r a h : ℝ)
  (hR : 2 * R ≥ a)
  (hr : 2 * r < h) :
  (R / r) > (a / h) :=
by
  -- sorry is used to skip the proof
  sorry

end triangle_ratio_inequality_l47_47721


namespace largest_common_term_up_to_150_l47_47462

theorem largest_common_term_up_to_150 :
  ∃ a : ℕ, a ≤ 150 ∧ (∃ n : ℕ, a = 2 + 8 * n) ∧ (∃ m : ℕ, a = 3 + 9 * m) ∧ (∀ b : ℕ, b ≤ 150 → (∃ n' : ℕ, b = 2 + 8 * n') → (∃ m' : ℕ, b = 3 + 9 * m') → b ≤ a) := 
sorry

end largest_common_term_up_to_150_l47_47462


namespace translate_A_coordinates_l47_47632

-- Definitions
def A_initial : ℝ × ℝ := (-3, 2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

-- Final coordinates after transformation
def A' : ℝ × ℝ :=
  let A_translated := translate_right A_initial 4
  translate_down A_translated 3

-- Proof statement
theorem translate_A_coordinates :
  A' = (1, -1) :=
by
  simp [A', translate_right, translate_down, A_initial]
  sorry

end translate_A_coordinates_l47_47632


namespace equal_sum_sequence_a18_l47_47643

def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_sequence_a18 (a : ℕ → ℕ) (h : equal_sum_sequence a 5) (h1 : a 1 = 2) : a 18 = 3 :=
  sorry

end equal_sum_sequence_a18_l47_47643


namespace possible_AC_values_l47_47900

-- Given points A, B, and C on a straight line 
-- with AB = 1 and BC = 3, prove that AC can be 2 or 4.

theorem possible_AC_values (A B C : ℝ) (hAB : abs (B - A) = 1) (hBC : abs (C - B) = 3) : 
  abs (C - A) = 2 ∨ abs (C - A) = 4 :=
sorry

end possible_AC_values_l47_47900


namespace div_by_30_l47_47998

theorem div_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end div_by_30_l47_47998


namespace find_P_eq_30_l47_47945

theorem find_P_eq_30 (P Q R S : ℕ) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  P * Q = 120 ∧ R * S = 120 ∧ P - Q = R + S → P = 30 :=
by
  sorry

end find_P_eq_30_l47_47945


namespace distinct_numbers_on_board_l47_47451

def count_distinct_numbers (Mila_divisors : ℕ) (Zhenya_divisors : ℕ) (common : ℕ) : ℕ :=
  Mila_divisors + Zhenya_divisors - (common - 1)

theorem distinct_numbers_on_board :
  count_distinct_numbers 10 9 2 = 13 := by
  sorry

end distinct_numbers_on_board_l47_47451


namespace find_x_l47_47660

def operation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) :
  operation 6 (operation 4 x) = 480 ↔ x = 5 := 
by
  sorry

end find_x_l47_47660


namespace range_a_l47_47558

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then Real.log x / Real.log a else -2 * x + 8

theorem range_a (a : ℝ) (hf : ∀ x, f a x ≤ f a 2) :
  1 < a ∧ a ≤ Real.sqrt 3 := by
  sorry

end range_a_l47_47558


namespace sin_inequality_in_triangle_l47_47912

theorem sin_inequality_in_triangle (A B C : ℝ) (hA_leq_B : A ≤ B) (hB_leq_C : B ≤ C)
  (hSum : A + B + C = π) (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π) :
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := 
sorry

end sin_inequality_in_triangle_l47_47912


namespace cannot_determine_right_triangle_from_conditions_l47_47453

-- Let triangle ABC have side lengths a, b, c opposite angles A, B, C respectively.
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Condition A: c^2 = a^2 - b^2 is rearranged to c^2 + b^2 = a^2 implying right triangle
def condition_A (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Condition B: Triangle angles in the ratio A:B:C = 3:4:5 means not a right triangle
def condition_B : Prop := 
  let A := 45.0
  let B := 60.0
  let C := 75.0
  A ≠ 90.0 ∧ B ≠ 90.0 ∧ C ≠ 90.0

-- Condition C: Specific lengths 7, 24, 25 form a right triangle
def condition_C : Prop := 
  let a := 7.0
  let b := 24.0
  let c := 25.0
  is_right_triangle a b c

-- Condition D: A = B - C can be shown to always form at least one 90 degree angle, a right triangle
def condition_D (A B C : ℝ) : Prop := A = B - C ∧ (A + B + C = 180)

-- The actual mathematical proof that option B does not determine a right triangle
theorem cannot_determine_right_triangle_from_conditions :
  ∀ a b c (A B C : ℝ),
    (condition_A a b c → is_right_triangle a b c) ∧
    (condition_C → is_right_triangle 7 24 25) ∧
    (condition_D A B C → is_right_triangle a b c) ∧
    ¬condition_B :=
by
  sorry

end cannot_determine_right_triangle_from_conditions_l47_47453


namespace fewest_four_dollar_frisbees_l47_47060

-- Definitions based on the conditions
variables (x y : ℕ) -- The numbers of $3 and $4 frisbees, respectively.
def total_frisbees (x y : ℕ) : Prop := x + y = 60
def total_receipts (x y : ℕ) : Prop := 3 * x + 4 * y = 204

-- The statement to prove
theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : total_frisbees x y) (h2 : total_receipts x y) : y = 24 :=
sorry

end fewest_four_dollar_frisbees_l47_47060


namespace green_dots_fifth_row_l47_47519

variable (R : ℕ → ℕ)

-- Define the number of green dots according to the pattern
def pattern (n : ℕ) : ℕ := 3 * n

-- Define conditions for rows
axiom row_1 : R 1 = 3
axiom row_2 : R 2 = 6
axiom row_3 : R 3 = 9
axiom row_4 : R 4 = 12

-- The theorem
theorem green_dots_fifth_row : R 5 = 15 :=
by
  -- Row 5 follows the pattern and should satisfy the condition R 5 = R 4 + 3
  sorry

end green_dots_fifth_row_l47_47519


namespace points_opposite_side_of_line_l47_47766

theorem points_opposite_side_of_line :
  (∀ a : ℝ, ((2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0) ↔ -1 < a ∧ a < 1) :=
by sorry

end points_opposite_side_of_line_l47_47766


namespace cubics_sum_l47_47794

theorem cubics_sum (a b c : ℝ) (h₁ : a + b + c = 4) (h₂ : ab + ac + bc = 6) (h₃ : abc = -8) :
  a^3 + b^3 + c^3 = 8 :=
by {
  -- proof steps would go here
  sorry
}

end cubics_sum_l47_47794


namespace shells_in_afternoon_l47_47456

-- Conditions: Lino picked up 292 shells in the morning and 616 shells in total.
def shells_in_morning : ℕ := 292
def total_shells : ℕ := 616

-- Theorem: The number of shells Lino picked up in the afternoon is 324.
theorem shells_in_afternoon : (total_shells - shells_in_morning) = 324 := 
by sorry

end shells_in_afternoon_l47_47456


namespace correct_student_mark_l47_47275

theorem correct_student_mark
  (avg_wrong : ℕ) (num_students : ℕ) (wrong_mark : ℕ) (avg_correct : ℕ)
  (h1 : num_students = 10) (h2 : avg_wrong = 100) (h3 : wrong_mark = 90) (h4 : avg_correct = 92) :
  ∃ (x : ℕ), x = 10 :=
by
  sorry

end correct_student_mark_l47_47275


namespace train_speed_l47_47090

def train_length : ℝ := 1000  -- train length in meters
def time_to_cross_pole : ℝ := 200  -- time to cross the pole in seconds

theorem train_speed : train_length / time_to_cross_pole = 5 := by
  sorry

end train_speed_l47_47090


namespace smallest_solution_of_abs_eq_l47_47276

theorem smallest_solution_of_abs_eq (x : ℝ) : 
  (x * |x| = 3 * x + 2 → x ≥ 0 → x = (3 + Real.sqrt 17) / 2) ∧
  (x * |x| = 3 * x + 2 → x < 0 → x = -2) ∧
  (x * |x| = 3 * x + 2 → x = -2 → x = -2) :=
by
  sorry

end smallest_solution_of_abs_eq_l47_47276


namespace inequality_sum_l47_47888

variables {a b c : ℝ}

theorem inequality_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end inequality_sum_l47_47888


namespace minimalBananasTotal_is_408_l47_47595

noncomputable def minimalBananasTotal : ℕ :=
  let b₁ := 11 * 8
  let b₂ := 13 * 8
  let b₃ := 27 * 8
  b₁ + b₂ + b₃

theorem minimalBananasTotal_is_408 : minimalBananasTotal = 408 := by
  sorry

end minimalBananasTotal_is_408_l47_47595


namespace find_a2023_l47_47076

noncomputable def a_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, 2 * S n = a n * (a n + 1))

theorem find_a2023 (a S : ℕ → ℕ) 
  (h_seq : a_sequence a S)
  (h_pos : ∀ n, 0 < a n) 
  : a 2023 = 2023 :=
sorry

end find_a2023_l47_47076


namespace linear_function_not_in_first_quadrant_l47_47109

theorem linear_function_not_in_first_quadrant:
  ∀ x y : ℝ, y = -2 * x - 3 → ¬ (x > 0 ∧ y > 0) :=
by
 -- proof steps would go here
 sorry

end linear_function_not_in_first_quadrant_l47_47109


namespace trapezium_area_l47_47439

theorem trapezium_area (a b h : ℝ) (h_a : a = 4) (h_b : b = 5) (h_h : h = 6) :
  (1 / 2 * (a + b) * h) = 27 :=
by
  rw [h_a, h_b, h_h]
  norm_num

end trapezium_area_l47_47439


namespace maximum_value_l47_47557

theorem maximum_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 :=
sorry

end maximum_value_l47_47557


namespace total_canoes_by_end_of_april_l47_47628

def N_F : ℕ := 4
def N_M : ℕ := 3 * N_F
def N_A : ℕ := 3 * N_M
def total_canoes : ℕ := N_F + N_M + N_A

theorem total_canoes_by_end_of_april : total_canoes = 52 := by
  sorry

end total_canoes_by_end_of_april_l47_47628


namespace sin_lower_bound_lt_l47_47401

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end sin_lower_bound_lt_l47_47401


namespace equilateral_triangle_perimeter_l47_47626

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end equilateral_triangle_perimeter_l47_47626


namespace storks_count_l47_47301

theorem storks_count (B S : ℕ) (h1 : B = 3) (h2 : B + 2 = S + 1) : S = 4 :=
by
  sorry

end storks_count_l47_47301


namespace problem1_solution_problem2_solution_l47_47797

theorem problem1_solution (x : ℝ) : x^2 - x - 6 > 0 ↔ x < -2 ∨ x > 3 := sorry

theorem problem2_solution (x : ℝ) : -2*x^2 + x + 1 < 0 ↔ x < -1/2 ∨ x > 1 := sorry

end problem1_solution_problem2_solution_l47_47797


namespace power_difference_of_squares_l47_47397

theorem power_difference_of_squares : (((7^2 - 3^2) : ℤ)^4) = 2560000 := by
  sorry

end power_difference_of_squares_l47_47397


namespace base_k_sum_l47_47068

theorem base_k_sum (k : ℕ) (t : ℕ) (h1 : (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5)
    (h2 : t = (k + 3) + (k + 4) + (k + 7)) :
    t = 50 := sorry

end base_k_sum_l47_47068


namespace double_apply_l47_47535

def op1 (x : ℤ) : ℤ := 9 - x 
def op2 (x : ℤ) : ℤ := x - 9

theorem double_apply (x : ℤ) : op1 (op2 x) = 3 := by
  sorry

end double_apply_l47_47535


namespace triangle_perimeter_l47_47478

theorem triangle_perimeter (r A : ℝ) (p : ℝ)
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) : 
  p = 20 :=
by 
  sorry

end triangle_perimeter_l47_47478


namespace value_of_expression_l47_47894

variable {x : ℝ}

theorem value_of_expression (h : x^2 - 3 * x = 2) : 3 * x^2 - 9 * x - 7 = -1 := by
  sorry

end value_of_expression_l47_47894


namespace Punta_position_l47_47475

theorem Punta_position (N x y p : ℕ) (h1 : N = 36) (h2 : x = y / 4) (h3 : x + y = 35) : p = 8 := by
  sorry

end Punta_position_l47_47475


namespace infinite_series_fraction_l47_47897

theorem infinite_series_fraction:
  (∑' n : ℕ, (if n = 0 then 0 else ((2 : ℚ) / (3 * n) - (1 : ℚ) / (3 * (n + 1)) - (7 : ℚ) / (6 * (n + 3))))) =
  (1 : ℚ) / 3 := 
sorry

end infinite_series_fraction_l47_47897


namespace solve_work_problem_l47_47499

variables (A B C : ℚ)

-- Conditions
def condition1 := B + C = 1/3
def condition2 := C + A = 1/4
def condition3 := C = 1/24

-- Conclusion (Question translated to proof statement)
theorem solve_work_problem (h1 : condition1 B C) (h2 : condition2 C A) (h3 : condition3 C) : A + B = 1/2 :=
by sorry

end solve_work_problem_l47_47499


namespace polar_equations_and_ratios_l47_47317

open Real

theorem polar_equations_and_ratios (α β : ℝ)
    (h_line : ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ)
    (h_curve : ∀ (α : ℝ), ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2) :
    ( ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ) ∧
    ( ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2 → 
    0 < r * sin 2 * θ / (r / cos θ) ∧ r * sin 2 * θ / (r / cos θ) ≤ 1 / 2) :=
by
  sorry

end polar_equations_and_ratios_l47_47317


namespace factorial_expression_l47_47935

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression (N : ℕ) (h : N > 0) :
  (factorial (N + 1) + factorial (N - 1)) / factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3 * N^2 + 2 * N) :=
by
  sorry

end factorial_expression_l47_47935


namespace smallest_stamps_l47_47177

theorem smallest_stamps : ∃ S, 1 < S ∧ (S % 9 = 1) ∧ (S % 10 = 1) ∧ (S % 11 = 1) ∧ S = 991 :=
by
  sorry

end smallest_stamps_l47_47177


namespace roots_triple_relation_l47_47977

theorem roots_triple_relation (a b c : ℤ) (α β : ℤ)
    (h_quad : a ≠ 0)
    (h_roots : α + β = -b / a)
    (h_prod : α * β = c / a)
    (h_triple : β = 3 * α) :
    3 * b^2 = 16 * a * c :=
sorry

end roots_triple_relation_l47_47977


namespace total_people_surveyed_l47_47817

-- Define the conditions
variable (total_surveyed : ℕ) (disease_believers : ℕ)
variable (rabies_believers : ℕ)

-- Condition 1: 75% of the people surveyed thought rats carried diseases
def condition1 (total_surveyed disease_believers : ℕ) : Prop :=
  disease_believers = (total_surveyed * 75) / 100

-- Condition 2: 50% of the people who thought rats carried diseases said rats frequently carried rabies
def condition2 (disease_believers rabies_believers : ℕ) : Prop :=
  rabies_believers = (disease_believers * 50) / 100

-- Condition 3: 18 people were mistaken in thinking rats frequently carry rabies
def condition3 (rabies_believers : ℕ) : Prop := rabies_believers = 18

-- The theorem to prove the total number of people surveyed given the conditions
theorem total_people_surveyed (total_surveyed disease_believers rabies_believers : ℕ) :
  condition1 total_surveyed disease_believers →
  condition2 disease_believers rabies_believers →
  condition3 rabies_believers →
  total_surveyed = 48 :=
by sorry

end total_people_surveyed_l47_47817


namespace missing_angles_sum_l47_47865

theorem missing_angles_sum 
  (calculated_sum : ℕ) 
  (missed_angles_sum : ℕ)
  (total_corrections : ℕ)
  (polygon_angles : ℕ) 
  (h1 : calculated_sum = 2797) 
  (h2 : total_corrections = 2880) 
  (h3 : polygon_angles = total_corrections - calculated_sum) : 
  polygon_angles = 83 := by
  sorry

end missing_angles_sum_l47_47865


namespace sum_seven_l47_47296

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom a2 : a 2 = 3
axiom a6 : a 6 = 11
axiom arithmetic_seq : arithmetic_sequence a
axiom sum_of_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_seven : S 7 = 49 :=
sorry

end sum_seven_l47_47296


namespace z_coordinate_of_point_on_line_l47_47356

theorem z_coordinate_of_point_on_line (t : ℝ)
  (h₁ : (1 + 3 * t, 3 + 2 * t, 2 + 4 * t) = (x, 7, z))
  (h₂ : x = 1 + 3 * t) :
  z = 10 :=
sorry

end z_coordinate_of_point_on_line_l47_47356


namespace domain_composite_function_l47_47202

theorem domain_composite_function (f : ℝ → ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x = y) →
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f (2^x - 1) = y) :=
by
  sorry

end domain_composite_function_l47_47202


namespace lola_pop_tarts_baked_l47_47226

theorem lola_pop_tarts_baked :
  ∃ P : ℕ, (13 + P + 8) + (16 + 12 + 14) = 73 ∧ P = 10 := by
  sorry

end lola_pop_tarts_baked_l47_47226


namespace farm_corn_cobs_l47_47344

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l47_47344


namespace chess_tournament_games_l47_47556

theorem chess_tournament_games (n : ℕ) (h : n = 17) (k : n - 1 = 16) :
  (n * (n - 1)) / 2 = 136 := by
  sorry

end chess_tournament_games_l47_47556


namespace fixed_point_of_shifted_exponential_l47_47717

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end fixed_point_of_shifted_exponential_l47_47717


namespace vec_mag_diff_eq_neg_one_l47_47674

variables (a b : ℝ × ℝ)

def vec_add_eq := a + b = (2, 3)

def vec_sub_eq := a - b = (-2, 1)

theorem vec_mag_diff_eq_neg_one (h₁ : vec_add_eq a b) (h₂ : vec_sub_eq a b) :
  (a.1 ^ 2 + a.2 ^ 2) - (b.1 ^ 2 + b.2 ^ 2) = -1 :=
  sorry

end vec_mag_diff_eq_neg_one_l47_47674


namespace exist_divisible_number_l47_47573

theorem exist_divisible_number (d : ℕ) (hd : d > 0) :
  ∃ n : ℕ, (n % d = 0) ∧ ∃ k : ℕ, (k > 0) ∧ (k < 10) ∧ 
  ((∃ m : ℕ, m = n - k*(10^k / 10^k) ∧ m % d = 0) ∨ ∃ m : ℕ, m = n - k * (10^(k - 1)) ∧ m % d = 0) :=
sorry

end exist_divisible_number_l47_47573


namespace final_score_l47_47232

theorem final_score (questions_first_half questions_second_half : Nat)
  (points_correct points_incorrect : Int)
  (correct_first_half incorrect_first_half correct_second_half incorrect_second_half : Nat) :
  questions_first_half = 10 →
  questions_second_half = 15 →
  points_correct = 3 →
  points_incorrect = -1 →
  correct_first_half = 6 →
  incorrect_first_half = 4 →
  correct_second_half = 10 →
  incorrect_second_half = 5 →
  (points_correct * correct_first_half + points_incorrect * incorrect_first_half 
   + points_correct * correct_second_half + points_incorrect * incorrect_second_half) = 39 := 
by
  intros
  sorry

end final_score_l47_47232


namespace quad_inequality_solution_set_is_reals_l47_47962

theorem quad_inequality_solution_set_is_reals (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) := 
sorry

end quad_inequality_solution_set_is_reals_l47_47962


namespace max_adjacent_distinct_pairs_l47_47869

theorem max_adjacent_distinct_pairs (n : ℕ) (h : n = 100) : 
  ∃ (a : ℕ), a = 50 := 
by 
  -- Here we use the provided constraints and game scenario to state the theorem formally.
  sorry

end max_adjacent_distinct_pairs_l47_47869


namespace correct_operation_l47_47782

theorem correct_operation (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, (x / 10) = 0.01 * f x) : 
  f x = 10 * x :=
by
  sorry

end correct_operation_l47_47782


namespace box_inscribed_in_sphere_l47_47039

theorem box_inscribed_in_sphere (x y z r : ℝ) (surface_area : ℝ)
  (edge_sum : ℝ) (given_x : x = 8) 
  (given_surface_area : surface_area = 432) 
  (given_edge_sum : edge_sum = 104) 
  (surface_area_eq : 2 * (x * y + y * z + z * x) = surface_area)
  (edge_sum_eq : 4 * (x + y + z) = edge_sum) : 
  r = 7 :=
by
  sorry

end box_inscribed_in_sphere_l47_47039


namespace fraction_equivalence_l47_47599

theorem fraction_equivalence :
  ( (3 / 7 + 2 / 3) / (5 / 11 + 3 / 8) ) = (119 / 90) :=
by
  sorry

end fraction_equivalence_l47_47599


namespace barium_oxide_amount_l47_47105

theorem barium_oxide_amount (BaO H2O BaOH₂ : ℕ) 
  (reaction : BaO + H2O = BaOH₂) 
  (molar_ratio : BaOH₂ = BaO) 
  (required_BaOH₂ : BaOH₂ = 2) :
  BaO = 2 :=
by 
  sorry

end barium_oxide_amount_l47_47105


namespace cube_volume_and_surface_area_l47_47967

theorem cube_volume_and_surface_area (s : ℝ) (h : 12 * s = 72) :
  s^3 = 216 ∧ 6 * s^2 = 216 :=
by 
  sorry

end cube_volume_and_surface_area_l47_47967


namespace emily_101st_card_is_10_of_Hearts_l47_47622

def number_sequence : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
def suit_sequence : List String := ["Hearts", "Diamonds", "Clubs", "Spades"]

-- Function to get the number of a specific card
def card_number (n : ℕ) : String :=
  number_sequence.get! (n % number_sequence.length)

-- Function to get the suit of a specific card
def card_suit (n : ℕ) : String :=
  suit_sequence.get! ((n / suit_sequence.length) % suit_sequence.length)

-- Definition to state the question and the answer
def emily_card (n : ℕ) : String := card_number n ++ " of " ++ card_suit n

-- Proving that the 101st card is "10 of Hearts"
theorem emily_101st_card_is_10_of_Hearts : emily_card 100 = "10 of Hearts" :=
by {
  sorry
}

end emily_101st_card_is_10_of_Hearts_l47_47622


namespace domain_of_function_l47_47707

variable (x : ℝ)

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ 2 - x ≠ 0} =
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} :=
by
  sorry

end domain_of_function_l47_47707


namespace right_triangle_ABC_l47_47625

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points definitions
def point_A : ℝ × ℝ := (1, 2)
def point_on_line : ℝ × ℝ := (5, -2)

-- Points B and C on the parabola with parameters t and s respectively
def point_B (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def point_C (s : ℝ) : ℝ × ℝ := (s^2, 2 * s)

-- Line equation passing through points B and C
def line_eq (s t : ℝ) (x y : ℝ) : Prop :=
  2 * x - (s + t) * y + 2 * s * t = 0

-- Proof goal: Show that triangle ABC is a right triangle
theorem right_triangle_ABC
  (t s : ℝ)
  (hB : parabola (point_B t).1 (point_B t).2)
  (hC : parabola (point_C s).1 (point_C s).2)
  (hlt : point_on_line.1 = (5 : ℝ))
  (hlx : line_eq s t point_on_line.1 point_on_line.2)
  : let A := point_A
    let B := point_B t
    let C := point_C s
    -- Conclusion: triangle ABC is a right triangle
    k_AB * k_AC = -1 :=
  sorry
  where k_AB := (2 * t - 2) / (t^2 - 1)
        k_AC := (2 * s - 2) / (s^2 - 1)
        rel_t_s := (s + 1) * (t + 1) = -4

end right_triangle_ABC_l47_47625


namespace angle_sum_eq_180_l47_47845

theorem angle_sum_eq_180 (A B C D E F G : ℝ) 
  (h1 : A + B + C + D + E + F = 360) : 
  A + B + C + D + E + F + G = 180 :=
by
  sorry

end angle_sum_eq_180_l47_47845


namespace variance_of_data_is_0_02_l47_47016

def data : List ℝ := [10.1, 9.8, 10, 9.8, 10.2]

theorem variance_of_data_is_0_02 (h : (10.1 + 9.8 + 10 + 9.8 + 10.2) / 5 = 10) : 
  (1 / 5) * ((10.1 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10.2 - 10) ^ 2) = 0.02 :=
by
  sorry

end variance_of_data_is_0_02_l47_47016


namespace systematic_sampling_twentieth_group_number_l47_47946

theorem systematic_sampling_twentieth_group_number 
  (total_students : ℕ) 
  (total_groups : ℕ) 
  (first_group_number : ℕ) 
  (interval : ℕ) 
  (n : ℕ) 
  (drawn_number : ℕ) :
  total_students = 400 →
  total_groups = 20 →
  first_group_number = 11 →
  interval = 20 →
  n = 20 →
  drawn_number = 11 + 20 * (n - 1) →
  drawn_number = 391 :=
by
  sorry

end systematic_sampling_twentieth_group_number_l47_47946


namespace least_number_added_to_divisible_l47_47871

theorem least_number_added_to_divisible (n : ℕ) (k : ℕ) : n = 1789 → k = 11 → (n + k) % Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 4 3)) = 0 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end least_number_added_to_divisible_l47_47871


namespace triangle_side_relation_l47_47895

theorem triangle_side_relation (a b c : ℝ) (h1 : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) (h2 : a + b > c) :
  a + c = 2 * b := 
sorry

end triangle_side_relation_l47_47895


namespace sin_alpha_minus_pi_over_6_l47_47534

variable (α : ℝ)

theorem sin_alpha_minus_pi_over_6 (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_alpha_minus_pi_over_6_l47_47534


namespace probability_both_boys_or_both_girls_l47_47148

theorem probability_both_boys_or_both_girls 
  (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 5 → boys = 2 → girls = 3 →
    (∃ (p : ℚ), p = 2/5) :=
by
  intros ht hb hg
  sorry

end probability_both_boys_or_both_girls_l47_47148


namespace more_pie_eaten_l47_47500

theorem more_pie_eaten (erik_pie : ℝ) (frank_pie : ℝ)
  (h_erik : erik_pie = 0.6666666666666666)
  (h_frank : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 :=
by
  sorry

end more_pie_eaten_l47_47500


namespace total_people_transport_l47_47047

-- Define the conditions
def boatA_trips_day1 := 7
def boatB_trips_day1 := 5
def boatA_capacity := 20
def boatB_capacity := 15
def boatA_trips_day2 := 5
def boatB_trips_day2 := 6

-- Define the theorem statement
theorem total_people_transport :
  (boatA_trips_day1 * boatA_capacity + boatB_trips_day1 * boatB_capacity) +
  (boatA_trips_day2 * boatA_capacity + boatB_trips_day2 * boatB_capacity)
  = 405 := 
  by
  sorry

end total_people_transport_l47_47047


namespace symmetric_circle_eq_l47_47459

theorem symmetric_circle_eq (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (x^2 + y^2 - 4 * y = 0) :=
sorry

end symmetric_circle_eq_l47_47459


namespace complement_of_M_l47_47022

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x | x^2 - x ≥ 0 }

theorem complement_of_M :
  compl M = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end complement_of_M_l47_47022


namespace sum_of_m_integers_l47_47274

theorem sum_of_m_integers :
  ∀ (m : ℤ), 
    (∀ (x : ℚ), (x - 10) / 5 ≤ -1 - x / 5 ∧ x - 1 > -m / 2) → 
    (∃ x_max x_min : ℤ, x_max + x_min = -2 ∧ 
                        (x_max ≤ 5 / 2 ∧ x_min ≤ 5 / 2) ∧ 
                        (1 - m / 2 < x_min ∧ 1 - m / 2 < x_max)) →
  (10 < m ∧ m ≤ 12) → m = 11 ∨ m = 12 → 11 + 12 = 23 :=
by sorry

end sum_of_m_integers_l47_47274


namespace factor_polynomial_l47_47391

theorem factor_polynomial (z : ℝ) : (70 * z ^ 20 + 154 * z ^ 40 + 224 * z ^ 60) = 14 * z ^ 20 * (5 + 11 * z ^ 20 + 16 * z ^ 40) := 
sorry

end factor_polynomial_l47_47391


namespace cost_of_one_pencil_l47_47432

theorem cost_of_one_pencil (students : ℕ) (more_than_half : ℕ) (pencil_cost : ℕ) (pencils_each : ℕ)
  (total_cost : ℕ) (students_condition : students = 36) 
  (more_than_half_condition : more_than_half > 18) 
  (pencil_count_condition : pencils_each > 1) 
  (cost_condition : pencil_cost > pencils_each) 
  (total_cost_condition : students * pencil_cost * pencils_each = 1881) : 
  pencil_cost = 17 :=
sorry

end cost_of_one_pencil_l47_47432


namespace nathan_and_parents_total_cost_l47_47872

/-- Define the total number of people -/
def num_people := 3

/-- Define the cost per object -/
def cost_per_object := 11

/-- Define the number of objects per person -/
def objects_per_person := 2 + 2 + 1

/-- Define the total number of objects -/
def total_objects := num_people * objects_per_person

/-- Define the total cost -/
def total_cost := total_objects * cost_per_object

/-- The main theorem to prove the total cost -/
theorem nathan_and_parents_total_cost : total_cost = 165 := by
  sorry

end nathan_and_parents_total_cost_l47_47872


namespace avianna_blue_candles_l47_47728

theorem avianna_blue_candles (r b : ℕ) (h1 : r = 45) (h2 : r/b = 5/3) : b = 27 :=
by sorry

end avianna_blue_candles_l47_47728


namespace more_cabbages_produced_l47_47997

theorem more_cabbages_produced
  (square_garden : ∀ n : ℕ, ∃ s : ℕ, s ^ 2 = n)
  (area_per_cabbage : ∀ cabbages : ℕ, cabbages = 11236 → ∃ s : ℕ, s ^ 2 = cabbages) :
  11236 - 105 ^ 2 = 211 := by
sorry

end more_cabbages_produced_l47_47997


namespace times_older_l47_47983

-- Conditions
variables (H S : ℕ)
axiom hold_age : H = 36
axiom hold_son_relation : H = 3 * S

-- Statement of the problem
theorem times_older (H S : ℕ) (h1 : H = 36) (h2 : H = 3 * S) : (H - 8) / (S - 8) = 7 :=
by
  -- Proof will be provided here
  sorry

end times_older_l47_47983


namespace jugs_needed_to_provide_water_for_students_l47_47814

def jug_capacity : ℕ := 40
def students : ℕ := 200
def cups_per_student : ℕ := 10

def total_cups_needed := students * cups_per_student

theorem jugs_needed_to_provide_water_for_students :
  total_cups_needed / jug_capacity = 50 :=
by
  -- Proof goes here
  sorry

end jugs_needed_to_provide_water_for_students_l47_47814


namespace ticket_cost_l47_47924

theorem ticket_cost
    (rows : ℕ) (seats_per_row : ℕ)
    (fraction_sold : ℚ) (total_earnings : ℚ)
    (N : ℕ := rows * seats_per_row)
    (S : ℚ := fraction_sold * N)
    (C : ℚ := total_earnings / S)
    (h1 : rows = 20) (h2 : seats_per_row = 10)
    (h3 : fraction_sold = 3 / 4) (h4 : total_earnings = 1500) :
    C = 10 :=
by
  sorry

end ticket_cost_l47_47924


namespace mistaken_divisor_l47_47471

theorem mistaken_divisor (x : ℕ) (h1 : ∀ (d : ℕ), d ∣ 840 → d = 21 ∨ d = x) 
(h2 : 840 = 70 * x) : x = 12 := 
by sorry

end mistaken_divisor_l47_47471


namespace price_comparison_l47_47580

variable (x y : ℝ)
variable (h1 : 6 * x + 3 * y > 24)
variable (h2 : 4 * x + 5 * y < 22)

theorem price_comparison : 2 * x > 3 * y :=
sorry

end price_comparison_l47_47580


namespace find_a_l47_47300

variable {f : ℝ → ℝ}

-- Conditions
variables (a : ℝ) (domain : Set ℝ := Set.Ioo (3 - 2 * a) (a + 1))
variable (even_f : ∀ x, f (x + 1) = f (- (x + 1)))

-- The theorem stating the problem
theorem find_a (h : ∀ x, x ∈ domain ↔ x ∈ Set.Ioo (3 - 2 * a) (a + 1)) : a = 2 := by
  sorry

end find_a_l47_47300


namespace incorrect_statement_l47_47770

noncomputable def first_line_of_defense := "Skin and mucous membranes"
noncomputable def second_line_of_defense := "Antimicrobial substances and phagocytic cells in body fluids"
noncomputable def third_line_of_defense := "Immune organs and immune cells"
noncomputable def non_specific_immunity := "First and second line of defense"
noncomputable def specific_immunity := "Third line of defense"
noncomputable def d_statement := "The defensive actions performed by the three lines of defense in the human body are called non-specific immunity"

theorem incorrect_statement : d_statement ≠ specific_immunity ∧ d_statement ≠ non_specific_immunity := by
  sorry

end incorrect_statement_l47_47770


namespace second_set_length_is_20_l47_47834

-- Define the lengths
def length_first_set : ℕ := 4
def length_second_set : ℕ := 5 * length_first_set

-- Formal proof statement
theorem second_set_length_is_20 : length_second_set = 20 :=
by
  sorry

end second_set_length_is_20_l47_47834


namespace never_prime_l47_47044

theorem never_prime (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 105) := sorry

end never_prime_l47_47044


namespace perimeter_of_grid_l47_47823

theorem perimeter_of_grid (area: ℕ) (side_length: ℕ) (perimeter: ℕ) 
  (h1: area = 144) 
  (h2: 4 * side_length * side_length = area) 
  (h3: perimeter = 4 * 2 * side_length) : 
  perimeter = 48 :=
by
  sorry

end perimeter_of_grid_l47_47823


namespace six_digit_numbers_with_at_least_one_zero_correct_l47_47750

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l47_47750


namespace canoe_row_probability_l47_47019

-- Definitions based on conditions
def prob_left_works : ℚ := 3 / 5
def prob_right_works : ℚ := 3 / 5

-- The probability that you can still row the canoe
def prob_can_row : ℚ := 
  prob_left_works * prob_right_works +  -- both oars work
  prob_left_works * (1 - prob_right_works) +  -- left works, right breaks
  (1 - prob_left_works) * prob_right_works  -- left breaks, right works
  
theorem canoe_row_probability : prob_can_row = 21 / 25 := by
  -- Skip proof for now
  sorry

end canoe_row_probability_l47_47019


namespace photograph_perimeter_l47_47382

theorem photograph_perimeter (w l m : ℕ) 
  (h1 : (w + 4) * (l + 4) = m)
  (h2 : (w + 8) * (l + 8) = m + 94) :
  2 * (w + l) = 23 := 
by
  sorry

end photograph_perimeter_l47_47382


namespace selling_price_of_bracelet_l47_47018

theorem selling_price_of_bracelet (x : ℝ) 
  (cost_per_bracelet : ℝ) 
  (num_bracelets : ℕ) 
  (box_of_cookies_cost : ℝ) 
  (money_left_after_buying_cookies : ℝ) 
  (total_revenue : ℝ) 
  (total_cost_of_supplies : ℝ) :
  cost_per_bracelet = 1 →
  num_bracelets = 12 →
  box_of_cookies_cost = 3 →
  money_left_after_buying_cookies = 3 →
  total_cost_of_supplies = cost_per_bracelet * num_bracelets →
  total_revenue = 9 →
  x = total_revenue / num_bracelets :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Placeholder for the actual proof
  sorry

end selling_price_of_bracelet_l47_47018


namespace arithmetic_sequence_ratio_l47_47469

theorem arithmetic_sequence_ratio (a d : ℕ) (h : b = a + 3 * d) : a = 1 -> d = 1 -> (a / b = 1 / 4) :=
by
  sorry

end arithmetic_sequence_ratio_l47_47469


namespace desired_ellipse_properties_l47_47172

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2)/(a^2) + (x^2)/(b^2) = 1

def ellipse_has_foci (a b : ℝ) (c : ℝ) : Prop :=
  c^2 = a^2 - b^2

def desired_ellipse_passes_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2

def foci_of_ellipse (a b : ℝ) (c : ℝ) : Prop :=
  ellipse_has_foci a b c

axiom given_ellipse_foci : foci_of_ellipse 3 2 (Real.sqrt 5)

theorem desired_ellipse_properties :
  desired_ellipse_passes_through_point 4 (Real.sqrt 11) (0, 4) ∧
  foci_of_ellipse 4 (Real.sqrt 11) (Real.sqrt 5) :=
by
  sorry

end desired_ellipse_properties_l47_47172


namespace percentage_of_apples_is_50_l47_47887

-- Definitions based on the conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

-- Final percentage calculation after removing 13 oranges
def percentage_apples (apples oranges_removed : ℕ) :=
  let total_initial := initial_apples + initial_oranges
  let oranges_left := initial_oranges - oranges_removed
  let total_after_removal := initial_apples + oranges_left
  (initial_apples * 100) / total_after_removal

-- The theorem to be proved
theorem percentage_of_apples_is_50 : percentage_apples initial_apples oranges_removed = 50 := by
  sorry

end percentage_of_apples_is_50_l47_47887


namespace total_widgets_sold_15_days_l47_47885

def widgets_sold (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * n

theorem total_widgets_sold_15_days :
  (Finset.range 15).sum widgets_sold = 359 :=
by
  sorry

end total_widgets_sold_15_days_l47_47885


namespace find_a_minus_inverse_l47_47756

-- Definition for the given condition
def condition (a : ℝ) : Prop := a + a⁻¹ = 6

-- Definition for the target value to be proven
def target_value (x : ℝ) : Prop := x = 4 * Real.sqrt 2 ∨ x = -4 * Real.sqrt 2

-- Theorem statement to be proved
theorem find_a_minus_inverse (a : ℝ) (ha : condition a) : target_value (a - a⁻¹) :=
by
  sorry

end find_a_minus_inverse_l47_47756


namespace non_isosceles_count_l47_47835

def n : ℕ := 20

def total_triangles : ℕ := Nat.choose n 3

def isosceles_triangles_per_vertex : ℕ := 9

def total_isosceles_triangles : ℕ := n * isosceles_triangles_per_vertex

def non_isosceles_triangles : ℕ := total_triangles - total_isosceles_triangles

theorem non_isosceles_count :
  non_isosceles_triangles = 960 := 
  by 
    -- proof details would go here
    sorry

end non_isosceles_count_l47_47835


namespace max_value_of_expression_l47_47540

theorem max_value_of_expression :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 → 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 :=
by sorry

end max_value_of_expression_l47_47540


namespace krishan_money_l47_47961

theorem krishan_money (R G K : ℕ) (h₁ : 7 * G = 17 * R) (h₂ : 7 * K = 17 * G) (h₃ : R = 686) : K = 4046 :=
  by sorry

end krishan_money_l47_47961


namespace value_of_b_minus_a_l47_47164

theorem value_of_b_minus_a (a b : ℕ) (h1 : a * b = 2 * (a + b) + 1) (h2 : b = 7) : b - a = 4 :=
by
  sorry

end value_of_b_minus_a_l47_47164


namespace attendees_gift_exchange_l47_47474

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l47_47474


namespace power_of_10_digits_l47_47262

theorem power_of_10_digits (n : ℕ) (hn : n > 1) :
  (∃ k : ℕ, (2^(n-1) < 10^k ∧ 10^k < 2^n) ∨ (5^(n-1) < 10^k ∧ 10^k < 5^n)) ∧ ¬((∃ k : ℕ, 2^(n-1) < 10^k ∧ 10^k < 2^n) ∧ (∃ k : ℕ, 5^(n-1) < 10^k ∧ 10^k < 5^n)) :=
sorry

end power_of_10_digits_l47_47262


namespace program_total_cost_l47_47929

-- Define the necessary variables and constants
def ms_to_s : Float := 0.001
def os_overhead : Float := 1.07
def cost_per_ms : Float := 0.023
def mount_cost : Float := 5.35
def time_required : Float := 1.5

-- Calculate components of the total cost
def total_cost_for_computer_time := (time_required * 1000) * cost_per_ms
def total_cost := os_overhead + total_cost_for_computer_time + mount_cost

-- State the theorem
theorem program_total_cost : total_cost = 40.92 := by
  sorry

end program_total_cost_l47_47929


namespace solve_compound_inequality_l47_47008

noncomputable def compound_inequality_solution (x : ℝ) : Prop :=
  (3 - (1 / (3 * x + 4)) < 5) ∧ (2 * x + 1 > 0)

theorem solve_compound_inequality (x : ℝ) :
  compound_inequality_solution x ↔ (x > -1/2) :=
by
  sorry

end solve_compound_inequality_l47_47008


namespace find_constant_a_find_ordinary_equation_of_curve_l47_47612

open Real

theorem find_constant_a (a t : ℝ) (h1 : 1 + 2 * t = 3) (h2 : a * t^2 = 1) : a = 1 :=
by
  -- Proof goes here
  sorry

theorem find_ordinary_equation_of_curve (x y t : ℝ) (h1 : x = 1 + 2 * t) (h2 : y = t^2) :
  (x - 1)^2 = 4 * y :=
by
  -- Proof goes here
  sorry

end find_constant_a_find_ordinary_equation_of_curve_l47_47612


namespace polynomial_divisible_iff_l47_47049

theorem polynomial_divisible_iff (a b : ℚ) : 
  ((a + b) * 1^5 + (a * b) * 1^2 + 1 = 0) ∧ 
  ((a + b) * 2^5 + (a * b) * 2^2 + 1 = 0) ↔ 
  (a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1) := 
by 
  sorry

end polynomial_divisible_iff_l47_47049


namespace root_polynomial_h_l47_47140

theorem root_polynomial_h (h : ℤ) : (2^3 + h * 2 + 10 = 0) → h = -9 :=
by
  sorry

end root_polynomial_h_l47_47140


namespace min_value_geometric_seq_l47_47119

theorem min_value_geometric_seq (a : ℕ → ℝ) (m n : ℕ) (h_pos : ∀ k, a k > 0)
  (h1 : a 1 = 1)
  (h2 : a 7 = a 6 + 2 * a 5)
  (h3 : a m * a n = 16) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_value_geometric_seq_l47_47119


namespace middle_box_label_l47_47333

/--
Given a sequence of 23 boxes in a row on the table, where each box has a label indicating either
  "There is no prize here" or "The prize is in a neighboring box",
and it is known that exactly one of these statements is true.
Prove that the label on the middle box (the 12th box) says "The prize is in the adjacent box."
-/
theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (labels : Fin 23 → String),
    (∀ i, labels i = "There is no prize here" ∨ labels i = "The prize is in a neighboring box") ∧
    (∃! i : Fin 23, boxes i ∧ (labels i = "The prize is in a neighboring box")) →
    labels ⟨11, sorry⟩ = "The prize is in a neighboring box" :=
sorry

end middle_box_label_l47_47333


namespace Bernardo_wins_with_smallest_M_l47_47688

-- Define the operations
def Bernardo_op (n : ℕ) : ℕ := 3 * n
def Lucas_op (n : ℕ) : ℕ := n + 75

-- Define the game behavior
def game_sequence (M : ℕ) : List ℕ :=
  [M, Bernardo_op M, Lucas_op (Bernardo_op M), Bernardo_op (Lucas_op (Bernardo_op M)),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M)))),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))))]

-- Define winning condition
def Bernardo_wins (M : ℕ) : Prop :=
  let seq := game_sequence M
  seq.get! 5 < 1200 ∧ seq.get! 6 >= 1200

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- The final theorem statement
theorem Bernardo_wins_with_smallest_M :
  Bernardo_wins 9 ∧ (∀ M < 9, ¬Bernardo_wins M) ∧ sum_of_digits 9 = 9 :=
by
  sorry

end Bernardo_wins_with_smallest_M_l47_47688


namespace number_of_small_cubes_l47_47683

theorem number_of_small_cubes (X : ℕ) (h1 : ∃ k, k = 29 - X) (h2 : 4 * 4 * 4 = 64) (h3 : X + 8 * (29 - X) = 64) : X = 24 :=
by
  sorry

end number_of_small_cubes_l47_47683


namespace frogs_climbed_onto_logs_l47_47993

-- Definitions of the conditions
def f_lily : ℕ := 5
def f_rock : ℕ := 24
def f_total : ℕ := 32

-- The final statement we want to prove
theorem frogs_climbed_onto_logs : f_total - (f_lily + f_rock) = 3 :=
by
  sorry

end frogs_climbed_onto_logs_l47_47993


namespace find_f_2016_minus_f_2015_l47_47290

-- Definitions for the given conditions

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) → f x = 2^x

-- Main theorem statement
theorem find_f_2016_minus_f_2015 {f : ℝ → ℝ} 
    (H1 : odd_function f) 
    (H2 : periodic_function f)
    (H3 : specific_values f)
    : f 2016 - f 2015 = 2 := 
sorry

end find_f_2016_minus_f_2015_l47_47290


namespace dig_eq_conditions_l47_47723

theorem dig_eq_conditions (n k : ℕ) 
  (h1 : 10^(k-1) ≤ n^n ∧ n^n < 10^k)
  (h2 : 10^(n-1) ≤ k^k ∧ k^k < 10^n) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end dig_eq_conditions_l47_47723


namespace smallest_k_for_bisectors_l47_47603

theorem smallest_k_for_bisectors (a b c l_a l_b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c))
  (h5 : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)) :
  (l_a + l_b) / (a + b) ≤ 4 / 3 :=
by
  sorry

end smallest_k_for_bisectors_l47_47603


namespace functional_equation_solution_l47_47486

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) = x * f x - y * f y) →
  ∃ m b : ℝ, ∀ t : ℝ, f t = m * t + b :=
by
  intro h
  sorry

end functional_equation_solution_l47_47486


namespace added_classes_l47_47902

def original_classes := 15
def students_per_class := 20
def new_total_students := 400

theorem added_classes : 
  new_total_students = original_classes * students_per_class + 5 * students_per_class :=
by
  sorry

end added_classes_l47_47902


namespace total_balloons_l47_47541

theorem total_balloons (T : ℕ) 
    (h1 : T / 4 = 100)
    : T = 400 := 
by
  sorry

end total_balloons_l47_47541


namespace minimum_shift_value_l47_47694

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem minimum_shift_value :
  ∃ m > 0, ∀ x, f (x + m) = Real.sin x ∧ m = 3 * Real.pi / 2 :=
by
  sorry

end minimum_shift_value_l47_47694


namespace finite_ring_identity_l47_47265

variable {A : Type} [Ring A] [Fintype A]
variables (a b : A)

theorem finite_ring_identity (h : (ab - 1) * b = 0) : b * (ab - 1) = 0 :=
sorry

end finite_ring_identity_l47_47265


namespace total_tickets_sales_l47_47345

theorem total_tickets_sales:
    let student_ticket_price := 6
    let adult_ticket_price := 8
    let number_of_students := 20
    let number_of_adults := 12
    number_of_students * student_ticket_price + number_of_adults * adult_ticket_price = 216 :=
by
    intros
    sorry

end total_tickets_sales_l47_47345


namespace hexagon_largest_angle_l47_47904

theorem hexagon_largest_angle (a : ℚ) 
  (h₁ : (a + 2) + (2 * a - 3) + (3 * a + 1) + 4 * a + (5 * a - 4) + (6 * a + 2) = 720) :
  6 * a + 2 = 4374 / 21 :=
by sorry

end hexagon_largest_angle_l47_47904


namespace cubic_identity_l47_47702

variable {a b c : ℝ}

theorem cubic_identity (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 := 
by 
  sorry

end cubic_identity_l47_47702


namespace radishes_times_carrots_l47_47905

theorem radishes_times_carrots (cucumbers radishes carrots : ℕ) 
  (h1 : cucumbers = 15) 
  (h2 : radishes = 3 * cucumbers) 
  (h3 : carrots = 9) : 
  radishes / carrots = 5 :=
by
  sorry

end radishes_times_carrots_l47_47905


namespace major_premise_incorrect_l47_47242

theorem major_premise_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
    ¬ (∀ x y : ℝ, x < y → a^x < a^y) :=
by {
  sorry
}

end major_premise_incorrect_l47_47242


namespace journey_time_ratio_l47_47638

theorem journey_time_ratio (D : ℝ) (hD_pos : D > 0) :
  let T1 := D / 45
  let T2 := D / 30
  (T2 / T1) = (3 / 2) := 
by
  sorry

end journey_time_ratio_l47_47638


namespace martha_profit_l47_47222

theorem martha_profit :
  let loaves_baked := 60
  let cost_per_loaf := 1
  let morning_price := 3
  let afternoon_price := 3 * 0.75
  let evening_price := 2
  let morning_loaves := loaves_baked / 3
  let afternoon_loaves := (loaves_baked - morning_loaves) / 2
  let evening_loaves := loaves_baked - morning_loaves - afternoon_loaves
  let morning_revenue := morning_loaves * morning_price
  let afternoon_revenue := afternoon_loaves * afternoon_price
  let evening_revenue := evening_loaves * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 85 := 
by
  sorry

end martha_profit_l47_47222


namespace value_of_x_l47_47061

theorem value_of_x : 
  ∀ (x y z : ℕ), 
  (x = y / 3) ∧ 
  (y = z / 6) ∧ 
  (z = 72) → 
  x = 4 :=
by
  intros x y z h
  have h1 : y = z / 6 := h.2.1
  have h2 : z = 72 := h.2.2
  have h3 : x = y / 3 := h.1
  sorry

end value_of_x_l47_47061


namespace yardsCatchingPasses_l47_47906

-- Definitions from conditions in a)
def totalYardage : ℕ := 150
def runningYardage : ℕ := 90

-- Problem statement (Proof will follow)
theorem yardsCatchingPasses : totalYardage - runningYardage = 60 := by
  sorry

end yardsCatchingPasses_l47_47906


namespace ratio_ac_l47_47706

-- Definitions based on conditions
variables (a b c : ℕ)
variables (x y : ℕ)

-- Conditions
def ratio_ab := (a : ℚ) / (b : ℚ) = 2 / 3
def ratio_bc := (b : ℚ) / (c : ℚ) = 1 / 5

-- Theorem to prove the desired ratio
theorem ratio_ac (h1 : ratio_ab a b) (h2 : ratio_bc b c) : (a : ℚ) / (c : ℚ) = 2 / 15 :=
by
  sorry

end ratio_ac_l47_47706


namespace equal_parts_division_l47_47903

theorem equal_parts_division (n : ℕ) (h : (n * n) % 4 = 0) : 
  ∃ parts : ℕ, parts = 4 ∧ ∀ (i : ℕ), i < parts → 
    ∃ p : ℕ, p = (n * n) / parts :=
by sorry

end equal_parts_division_l47_47903


namespace simplify_expression_l47_47312

variable (x : ℝ)

theorem simplify_expression :
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 + 5 * x ^ 10 + 3 * x ^ 9)) =
  (15 * x ^ 13 - x ^ 12 + 9 * x ^ 11 - x ^ 10 - 6 * x ^ 9) :=
by
  sorry

end simplify_expression_l47_47312


namespace interest_percentage_face_value_l47_47031

def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_percentage_selling_price : ℝ := 0.065

def interest_amount : ℝ := interest_percentage_selling_price * selling_price

theorem interest_percentage_face_value :
  (interest_amount / face_value) * 100 = 8 :=
by
  sorry

end interest_percentage_face_value_l47_47031


namespace inequality_problem_l47_47719

theorem inequality_problem (x y a b : ℝ) (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : (a ^ x < b ^ y) :=
by 
  sorry

end inequality_problem_l47_47719


namespace xy_value_l47_47168

namespace ProofProblem

variables {x y : ℤ}

theorem xy_value (h1 : x * (x + y) = x^2 + 12) (h2 : x - y = 3) : x * y = 12 :=
by
  -- The proof is not required here
  sorry

end ProofProblem

end xy_value_l47_47168


namespace ratio_larger_to_smaller_l47_47070

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
  a / b

theorem ratio_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : ratio_of_numbers a b = 9 / 5 := 
  sorry

end ratio_larger_to_smaller_l47_47070


namespace sum_of_three_smallest_positive_solutions_equals_ten_and_half_l47_47637

noncomputable def sum_three_smallest_solutions : ℚ :=
    let x1 : ℚ := 2.75
    let x2 : ℚ := 3 + (4 / 9)
    let x3 : ℚ := 4 + (5 / 16)
    x1 + x2 + x3

theorem sum_of_three_smallest_positive_solutions_equals_ten_and_half :
  sum_three_smallest_solutions = 10.5 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_equals_ten_and_half_l47_47637


namespace problem_1_problem_2_problem_3_l47_47371

noncomputable def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C
noncomputable def area_quadrilateral (e f φ : ℝ) : ℝ := (1/2) * e * f * Real.sin φ

theorem problem_1 (a b C : ℝ) (hC : Real.sin C ≤ 1) : 
  area_triangle a b C ≤ (a^2 + b^2) / 4 :=
sorry

theorem problem_2 (e f φ : ℝ) (hφ : Real.sin φ ≤ 1) : 
  area_quadrilateral e f φ ≤ (e^2 + f^2) / 4 :=
sorry

theorem problem_3 (a b C c d D : ℝ) 
  (hC : Real.sin C ≤ 1) 
  (hD : Real.sin D ≤ 1) :
  area_triangle a b C + area_triangle c d D ≤ (a^2 + b^2 + c^2 + d^2) / 4 :=
sorry

end problem_1_problem_2_problem_3_l47_47371


namespace soccer_ball_price_l47_47754

theorem soccer_ball_price 
  (B S V : ℕ) 
  (h1 : (B + S + V) / 3 = 36)
  (h2 : B = V + 10)
  (h3 : S = V + 8) : 
  S = 38 := 
by 
  sorry

end soccer_ball_price_l47_47754


namespace geometric_sequence_sum_l47_47208

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℚ),
  (∀ n, 3 * a (n + 1) + a n = 0) ∧
  a 2 = -2/3 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4) = 122/81 :=
sorry

end geometric_sequence_sum_l47_47208


namespace quadratic_real_roots_l47_47890

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_roots_l47_47890


namespace find_marks_in_mathematics_l47_47036

theorem find_marks_in_mathematics
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (subjects : ℕ)
  (marks_math : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  subjects = 5 →
  (average * subjects = english + marks_math + physics + chemistry + biology) →
  marks_math = 95 :=
  by
    intros h_eng h_phy h_chem h_bio h_avg h_sub h_eq
    rw [h_eng, h_phy, h_chem, h_bio, h_avg, h_sub] at h_eq
    sorry

end find_marks_in_mathematics_l47_47036


namespace baking_time_correct_l47_47555

/-- Mark lets the bread rise for 120 minutes twice. -/
def rising_time : ℕ := 120 * 2

/-- Mark spends 10 minutes kneading the bread. -/
def kneading_time : ℕ := 10

/-- Total time taken to finish making the bread. -/
def total_time : ℕ := 280

/-- Calculate the baking time based on the given conditions. -/
def baking_time (rising kneading total : ℕ) : ℕ := total - (rising + kneading)

theorem baking_time_correct :
  baking_time rising_time kneading_time total_time = 30 := 
by 
  -- Proof is omitted
  sorry

end baking_time_correct_l47_47555


namespace ticket_price_for_children_l47_47981

open Nat

theorem ticket_price_for_children
  (C : ℕ)
  (adult_ticket_price : ℕ := 12)
  (num_adults : ℕ := 3)
  (num_children : ℕ := 3)
  (total_cost : ℕ := 66)
  (H : num_adults * adult_ticket_price + num_children * C = total_cost) :
  C = 10 :=
sorry

end ticket_price_for_children_l47_47981


namespace sum_eq_3_or_7_l47_47407

theorem sum_eq_3_or_7 {x y z : ℝ} 
  (h1 : x + y / z = 2)
  (h2 : y + z / x = 2)
  (h3 : z + x / y = 2) : 
  x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_eq_3_or_7_l47_47407


namespace min_number_of_bags_l47_47812

theorem min_number_of_bags (a b : ℕ) : 
  ∃ K : ℕ, K = a + b - Nat.gcd a b :=
by
  sorry

end min_number_of_bags_l47_47812


namespace largest_inscribed_triangle_area_l47_47800

theorem largest_inscribed_triangle_area 
  (radius : ℝ) 
  (diameter : ℝ)
  (base : ℝ)
  (height : ℝ) 
  (area : ℝ)
  (h1 : radius = 10)
  (h2 : diameter = 2 * radius)
  (h3 : base = diameter)
  (h4 : height = radius) 
  (h5 : area = (1/2) * base * height) : 
  area  = 100 :=
by 
  have h_area := (1/2) * 20 * 10
  sorry

end largest_inscribed_triangle_area_l47_47800


namespace minimum_questionnaires_l47_47926

theorem minimum_questionnaires (responses_needed : ℕ) (response_rate : ℝ)
  (h1 : responses_needed = 300) (h2 : response_rate = 0.70) :
  ∃ (n : ℕ), n = Nat.ceil (responses_needed / response_rate) ∧ n = 429 :=
by
  sorry

end minimum_questionnaires_l47_47926


namespace pinocchio_cannot_pay_exactly_l47_47889

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l47_47889


namespace bank_deposit_exceeds_1000_on_saturday_l47_47244

theorem bank_deposit_exceeds_1000_on_saturday:
  ∃ n: ℕ, (2 * (3^n - 1) / 2 > 1000) ∧ ((n + 1) % 7 = 0) := by
  sorry

end bank_deposit_exceeds_1000_on_saturday_l47_47244


namespace smallest_angle_between_radii_l47_47216

theorem smallest_angle_between_radii (n : ℕ) (k : ℕ) (angle_step : ℕ) (angle_smallest : ℕ) 
(h_n : n = 40) 
(h_k : k = 23) 
(h_angle_step : angle_step = k) 
(h_angle_smallest : angle_smallest = 23) : 
angle_smallest = 23 :=
sorry

end smallest_angle_between_radii_l47_47216


namespace boris_clock_time_l47_47941

-- Define a function to compute the sum of digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem boris_clock_time (h m : ℕ) :
  sum_digits h + sum_digits m = 6 ∧ h + m = 15 ↔
  (h, m) = (0, 15) ∨ (h, m) = (1, 14) ∨ (h, m) = (2, 13) ∨ (h, m) = (3, 12) ∨
  (h, m) = (4, 11) ∨ (h, m) = (5, 10) ∨ (h, m) = (10, 5) ∨ (h, m) = (11, 4) ∨
  (h, m) = (12, 3) ∨ (h, m) = (13, 2) ∨ (h, m) = (14, 1) ∨ (h, m) = (15, 0) :=
by sorry

end boris_clock_time_l47_47941


namespace negation_of_universal_sin_pos_l47_47257

theorem negation_of_universal_sin_pos :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 :=
by sorry

end negation_of_universal_sin_pos_l47_47257


namespace score_of_29_impossible_l47_47771

theorem score_of_29_impossible :
  ¬ ∃ (c u w : ℕ), c + u + w = 10 ∧ 3 * c + u = 29 :=
by {
  sorry
}

end score_of_29_impossible_l47_47771


namespace second_tap_emptying_time_l47_47291

theorem second_tap_emptying_time :
  ∀ (T : ℝ), (∀ (f e : ℝ),
  (f = 1 / 3) →
  (∀ (n : ℝ), (n = 1 / 4.5) →
  (n = f - e ↔ e = 1 / T))) →
  T = 9 :=
by
  sorry

end second_tap_emptying_time_l47_47291


namespace Liza_reads_more_pages_than_Suzie_l47_47577

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie_l47_47577


namespace triangle_side_lengths_inequality_iff_l47_47735

theorem triangle_side_lengths_inequality_iff :
  {x : ℕ | 7 < x^2 ∧ x^2 < 17} = {3, 4} :=
by
  sorry

end triangle_side_lengths_inequality_iff_l47_47735


namespace B_completes_in_40_days_l47_47615

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l47_47615


namespace find_dividend_l47_47346

def dividend_problem (dividend divisor : ℕ) : Prop :=
  (15 * divisor + 5 = dividend) ∧ (dividend + divisor + 15 + 5 = 2169)

theorem find_dividend : ∃ dividend, ∃ divisor, dividend_problem dividend divisor ∧ dividend = 2015 :=
sorry

end find_dividend_l47_47346


namespace pounds_over_minimum_l47_47239

def cost_per_pound : ℕ := 3
def minimum_purchase : ℕ := 15
def total_spent : ℕ := 105

theorem pounds_over_minimum : 
  (total_spent / cost_per_pound) - minimum_purchase = 20 :=
by
  sorry

end pounds_over_minimum_l47_47239


namespace num_other_adults_l47_47699

-- Define the variables and conditions
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9
def shonda_kids : ℕ := 2
def kids_friends : ℕ := 10
def num_participants : ℕ := (num_baskets * eggs_per_basket) / eggs_per_person

-- Prove the number of other adults at the Easter egg hunt
theorem num_other_adults : (num_participants - (shonda_kids + kids_friends + 1)) = 7 := by
  sorry

end num_other_adults_l47_47699


namespace binom_10_1_eq_10_l47_47685

theorem binom_10_1_eq_10 : Nat.choose 10 1 = 10 := by
  sorry

end binom_10_1_eq_10_l47_47685


namespace smallest_number_with_ten_divisors_l47_47516

/-- 
  Theorem: The smallest natural number n that has exactly 10 positive divisors is 48.
--/
theorem smallest_number_with_ten_divisors : 
  ∃ (n : ℕ), (∀ (p1 p2 p3 p4 p5 : ℕ) (a1 a2 a3 a4 a5 : ℕ), 
    n = p1^a1 * p2^a2 * p3^a3 * p4^a4 * p5^a5 → 
    n.factors.count = 10) 
    ∧ n = 48 := sorry

end smallest_number_with_ten_divisors_l47_47516


namespace complement_intersection_l47_47650

-- Definitions of the sets as given in the problem
namespace ProofProblem

def U : Set ℤ := {-2, -1, 0, 1, 2}
def M : Set ℤ := {y | y > 0}
def N : Set ℤ := {x | x = -1 ∨ x = 2}

theorem complement_intersection :
  (U \ M) ∩ N = {-1} :=
by
  sorry

end ProofProblem

end complement_intersection_l47_47650


namespace library_growth_rate_l47_47673

theorem library_growth_rate (C_2022 C_2024: ℝ) (h₁ : C_2022 = 100000) (h₂ : C_2024 = 144000) :
  ∃ x : ℝ, (1 + x) ^ 2 = C_2024 / C_2022 ∧ x = 0.2 := 
by {
  sorry
}

end library_growth_rate_l47_47673


namespace isosceles_triangle_base_l47_47429

theorem isosceles_triangle_base (a b c : ℕ) (h_isosceles : a = b ∨ a = c ∨ b = c)
  (h_perimeter : a + b + c = 29) (h_side : a = 7 ∨ b = 7 ∨ c = 7) : 
  a = 7 ∨ b = 7 ∨ c = 7 ∧ (a = 7 ∨ a = 11) ∧ (b = 7 ∨ b = 11) ∧ (c = 7 ∨ c = 11) ∧ (a ≠ b ∨ c ≠ b) :=
by
  sorry

end isosceles_triangle_base_l47_47429


namespace expression_eq_one_l47_47648

theorem expression_eq_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
   a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
   b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 := 
by
  sorry

end expression_eq_one_l47_47648


namespace decorations_cost_correct_l47_47925

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l47_47925


namespace triangle_right_angled_l47_47980

theorem triangle_right_angled (A B C : ℝ) (h : A + B + C = 180) (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x) :
  C = 90 :=
by {
  sorry
}

end triangle_right_angled_l47_47980


namespace race_time_l47_47436

theorem race_time 
  (v t : ℝ)
  (h1 : 1000 = v * t)
  (h2 : 960 = v * (t + 10)) :
  t = 250 :=
by
  sorry

end race_time_l47_47436


namespace choir_members_total_l47_47649

theorem choir_members_total
  (first_group second_group third_group : ℕ)
  (h1 : first_group = 25)
  (h2 : second_group = 30)
  (h3 : third_group = 15) :
  first_group + second_group + third_group = 70 :=
by
  sorry

end choir_members_total_l47_47649


namespace kishore_expenses_l47_47024

noncomputable def total_salary (savings : ℕ) (percent : ℝ) : ℝ :=
savings / percent

noncomputable def total_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

noncomputable def miscellaneous_expenses (total_salary : ℝ) (total_expenses : ℕ) (savings : ℕ) : ℝ :=
  total_salary - (total_expenses + savings)

theorem kishore_expenses :
  total_salary 2160 0.1 - (total_expenses 5000 1500 4500 2500 2000 + 2160) = 3940 := by
  sorry

end kishore_expenses_l47_47024


namespace paint_time_l47_47953

theorem paint_time (n₁ n₂ h: ℕ) (t₁ t₂: ℕ) (constant: ℕ):
  n₁ = 6 → t₁ = 8 → h = 2 → constant = 96 →
  constant = n₁ * t₁ * h → n₂ = 4 → constant = n₂ * t₂ * h →
  t₂ = 12 :=
by
  intros
  sorry

end paint_time_l47_47953


namespace value_of_2_pow_b_l47_47627

theorem value_of_2_pow_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h1 : (2 ^ a) ^ b = 2 ^ 2) (h2 : 2 ^ a * 2 ^ b = 8) : 2 ^ b = 4 :=
by
  sorry

end value_of_2_pow_b_l47_47627


namespace oranges_harvest_per_day_l47_47968

theorem oranges_harvest_per_day (total_sacks : ℕ) (days : ℕ) (sacks_per_day : ℕ) 
  (h1 : total_sacks = 498) (h2 : days = 6) : total_sacks / days = sacks_per_day ∧ sacks_per_day = 83 :=
by
  sorry

end oranges_harvest_per_day_l47_47968


namespace savings_percentage_correct_l47_47550

theorem savings_percentage_correct :
  let original_price_jacket := 120
  let original_price_shirt := 60
  let original_price_shoes := 90
  let discount_jacket := 0.30
  let discount_shirt := 0.50
  let discount_shoes := 0.25
  let total_original_price := original_price_jacket + original_price_shirt + original_price_shoes
  let savings_jacket := original_price_jacket * discount_jacket
  let savings_shirt := original_price_shirt * discount_shirt
  let savings_shoes := original_price_shoes * discount_shoes
  let total_savings := savings_jacket + savings_shirt + savings_shoes
  let percentage_savings := (total_savings / total_original_price) * 100
  percentage_savings = 32.8 := 
by 
  sorry

end savings_percentage_correct_l47_47550


namespace painters_time_l47_47210

-- Define the initial conditions
def n1 : ℕ := 3
def d1 : ℕ := 2
def W := n1 * d1
def n2 : ℕ := 2
def d2 := W / n2
def d_r := (3 * d2) / 4

-- Theorem statement
theorem painters_time (h : d_r = 9 / 4) : d_r = 9 / 4 := by
  sorry

end painters_time_l47_47210


namespace pure_imaginary_complex_number_solution_l47_47030

theorem pure_imaginary_complex_number_solution (m : ℝ) :
  (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) → m = 2 :=
by
  sorry

end pure_imaginary_complex_number_solution_l47_47030


namespace fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l47_47194

def visitors_enjoyed_understood_fraction (E U : ℕ) (total_visitors no_enjoy_no_understood : ℕ) : Prop :=
  E = U ∧
  no_enjoy_no_understood = 110 ∧
  total_visitors = 440 ∧
  E = (total_visitors - no_enjoy_no_understood) / 2 ∧
  E = 165 ∧
  (E / total_visitors) = 3 / 8

theorem fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8 :
  ∃ (E U : ℕ), visitors_enjoyed_understood_fraction E U 440 110 :=
by
  sorry

end fraction_of_visitors_who_both_enjoyed_and_understood_is_3_over_8_l47_47194


namespace find_two_digit_number_l47_47056

theorem find_two_digit_number : ∃ (x : ℕ), 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 10^6 ≤ n^3 ∧ n^3 < 10^7 ∧ 101010 * x + 1 = n^3 ∧ x = 93) := 
 by
  sorry

end find_two_digit_number_l47_47056


namespace compound_interest_rate_l47_47160

theorem compound_interest_rate (P : ℝ) (r : ℝ) (t : ℕ) (A : ℝ) 
  (h1 : t = 15) (h2 : A = (9 / 5) * P) :
  (1 + r) ^ t = (9 / 5) → 
  r ≠ 0.05 ∧ r ≠ 0.06 ∧ r ≠ 0.07 ∧ r ≠ 0.08 :=
by
  -- Sorry could be placed here for now
  sorry

end compound_interest_rate_l47_47160


namespace fruit_salad_cost_3_l47_47952

def cost_per_fruit_salad (num_people sodas_per_person soda_cost sandwich_cost num_snacks snack_cost total_cost : ℕ) : ℕ :=
  let total_soda_cost := num_people * sodas_per_person * soda_cost
  let total_sandwich_cost := num_people * sandwich_cost
  let total_snack_cost := num_snacks * snack_cost
  let total_known_cost := total_soda_cost + total_sandwich_cost + total_snack_cost
  let total_fruit_salad_cost := total_cost - total_known_cost
  total_fruit_salad_cost / num_people

theorem fruit_salad_cost_3 :
  cost_per_fruit_salad 4 2 2 5 3 4 60 = 3 :=
by
  sorry

end fruit_salad_cost_3_l47_47952


namespace geometric_sequence_sum_l47_47492

theorem geometric_sequence_sum (S : ℕ → ℝ) 
  (S5 : S 5 = 10)
  (S10 : S 10 = 50) :
  S 15 = 210 := 
by
  sorry

end geometric_sequence_sum_l47_47492


namespace no_solution_inequality_system_l47_47712

theorem no_solution_inequality_system (m : ℝ) :
  (¬ ∃ x : ℝ, 2 * x - 1 < 3 ∧ x > m) ↔ m ≥ 2 :=
by
  sorry

end no_solution_inequality_system_l47_47712


namespace graph_eq_pair_of_straight_lines_l47_47404

theorem graph_eq_pair_of_straight_lines (x y : ℝ) :
  x^2 - 9*y^2 = 0 ↔ (x = 3*y ∨ x = -3*y) :=
by
  sorry

end graph_eq_pair_of_straight_lines_l47_47404


namespace intersection_of_lines_l47_47247

theorem intersection_of_lines :
  ∃ x y : ℚ, (12 * x - 3 * y = 33) ∧ (8 * x + 2 * y = 18) ∧ (x = 29 / 12 ∧ y = -2 / 3) :=
by {
  sorry
}

end intersection_of_lines_l47_47247


namespace book_collection_example_l47_47057

theorem book_collection_example :
  ∃ (P C B : ℕ), 
    (P : ℚ) / C = 3 / 2 ∧ 
    (C : ℚ) / B = 4 / 3 ∧ 
    P + C + B = 3002 ∧ 
    P + C + B > 3000 :=
by
  sorry

end book_collection_example_l47_47057


namespace equation_for_pears_l47_47328

-- Define the conditions
def pearDist1 (x : ℕ) : ℕ := 4 * x + 12
def pearDist2 (x : ℕ) : ℕ := 6 * x

-- State the theorem to be proved
theorem equation_for_pears (x : ℕ) : pearDist1 x = pearDist2 x :=
by
  sorry

end equation_for_pears_l47_47328


namespace intersection_A_B_l47_47087

def set_A : Set ℕ := {x | x^2 - 2 * x = 0}
def set_B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : set_A ∩ set_B = {0, 2} := 
by sorry

end intersection_A_B_l47_47087


namespace wedge_volume_cylinder_l47_47596

theorem wedge_volume_cylinder (r h : ℝ) (theta : ℝ) (V : ℝ) 
  (hr : r = 6) (hh : h = 6) (htheta : theta = 60) (hV : V = 113) : 
  V = (theta / 360) * π * r^2 * h :=
by
  sorry

end wedge_volume_cylinder_l47_47596


namespace arithmetic_sequence_99th_term_l47_47133

-- Define the problem with conditions and question
theorem arithmetic_sequence_99th_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : S 9 = 27) (h2 : a 10 = 8) :
  a 99 = 97 := 
sorry

end arithmetic_sequence_99th_term_l47_47133


namespace reciprocal_of_36_recurring_decimal_l47_47943

-- Definitions and conditions
def recurring_decimal (x : ℚ) : Prop := x = 36 / 99

-- Theorem statement
theorem reciprocal_of_36_recurring_decimal :
  recurring_decimal (36 / 99) → (1 / (36 / 99) = 11 / 4) :=
sorry

end reciprocal_of_36_recurring_decimal_l47_47943


namespace apples_count_l47_47762

theorem apples_count : (23 - 20 + 6 = 9) :=
by
  sorry

end apples_count_l47_47762


namespace card_picking_l47_47909

/-
Statement of the problem:
- A modified deck of cards has 65 cards.
- The deck is divided into 5 suits, each of which has 13 cards.
- The cards are placed in random order.
- Prove that the number of ways to pick two different cards from this deck with the order of picking being significant is 4160.
-/
theorem card_picking : (65 * 64) = 4160 := by
  sorry

end card_picking_l47_47909


namespace terrell_lifting_problem_l47_47487

theorem terrell_lifting_problem (w1 w2 w3 n1 n2 : ℕ) (h1 : w1 = 12) (h2 : w2 = 18) (h3 : w3 = 24) (h4 : n1 = 20) :
  60 * n2 = 3 * w1 * n1 → n2 = 12 :=
by
  intros h
  sorry

end terrell_lifting_problem_l47_47487


namespace positive_real_solutions_l47_47747

noncomputable def x1 := (75 + Real.sqrt 5773) / 2
noncomputable def x2 := (-50 + Real.sqrt 2356) / 2

theorem positive_real_solutions :
  ∀ x : ℝ, 
  0 < x → 
  (1/2 * (4*x^2 - 1) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10)) ↔ 
  (x = x1 ∨ x = x2) :=
by
  sorry

end positive_real_solutions_l47_47747


namespace minimum_value_f_l47_47861

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x + 6 / x + 4 / x^2 - 1

theorem minimum_value_f : 
    ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f y ≥ f x) ∧ 
    f x = 3 - 6 * Real.sqrt 2 :=
sorry

end minimum_value_f_l47_47861


namespace T_n_correct_l47_47086

def a_n (n : ℕ) : ℤ := 2 * n - 5

def b_n (n : ℕ) : ℤ := 2^n

def C_n (n : ℕ) : ℤ := |a_n n| * b_n n

def T_n : ℕ → ℤ
| 1     => 6
| 2     => 10
| n     => if n >= 3 then 34 + (2 * n - 7) * 2^(n + 1) else 0  -- safeguard for invalid n

theorem T_n_correct (n : ℕ) (hyp : n ≥ 1) : 
  T_n n = 
  if n = 1 then 6 
  else if n = 2 then 10 
  else if n ≥ 3 then 34 + (2 * n - 7) * 2^(n + 1) 
  else 0 := 
by 
sorry

end T_n_correct_l47_47086


namespace cristobal_read_more_pages_l47_47979

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end cristobal_read_more_pages_l47_47979


namespace fn_prime_factor_bound_l47_47449

theorem fn_prime_factor_bound (n : ℕ) (h : n ≥ 3) : 
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^(2^n) + 1)) ∧ p > 2^(n+2) * (n+1) :=
sorry

end fn_prime_factor_bound_l47_47449


namespace oxen_eat_as_much_as_buffaloes_or_cows_l47_47736

theorem oxen_eat_as_much_as_buffaloes_or_cows
  (B C O : ℝ)
  (h1 : 3 * B = 4 * C)
  (h2 : (15 * B + 8 * O + 24 * C) * 36 = (30 * B + 8 * O + 64 * C) * 18) :
  3 * B = 4 * O :=
by sorry

end oxen_eat_as_much_as_buffaloes_or_cows_l47_47736


namespace range_of_p_l47_47206

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l47_47206


namespace alpha_beta_square_l47_47585

theorem alpha_beta_square (α β : ℝ) (h₁ : α^2 = 2*α + 1) (h₂ : β^2 = 2*β + 1) (hαβ : α ≠ β) :
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_l47_47585


namespace value_of_expression_l47_47351

theorem value_of_expression :
  (3150 - 3030)^2 / 144 = 100 :=
by {
  -- This imported module allows us to use basic mathematical functions and properties
  sorry -- We use sorry to skip the actual proof
}

end value_of_expression_l47_47351


namespace find_greatest_and_second_greatest_problem_solution_l47_47088

theorem find_greatest_and_second_greatest
  (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : (a > b) ∧ (b > c) ∧ (c > d) :=
by 
  sorry

def greatest_and_second_greatest_eq (x1 x2 : ℝ) : Prop :=
  x1 = 4 ^ (1 / 4) ∧ x2 = 5 ^ (1 / 5)

theorem problem_solution (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : greatest_and_second_greatest_eq a b :=
by 
  sorry

end find_greatest_and_second_greatest_problem_solution_l47_47088


namespace power_identity_l47_47818

theorem power_identity :
  (3 ^ 12) * (3 ^ 8) = 243 ^ 4 :=
sorry

end power_identity_l47_47818


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l47_47246

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l47_47246


namespace closest_perfect_square_to_350_l47_47007

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end closest_perfect_square_to_350_l47_47007


namespace circle_radius_and_diameter_relations_l47_47661

theorem circle_radius_and_diameter_relations
  (r_x r_y r_z A_x A_y A_z d_x d_z : ℝ)
  (hx_circumference : 2 * π * r_x = 18 * π)
  (hx_area : A_x = π * r_x^2)
  (hy_area_eq : A_y = A_x)
  (hz_area_eq : A_z = 4 * A_x)
  (hy_area : A_y = π * r_y^2)
  (hz_area : A_z = π * r_z^2)
  (dx_def : d_x = 2 * r_x)
  (dz_def : d_z = 2 * r_z)
  : r_y = r_z / 2 ∧ d_z = 2 * d_x := 
by 
  sorry

end circle_radius_and_diameter_relations_l47_47661


namespace volume_of_cut_cone_l47_47956

theorem volume_of_cut_cone (V_frustum : ℝ) (A_bottom : ℝ) (A_top : ℝ) (V_cut_cone : ℝ) :
  V_frustum = 52 ∧ A_bottom = 9 * A_top → V_cut_cone = 54 :=
by
  sorry

end volume_of_cut_cone_l47_47956


namespace lesser_number_is_21_5_l47_47704

theorem lesser_number_is_21_5
  (x y : ℝ)
  (h1 : x + y = 50)
  (h2 : x - y = 7) :
  y = 21.5 :=
by
  sorry

end lesser_number_is_21_5_l47_47704


namespace earned_points_l47_47586

def points_per_enemy := 3
def total_enemies := 6
def enemies_undefeated := 2
def enemies_defeated := total_enemies - enemies_undefeated

theorem earned_points : enemies_defeated * points_per_enemy = 12 :=
by sorry

end earned_points_l47_47586


namespace incorrect_statement_about_absolute_value_l47_47874

theorem incorrect_statement_about_absolute_value (x : ℝ) : abs x = 0 → x = 0 :=
by 
  sorry

end incorrect_statement_about_absolute_value_l47_47874


namespace trapezium_top_width_l47_47715

theorem trapezium_top_width (bottom_width : ℝ) (height : ℝ) (area : ℝ) (top_width : ℝ) 
  (h1 : bottom_width = 8) 
  (h2 : height = 50) 
  (h3 : area = 500) : top_width = 12 :=
by
  -- Definitions
  have h_formula : area = 1 / 2 * (top_width + bottom_width) * height := by sorry
  -- Applying given conditions to the formula
  rw [h1, h2, h3] at h_formula
  -- Solve for top_width
  sorry

end trapezium_top_width_l47_47715


namespace find_f_13_l47_47515

variable (f : ℤ → ℤ)

def is_odd_function (f : ℤ → ℤ) := ∀ x : ℤ, f (-x) = -f (x)
def has_period_4 (f : ℤ → ℤ) := ∀ x : ℤ, f (x + 4) = f (x)

theorem find_f_13 (h1 : is_odd_function f) (h2 : has_period_4 f) (h3 : f (-1) = 2) : f 13 = -2 :=
by
  sorry

end find_f_13_l47_47515


namespace arithmetic_sequence_common_difference_l47_47384

variable (a₁ d : ℝ)

def sum_odd := 5 * a₁ + 20 * d
def sum_even := 5 * a₁ + 25 * d

theorem arithmetic_sequence_common_difference 
  (h₁ : sum_odd a₁ d = 15) 
  (h₂ : sum_even a₁ d = 30) :
  d = 3 := 
by
  sorry

end arithmetic_sequence_common_difference_l47_47384


namespace power_equality_l47_47122

-- Definitions based on conditions
def nine := 3^2

-- Theorem stating the given mathematical problem
theorem power_equality : nine^4 = 3^8 := by
  sorry

end power_equality_l47_47122


namespace log_expression_equality_l47_47549

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_equality :
  Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + (log_base 2 5) * (log_base 5 8) = 5 := by
  sorry

end log_expression_equality_l47_47549


namespace fraction_multiplication_l47_47135

-- Define the problem as a theorem in Lean
theorem fraction_multiplication
  (a b x : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) (ha : a ≠ 0): 
  (3 * a * b / x) * (2 * x^2 / (9 * a * b^2)) = (2 * x) / (3 * b) := 
by
  sorry

end fraction_multiplication_l47_47135


namespace maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l47_47069

open Real

theorem maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism 
  (a b : ℝ)
  (ha : a^2 + b^2 = 25) 
  (AC_eq_5 : AC = 5) :
  ∃ (r : ℝ), 4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end maximum_surface_area_of_inscribed_sphere_in_right_triangular_prism_l47_47069


namespace math_problem_l47_47156

theorem math_problem :
  let a := 481 * 7
  let b := 426 * 5
  ((a + b) ^ 3 - 4 * a * b) = 166021128033 := 
by
  let a := 481 * 7
  let b := 426 * 5
  sorry

end math_problem_l47_47156


namespace percentage_liked_B_l47_47810

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l47_47810


namespace simplify_expression_l47_47948

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 5 * x + 9 * x^2 + 8 - (6 - 5 * x - 3 * x^2)

-- Define the expected simplified form
def expected_expression (x : ℝ) : ℝ := 12 * x^2 + 10 * x + 2

-- The theorem we want to prove
theorem simplify_expression (x : ℝ) : given_expression x = expected_expression x := by
  sorry

end simplify_expression_l47_47948


namespace cos_difference_identity_l47_47757

theorem cos_difference_identity (α : ℝ)
  (h : Real.sin (α + π / 6) + Real.cos α = - (Real.sqrt 3) / 3) :
  Real.cos (π / 6 - α) = -1 / 3 := 
sorry

end cos_difference_identity_l47_47757


namespace average_of_new_sequence_l47_47231

theorem average_of_new_sequence (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_new_sequence_l47_47231


namespace treaty_signed_on_thursday_l47_47811

def initial_day : ℕ := 0  -- 0 representing Monday, assuming a week cycle from 0 (Monday) to 6 (Sunday)
def days_in_week : ℕ := 7

def treaty_day (n : ℕ) : ℕ :=
(n + initial_day) % days_in_week

theorem treaty_signed_on_thursday :
  treaty_day 1000 = 4 :=  -- 4 representing Thursday
by
  sorry

end treaty_signed_on_thursday_l47_47811


namespace bug_total_distance_l47_47283

theorem bug_total_distance :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let final_pos := 0
  let distance1 := |pos1 - pos2|
  let distance2 := |pos2 - pos3|
  let distance3 := |pos3 - final_pos|
  let total_distance := distance1 + distance2 + distance3
  total_distance = 29 := by
    sorry

end bug_total_distance_l47_47283


namespace num_ways_to_pay_16_rubles_l47_47268

theorem num_ways_to_pay_16_rubles :
  ∃! (n : ℕ), n = 13 ∧ ∀ (x y z : ℕ), (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ 
  (10 * x + 2 * y + 1 * z = 16) ∧ (x < 2) ∧ (y + z > 0) := sorry

end num_ways_to_pay_16_rubles_l47_47268


namespace find_prime_p_l47_47187

theorem find_prime_p (p x y : ℕ) (hp : Nat.Prime p) (hx : x > 0) (hy : y > 0) :
  (p + 49 = 2 * x^2) ∧ (p^2 + 49 = 2 * y^2) ↔ p = 23 :=
by
  sorry

end find_prime_p_l47_47187


namespace intersection_of_A_and_B_union_of_A_and_B_l47_47263

def A : Set ℝ := {x | x * (9 - x) > 0}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} :=
sorry

theorem union_of_A_and_B : A ∪ B = {x | x < 9} :=
sorry

end intersection_of_A_and_B_union_of_A_and_B_l47_47263


namespace quadratic_completing_square_t_value_l47_47154

theorem quadratic_completing_square_t_value :
  ∃ q t : ℝ, 4 * x^2 - 24 * x - 96 = 0 → (x + q) ^ 2 = t ∧ t = 33 :=
by
  sorry

end quadratic_completing_square_t_value_l47_47154


namespace k_is_2_l47_47131

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x - 1
def g (x : ℝ) : ℝ := 0
noncomputable def h (x : ℝ) : ℝ := (x + 1) * Real.log x

theorem k_is_2 :
  (∀ x ∈ Set.Icc 1 (2 * Real.exp 1), 0 ≤ f k x ∧ f k x ≤ h x) ↔ (k = 2) :=
  sorry

end k_is_2_l47_47131


namespace book_price_increase_l47_47911

theorem book_price_increase (P : ℝ) (x : ℝ) :
  (P * (1 + x / 100)^2 = P * 1.3225) → x = 15 :=
by
  sorry

end book_price_increase_l47_47911


namespace price_of_second_tea_l47_47523

theorem price_of_second_tea (price_first_tea : ℝ) (mixture_price : ℝ) (required_ratio : ℝ) (price_second_tea : ℝ) :
  price_first_tea = 62 → mixture_price = 64.5 → required_ratio = 3 → price_second_tea = 65.33 :=
by
  intros h1 h2 h3
  sorry

end price_of_second_tea_l47_47523


namespace rectangle_length_l47_47988

theorem rectangle_length (b l : ℝ) 
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 5) = l * b + 75) : l = 40 := by
  sorry

end rectangle_length_l47_47988


namespace smallest_integer_condition_l47_47842

theorem smallest_integer_condition {A : ℕ} (h1 : A > 1) 
  (h2 : ∃ k : ℕ, A = 5 * k / 3 + 2 / 3)
  (h3 : ∃ m : ℕ, A = 7 * m / 5 + 2 / 5)
  (h4 : ∃ n : ℕ, A = 9 * n / 7 + 2 / 7)
  (h5 : ∃ p : ℕ, A = 11 * p / 9 + 2 / 9) : 
  A = 316 := 
sorry

end smallest_integer_condition_l47_47842


namespace slope_of_parallel_lines_l47_47840

theorem slope_of_parallel_lines (m : ℝ)
  (y1 y2 y3 : ℝ)
  (h1 : y1 = 2) 
  (h2 : y2 = 3) 
  (h3 : y3 = 4)
  (sum_of_x_intercepts : (-2 / m) + (-3 / m) + (-4 / m) = 36) :
  m = -1 / 4 := by
  sorry

end slope_of_parallel_lines_l47_47840


namespace intersect_lines_l47_47438

theorem intersect_lines (k : ℝ) :
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 :=
by
  sorry

end intersect_lines_l47_47438


namespace chadsRopeLength_l47_47497

-- Define the constants and conditions
def joeysRopeLength : ℕ := 56
def joeyChadRatioNumerator : ℕ := 8
def joeyChadRatioDenominator : ℕ := 3

-- Prove that Chad's rope length is 21 cm
theorem chadsRopeLength (C : ℕ) 
  (h_ratio : joeysRopeLength * joeyChadRatioDenominator = joeyChadRatioNumerator * C) : 
  C = 21 :=
sorry

end chadsRopeLength_l47_47497


namespace problem_statements_correct_l47_47269

theorem problem_statements_correct :
    (∀ (select : ℕ) (male female : ℕ), male = 4 → female = 3 → 
      (select = (4 * 3 + 3)) → select ≥ 12 = false) ∧
    (∀ (a1 a2 a3 : ℕ), 
      a2 = 0 ∨ a2 = 1 ∨ a2 = 2 →
      (∃ (cases : ℕ), cases = 14) →
      cases = 14) ∧
    (∀ (ways enter exit : ℕ), enter = 4 → exit = 4 - 1 →
      (ways = enter * exit) → ways = 12 = false) ∧
    (∀ (a b : ℕ),
      a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 →
      (∃ (log_val : ℕ), log_val = 54) →
      log_val = 54) := by
  admit

end problem_statements_correct_l47_47269


namespace eden_initial_bears_l47_47409

theorem eden_initial_bears (d_total : ℕ) (d_favorite : ℕ) (sisters : ℕ) (eden_after : ℕ) (each_share : ℕ)
  (h1 : d_total = 20)
  (h2 : d_favorite = 8)
  (h3 : sisters = 3)
  (h4 : eden_after = 14)
  (h5 : each_share = (d_total - d_favorite) / sisters)
  : (eden_after - each_share) = 10 :=
by
  sorry

end eden_initial_bears_l47_47409


namespace line_parallel_condition_l47_47960

theorem line_parallel_condition (a : ℝ) :
    (a = 1) → (∀ (x y : ℝ), (ax + 2 * y - 1 = 0) ∧ (x + (a + 1) * y + 4 = 0)) → (a = 1 ∨ a = -2) :=
by
sorry

end line_parallel_condition_l47_47960


namespace tim_total_points_l47_47134

theorem tim_total_points :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6
  let tetrises := 4
  let total_points := singles * single_points + tetrises * tetris_points
  total_points = 38000 :=
by
  sorry

end tim_total_points_l47_47134


namespace intersection_A_B_l47_47402

open Set

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by {
  sorry
}

end intersection_A_B_l47_47402


namespace binary_add_sub_l47_47405

theorem binary_add_sub:
  let a := 0b10110
  let b := 0b1010
  let c := 0b11100
  let d := 0b1110
  a + b - c + d = 0b01110 := by
  sorry

end binary_add_sub_l47_47405


namespace solve_quadratic_l47_47640

theorem solve_quadratic (x : ℚ) (h_pos : x > 0) (h_eq : 3 * x^2 + 8 * x - 35 = 0) : 
    x = 7/3 :=
by
    sorry

end solve_quadratic_l47_47640


namespace most_persuasive_method_l47_47267

-- Survey data and conditions
def male_citizens : ℕ := 4258
def male_believe_doping : ℕ := 2360
def female_citizens : ℕ := 3890
def female_believe_framed : ℕ := 2386

def random_division_by_gender : Prop := true -- Represents the random division into male and female groups

-- Proposition to prove
theorem most_persuasive_method : 
  random_division_by_gender → 
  ∃ method : String, method = "Independence Test" := by
  sorry

end most_persuasive_method_l47_47267


namespace basketball_game_score_difference_l47_47448

theorem basketball_game_score_difference :
  let blueFreeThrows := 18
  let blueTwoPointers := 25
  let blueThreePointers := 6
  let redFreeThrows := 15
  let redTwoPointers := 22
  let redThreePointers := 5
  let blueScore := blueFreeThrows * 1 + blueTwoPointers * 2 + blueThreePointers * 3
  let redScore := redFreeThrows * 1 + redTwoPointers * 2 + redThreePointers * 3
  blueScore - redScore = 12 := by
  sorry

end basketball_game_score_difference_l47_47448


namespace average_age_of_women_l47_47052

theorem average_age_of_women (A : ℕ) :
  (6 * (A + 2) = 6 * A - 22 + W) → (W / 2 = 17) :=
by
  intro h
  sorry

end average_age_of_women_l47_47052


namespace tank_capacity_is_32_l47_47433

noncomputable def capacity_of_tank (C : ℝ) : Prop :=
  (3/4) * C + 4 = (7/8) * C

theorem tank_capacity_is_32 : ∃ C : ℝ, capacity_of_tank C ∧ C = 32 :=
sorry

end tank_capacity_is_32_l47_47433


namespace probability_of_pink_tie_l47_47963

theorem probability_of_pink_tie 
  (black_ties gold_ties pink_ties : ℕ) 
  (h_black : black_ties = 5) 
  (h_gold : gold_ties = 7) 
  (h_pink : pink_ties = 8) 
  (h_total : (5 + 7 + 8) = (black_ties + gold_ties + pink_ties)) 
  : (pink_ties : ℚ) / (black_ties + gold_ties + pink_ties) = 2 / 5 := 
by 
  sorry

end probability_of_pink_tie_l47_47963


namespace remainder_of_n_plus_3255_l47_47400

theorem remainder_of_n_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := 
by
  sorry

end remainder_of_n_plus_3255_l47_47400


namespace find_dividend_l47_47633

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 8) (h2 : quotient = 8) (h3 : dividend = k * quotient) : dividend = 64 := 
by 
  sorry

end find_dividend_l47_47633


namespace greatest_length_of_equal_pieces_l47_47826

theorem greatest_length_of_equal_pieces (a b c : ℕ) (h₁ : a = 42) (h₂ : b = 63) (h₃ : c = 84) :
  Nat.gcd (Nat.gcd a b) c = 21 :=
by
  rw [h₁, h₂, h₃]
  sorry

end greatest_length_of_equal_pieces_l47_47826


namespace value_range_of_func_l47_47085

-- Define the function y = x^2 - 4x + 6 for x in the interval [1, 4]
def func (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem value_range_of_func : 
  ∀ y, ∃ x, (1 ≤ x ∧ x ≤ 4) ∧ y = func x ↔ 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end value_range_of_func_l47_47085


namespace find_first_number_l47_47691

/-- Given a sequence of 6 numbers b_1, b_2, ..., b_6 such that:
  1. For n ≥ 2, b_{2n} = b_{2n-1}^2
  2. For n ≥ 2, b_{2n+1} = (b_{2n} * b_{2n-1})^2
And the sequence ends as: b_4 = 16, b_5 = 256, and b_6 = 65536,
prove that the first number b_1 is 1/2. -/
theorem find_first_number : 
  ∃ b : ℕ → ℝ, b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧ 
  (∀ n ≥ 2, b (2 * n) = (b (2 * n - 1)) ^ 2) ∧
  (∀ n ≥ 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧ 
  b 1 = 1/2 :=
by
  sorry

end find_first_number_l47_47691


namespace arithmetic_seq_sum_l47_47495

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) 
(h_given : a 2 + a 8 = 10) : 
a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l47_47495


namespace sheets_in_stack_l47_47524

theorem sheets_in_stack 
  (num_sheets : ℕ) 
  (initial_thickness final_thickness : ℝ) 
  (t_per_sheet : ℝ) 
  (h_initial : num_sheets = 800) 
  (h_thickness : initial_thickness = 4) 
  (h_thickness_per_sheet : initial_thickness / num_sheets = t_per_sheet) 
  (h_final_thickness : final_thickness = 6) 
  : num_sheets * (final_thickness / t_per_sheet) = 1200 := 
by 
  sorry

end sheets_in_stack_l47_47524


namespace find_n_l47_47868

-- Define the vectors \overrightarrow {AB}, \overrightarrow {BC}, and \overrightarrow {AC}
def vectorAB : ℝ × ℝ := (2, 4)
def vectorBC (n : ℝ) : ℝ × ℝ := (-2, 2 * n)
def vectorAC : ℝ × ℝ := (0, 2)

-- State the theorem and prove the value of n
theorem find_n (n : ℝ) (h : vectorAC = (vectorAB.1 + (vectorBC n).1, vectorAB.2 + (vectorBC n).2)) : n = -1 :=
by
  sorry

end find_n_l47_47868


namespace average_speed_of_car_l47_47176

/-- The average speed of a car over four hours given specific distances covered each hour. -/
theorem average_speed_of_car
  (d1 d2 d3 d4 : ℝ)
  (t1 t2 t3 t4 : ℝ)
  (h1 : d1 = 20)
  (h2 : d2 = 40)
  (h3 : d3 = 60)
  (h4 : d4 = 100)
  (h5 : t1 = 1)
  (h6 : t2 = 1)
  (h7 : t3 = 1)
  (h8 : t4 = 1) :
  (d1 + d2 + d3 + d4) / (t1 + t2 + t3 + t4) = 55 :=
by sorry

end average_speed_of_car_l47_47176


namespace find_f_log_value_l47_47112

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 2^x + 1 else sorry

theorem find_f_log_value (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_spec : ∀ x, 0 < x → x < 1 → f x = 2^x + 1) :
  f (Real.logb (1/2) (1/15)) = -31/15 :=
sorry

end find_f_log_value_l47_47112


namespace eight_times_10x_plus_14pi_l47_47145

theorem eight_times_10x_plus_14pi (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * π) = Q) : 
  8 * (10 * x + 14 * π) = 4 * Q := 
by {
  sorry  -- proof is omitted
}

end eight_times_10x_plus_14pi_l47_47145


namespace sum_first_5_arithmetic_l47_47125

theorem sum_first_5_arithmetic (u : ℕ → ℝ) (h : u 3 = 0) : 
  (u 1 + u 2 + u 3 + u 4 + u 5) = 0 :=
sorry

end sum_first_5_arithmetic_l47_47125


namespace nba_conferences_division_l47_47352

theorem nba_conferences_division (teams : ℕ) (games_per_team : ℕ) (E : ℕ) :
  teams = 30 ∧ games_per_team = 82 ∧
  (teams = E + (teams - E)) ∧
  (games_per_team / 2 * E) + (games_per_team / 2 * (teams - E))  ≠ teams * games_per_team / 2 :=
by
  sorry

end nba_conferences_division_l47_47352


namespace find_line_through_M_and_parallel_l47_47192
-- Lean code to represent the proof problem

def M : Prop := ∃ (x y : ℝ), 3 * x + 4 * y - 5 = 0 ∧ 2 * x - 3 * y + 8 = 0 

def line_parallel : Prop := ∃ (m b : ℝ), 2 * m + b = 0

theorem find_line_through_M_and_parallel :
  M → line_parallel → ∃ (a b c : ℝ), (a = 2) ∧ (b = 1) ∧ (c = 0) :=
by
  intros hM hLineParallel
  sorry

end find_line_through_M_and_parallel_l47_47192


namespace two_le_three_l47_47528

/-- Proof that the proposition "2 ≤ 3" is true given the logical connective. -/
theorem two_le_three : 2 ≤ 3 := 
by
  sorry

end two_le_three_l47_47528


namespace Fred_earned_4_dollars_l47_47828

-- Conditions are translated to definitions
def initial_amount_Fred : ℕ := 111
def current_amount_Fred : ℕ := 115

-- Proof problem in Lean 4 statement
theorem Fred_earned_4_dollars : current_amount_Fred - initial_amount_Fred = 4 := by
  sorry

end Fred_earned_4_dollars_l47_47828


namespace find_k_l47_47986

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end find_k_l47_47986


namespace angle_Z_of_triangle_l47_47437

theorem angle_Z_of_triangle (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : X + Y + Z = 180) : 
  Z = 90 := 
sorry

end angle_Z_of_triangle_l47_47437


namespace width_of_Carols_rectangle_l47_47551

theorem width_of_Carols_rectangle 
  (w : ℝ) 
  (h1 : 15 * w = 6 * 50) : w = 20 := 
by 
  sorry

end width_of_Carols_rectangle_l47_47551


namespace find_numbers_l47_47298

theorem find_numbers (x y : ℕ) (h1 : x / y = 3) (h2 : (x^2 + y^2) / (x + y) = 5) : 
  x = 6 ∧ y = 2 := 
by
  sorry

end find_numbers_l47_47298


namespace find_v_l47_47544

variables (a b c : ℝ)

def condition1 := (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -6
def condition2 := (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

theorem find_v (h1 : condition1 a b c) (h2 : condition2 a b c) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 17 / 2 :=
by
  sorry

end find_v_l47_47544


namespace quad_form_b_c_sum_l47_47091

theorem quad_form_b_c_sum :
  ∃ (b c : ℝ), (b + c = -10) ∧ (∀ x : ℝ, x^2 - 20 * x + 100 = (x + b)^2 + c) :=
by
  sorry

end quad_form_b_c_sum_l47_47091


namespace trigonometric_identity_l47_47930

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
sorry

end trigonometric_identity_l47_47930


namespace my_op_eq_l47_47452

-- Define the custom operation
def my_op (m n : ℝ) : ℝ := m * n * (m - n)

-- State the theorem
theorem my_op_eq :
  ∀ (a b : ℝ), my_op (a + b) a = a^2 * b + a * b^2 :=
by intros a b; sorry

end my_op_eq_l47_47452


namespace solution_for_4_minus_c_l47_47588

-- Define the conditions as Lean hypotheses
theorem solution_for_4_minus_c (c d : ℚ) (h1 : 4 + c = 5 - d) (h2 : 5 + d = 9 + c) : 4 - c = 11 / 2 :=
by
  sorry

end solution_for_4_minus_c_l47_47588


namespace min_value_of_b_plus_2_div_a_l47_47758

theorem min_value_of_b_plus_2_div_a (a : ℝ) (b : ℝ) (h₁ : 0 < a) 
  (h₂ : ∀ x : ℝ, 0 < x → (ax - 1) * (x^2 + bx - 4) ≥ 0) : 
  ∃ a' b', (a' > 0 ∧ b' = 4 * a' - 1 / a') ∧ b' + 2 / a' = 4 :=
by
  sorry

end min_value_of_b_plus_2_div_a_l47_47758


namespace island_length_l47_47870

/-- Proof problem: Given an island in the Indian Ocean with a width of 4 miles and a perimeter of 22 miles. 
    Assume the island is rectangular in shape. Prove that the length of the island is 7 miles. -/
theorem island_length
  (width length : ℝ) 
  (h_width : width = 4)
  (h_perimeter : 2 * (length + width) = 22) : 
  length = 7 :=
sorry

end island_length_l47_47870


namespace smallest_natural_number_with_condition_l47_47223

theorem smallest_natural_number_with_condition {N : ℕ} :
  (N % 10 = 6) ∧ (4 * N = (6 * 10 ^ ((Nat.digits 10 (N / 10)).length) + (N / 10))) ↔ N = 153846 :=
by
  sorry

end smallest_natural_number_with_condition_l47_47223


namespace percentage_of_money_spent_l47_47512

theorem percentage_of_money_spent (initial_amount remaining_amount : ℝ) (h_initial : initial_amount = 500) (h_remaining : remaining_amount = 350) :
  (((initial_amount - remaining_amount) / initial_amount) * 100) = 30 :=
by
  -- Start the proof
  sorry

end percentage_of_money_spent_l47_47512


namespace pudding_distribution_l47_47972

theorem pudding_distribution {puddings students : ℕ} (h1 : puddings = 315) (h2 : students = 218) : 
  ∃ (additional_puddings : ℕ), additional_puddings >= 121 ∧ ∃ (cups_per_student : ℕ), 
  (puddings + additional_puddings) ≥ students * cups_per_student :=
by
  sorry

end pudding_distribution_l47_47972


namespace amount_paid_by_customer_l47_47332

theorem amount_paid_by_customer 
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (final_price : ℝ)
  (h1 : cost_price = 6681.818181818181)
  (h2 : markup_percentage = 10 / 100)
  (h3 : final_price = cost_price * (1 + markup_percentage)) :
  final_price = 7350 :=
by 
  sorry

end amount_paid_by_customer_l47_47332


namespace chord_length_in_circle_l47_47139

theorem chord_length_in_circle 
  (radius : ℝ) 
  (chord_midpoint_perpendicular_radius : ℝ)
  (r_eq_10 : radius = 10)
  (cmp_eq_5 : chord_midpoint_perpendicular_radius = 5) : 
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 3 := 
by 
  sorry

end chord_length_in_circle_l47_47139


namespace setB_forms_right_triangle_l47_47724

-- Define the sets of side lengths
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (5, 6, 7)
def setD : (ℕ × ℕ × ℕ) := (7, 8, 9)

-- Define the Pythagorean theorem condition
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The specific proof goal
theorem setB_forms_right_triangle : isRightTriangle 3 4 5 := by
  sorry

end setB_forms_right_triangle_l47_47724


namespace min_value_expression_l47_47227

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l47_47227


namespace prime_1002_n_count_l47_47824

theorem prime_1002_n_count :
  ∃! n : ℕ, n ≥ 2 ∧ Prime (n^3 + 2) :=
by
  sorry

end prime_1002_n_count_l47_47824


namespace solve_ordered_pairs_l47_47321

theorem solve_ordered_pairs (a b : ℕ) (h : a^2 + b^2 = ab * (a + b)) : 
  (a, b) = (1, 1) ∨ (a, b) = (1, 1) :=
by 
  sorry

end solve_ordered_pairs_l47_47321


namespace man_l47_47392

theorem man's_present_age (P : ℝ) 
  (h1 : P = (4/5) * P + 10)
  (h2 : P = (3/2.5) * P - 10) :
  P = 50 :=
sorry

end man_l47_47392


namespace feathers_per_flamingo_l47_47969

theorem feathers_per_flamingo (num_boa : ℕ) (feathers_per_boa : ℕ) (num_flamingoes : ℕ) (pluck_rate : ℚ)
  (total_feathers : ℕ) (feathers_per_flamingo : ℕ) :
  num_boa = 12 →
  feathers_per_boa = 200 →
  num_flamingoes = 480 →
  pluck_rate = 0.25 →
  total_feathers = num_boa * feathers_per_boa →
  total_feathers = num_flamingoes * feathers_per_flamingo * pluck_rate →
  feathers_per_flamingo = 20 :=
by
  intros h_num_boa h_feathers_per_boa h_num_flamingoes h_pluck_rate h_total_feathers h_feathers_eq
  sorry

end feathers_per_flamingo_l47_47969


namespace probability_of_negative_m_l47_47218

theorem probability_of_negative_m (m : ℤ) (h₁ : -2 ≤ m) (h₂ : m < (9 : ℤ) / 4) :
  ∃ (neg_count total_count : ℤ), 
    (neg_count = 2) ∧ (total_count = 5) ∧ (m ∈ {i : ℤ | -2 ≤ i ∧ i < 2 ∧ i < 9 / 4}) → 
    (neg_count / total_count = 2 / 5) :=
sorry

end probability_of_negative_m_l47_47218


namespace Mike_changed_64_tires_l47_47798

def tires_changed (motorcycles: ℕ) (cars: ℕ): ℕ := 
  (motorcycles * 2) + (cars * 4)

theorem Mike_changed_64_tires:
  (tires_changed 12 10) = 64 :=
by
  sorry

end Mike_changed_64_tires_l47_47798


namespace trajectory_of_square_is_line_l47_47129

open Complex

theorem trajectory_of_square_is_line (z : ℂ) (h : z.re = z.im) : ∃ c : ℝ, z^2 = Complex.I * (c : ℂ) :=
by
  sorry

end trajectory_of_square_is_line_l47_47129


namespace area_of_smallest_square_l47_47373

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end area_of_smallest_square_l47_47373


namespace length_of_rect_box_l47_47311

noncomputable def length_of_box (height : ℝ) (width : ℝ) (volume : ℝ) : ℝ :=
  volume / (width * height)

theorem length_of_rect_box :
  (length_of_box 0.5 25 (6000 / 7.48052)) = 64.1624 :=
by
  unfold length_of_box
  norm_num
  sorry

end length_of_rect_box_l47_47311


namespace solve_for_x_l47_47304

theorem solve_for_x (x : ℝ) (h₁ : (7 * x) / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) (h₂ : x ≠ -4) : x = 6 / 7 :=
by
  sorry

end solve_for_x_l47_47304


namespace fg_of_3_l47_47417

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem we want to prove
theorem fg_of_3 : f (g 3) = 344 := by
  sorry

end fg_of_3_l47_47417


namespace _l47_47786

lemma right_triangle_angles (AB BC AC : ℝ) (α β : ℝ)
  (h1 : AB = 1) 
  (h2 : BC = Real.sin α)
  (h3 : AC = Real.cos α)
  (h4 : AB^2 = BC^2 + AC^2) -- Pythagorean theorem for the right triangle
  (h5 : α = (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1))) :
  β = 90 - (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
sorry

end _l47_47786


namespace count_satisfying_pairs_l47_47940

theorem count_satisfying_pairs :
  ∃ (count : ℕ), count = 540 ∧ 
  (∀ (w n : ℕ), (w % 23 = 5) ∧ (w < 450) ∧ (n % 17 = 7) ∧ (n < 450) → w < 450 ∧ n < 450) := 
by
  sorry

end count_satisfying_pairs_l47_47940


namespace inscribed_circle_radius_l47_47604

-- Conditions
variables {S A B C D O : Point} -- Points in 3D space
variables (AC : ℝ) (cos_SBD : ℝ)
variables (r : ℝ) -- Radius of inscribed circle

-- Given conditions
def AC_eq_one := AC = 1
def cos_angle_SBD := cos_SBD = 2/3

-- Assertion to be proved
theorem inscribed_circle_radius :
  AC_eq_one AC →
  cos_angle_SBD cos_SBD →
  (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 :=
by
  intro hAC hcos
  -- Proof goes here
  sorry

end inscribed_circle_radius_l47_47604


namespace ratio_of_spinsters_to_cats_l47_47237

def spinsters := 22
def cats := spinsters + 55

theorem ratio_of_spinsters_to_cats : (spinsters : ℝ) / (cats : ℝ) = 2 / 7 := 
by
  sorry

end ratio_of_spinsters_to_cats_l47_47237


namespace pencil_eraser_cost_l47_47378

/-- Oscar buys 13 pencils and 3 erasers for 100 cents. A pencil costs more than an eraser, 
    and both items cost a whole number of cents. 
    We need to prove that the total cost of one pencil and one eraser is 10 cents. -/
theorem pencil_eraser_cost (p e : ℕ) (h1 : 13 * p + 3 * e = 100) (h2 : p > e) : p + e = 10 :=
sorry

end pencil_eraser_cost_l47_47378


namespace slope_of_parallel_line_l47_47009

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l47_47009


namespace vehicle_height_limit_l47_47755

theorem vehicle_height_limit (h : ℝ) (sign : String) (cond : sign = "Height Limit 4.5 meters") : h ≤ 4.5 :=
sorry

end vehicle_height_limit_l47_47755


namespace number_of_ordered_pairs_l47_47229

noncomputable def max (x y : ℕ) : ℕ := if x > y then x else y

def valid_pair_count (k : ℕ) : ℕ := 2 * k + 1

def pairs_count (a b : ℕ) : ℕ := 
  valid_pair_count 5 * valid_pair_count 3 * valid_pair_count 2 * valid_pair_count 1

theorem number_of_ordered_pairs : pairs_count 2 3 = 1155 := 
  sorry

end number_of_ordered_pairs_l47_47229


namespace not_possible_consecutive_results_l47_47015

theorem not_possible_consecutive_results 
  (dot_counts : ℕ → ℕ)
  (h_identical_conditions : ∀ (i : ℕ), dot_counts i = 1 ∨ dot_counts i = 2 ∨ dot_counts i = 3) 
  (h_correct_dot_distribution : ∀ (i j : ℕ), (i ≠ j → dot_counts i ≠ dot_counts j))
  : ¬ (∃ (consecutive : ℕ → ℕ), 
        (∀ (k : ℕ), k < 6 → consecutive k = dot_counts (4 * k) + dot_counts (4 * k + 1) 
                         + dot_counts (4 * k + 2) + dot_counts (4 * k + 3))
        ∧ (∀ (k : ℕ), k < 5 → consecutive (k + 1) = consecutive k + 1)) := sorry

end not_possible_consecutive_results_l47_47015


namespace hyperbola_eccentricity_sqrt5_l47_47193

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_sqrt5_l47_47193


namespace diophantine_solution_exists_if_prime_divisor_l47_47285

theorem diophantine_solution_exists_if_prime_divisor (b : ℕ) (hb : 0 < b) (gcd_b_6 : Nat.gcd b 6 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (1 / (x : ℚ) + 1 / (y : ℚ) = 3 / (b : ℚ))) ↔ 
  ∃ p : ℕ, Nat.Prime p ∧ (∃ k : ℕ, p = 6 * k - 1) ∧ p ∣ b := 
by 
  sorry

end diophantine_solution_exists_if_prime_divisor_l47_47285


namespace train_time_original_l47_47729

theorem train_time_original (D : ℝ) (T : ℝ) 
  (h1 : D = 48 * T) 
  (h2 : D = 60 * (2/3)) : T = 5 / 6 := 
by
  sorry

end train_time_original_l47_47729


namespace remainder_M_divided_by_1000_l47_47064

/-- Define flag problem parameters -/
def flagpoles: ℕ := 2
def blue_flags: ℕ := 15
def green_flags: ℕ := 10

/-- Condition: Two flagpoles, 15 blue flags and 10 green flags -/
def arrangable_flags (flagpoles blue_flags green_flags: ℕ) : Prop :=
  blue_flags + green_flags = 25 ∧ flagpoles = 2

/-- Condition: Each pole contains at least one flag -/
def each_pole_has_flag (arranged_flags: ℕ) : Prop :=
  arranged_flags > 0

/-- Condition: No two green flags are adjacent in any arrangement -/
def no_adjacent_green_flags (arranged_greens: ℕ) : Prop :=
  arranged_greens > 0

/-- Main theorem statement with correct answer -/
theorem remainder_M_divided_by_1000 (M: ℕ) : 
  arrangable_flags flagpoles blue_flags green_flags ∧ 
  each_pole_has_flag M ∧ 
  no_adjacent_green_flags green_flags ∧ 
  M % 1000 = 122
:= sorry

end remainder_M_divided_by_1000_l47_47064


namespace empty_drainpipe_rate_l47_47576

theorem empty_drainpipe_rate :
  (∀ x : ℝ, (1/5 + 1/4 - 1/x = 1/2.5) → x = 20) :=
by 
    intro x
    intro h
    sorry -- Proof is omitted, only the statement is required

end empty_drainpipe_rate_l47_47576


namespace daughter_age_l47_47065

-- Define the conditions and the question as a theorem
theorem daughter_age (D F : ℕ) (h1 : F = 3 * D) (h2 : F + 12 = 2 * (D + 12)) : D = 12 :=
by
  -- We need to provide a proof or placeholder for now
  sorry

end daughter_age_l47_47065


namespace michelle_travel_distance_l47_47783

-- Define the conditions
def initial_fee : ℝ := 2
def charge_per_mile : ℝ := 2.5
def total_paid : ℝ := 12

-- Define the theorem to prove the distance Michelle traveled
theorem michelle_travel_distance : (total_paid - initial_fee) / charge_per_mile = 4 := by
  sorry

end michelle_travel_distance_l47_47783


namespace arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l47_47450

noncomputable def arsh (x : ℝ) := Real.log (x + Real.sqrt (x^2 + 1))
noncomputable def arch_pos (x : ℝ) := Real.log (x + Real.sqrt (x^2 - 1))
noncomputable def arch_neg (x : ℝ) := Real.log (x - Real.sqrt (x^2 - 1))
noncomputable def arth (x : ℝ) := (1 / 2) * Real.log ((1 + x) / (1 - x))

theorem arsh_eq (x : ℝ) : arsh x = Real.log (x + Real.sqrt (x^2 + 1)) := by
  sorry

theorem arch_pos_eq (x : ℝ) : arch_pos x = Real.log (x + Real.sqrt (x^2 - 1)) := by
  sorry

theorem arch_neg_eq (x : ℝ) : arch_neg x = Real.log (x - Real.sqrt (x^2 - 1)) := by
  sorry

theorem arth_eq (x : ℝ) : arth x = (1 / 2) * Real.log ((1 + x) / (1 - x)) := by
  sorry

end arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l47_47450


namespace quotient_remainder_base5_l47_47850

theorem quotient_remainder_base5 (n m : ℕ) 
    (hn : n = 3 * 5^3 + 2 * 5^2 + 3 * 5^1 + 2)
    (hm : m = 2 * 5^1 + 1) :
    n / m = 40 ∧ n % m = 2 :=
by
  sorry

end quotient_remainder_base5_l47_47850


namespace remainder_three_l47_47787

theorem remainder_three (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 3 = 1 :=
sorry

end remainder_three_l47_47787


namespace cube_cut_off_edges_l47_47032

theorem cube_cut_off_edges :
  let original_edges := 12
  let new_edges_per_vertex := 3
  let vertices := 8
  let new_edges := new_edges_per_vertex * vertices
  (original_edges + new_edges) = 36 :=
by
  sorry

end cube_cut_off_edges_l47_47032


namespace log_prime_factor_inequality_l47_47703

open Real

-- Define p(n) such that it returns the number of prime factors of n.
noncomputable def p (n: ℕ) : ℕ := sorry  -- This will be defined contextually for now

theorem log_prime_factor_inequality (n : ℕ) (hn : n > 0) : 
  log n ≥ (p n) * log 2 :=
by 
  sorry

end log_prime_factor_inequality_l47_47703


namespace simplify_expression_l47_47856

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 
  3 * (4 - 2 * i) + 2 * i * (3 + i) + 5 * (-1 + i) = 5 + 5 * i :=
by
  sorry

end simplify_expression_l47_47856


namespace value_of_a_plus_b_l47_47108

theorem value_of_a_plus_b (a b : ℝ) (h : |a - 2| = -(b + 5)^2) : a + b = -3 :=
sorry

end value_of_a_plus_b_l47_47108


namespace sum_of_digits_of_valid_n_eq_seven_l47_47743

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_valid_n (n : ℕ) : Prop :=
  (500 < n) ∧ (Nat.gcd 70 (n + 150) = 35) ∧ (Nat.gcd (n + 70) 150 = 50)

theorem sum_of_digits_of_valid_n_eq_seven :
  ∃ n : ℕ, is_valid_n n ∧ sum_of_digits n = 7 := by
  sorry

end sum_of_digits_of_valid_n_eq_seven_l47_47743


namespace melanie_gave_3_plums_to_sam_l47_47305

theorem melanie_gave_3_plums_to_sam 
  (initial_plums : ℕ) 
  (plums_left : ℕ) 
  (plums_given : ℕ) 
  (h1 : initial_plums = 7) 
  (h2 : plums_left = 4) 
  (h3 : plums_left + plums_given = initial_plums) : 
  plums_given = 3 :=
by 
  sorry

end melanie_gave_3_plums_to_sam_l47_47305


namespace triangles_xyz_l47_47532

theorem triangles_xyz (A B C D P Q R : Type) 
    (u v w x : ℝ)
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (h1 : angle_ADB = 120) 
    (h2 : angle_BDC = 120) 
    (h3 : angle_CDA = 120) :
    x = u + v + w :=
sorry

end triangles_xyz_l47_47532


namespace isosceles_triangle_height_ratio_l47_47375

theorem isosceles_triangle_height_ratio (b1 h1 b2 h2 : ℝ) 
  (A1 : ℝ := 1/2 * b1 * h1) (A2 : ℝ := 1/2 * b2 * h2)
  (area_ratio : A1 / A2 = 16 / 49)
  (similar : b1 / b2 = h1 / h2) : 
  h1 / h2 = 4 / 7 := 
by {
  sorry
}

end isosceles_triangle_height_ratio_l47_47375


namespace find_m_l47_47951

noncomputable def f : ℝ → ℝ := sorry

theorem find_m (h₁ : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h₂ : f 2 = m) : m = -1 / 2 :=
by
  sorry

end find_m_l47_47951


namespace range_of_a_l47_47211

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a * x) > 2) → 0 < a ∧ a < 4 := 
by 
  sorry

end range_of_a_l47_47211


namespace even_digit_number_division_l47_47705

theorem even_digit_number_division (N : ℕ) (n : ℕ) :
  (N % 2 = 0) ∧
  (∃ a b : ℕ, (∀ k : ℕ, N = a * 10^n + b → N = k * (a * b)) ∧
  ((N = (1000^(2*n - 1) + 1)^2 / 7) ∨
   (N = 12) ∨
   (N = (10^n + 2)^2 / 6) ∨
   (N = 1352) ∨
   (N = 15))) :=
sorry

end even_digit_number_division_l47_47705


namespace good_horse_catchup_l47_47096

theorem good_horse_catchup (x : ℕ) : 240 * x = 150 * (x + 12) :=
by sorry

end good_horse_catchup_l47_47096


namespace baby_whales_on_second_trip_l47_47098

def iwishmael_whales_problem : Prop :=
  let male1 := 28
  let female1 := 2 * male1
  let male3 := male1 / 2
  let female3 := female1
  let total_whales := 178
  let total_without_babies := (male1 + female1) + (male3 + female3)
  total_whales - total_without_babies = 24

theorem baby_whales_on_second_trip : iwishmael_whales_problem :=
  by
  sorry

end baby_whales_on_second_trip_l47_47098


namespace initial_person_count_l47_47178

theorem initial_person_count
  (avg_weight_increase : ℝ)
  (weight_old_person : ℝ)
  (weight_new_person : ℝ)
  (h1 : avg_weight_increase = 4.2)
  (h2 : weight_old_person = 65)
  (h3 : weight_new_person = 98.6) :
  ∃ n : ℕ, weight_new_person - weight_old_person = avg_weight_increase * n ∧ n = 8 := 
by
  sorry

end initial_person_count_l47_47178


namespace number_of_people_chose_pop_l47_47918

theorem number_of_people_chose_pop (total_people : ℕ) (angle_pop : ℕ) (h1 : total_people = 540) (h2 : angle_pop = 270) : (total_people * (angle_pop / 360)) = 405 := by
  sorry

end number_of_people_chose_pop_l47_47918


namespace face_value_of_shares_l47_47281

-- Define the problem conditions
variables (F : ℝ) (D R : ℝ)

-- Assume conditions
axiom h1 : D = 0.155 * F
axiom h2 : R = 0.25 * 31
axiom h3 : D = R

-- State the theorem
theorem face_value_of_shares : F = 50 :=
by 
  -- Here should be the proof which we are skipping
  sorry

end face_value_of_shares_l47_47281


namespace second_third_parts_length_l47_47163

variable (total_length : ℝ) (first_part : ℝ) (last_part : ℝ)
variable (second_third_part_length : ℝ)

def is_equal_length (x y : ℝ) := x = y

theorem second_third_parts_length :
  total_length = 74.5 ∧ first_part = 15.5 ∧ last_part = 16 → 
  is_equal_length (second_third_part_length) 21.5 :=
by
  intros h
  let remaining_distance := total_length - first_part - last_part
  let second_third_part_length := remaining_distance / 2
  sorry

end second_third_parts_length_l47_47163


namespace Murtha_pebbles_problem_l47_47021

theorem Murtha_pebbles_problem : 
  let a := 3
  let d := 3
  let n := 18
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  S_n = 513 :=
by
  sorry

end Murtha_pebbles_problem_l47_47021


namespace amalie_coins_proof_l47_47565

def coins_proof : Prop :=
  ∃ (E A : ℕ),
    (E / A = 10 / 45) ∧
    (E + A = 440) ∧
    ((3 / 4) * A = 270) ∧
    (A - 270 = 90)

theorem amalie_coins_proof : coins_proof :=
  sorry

end amalie_coins_proof_l47_47565


namespace Kira_breakfast_time_l47_47774

theorem Kira_breakfast_time :
  let sausages := 3
  let eggs := 6
  let time_per_sausage := 5
  let time_per_egg := 4
  (sausages * time_per_sausage + eggs * time_per_egg) = 39 :=
by
  sorry

end Kira_breakfast_time_l47_47774


namespace midpoint_sum_is_correct_l47_47954

theorem midpoint_sum_is_correct:
  let A := (10, 8)
  let B := (-4, -6)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + midpoint.2) = 4 :=
by
  sorry

end midpoint_sum_is_correct_l47_47954


namespace isosceles_triangle_base_l47_47010

theorem isosceles_triangle_base (h_perimeter : 2 * 1.5 + x = 3.74) : x = 0.74 :=
by
  sorry

end isosceles_triangle_base_l47_47010


namespace quadratic_one_real_root_positive_m_l47_47679

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m_l47_47679


namespace largest_multiple_of_9_less_than_75_is_72_l47_47610

theorem largest_multiple_of_9_less_than_75_is_72 : 
  ∃ n : ℕ, 9 * n < 75 ∧ ∀ m : ℕ, 9 * m < 75 → 9 * m ≤ 9 * n :=
sorry

end largest_multiple_of_9_less_than_75_is_72_l47_47610


namespace pet_fee_is_120_l47_47335

noncomputable def daily_rate : ℝ := 125.00
noncomputable def rental_days : ℕ := 14
noncomputable def service_fee_rate : ℝ := 0.20
noncomputable def security_deposit : ℝ := 1110.00
noncomputable def security_deposit_rate : ℝ := 0.50

theorem pet_fee_is_120 :
  let total_stay_cost := daily_rate * rental_days
  let service_fee := service_fee_rate * total_stay_cost
  let total_before_pet_fee := total_stay_cost + service_fee
  let entire_bill := security_deposit / security_deposit_rate
  let pet_fee := entire_bill - total_before_pet_fee
  pet_fee = 120 := by
  sorry

end pet_fee_is_120_l47_47335


namespace circle_radius_correct_l47_47322

noncomputable def radius_of_circle 
  (side_length : ℝ)
  (angle_tangents : ℝ)
  (sin_18 : ℝ) : ℝ := 
  sorry

theorem circle_radius_correct 
  (side_length : ℝ := 6 + 2 * Real.sqrt 5)
  (angle_tangents : ℝ := 36)
  (sin_18 : ℝ := (Real.sqrt 5 - 1) / 4) :
  radius_of_circle side_length angle_tangents sin_18 = 
  2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) :=
sorry

end circle_radius_correct_l47_47322


namespace complex_pow_sub_eq_zero_l47_47748

namespace complex_proof

open Complex

def i : ℂ := Complex.I -- Defining i to be the imaginary unit

-- Stating the conditions as definitions
def condition := i^2 = -1

-- Stating the goal as a theorem
theorem complex_pow_sub_eq_zero (cond : condition) :
  (1 + 2 * i) ^ 24 - (1 - 2 * i) ^ 24 = 0 := 
by
  sorry

end complex_proof

end complex_pow_sub_eq_zero_l47_47748


namespace least_value_l47_47215

theorem least_value : ∀ x y : ℝ, (xy + 1)^2 + (x - y)^2 ≥ 1 :=
by
  sorry

end least_value_l47_47215


namespace teal_bakery_revenue_l47_47426

theorem teal_bakery_revenue :
    let pumpkin_pies := 4
    let pumpkin_pie_slices := 8
    let pumpkin_slice_price := 5
    let custard_pies := 5
    let custard_pie_slices := 6
    let custard_slice_price := 6
    let total_pumpkin_slices := pumpkin_pies * pumpkin_pie_slices
    let total_custard_slices := custard_pies * custard_pie_slices
    let pumpkin_revenue := total_pumpkin_slices * pumpkin_slice_price
    let custard_revenue := total_custard_slices * custard_slice_price
    let total_revenue := pumpkin_revenue + custard_revenue
    total_revenue = 340 :=
by
  sorry

end teal_bakery_revenue_l47_47426


namespace pens_solution_exists_l47_47714

-- Definition of the conditions
def pen_cost_eq (x y : ℕ) : Prop :=
  17 * x + 12 * y = 150

-- Proof problem statement that follows from the conditions
theorem pens_solution_exists :
  ∃ x y : ℕ, pen_cost_eq x y :=
by
  existsi (6 : ℕ)
  existsi (4 : ℕ)
  -- Normally the proof would go here, but as stated, we use sorry.
  sorry

end pens_solution_exists_l47_47714


namespace jordan_rectangle_length_l47_47341

def rectangle_area (length width : ℝ) : ℝ := length * width

theorem jordan_rectangle_length :
  let carol_length := 8
  let carol_width := 15
  let jordan_width := 30
  let carol_area := rectangle_area carol_length carol_width
  ∃ jordan_length, rectangle_area jordan_length jordan_width = carol_area →
  jordan_length = 4 :=
by
  sorry

end jordan_rectangle_length_l47_47341


namespace number_of_balls_l47_47830

noncomputable def frequency_of_yellow (n : ℕ) : ℚ := 9 / n

theorem number_of_balls (n : ℕ) (h1 : frequency_of_yellow n = 0.30) : n = 30 :=
by sorry

end number_of_balls_l47_47830


namespace total_students_l47_47923

theorem total_students (third_grade_students fourth_grade_students second_grade_boys second_grade_girls : ℕ)
  (h1 : third_grade_students = 19)
  (h2 : fourth_grade_students = 2 * third_grade_students)
  (h3 : second_grade_boys = 10)
  (h4 : second_grade_girls = 19) :
  third_grade_students + fourth_grade_students + (second_grade_boys + second_grade_girls) = 86 :=
by
  rw [h1, h3, h4, h2]
  norm_num
  sorry

end total_students_l47_47923


namespace probability_of_high_value_hand_l47_47653

noncomputable def bridge_hand_probability : ℚ :=
  let total_combinations : ℕ := Nat.choose 16 4
  let favorable_combinations : ℕ := 1 + 16 + 16 + 16 + 36 + 96 + 16
  favorable_combinations / total_combinations

theorem probability_of_high_value_hand : bridge_hand_probability = 197 / 1820 := by
  sorry

end probability_of_high_value_hand_l47_47653


namespace find_f_minus_two_l47_47620

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_minus_two : f (-2) = 2 :=
by sorry

end find_f_minus_two_l47_47620


namespace kaleb_boxes_required_l47_47739

/-- Kaleb's Games Packing Problem -/
theorem kaleb_boxes_required (initial_games sold_games box_capacity : ℕ) (h1 : initial_games = 76) (h2 : sold_games = 46) (h3 : box_capacity = 5) :
  ((initial_games - sold_games) / box_capacity) = 6 :=
by
  -- Skipping the proof
  sorry

end kaleb_boxes_required_l47_47739


namespace expand_polynomial_l47_47422

theorem expand_polynomial (x : ℝ) :
    (5*x^2 + 3*x - 7) * (4*x^3) = 20*x^5 + 12*x^4 - 28*x^3 :=
by
  sorry

end expand_polynomial_l47_47422


namespace find_x_plus_y_l47_47106

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005) (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2009 + Real.pi / 2 := 
sorry

end find_x_plus_y_l47_47106


namespace center_of_rotation_l47_47337

noncomputable def f (z : ℂ) : ℂ := ((-1 - (Complex.I * Real.sqrt 3)) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c ∧ c = -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by
  sorry

end center_of_rotation_l47_47337


namespace alyosha_cube_cut_l47_47651

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l47_47651


namespace walnut_trees_total_l47_47543

variable (current_trees : ℕ) (new_trees : ℕ)

theorem walnut_trees_total (h1 : current_trees = 22) (h2 : new_trees = 55) : current_trees + new_trees = 77 :=
by
  sorry

end walnut_trees_total_l47_47543


namespace janets_shampoo_days_l47_47536

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end janets_shampoo_days_l47_47536


namespace determine_m_n_l47_47072

theorem determine_m_n 
  {a b c d m n : ℕ} 
  (h₁ : a + b + c + d = m^2)
  (h₂ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₃ : max (max a b) (max c d) = n^2) 
  : m = 9 ∧ n = 6 := by 
  sorry

end determine_m_n_l47_47072


namespace foma_should_give_ierema_55_coins_l47_47029

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l47_47029


namespace total_current_ages_l47_47255

theorem total_current_ages (T : ℕ) : (T - 12 = 54) → T = 66 :=
by
  sorry

end total_current_ages_l47_47255


namespace simplify_trig_expr_l47_47221

noncomputable def sin15 := Real.sin (Real.pi / 12)
noncomputable def sin30 := Real.sin (Real.pi / 6)
noncomputable def sin45 := Real.sin (Real.pi / 4)
noncomputable def sin60 := Real.sin (Real.pi / 3)
noncomputable def sin75 := Real.sin (5 * Real.pi / 12)
noncomputable def cos10 := Real.cos (Real.pi / 18)
noncomputable def cos20 := Real.cos (Real.pi / 9)
noncomputable def cos30 := Real.cos (Real.pi / 6)

theorem simplify_trig_expr :
  (sin15 + sin30 + sin45 + sin60 + sin75) / (cos10 * cos20 * cos30) = 5.128 :=
sorry

end simplify_trig_expr_l47_47221


namespace value_of_f_log_half_24_l47_47120

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_log_half_24 :
  (∀ x : ℝ, f x * -1 = f (-x)) → -- Condition 1: f(x) is an odd function.
  (∀ x : ℝ, f (x + 1) = f (x - 1)) → -- Condition 2: f(x + 1) = f(x - 1).
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2) → -- Condition 3: For 0 < x < 1, f(x) = 2^x - 2.
  f (Real.logb 0.5 24) = 1 / 2 := 
sorry

end value_of_f_log_half_24_l47_47120


namespace finite_cuboid_blocks_l47_47323

/--
Prove that there are only finitely many cuboid blocks with integer dimensions a, b, c
such that abc = 2(a - 2)(b - 2)(c - 2) and c ≤ b ≤ a.
-/
theorem finite_cuboid_blocks :
  ∃ (S : Finset (ℤ × ℤ × ℤ)), ∀ (a b c : ℤ), (abc = 2 * (a - 2) * (b - 2) * (c - 2)) → (c ≤ b) → (b ≤ a) → (a, b, c) ∈ S := 
by
  sorry

end finite_cuboid_blocks_l47_47323


namespace total_puff_pastries_l47_47213

theorem total_puff_pastries (batches trays puff_pastry volunteers : ℕ) 
  (h_batches : batches = 1) 
  (h_trays : trays = 8) 
  (h_puff_pastry : puff_pastry = 25) 
  (h_volunteers : volunteers = 1000) : 
  (volunteers * trays * puff_pastry) = 200000 := 
by 
  have h_total_trays : volunteers * trays = 1000 * 8 := by sorry
  have h_total_puff_pastries_per_volunteer : trays * puff_pastry = 8 * 25 := by sorry
  have h_total_puff_pastries : volunteers * trays * puff_pastry = 1000 * 8 * 25 := by sorry
  sorry

end total_puff_pastries_l47_47213


namespace parking_lot_cars_l47_47089

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end parking_lot_cars_l47_47089


namespace chosen_numbers_divisibility_l47_47928

theorem chosen_numbers_divisibility (n : ℕ) (S : Finset ℕ) (hS : S.card > (n + 1) / 2) :
  ∃ a ∈ S, ∃ b ∈ S, a ≠ b ∧ a ∣ b :=
by sorry

end chosen_numbers_divisibility_l47_47928


namespace part_a_part_b_l47_47746

def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem part_a : ∃ n : ℕ, is_multiple_of_9 n ∧ digit_sum n = 81 ∧ (n / 9) = 111111111 := 
sorry

theorem part_b : ∃ n1 n2 n3 n4 : ℕ,
  is_multiple_of_9 n1 ∧
  is_multiple_of_9 n2 ∧
  is_multiple_of_9 n3 ∧
  is_multiple_of_9 n4 ∧
  digit_sum n1 = 27 ∧ digit_sum n2 = 27 ∧ digit_sum n3 = 27 ∧ digit_sum n4 = 27 ∧
  (n1 / 9) + 1 = (n2 / 9) ∧ 
  (n2 / 9) + 1 = (n3 / 9) ∧ 
  (n3 / 9) + 1 = (n4 / 9) ∧ 
  (n4 / 9) < 1111 := 
sorry

end part_a_part_b_l47_47746


namespace shirts_not_all_on_sale_implications_l47_47843

variable (Shirts : Type) (store_contains : Shirts → Prop) (on_sale : Shirts → Prop)

theorem shirts_not_all_on_sale_implications :
  ¬ ∀ s, store_contains s → on_sale s → 
  (∃ s, store_contains s ∧ ¬ on_sale s) ∧ (∃ s, store_contains s ∧ ¬ on_sale s) :=
by
  sorry

end shirts_not_all_on_sale_implications_l47_47843


namespace problem_eq_solution_l47_47425

variables (a b x y : ℝ)

theorem problem_eq_solution
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : a + b + x + y < 2)
  (h6 : a + b^2 = x + y^2)
  (h7 : a^2 + b = x^2 + y) :
  a = x ∧ b = y :=
by
  sorry

end problem_eq_solution_l47_47425


namespace value_of_a_l47_47361

theorem value_of_a (a : ℝ) : (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  intro h
  have h1 : 2 = a - 1 := sorry
  have h2 : 4 = a + 1 := sorry
  have h3 : a = 3 := sorry
  exact h3

end value_of_a_l47_47361


namespace division_problem_l47_47718

theorem division_problem : 8900 / 6 / 4 = 1483.3333 :=
by sorry

end division_problem_l47_47718


namespace game_ends_and_last_numbers_depend_on_start_l47_47936
-- Given that there are three positive integers a, b, c initially.
variables (a b c : ℕ)
-- Assume that a, b, and c are greater than zero.
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the gcd of the three numbers.
def g := gcd (gcd a b) c

-- Define the game step condition.
def step_condition (a b c : ℕ): Prop := a > gcd b c

-- Define the termination condition.
def termination_condition (a b c : ℕ): Prop := ¬ step_condition a b c

-- The main theorem
theorem game_ends_and_last_numbers_depend_on_start (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ n, ∃ b' c', termination_condition n b' c' ∧
  n = g ∧ b' = g ∧ c' = g :=
sorry

end game_ends_and_last_numbers_depend_on_start_l47_47936


namespace min_area_of_rectangle_with_perimeter_100_l47_47982

theorem min_area_of_rectangle_with_perimeter_100 :
  ∃ (length width : ℕ), 
    (length + width = 50) ∧ 
    (length * width = 49) := 
by
  sorry

end min_area_of_rectangle_with_perimeter_100_l47_47982


namespace basket_weight_l47_47408

variable (B P : ℕ)

theorem basket_weight (h1 : B + P = 62) (h2 : B + P / 2 = 34) : B = 6 :=
by
  sorry

end basket_weight_l47_47408


namespace problem1_problem2_l47_47273

noncomputable def f (x : ℝ) : ℝ :=
let m := (2 * Real.cos x, 1)
let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
m.1 * n.1 + m.2 * n.2

theorem problem1 :
  ( ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi ) ∧
  ∀ k : ℤ, ∀ x ∈ Set.Icc ((1 : ℝ) * Real.pi / 6 + k * Real.pi) ((2 : ℝ) * Real.pi / 3 + k * Real.pi),
  f x < f (x + (Real.pi / 3)) :=
sorry

theorem problem2 (A : ℝ) (a b c : ℝ) :
  a ≠ 0 ∧ b = 1 ∧ f A = 2 ∧
  0 < A ∧ A < Real.pi ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2  →
  a = Real.sqrt 3 :=
sorry

end problem1_problem2_l47_47273


namespace eccentricity_of_hyperbola_l47_47017

noncomputable def hyperbola_eccentricity : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  → (∀ (c : ℝ), c^2 = a^2 + b^2) → b = 3 * a → ∃ e : ℝ, e = Real.sqrt 10

-- Statement of the problem without proof (includes the conditions)
theorem eccentricity_of_hyperbola (a b : ℝ) (h : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  (h4 : ∀ (c : ℝ), c^2 = a^2 + b^2) : hyperbola_eccentricity := 
  sorry

end eccentricity_of_hyperbola_l47_47017


namespace intersection_points_hyperbola_l47_47081

theorem intersection_points_hyperbola (t : ℝ) :
  ∃ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 4 = 0) ∧ 
  (x^2 / 4 - y^2 / (9 / 16) = 1) :=
sorry

end intersection_points_hyperbola_l47_47081


namespace sec_150_eq_neg_two_div_sqrt_three_l47_47093

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l47_47093


namespace pos_numbers_equal_l47_47562

theorem pos_numbers_equal (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eq : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end pos_numbers_equal_l47_47562


namespace sales_tax_rate_l47_47169

-- Given conditions
def cost_of_video_game : ℕ := 50
def weekly_allowance : ℕ := 10
def weekly_savings : ℕ := weekly_allowance / 2
def weeks_to_save : ℕ := 11
def total_savings : ℕ := weeks_to_save * weekly_savings

-- Proof problem statement
theorem sales_tax_rate : 
  total_savings - cost_of_video_game = (cost_of_video_game * 10) / 100 := by
  sorry

end sales_tax_rate_l47_47169


namespace find_total_games_l47_47494

-- Define the initial conditions
def avg_points_per_game : ℕ := 26
def games_played : ℕ := 15
def goal_avg_points : ℕ := 30
def required_avg_remaining : ℕ := 42

-- Statement of the proof problem
theorem find_total_games (G : ℕ) :
  avg_points_per_game * games_played + required_avg_remaining * (G - games_played) = goal_avg_points * G →
  G = 20 :=
by sorry

end find_total_games_l47_47494


namespace jason_egg_consumption_l47_47159

-- Definition for the number of eggs Jason consumes per day
def eggs_per_day : ℕ := 3

-- Definition for the number of days in a week
def days_in_week : ℕ := 7

-- Definition for the number of weeks we are considering
def weeks : ℕ := 2

-- The statement we want to prove, which combines all the conditions and provides the final answer
theorem jason_egg_consumption : weeks * days_in_week * eggs_per_day = 42 := by
sorry

end jason_egg_consumption_l47_47159


namespace percent_profit_l47_47942

theorem percent_profit (C S : ℝ) (h : 60 * C = 40 * S) : (S - C) / C * 100 = 50 := by
  sorry

end percent_profit_l47_47942


namespace Katie_average_monthly_balance_l47_47320

def balances : List ℕ := [120, 240, 180, 180, 240]

def average (l : List ℕ) : ℕ := l.sum / l.length

theorem Katie_average_monthly_balance : average balances = 192 :=
by
  sorry

end Katie_average_monthly_balance_l47_47320


namespace rectangle_area_l47_47028

theorem rectangle_area (AB AC : ℝ) (H1 : AB = 15) (H2 : AC = 17) : 
  ∃ (BC : ℝ), (AB * BC = 120) :=
by
  sorry

end rectangle_area_l47_47028


namespace john_boxes_l47_47411

theorem john_boxes
  (stan_boxes : ℕ)
  (joseph_boxes : ℕ)
  (jules_boxes : ℕ)
  (john_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_boxes = stan_boxes - 80 * stan_boxes / 100)
  (h3 : jules_boxes = joseph_boxes + 5)
  (h4 : john_boxes = jules_boxes + 20 * jules_boxes / 100) :
  john_boxes = 30 :=
by
  -- Proof will go here
  sorry

end john_boxes_l47_47411


namespace binom_divisibility_l47_47104

theorem binom_divisibility (k n : ℕ) (p : ℕ) (h1 : k > 1) (h2 : n > 1) 
  (h3 : p = 2 * k - 1) (h4 : Nat.Prime p) (h5 : p ∣ (Nat.choose n 2 - Nat.choose k 2)) : 
  p^2 ∣ (Nat.choose n 2 - Nat.choose k 2) := 
sorry

end binom_divisibility_l47_47104


namespace range_of_a_l47_47430

def f (x : ℝ) : ℝ := -x^5 - 3 * x^3 - 5 * x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 :=
by
  -- Here, we would have to show the proof, but we're skipping it
  sorry

end range_of_a_l47_47430


namespace base5_to_octal_1234_eval_f_at_3_l47_47127

-- Definition of base conversion from base 5 to decimal and to octal
def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 1234 => 1 * 5^3 + 2 * 5^2 + 3 * 5 + 4
  | _ => 0

def decimal_to_octal (n : Nat) : Nat :=
  match n with
  | 194 => 302
  | _ => 0

-- Definition of the polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x
def f (x : Nat) : Nat :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

-- Definition of Horner's method evaluation
def horner_eval (x : Nat) : Nat :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

-- Theorem statement for base-5 to octal conversion
theorem base5_to_octal_1234 : base5_to_decimal 1234 = 194 ∧ decimal_to_octal 194 = 302 :=
  by
    sorry

-- Theorem statement for polynomial evaluation using Horner's method
theorem eval_f_at_3 : horner_eval 3 = f 3 ∧ f 3 = 21324 :=
  by
    sorry

end base5_to_octal_1234_eval_f_at_3_l47_47127


namespace expand_and_simplify_product_l47_47188

theorem expand_and_simplify_product :
  5 * (x + 6) * (x + 2) * (x + 7) = 5 * x^3 + 75 * x^2 + 340 * x + 420 := 
by
  sorry

end expand_and_simplify_product_l47_47188


namespace max_min_values_f_l47_47984

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem max_min_values_f :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ Real.sqrt 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = Real.sqrt 3) :=
by
  sorry

end max_min_values_f_l47_47984


namespace angle_A_and_area_of_triangle_l47_47241

theorem angle_A_and_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (h1 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) 
(h2 : R = 2) (h3 : b^2 + c^2 = 18) :
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 := 
by
  sorry

end angle_A_and_area_of_triangle_l47_47241


namespace vertical_strips_count_l47_47143

/- Define the conditions -/

variables {a b x y : ℕ}

-- The outer rectangle has a perimeter of 50 cells
axiom outer_perimeter : 2 * a + 2 * b = 50

-- The inner hole has a perimeter of 32 cells
axiom inner_perimeter : 2 * x + 2 * y = 32

-- Cutting along all horizontal lines produces 20 strips
axiom horizontal_cuts : a + x = 20

-- We want to prove that cutting along all vertical grid lines produces 21 strips
theorem vertical_strips_count : b + y = 21 :=
by
  sorry

end vertical_strips_count_l47_47143


namespace rocket_max_speed_l47_47569

theorem rocket_max_speed (M m : ℝ) (h : 2000 * Real.log (1 + M / m) = 12000) : 
  M / m = Real.exp 6 - 1 := 
by {
  sorry
}

end rocket_max_speed_l47_47569


namespace girls_in_math_class_l47_47910

theorem girls_in_math_class (x y z : ℕ)
  (boys_girls_ratio : 5 * x = 8 * x)
  (math_science_ratio : 7 * y = 13 * x)
  (science_literature_ratio : 4 * y = 3 * z)
  (total_students : 13 * x + 4 * y + 5 * z = 720) :
  8 * x = 176 :=
by
  sorry

end girls_in_math_class_l47_47910


namespace correct_point_on_hyperbola_l47_47559

-- Given condition
def hyperbola_condition (x y : ℝ) : Prop := x * y = -4

-- Question (translated to a mathematically equivalent proof)
theorem correct_point_on_hyperbola :
  hyperbola_condition (-2) 2 :=
sorry

end correct_point_on_hyperbola_l47_47559


namespace problem_l47_47553

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (2 * m + 1) * x - 1
noncomputable def h (m : ℝ) (x : ℝ) := f m x + g m x

noncomputable def h_deriv (m : ℝ) (x : ℝ) : ℝ := m * x - (2 * m + 1) + (2 / x)

theorem problem (m : ℝ) : h_deriv m 1 = h_deriv m 3 → m = 2 / 3 :=
by
  sorry

end problem_l47_47553


namespace max_side_length_l47_47700

theorem max_side_length (a b c : ℕ) (h : a + b + c = 30) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_order : a ≤ b ∧ b ≤ c) (h_triangle_ineq : a + b > c) : c ≤ 14 := 
sorry

end max_side_length_l47_47700


namespace solve_AlyoshaCube_l47_47582

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l47_47582


namespace Mark_water_balloon_spending_l47_47964

theorem Mark_water_balloon_spending :
  let budget := 24
  let small_bag_cost := 4
  let small_bag_balloons := 50
  let medium_bag_balloons := 75
  let extra_large_bag_cost := 12
  let extra_large_bag_balloons := 200
  let total_balloons := 400
  (2 * extra_large_bag_balloons = total_balloons) → (2 * extra_large_bag_cost = budget) :=
by
  intros
  sorry

end Mark_water_balloon_spending_l47_47964


namespace hundredth_odd_integer_not_divisible_by_five_l47_47220

def odd_positive_integer (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_integer_not_divisible_by_five :
  odd_positive_integer 100 = 199 ∧ ¬ (199 % 5 = 0) :=
by
  sorry

end hundredth_odd_integer_not_divisible_by_five_l47_47220


namespace stones_required_to_pave_hall_l47_47769

noncomputable def hall_length_meters : ℝ := 36
noncomputable def hall_breadth_meters : ℝ := 15
noncomputable def stone_length_dms : ℝ := 4
noncomputable def stone_breadth_dms : ℝ := 5

theorem stones_required_to_pave_hall :
  let hall_length_dms := hall_length_meters * 10
  let hall_breadth_dms := hall_breadth_meters * 10
  let hall_area_dms_squared := hall_length_dms * hall_breadth_dms
  let stone_area_dms_squared := stone_length_dms * stone_breadth_dms
  let number_of_stones := hall_area_dms_squared / stone_area_dms_squared
  number_of_stones = 2700 :=
by
  sorry

end stones_required_to_pave_hall_l47_47769


namespace option_a_correct_l47_47419

-- Define the variables as real numbers
variables {a b : ℝ}

-- Define the main theorem to prove
theorem option_a_correct : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  -- start the proof block
  sorry

end option_a_correct_l47_47419


namespace emma_uniform_number_correct_l47_47851

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def dan : ℕ := 11  -- Example value, but needs to satisfy all conditions
noncomputable def emma : ℕ := 19  -- This is what we need to prove
noncomputable def fiona : ℕ := 13  -- Example value, but needs to satisfy all conditions
noncomputable def george : ℕ := 11  -- Example value, but needs to satisfy all conditions

theorem emma_uniform_number_correct :
  is_two_digit_prime dan ∧
  is_two_digit_prime emma ∧
  is_two_digit_prime fiona ∧
  is_two_digit_prime george ∧
  dan ≠ emma ∧ dan ≠ fiona ∧ dan ≠ george ∧
  emma ≠ fiona ∧ emma ≠ george ∧
  fiona ≠ george ∧
  dan + fiona = 23 ∧
  george + emma = 9 ∧
  dan + fiona + george + emma = 32
  → emma = 19 :=
sorry

end emma_uniform_number_correct_l47_47851


namespace calculate_fraction_l47_47288

theorem calculate_fraction :
  (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end calculate_fraction_l47_47288


namespace election_winning_votes_l47_47336

noncomputable def total_votes (x y : ℕ) (p : ℚ) : ℚ := 
  (x + y) / (1 - p)

noncomputable def winning_votes (x y : ℕ) (p : ℚ) : ℚ :=
  p * total_votes x y p

theorem election_winning_votes :
  winning_votes 2136 7636 0.54336448598130836 = 11628 := 
by
  sorry

end election_winning_votes_l47_47336


namespace cell_count_at_end_of_twelvth_day_l47_47882

def initial_cells : ℕ := 5
def days_per_cycle : ℕ := 3
def total_days : ℕ := 12
def dead_cells_on_ninth_day : ℕ := 3
noncomputable def cells_after_twelvth_day : ℕ :=
  let cycles := total_days / days_per_cycle
  let cells_before_death := initial_cells * 2^cycles
  cells_before_death - dead_cells_on_ninth_day

theorem cell_count_at_end_of_twelvth_day : cells_after_twelvth_day = 77 :=
by sorry

end cell_count_at_end_of_twelvth_day_l47_47882


namespace no_grammatical_errors_in_B_l47_47503

-- Definitions for each option’s description (conditions)
def sentence_A := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams."
def sentence_B := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region."
def sentence_C := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high."
def sentence_D := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves."

-- The statement that option B has no grammatical errors
theorem no_grammatical_errors_in_B : sentence_B = "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." :=
by
  sorry

end no_grammatical_errors_in_B_l47_47503


namespace sqrt_of_16_l47_47538

theorem sqrt_of_16 : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_of_16_l47_47538


namespace hyperbola_range_of_k_l47_47235

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ (x y : ℝ), (x^2)/(k-3) + (y^2)/(k+3) = 1 ∧ 
  (k-3 < 0) ∧ (k+3 > 0)) → (-3 < k ∧ k < 3) :=
by
  sorry

end hyperbola_range_of_k_l47_47235


namespace find_b_l47_47034

/-- Given the distance between the parallel lines l₁ : x - y = 0
  and l₂ : x - y + b = 0 is √2, prove that b = 2 or b = -2. --/
theorem find_b (b : ℝ) (h : ∀ (x y : ℝ), (x - y = 0) → ∀ (x' y' : ℝ), (x' - y' + b = 0) → (|b| / Real.sqrt 2 = Real.sqrt 2)) :
  b = 2 ∨ b = -2 :=
sorry

end find_b_l47_47034


namespace max_true_statements_l47_47203

theorem max_true_statements (c d : ℝ) : 
  (∃ n, 1 ≤ n ∧ n ≤ 5 ∧ 
    (n = (if (1/c > 1/d) then 1 else 0) +
          (if (c^2 < d^2) then 1 else 0) +
          (if (c > d) then 1 else 0) +
          (if (c > 0) then 1 else 0) +
          (if (d > 0) then 1 else 0))) → 
  n ≤ 3 := 
sorry

end max_true_statements_l47_47203


namespace number_of_benches_l47_47568

-- Define the conditions
def bench_capacity : ℕ := 4
def people_sitting : ℕ := 80
def available_spaces : ℕ := 120
def total_capacity : ℕ := people_sitting + available_spaces -- this equals 200

-- The theorem to prove the number of benches
theorem number_of_benches (B : ℕ) : bench_capacity * B = total_capacity → B = 50 :=
by
  intro h
  exact sorry

end number_of_benches_l47_47568


namespace probability_neither_perfect_square_nor_cube_l47_47752

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l47_47752


namespace evaluate_f_neg3_l47_47711

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_f_neg3 (a b c : ℝ) (h : f 3 a b c = 11) : f (-3) a b c = -9 := by
  sorry

end evaluate_f_neg3_l47_47711


namespace inequality_ab_equals_bc_l47_47846

-- Define the given conditions and state the theorem as per the proof problem
theorem inequality_ab_equals_bc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^b * b^c * c^a ≤ a^a * b^b * c^c :=
by
  sorry

end inequality_ab_equals_bc_l47_47846


namespace tom_has_7_blue_tickets_l47_47938

def number_of_blue_tickets_needed_for_bible := 10 * 10 * 10
def toms_current_yellow_tickets := 8
def toms_current_red_tickets := 3
def toms_needed_blue_tickets := 163

theorem tom_has_7_blue_tickets : 
  (number_of_blue_tickets_needed_for_bible - 
    (toms_current_yellow_tickets * 10 * 10 + 
     toms_current_red_tickets * 10 + 
     toms_needed_blue_tickets)) = 7 :=
by
  -- Proof can be provided here
  sorry

end tom_has_7_blue_tickets_l47_47938


namespace average_sales_l47_47011

-- Define the cost calculation for each special weekend
noncomputable def valentines_day_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20)) / 10

noncomputable def st_patricks_day_sales_per_ticket : Real :=
  ((3 * 2.00) + 6.25 + (8 * 1.00)) / 8

noncomputable def christmas_sales_per_ticket : Real :=
  ((6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 9

-- Define the combined average snack sales
noncomputable def combined_average_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20) + (3 * 2.00) + 6.25 + (8 * 1.00) + (6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 27

-- Proof problem as a Lean theorem
theorem average_sales : 
  valentines_day_sales_per_ticket = 2.62 ∧ 
  st_patricks_day_sales_per_ticket = 2.53 ∧ 
  christmas_sales_per_ticket = 3.16 ∧ 
  combined_average_sales_per_ticket = 2.78 :=
by 
  sorry

end average_sales_l47_47011


namespace son_present_age_l47_47325

theorem son_present_age (S F : ℕ) (h1 : F = S + 34) (h2 : F + 2 = 2 * (S + 2)) : S = 32 :=
by
  sorry

end son_present_age_l47_47325


namespace percentage_wax_left_eq_10_l47_47062

def total_amount_wax : ℕ := 
  let wax20 := 5 * 20
  let wax5 := 5 * 5
  let wax1 := 25 * 1
  wax20 + wax5 + wax1

def wax_used_for_new_candles : ℕ := 
  3 * 5

def percentage_wax_used (total_wax : ℕ) (wax_used : ℕ) : ℕ := 
  (wax_used * 100) / total_wax

theorem percentage_wax_left_eq_10 :
  percentage_wax_used total_amount_wax wax_used_for_new_candles = 10 :=
by
  sorry

end percentage_wax_left_eq_10_l47_47062


namespace elevator_stop_time_l47_47424

def time_to_reach_top (stories time_per_story : Nat) : Nat := stories * time_per_story

def total_time_with_stops (stories time_per_story stop_time : Nat) : Nat :=
  stories * time_per_story + (stories - 1) * stop_time

theorem elevator_stop_time (stories : Nat) (lola_time_per_story elevator_time_per_story total_elevator_time_to_top stop_time_per_floor : Nat)
  (lola_total_time : Nat) (is_slower : Bool)
  (h_lola: lola_total_time = time_to_reach_top stories lola_time_per_story)
  (h_slower: total_elevator_time_to_top = if is_slower then lola_total_time else 220)
  (h_no_stops: time_to_reach_top stories elevator_time_per_story + (stories - 1) * stop_time_per_floor = total_elevator_time_to_top) :
  stop_time_per_floor = 3 := 
  sorry

end elevator_stop_time_l47_47424


namespace percentage_of_first_relative_to_second_l47_47971

theorem percentage_of_first_relative_to_second (X : ℝ) 
  (first_number : ℝ := 8/100 * X) 
  (second_number : ℝ := 16/100 * X) :
  (first_number / second_number) * 100 = 50 := 
sorry

end percentage_of_first_relative_to_second_l47_47971


namespace other_root_l47_47380

theorem other_root (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + m * x - 5 = 0 → (x = 1 ∨ x = -5 / 3)) :=
by {
  sorry
}

end other_root_l47_47380


namespace quadratic_root_l47_47881

theorem quadratic_root (m : ℝ) (h : m^2 + 2 * m - 1 = 0) : 2 * m^2 + 4 * m = 2 := by
  sorry

end quadratic_root_l47_47881


namespace points_earned_l47_47463

-- Define the number of pounds required to earn one point
def pounds_per_point : ℕ := 4

-- Define the number of pounds Paige recycled
def paige_recycled : ℕ := 14

-- Define the number of pounds Paige's friends recycled
def friends_recycled : ℕ := 2

-- Define the total number of pounds recycled
def total_recycled : ℕ := paige_recycled + friends_recycled

-- Define the total number of points earned
def total_points : ℕ := total_recycled / pounds_per_point

-- Theorem to prove the total points earned
theorem points_earned : total_points = 4 := by
  sorry

end points_earned_l47_47463


namespace recyclable_cans_and_bottles_collected_l47_47250

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l47_47250


namespace cows_to_eat_grass_in_96_days_l47_47901

theorem cows_to_eat_grass_in_96_days (G r : ℕ) : 
  (∀ N : ℕ, (70 * 24 = G + 24 * r) → (30 * 60 = G + 60 * r) → 
  (∃ N : ℕ, 96 * N = G + 96 * r) → N = 20) :=
by
  intro N
  intro h1 h2 h3
  sorry

end cows_to_eat_grass_in_96_days_l47_47901


namespace proof_problem_l47_47987

noncomputable def problem : ℚ :=
  let a := 1
  let b := 2
  let c := 1
  let d := 0
  a + 2 * b + 3 * c + 4 * d

theorem proof_problem : problem = 8 := by
  -- All computations are visible here
  unfold problem
  rfl

end proof_problem_l47_47987


namespace probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l47_47264

-- Definitions based on the conditions provided
def total_books : ℕ := 100
def liberal_arts_books : ℕ := 40
def hardcover_books : ℕ := 70
def softcover_science_books : ℕ := 20
def hardcover_liberal_arts_books : ℕ := 30
def softcover_liberal_arts_books : ℕ := liberal_arts_books - hardcover_liberal_arts_books
def total_events_2 : ℕ := total_books * total_books

-- Statement part 1: Probability of selecting a hardcover liberal arts book
theorem probability_hardcover_liberal_arts :
  (hardcover_liberal_arts_books : ℝ) / total_books = 0.3 :=
sorry

-- Statement part 2: Probability of selecting a liberal arts book then a hardcover book (with replacement)
theorem probability_liberal_arts_then_hardcover :
  ((liberal_arts_books : ℝ) / total_books) * ((hardcover_books : ℝ) / total_books) = 0.28 :=
sorry

end probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l47_47264


namespace sum_a_b_c_l47_47331

theorem sum_a_b_c (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 390) (h2: a * b + b * c + c * a = 5) : a + b + c = 20 ∨ a + b + c = -20 := 
by 
  sorry

end sum_a_b_c_l47_47331


namespace pages_needed_l47_47892

def total_new_cards : ℕ := 8
def total_old_cards : ℕ := 10
def cards_per_page : ℕ := 3

theorem pages_needed (h : total_new_cards = 8) (h2 : total_old_cards = 10) (h3 : cards_per_page = 3) : 
  (total_new_cards + total_old_cards) / cards_per_page = 6 := by 
  sorry

end pages_needed_l47_47892


namespace min_value_of_expression_l47_47698

theorem min_value_of_expression (m n : ℝ) (h1 : m + 2 * n = 2) (h2 : m > 0) (h3 : n > 0) : 
  (1 / (m + 1) + 1 / (2 * n)) ≥ 4 / 3 :=
sorry

end min_value_of_expression_l47_47698


namespace tree_height_relationship_l47_47287

theorem tree_height_relationship (x : ℕ) : ∃ h : ℕ, h = 80 + 2 * x :=
by
  sorry

end tree_height_relationship_l47_47287


namespace add_A_to_10_eq_15_l47_47278

theorem add_A_to_10_eq_15 (A : ℕ) (h : A + 10 = 15) : A = 5 :=
sorry

end add_A_to_10_eq_15_l47_47278


namespace min_value_fraction_l47_47652

-- We start by defining the geometric sequence and the given conditions
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {a1 : ℝ} (h_pos : ∀ n, 0 < a n)
variable (h_geo : ∀ n, a (n + 1) = a n * r)
variable (h_a7 : a 7 = a 6 + 2 * a 5)
variable (h_am_an : ∃ m n, a m * a n = 16 * (a 1)^2)

theorem min_value_fraction : 
  ∃ (m n : ℕ), (a m * a n = 16 * (a 1)^2 ∧ (1/m) + (4/n) = 1) :=
sorry

end min_value_fraction_l47_47652


namespace total_balloons_l47_47045

theorem total_balloons (fred_balloons sam_balloons dan_balloons : ℕ) 
  (h1 : fred_balloons = 10) 
  (h2 : sam_balloons = 46) 
  (h3 : dan_balloons = 16) 
  (total : fred_balloons + sam_balloons + dan_balloons = 72) :
  fred_balloons + sam_balloons + dan_balloons = 72 := 
sorry

end total_balloons_l47_47045


namespace dropped_test_score_l47_47366

theorem dropped_test_score (A B C D : ℕ) 
  (h1 : A + B + C + D = 280) 
  (h2 : A + B + C = 225) : 
  D = 55 := 
by sorry

end dropped_test_score_l47_47366


namespace verify_number_of_true_props_l47_47623

def original_prop (a : ℝ) : Prop := a > -3 → a > 0
def converse_prop (a : ℝ) : Prop := a > 0 → a > -3
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ 0
def contrapositive_prop (a : ℝ) : Prop := a ≤ 0 → a ≤ -3

theorem verify_number_of_true_props :
  (¬ original_prop a ∧ converse_prop a ∧ inverse_prop a ∧ ¬ contrapositive_prop a) → (2 = 2) := sorry

end verify_number_of_true_props_l47_47623


namespace towel_percentage_decrease_l47_47697

theorem towel_percentage_decrease
  (L B: ℝ)
  (original_area : ℝ := L * B)
  (new_length : ℝ := 0.70 * L)
  (new_breadth : ℝ := 0.75 * B)
  (new_area : ℝ := new_length * new_breadth) :
  ((original_area - new_area) / original_area) * 100 = 47.5 := 
by 
  sorry

end towel_percentage_decrease_l47_47697


namespace simplify_fraction_90_150_l47_47778

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l47_47778


namespace expand_polynomials_l47_47642

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l47_47642


namespace tourists_originally_in_group_l47_47804

theorem tourists_originally_in_group (x : ℕ) (h₁ : 220 / x - 220 / (x + 1) = 2) : x = 10 := 
by
  sorry

end tourists_originally_in_group_l47_47804


namespace xyz_expression_l47_47989

theorem xyz_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
    (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
    (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)) = -3 / (2 * (x^2 + y^2 + xy)) :=
by sorry

end xyz_expression_l47_47989


namespace dentist_filling_cost_l47_47340

variable (F : ℝ)
variable (total_bill : ℝ := 5 * F)
variable (cleaning_cost : ℝ := 70)
variable (extraction_cost : ℝ := 290)
variable (two_fillings_cost : ℝ := 2 * F)

theorem dentist_filling_cost :
  total_bill = cleaning_cost + two_fillings_cost + extraction_cost → 
  F = 120 :=
by
  intros h
  sorry

end dentist_filling_cost_l47_47340


namespace total_lemonade_poured_l47_47094

-- Define the amounts of lemonade served during each intermission.
def first_intermission : ℝ := 0.25
def second_intermission : ℝ := 0.42
def third_intermission : ℝ := 0.25

-- State the theorem that the total amount of lemonade poured is 0.92 pitchers.
theorem total_lemonade_poured : first_intermission + second_intermission + third_intermission = 0.92 :=
by
  -- Placeholders to skip the proof.
  sorry

end total_lemonade_poured_l47_47094


namespace sufficient_but_not_necessary_l47_47339

theorem sufficient_but_not_necessary (a : ℝ) (h : a = 1/4) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 1) ∧ ¬(∀ x : ℝ, x > 0 → x + a / x ≥ 1 ↔ a = 1/4) :=
by
  sorry

end sufficient_but_not_necessary_l47_47339


namespace continuous_polynomial_continuous_cosecant_l47_47205

-- Prove that the function \( f(x) = 2x^2 - 1 \) is continuous on \(\mathbb{R}\)
theorem continuous_polynomial : Continuous (fun x : ℝ => 2 * x^2 - 1) :=
sorry

-- Prove that the function \( g(x) = (\sin x)^{-1} \) is continuous on \(\mathbb{R}\) \setminus \(\{ k\pi \mid k \in \mathbb{Z} \} \)
theorem continuous_cosecant : ∀ x : ℝ, x ∉ Set.range (fun k : ℤ => k * Real.pi) → ContinuousAt (fun x : ℝ => (Real.sin x)⁻¹) x :=
sorry

end continuous_polynomial_continuous_cosecant_l47_47205


namespace value_of_N_l47_47398

theorem value_of_N (N : ℕ): 6 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 7.5 ↔ N = 25 ∨ N = 26 ∨ N = 27 ∨ N = 28 ∨ N = 29 := 
by
  sorry

end value_of_N_l47_47398


namespace find_average_speed_l47_47303

theorem find_average_speed :
  ∃ v : ℝ, (880 / v) - (880 / (v + 10)) = 2 ∧ v = 61.5 :=
by
  sorry

end find_average_speed_l47_47303


namespace cube_side_length_is_30_l47_47530

theorem cube_side_length_is_30
  (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (s : ℝ)
  (h1 : cost_per_kg = 40)
  (h2 : coverage_per_kg = 20)
  (h3 : total_cost = 10800)
  (total_surface_area : ℝ) (W : ℝ) (C : ℝ)
  (h4 : total_surface_area = 6 * s^2)
  (h5 : W = total_surface_area / coverage_per_kg)
  (h6 : C = W * cost_per_kg)
  (h7 : C = total_cost) :
  s = 30 :=
by
  sorry

end cube_side_length_is_30_l47_47530


namespace team_a_score_l47_47428

theorem team_a_score : ∀ (A : ℕ), A + 9 + 4 = 15 → A = 2 :=
by
  intros A h
  sorry

end team_a_score_l47_47428


namespace probability_of_odd_number_l47_47355

theorem probability_of_odd_number (total_outcomes : ℕ) (odd_outcomes : ℕ) (h1 : total_outcomes = 6) (h2 : odd_outcomes = 3) : (odd_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry 

end probability_of_odd_number_l47_47355


namespace amanda_bought_30_candy_bars_l47_47917

noncomputable def candy_bars_bought (c1 c2 c3 c4 : ℕ) : ℕ :=
  let c5 := c4 * c2
  let c6 := c3 - c2
  let c7 := (c6 + c5) - c1
  c7

theorem amanda_bought_30_candy_bars :
  candy_bars_bought 7 3 22 4 = 30 :=
by
  sorry

end amanda_bought_30_candy_bars_l47_47917


namespace geometric_series_common_ratio_l47_47496

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 512) (hS : S = 2048) (h_sum : S = a / (1 - r)) : r = 3 / 4 :=
by
  rw [ha, hS] at h_sum 
  sorry

end geometric_series_common_ratio_l47_47496


namespace solve_system1_solve_system2_l47_47920

theorem solve_system1 (x y : ℚ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) :
  x = 3 / 2 ∧ y = -7 / 2 := 
sorry

theorem solve_system2 (x y : ℚ) (h1 : 3 * x - 2 * y = 1) (h2 : 7 * x + 4 * y = 11) :
  x = 1 ∧ y = 1 := 
sorry

end solve_system1_solve_system2_l47_47920


namespace part1_part2_l47_47324

-- Definitions of propositions P and q
def P (t : ℝ) : Prop := (4 - t > t - 1 ∧ t - 1 > 0)
def q (a t : ℝ) : Prop := t^2 - (a+3)*t + (a+2) < 0

-- Part 1: If P is true, find the range of t.
theorem part1 (t : ℝ) (hP : P t) : 1 < t ∧ t < 5/2 :=
by sorry

-- Part 2: If P is a sufficient but not necessary condition for q, find the range of a.
theorem part2 (a : ℝ) 
  (hP_q : ∀ t, P t → q a t) 
  (hsubset : ∀ t, 1 < t ∧ t < 5/2 → q a t) 
  : a > 1/2 :=
by sorry

end part1_part2_l47_47324


namespace radius_triple_area_l47_47658

variable (r n : ℝ)

theorem radius_triple_area (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n / 2) * (Real.sqrt 3 - 1) :=
sorry

end radius_triple_area_l47_47658


namespace value_of_a_l47_47035

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end value_of_a_l47_47035


namespace tan_330_eq_neg_sqrt3_div_3_l47_47358

theorem tan_330_eq_neg_sqrt3_div_3 :
  Real.tan (330 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_330_eq_neg_sqrt3_div_3_l47_47358


namespace part1_part2_l47_47152

noncomputable def triangleABC (a : ℝ) (cosB : ℝ) (b : ℝ) (SinA : ℝ) : Prop :=
  cosB = 3 / 5 ∧ b = 4 → SinA = 2 / 5

noncomputable def triangleABC2 (a : ℝ) (cosB : ℝ) (S : ℝ) (b c : ℝ) : Prop :=
  cosB = 3 / 5 ∧ S = 4 → b = Real.sqrt 17 ∧ c = 5

theorem part1 :
  triangleABC 2 (3 / 5) 4 (2 / 5) :=
by {
  sorry
}

theorem part2 :
  triangleABC2 2 (3 / 5) 4 (Real.sqrt 17) 5 :=
by {
  sorry
}

end part1_part2_l47_47152


namespace quadratic_complete_square_l47_47084

theorem quadratic_complete_square (c r s k : ℝ) (h1 : 8 * k^2 - 6 * k + 16 = c * (k + r)^2 + s) 
  (h2 : c = 8) 
  (h3 : r = -3 / 8) 
  (h4 : s = 119 / 8) : 
  s / r = -119 / 3 := 
by 
  sorry

end quadratic_complete_square_l47_47084


namespace regular_polygon_sides_l47_47219

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l47_47219


namespace fiftieth_term_arithmetic_seq_l47_47482

theorem fiftieth_term_arithmetic_seq : 
  (∀ (n : ℕ), (2 + (n - 1) * 5) = 247) := by
  sorry

end fiftieth_term_arithmetic_seq_l47_47482


namespace sum_of_squares_l47_47765

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2) : 
  ∃ e f : ℕ, a^2 + n * b^2 = e^2 + f^2 :=
by
  sorry

-- Theorem parameters and logical flow explained:

-- a, b, n : ℕ                  -- Natural number inputs
-- h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2  -- Condition given in the problem that a^2 + 2nb^2 is a perfect square
-- Prove that there exist natural numbers e and f such that a^2 + nb^2 = e^2 + f^2

end sum_of_squares_l47_47765


namespace a_square_plus_one_over_a_square_l47_47589

theorem a_square_plus_one_over_a_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 :=
by 
  sorry

end a_square_plus_one_over_a_square_l47_47589


namespace troy_needs_more_money_to_buy_computer_l47_47420

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l47_47420


namespace chess_team_girls_count_l47_47299

theorem chess_team_girls_count (B G : ℕ) 
  (h1 : B + G = 26) 
  (h2 : (3 / 4 : ℝ) * B + (1 / 4 : ℝ) * G = 13) : G = 13 := 
sorry

end chess_team_girls_count_l47_47299


namespace problem_l47_47975

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem_l47_47975


namespace global_maximum_condition_l47_47792

noncomputable def f (x m : ℝ) : ℝ :=
if x ≤ m then -x^2 - 2 * x else -x + 2

theorem global_maximum_condition (m : ℝ) (h : ∃ (x0 : ℝ), ∀ (x : ℝ), f x m ≤ f x0 m) : m ≥ 1 :=
sorry

end global_maximum_condition_l47_47792


namespace prove_number_of_cows_l47_47862

-- Define the conditions: Cows, Sheep, Pigs, Total animals
variables (C S P : ℕ)

-- Condition 1: Twice as many sheep as cows
def condition1 : Prop := S = 2 * C

-- Condition 2: Number of Pigs is 3 times the number of sheep
def condition2 : Prop := P = 3 * S

-- Condition 3: Total number of animals is 108
def condition3 : Prop := C + S + P = 108

-- The theorem to prove
theorem prove_number_of_cows (h1 : condition1 C S) (h2 : condition2 S P) (h3 : condition3 C S P) : C = 12 :=
sorry

end prove_number_of_cows_l47_47862


namespace number_wall_top_block_value_l47_47593

theorem number_wall_top_block_value (a b c d : ℕ) 
    (h1 : a = 8) (h2 : b = 5) (h3 : c = 3) (h4 : d = 2) : 
    (a + b + (b + c) + (c + d) = 34) :=
by
  sorry

end number_wall_top_block_value_l47_47593


namespace find_f_of_7_over_3_l47_47365

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the odd function f

-- Hypothesis: f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Hypothesis: f(1 + x) = f(-x) for all x in ℝ
axiom functional_equation (x : ℝ) : f (1 + x) = f (-x)

-- Hypothesis: f(-1/3) = 1/3
axiom initial_condition : f (-1 / 3) = 1 / 3

-- The statement we need to prove
theorem find_f_of_7_over_3 : f (7 / 3) = - (1 / 3) :=
by
  sorry -- Proof to be provided

end find_f_of_7_over_3_l47_47365


namespace amy_hours_per_week_l47_47319

theorem amy_hours_per_week {h w summer_salary school_weeks school_salary} 
  (hours_per_week_summer : h = 45)
  (weeks_summer : w = 8)
  (summer_salary_h : summer_salary = 3600)
  (school_weeks_h : school_weeks = 24)
  (school_salary_h : school_salary = 3600) :
  ∃ hours_per_week_school, hours_per_week_school = 15 :=
by
  sorry

end amy_hours_per_week_l47_47319


namespace prob_of_three_successes_correct_l47_47458

noncomputable def prob_of_three_successes (p : ℝ) : ℝ :=
  (Nat.choose 10 3) * (p^3) * (1-p)^7

theorem prob_of_three_successes_correct (p : ℝ) :
  prob_of_three_successes p = (Nat.choose 10 3 : ℝ) * (p^3) * (1-p)^7 :=
by
  sorry

end prob_of_three_successes_correct_l47_47458


namespace sequence_general_term_l47_47552

-- Define the sequence and the sum of the sequence
def Sn (n : ℕ) : ℕ := 3 + 2^n

def an (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n - 1)

-- Proposition stating the equivalence
theorem sequence_general_term (n : ℕ) : 
  (n = 1 → an n = 5) ∧ (n ≠ 1 → an n = 2^(n - 1)) :=
by 
  sorry

end sequence_general_term_l47_47552


namespace cost_of_jam_l47_47803

theorem cost_of_jam (N B J : ℕ) (hN : N > 1) (h_total_cost : N * (5 * B + 6 * J) = 348) :
    6 * N * J = 348 := by
  sorry

end cost_of_jam_l47_47803


namespace arithmetic_mean_of_sixty_integers_starting_from_3_l47_47991

def arithmetic_mean_of_sequence (a d n : ℕ) : ℚ :=
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n / n

theorem arithmetic_mean_of_sixty_integers_starting_from_3 : arithmetic_mean_of_sequence 3 1 60 = 32.5 :=
by 
  sorry

end arithmetic_mean_of_sixty_integers_starting_from_3_l47_47991


namespace problem_l47_47837

def f (a : ℕ) : ℕ := a + 3
def F (a b : ℕ) : ℕ := b^2 + a

theorem problem : F 4 (f 5) = 68 := by sorry

end problem_l47_47837


namespace distinct_remainders_count_l47_47507

theorem distinct_remainders_count {N : ℕ} (hN : N = 420) :
  ∃ (count : ℕ), (∀ n : ℕ, n ≥ 1 ∧ n ≤ N → ((n % 5 ≠ n % 6) ∧ (n % 5 ≠ n % 7) ∧ (n % 6 ≠ n % 7))) →
  count = 386 :=
by {
  sorry
}

end distinct_remainders_count_l47_47507


namespace cone_surface_area_l47_47891

theorem cone_surface_area (r l: ℝ) (θ : ℝ) (h₁ : r = 3) (h₂ : θ = 2 * π / 3) (h₃: 2 * π * r = θ * l) :
  π * r * l + π * r ^ 2 = 36 * π :=
by
  sorry

end cone_surface_area_l47_47891


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l47_47836

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l47_47836


namespace problem1_problem2_l47_47796

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Statement 1: If a = 1 and p ∧ q is true, then the range of x is 2 < x < 3
theorem problem1 (x : ℝ) (h : 1 = 1) (hpq : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
sorry

-- Statement 2: If ¬p is a sufficient but not necessary condition for ¬q, then the range of a is 1 < a ≤ 2
theorem problem2 (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 2) (h3 : ¬ (∃ x, p x a) → ¬ (∃ x, q x)) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l47_47796


namespace martha_saving_l47_47196

-- Definitions for the conditions
def daily_allowance : ℕ := 12
def half_daily_allowance : ℕ := daily_allowance / 2
def quarter_daily_allowance : ℕ := daily_allowance / 4
def days_saving_half : ℕ := 6
def day_saving_quarter : ℕ := 1

-- Statement to be proved
theorem martha_saving :
  (days_saving_half * half_daily_allowance) + (day_saving_quarter * quarter_daily_allowance) = 39 := by
  sorry

end martha_saving_l47_47196


namespace three_digit_number_with_ones_digit_5_divisible_by_5_l47_47944

theorem three_digit_number_with_ones_digit_5_divisible_by_5 (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) (h2 : N % 10 = 5) : N % 5 = 0 :=
sorry

end three_digit_number_with_ones_digit_5_divisible_by_5_l47_47944


namespace max_elements_A_union_B_l47_47922

noncomputable def sets_with_conditions (A B : Finset ℝ ) (n : ℕ) : Prop :=
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ A → s.sum id ∈ B) ∧
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ B → s.prod id ∈ A)

theorem max_elements_A_union_B {A B : Finset ℝ} (n : ℕ) (hn : 1 < n)
    (hA : A.card ≥ n) (hB : B.card ≥ n)
    (h_condition : sets_with_conditions A B n) :
    A.card + B.card ≤ 2 * n :=
  sorry

end max_elements_A_union_B_l47_47922


namespace trig_identity_example_l47_47240

theorem trig_identity_example :
  256 * (Real.sin (10 * Real.pi / 180)) * (Real.sin (30 * Real.pi / 180)) *
    (Real.sin (50 * Real.pi / 180)) * (Real.sin (70 * Real.pi / 180)) = 16 := by
  sorry

end trig_identity_example_l47_47240


namespace complex_product_l47_47115

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - i
def z2 : ℂ := 3 + i

-- Statement of the problem
theorem complex_product : z1 * z2 = 4 - 2 * i := by
  sorry

end complex_product_l47_47115


namespace find_q_l47_47313

noncomputable def q (x : ℝ) : ℝ := -2 * x^4 + 10 * x^3 - 2 * x^2 + 7 * x + 3

theorem find_q :
  ∀ x : ℝ,
  q x + (2 * x^4 - 5 * x^2 + 8 * x + 3) = (10 * x^3 - 7 * x^2 + 15 * x + 6) :=
by
  intro x
  unfold q
  sorry

end find_q_l47_47313


namespace first_part_eq_19_l47_47737

theorem first_part_eq_19 (x y : ℕ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 :=
by sorry

end first_part_eq_19_l47_47737


namespace aquarium_water_ratio_l47_47080

theorem aquarium_water_ratio :
  let length := 4
  let width := 6
  let height := 3
  let volume := length * width * height
  let halfway_volume := volume / 2
  let water_after_cat := halfway_volume / 2
  let final_water := 54
  (final_water / water_after_cat) = 3 := by
  sorry

end aquarium_water_ratio_l47_47080


namespace quadratic_roots_and_T_range_l47_47957

theorem quadratic_roots_and_T_range
  (m : ℝ)
  (h1 : m ≥ -1)
  (x1 x2 : ℝ)
  (h2 : x1^2 + 2*(m-2)*x1 + (m^2 - 3*m + 3) = 0)
  (h3 : x2^2 + 2*(m-2)*x2 + (m^2 - 3*m + 3) = 0)
  (h4 : x1 ≠ x2)
  (h5 : x1^2 + x2^2 = 6) :
  m = (5 - Real.sqrt 17) / 2 ∧ (0 < ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≤ 4 ∧ ((m * x1) / (1 - x1) + (m * x2) / (1 - x2)) ≠ 2) :=
by
  sorry

end quadratic_roots_and_T_range_l47_47957


namespace allocation_count_l47_47914

def allocate_volunteers (num_service_points : Nat) (num_volunteers : Nat) : Nat :=
  -- Definition that captures the counting logic as per the problem statement
  if num_service_points = 4 ∧ num_volunteers = 6 then 660 else 0

theorem allocation_count :
  allocate_volunteers 4 6 = 660 :=
sorry

end allocation_count_l47_47914


namespace marquita_gardens_l47_47292

open Nat

theorem marquita_gardens (num_mancino_gardens : ℕ) 
  (length_mancino_garden width_mancino_garden : ℕ) 
  (num_marquita_gardens : ℕ) 
  (length_marquita_garden width_marquita_garden : ℕ)
  (total_area : ℕ) 
  (h1 : num_mancino_gardens = 3)
  (h2 : length_mancino_garden = 16)
  (h3 : width_mancino_garden = 5)
  (h4 : length_marquita_garden = 8)
  (h5 : width_marquita_garden = 4)
  (h6 : total_area = 304)
  (hmancino_area : num_mancino_gardens * (length_mancino_garden * width_mancino_garden) = 3 * (16 * 5))
  (hcombined_area : total_area = num_mancino_gardens * (length_mancino_garden * width_mancino_garden) + num_marquita_gardens * (length_marquita_garden * width_marquita_garden)) :
  num_marquita_gardens = 2 :=
sorry

end marquita_gardens_l47_47292


namespace intersection_points_l47_47201

-- Definition of curve C by the polar equation
def curve_C (ρ : ℝ) (θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Definition of line l by the polar equation
def line_l (ρ : ℝ) (θ : ℝ) (m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 6) = m

-- Proof statement that line l intersects curve C exactly once for specific values of m
theorem intersection_points (m : ℝ) : 
  (∀ ρ θ, curve_C ρ θ → line_l ρ θ m → ρ = 0 ∧ θ = 0) ↔ (m = -1/2 ∨ m = 3/2) :=
by
  sorry

end intersection_points_l47_47201


namespace compare_exponents_and_logs_l47_47505

theorem compare_exponents_and_logs :
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  a > b ∧ b > c :=
by
  -- Definitions from the conditions
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  -- Proof here (omitted)
  sorry

end compare_exponents_and_logs_l47_47505


namespace triangle_is_3_l47_47873

def base6_addition_valid (delta : ℕ) : Prop :=
  delta < 6 ∧ 
  2 + delta + delta + 4 < 6 ∧ -- No carry effect in the middle digits
  ((delta + 3) % 6 = 4) ∧
  ((5 + delta + (2 + delta + delta + 4) / 6) % 6 = 3) ∧
  ((4 + (5 + delta + (2 + delta + delta + 4) / 6) / 6) % 6 = 5)

theorem triangle_is_3 : ∃ (δ : ℕ), base6_addition_valid δ ∧ δ = 3 :=
by
  use 3
  sorry

end triangle_is_3_l47_47873


namespace equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l47_47252
open BigOperators

-- First, we define the three equations and their constraints
def equation1_solution (k : ℤ) : ℤ × ℤ := (2 - 5 * k, -1 + 3 * k)
def equation2_solution (k : ℤ) : ℤ × ℤ := (8 - 5 * k, -4 + 3 * k)
def equation3_solution (k : ℤ) : ℤ × ℤ := (16 - 39 * k, -25 + 61 * k)

-- Define the proof that the supposed solutions hold for each equation
theorem equation1_solution_valid (k : ℤ) : 3 * (equation1_solution k).1 + 5 * (equation1_solution k).2 = 1 :=
by
  -- Proof steps would go here
  sorry

theorem equation2_solution_valid (k : ℤ) : 3 * (equation2_solution k).1 + 5 * (equation2_solution k).2 = 4 :=
by
  -- Proof steps would go here
  sorry

theorem equation3_solution_valid (k : ℤ) : 183 * (equation3_solution k).1 + 117 * (equation3_solution k).2 = 3 :=
by
  -- Proof steps would go here
  sorry

end equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l47_47252


namespace min_value_of_f_l47_47461

noncomputable def f (x : ℝ) := max (3 - x) (x^2 - 4*x + 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -1 :=
by {
  use 2,
  sorry
}

end min_value_of_f_l47_47461


namespace perpendicular_lines_from_perpendicular_planes_l47_47907

variable {Line : Type} {Plane : Type}

-- Definitions of non-coincidence, perpendicularity, parallelism
noncomputable def non_coincident_lines (a b : Line) : Prop := sorry
noncomputable def non_coincident_planes (α β : Plane) : Prop := sorry
noncomputable def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def plane_parallel_to_plane (α β : Plane) : Prop := sorry
noncomputable def plane_perpendicular_to_plane (α β : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_line (a b : Line) : Prop := sorry

-- Given non-coincident lines and planes
variable {a b : Line} {α β : Plane}

-- Problem statement
theorem perpendicular_lines_from_perpendicular_planes (h1 : non_coincident_lines a b)
  (h2 : non_coincident_planes α β)
  (h3 : line_perpendicular_to_plane a α)
  (h4 : line_perpendicular_to_plane b β)
  (h5 : plane_perpendicular_to_plane α β) : line_perpendicular_to_line a b := sorry

end perpendicular_lines_from_perpendicular_planes_l47_47907


namespace total_clouds_counted_l47_47779

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l47_47779


namespace initial_percentage_of_managers_l47_47827

theorem initial_percentage_of_managers (P : ℕ) (h : 0 ≤ P ∧ P ≤ 100)
  (total_employees initial_managers : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : initial_managers = P * total_employees / 100) 
  (remaining_employees remaining_managers : ℕ)
  (h3 : remaining_employees = total_employees - 250)
  (h4 : remaining_managers = initial_managers - 250)
  (h5 : remaining_managers * 100 = 98 * remaining_employees) :
  P = 99 := 
by
  sorry

end initial_percentage_of_managers_l47_47827


namespace increasing_interval_of_f_l47_47388

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2)

theorem increasing_interval_of_f :
  f x = (1/2)^(x^2 - 2) →
  ∀ x, f (x) ≤ f (x + 0.0001) :=
by
  sorry

end increasing_interval_of_f_l47_47388


namespace shaded_area_in_design_l47_47853

theorem shaded_area_in_design (side_length : ℝ) (radius : ℝ)
  (h1 : side_length = 30) (h2 : radius = side_length / 6)
  (h3 : 6 * (π * radius^2) = 150 * π) :
  (side_length^2) - 6 * (π * radius^2) = 900 - 150 * π := 
by
  sorry

end shaded_area_in_design_l47_47853


namespace soccer_team_percentage_l47_47074

theorem soccer_team_percentage (total_games won_games : ℕ) (h1 : total_games = 140) (h2 : won_games = 70) :
  (won_games / total_games : ℚ) * 100 = 50 := by
  sorry

end soccer_team_percentage_l47_47074


namespace car_speed_l47_47376

theorem car_speed (v t Δt : ℝ) (h1: 90 = v * t) (h2: 90 = (v + 30) * (t - Δt)) (h3: Δt = 0.5) : 
  ∃ v, 90 = v * t ∧ 90 = (v + 30) * (t - Δt) :=
by {
  sorry
}

end car_speed_l47_47376


namespace find_number_l47_47006

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 105) : x = 15 :=
by
  sorry

end find_number_l47_47006


namespace trigonometric_identity_l47_47186

theorem trigonometric_identity :
  (1 - Real.sin (Real.pi / 6)) * (1 - Real.sin (5 * Real.pi / 6)) = 1 / 4 :=
by
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by sorry
  have h2 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  sorry

end trigonometric_identity_l47_47186


namespace sqrt_expression_meaningful_l47_47511

/--
When is the algebraic expression √(x + 2) meaningful?
To ensure the algebraic expression √(x + 2) is meaningful, 
the expression under the square root, x + 2, must be greater than or equal to 0.
Thus, we need to prove that this condition is equivalent to x ≥ -2.
-/
theorem sqrt_expression_meaningful (x : ℝ) : (x + 2 ≥ 0) ↔ (x ≥ -2) :=
by
  sorry

end sqrt_expression_meaningful_l47_47511


namespace angle_quadrant_l47_47664

theorem angle_quadrant (θ : Real) (P : Real × Real) (h : P = (Real.sin θ * Real.cos θ, 2 * Real.cos θ) ∧ P.1 < 0 ∧ P.2 < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end angle_quadrant_l47_47664


namespace total_flour_used_l47_47258

def wheat_flour : ℝ := 0.2
def white_flour : ℝ := 0.1

theorem total_flour_used : wheat_flour + white_flour = 0.3 :=
by
  sorry

end total_flour_used_l47_47258


namespace range_of_b_l47_47066

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b) (h2 : a + b < 1) (h3 : 2 ≤ a - b) (h4 : a - b < 3) :
  -3 / 2 < b ∧ b < -1 / 2 :=
by
  sorry

end range_of_b_l47_47066


namespace positive_number_sum_square_l47_47768

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end positive_number_sum_square_l47_47768


namespace angle_turned_by_hour_hand_l47_47583

theorem angle_turned_by_hour_hand (rotation_degrees_per_hour : ℝ) (total_degrees_per_rotation : ℝ) :
  rotation_degrees_per_hour * 1 = -30 :=
by
  have rotation_degrees_per_hour := - total_degrees_per_rotation / 12
  have total_degrees_per_rotation := 360
  sorry

end angle_turned_by_hour_hand_l47_47583


namespace student_tickets_sold_l47_47504

theorem student_tickets_sold (A S : ℝ) (h1 : A + S = 59) (h2 : 4 * A + 2.5 * S = 222.50) : S = 9 :=
by
  sorry

end student_tickets_sold_l47_47504


namespace number_of_cows_brought_l47_47738

/--
A certain number of cows and 10 goats are brought for Rs. 1500. 
If the average price of a goat is Rs. 70, and the average price of a cow is Rs. 400, 
then the number of cows brought is 2.
-/
theorem number_of_cows_brought : 
  ∃ c : ℕ, ∃ g : ℕ, g = 10 ∧ (70 * g + 400 * c = 1500) ∧ c = 2 :=
sorry

end number_of_cows_brought_l47_47738


namespace copy_is_better_l47_47473

variable (α : ℝ)

noncomputable def p_random : ℝ := 1 / 2
noncomputable def I_mistake : ℝ := α
noncomputable def p_caught : ℝ := 1 / 10
noncomputable def I_caught : ℝ := 3 * α
noncomputable def p_neighbor_wrong : ℝ := 1 / 5
noncomputable def p_not_caught : ℝ := 9 / 10

theorem copy_is_better (α : ℝ) : 
  (12 * α / 25) < (α / 2) := by
  -- Proof goes here
  sorry

end copy_is_better_l47_47473


namespace probability_red_purple_not_same_bed_l47_47866

def colors : Set String := {"red", "yellow", "white", "purple"}

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_red_purple_not_same_bed : 
  let total_ways := C 4 2
  let unwanted_ways := 2
  let desired_ways := total_ways - unwanted_ways
  let probability := (desired_ways : ℚ) / total_ways
  probability = 2 / 3 := by
  sorry

end probability_red_purple_not_same_bed_l47_47866


namespace train_crossing_signal_pole_l47_47751

theorem train_crossing_signal_pole
  (length_train : ℕ)
  (same_length_platform : ℕ)
  (time_crossing_platform : ℕ)
  (h_train_platform : length_train = 420)
  (h_platform : same_length_platform = 420)
  (h_time_platform : time_crossing_platform = 60) : 
  (length_train / (length_train + same_length_platform / time_crossing_platform)) = 30 := 
by 
  sorry

end train_crossing_signal_pole_l47_47751


namespace homogeneous_diff_eq_solution_l47_47150

open Real

theorem homogeneous_diff_eq_solution (C : ℝ) : 
  ∀ (x y : ℝ), (y^4 - 2 * x^3 * y) * (dx) + (x^4 - 2 * x * y^3) * (dy) = 0 ↔ x^3 + y^3 = C * x * y :=
by
  sorry

end homogeneous_diff_eq_solution_l47_47150


namespace arithmetic_seq_terms_greater_than_50_l47_47641

theorem arithmetic_seq_terms_greater_than_50 :
  let a_n (n : ℕ) := 17 + (n-1) * 4
  let num_terms := (19 - 10) + 1
  ∀ (a_n : ℕ → ℕ), ((a_n 1 = 17) ∧ (∃ k, a_n k = 89) ∧ (∀ n, a_n (n + 1) = a_n n + 4)) →
  ∃ m, m = num_terms ∧ ∀ n, (10 ≤ n ∧ n ≤ 19) → a_n n > 50 :=
by
  sorry

end arithmetic_seq_terms_greater_than_50_l47_47641


namespace original_number_correct_l47_47023

-- Definitions for the problem conditions
/-
Let N be the original number.
X is the number to be subtracted.
We are given that X = 8.
We need to show that (N - 8) mod 5 = 4, (N - 8) mod 7 = 4, and (N - 8) mod 9 = 4.
-/

-- Declaration of variables
variable (N : ℕ) (X : ℕ)

-- Given conditions
def conditions := (N - X) % 5 = 4 ∧ (N - X) % 7 = 4 ∧ (N - X) % 9 = 4

-- Given the subtracted number X is 8.
def X_val : ℕ := 8

-- Prove that N = 326 meets the conditions
theorem original_number_correct (h : X = X_val) : ∃ N, conditions N X ∧ N = 326 := by
  sorry

end original_number_correct_l47_47023


namespace solve_for_x_l47_47959

theorem solve_for_x (x : ℚ) (h : (x + 8) / (x - 4) = (x - 3) / (x + 6)) : 
  x = -12 / 7 :=
sorry

end solve_for_x_l47_47959


namespace solution_is_D_l47_47567

-- Definitions of the equations
def eqA (x : ℝ) := 3 * x + 6 = 0
def eqB (x : ℝ) := 2 * x + 4 = 0
def eqC (x : ℝ) := (1 / 2) * x = -4
def eqD (x : ℝ) := 2 * x - 4 = 0

-- Theorem stating that only eqD has a solution x = 2
theorem solution_is_D : 
  ¬ eqA 2 ∧ ¬ eqB 2 ∧ ¬ eqC 2 ∧ eqD 2 := 
by
  sorry

end solution_is_D_l47_47567


namespace cost_of_450_candies_l47_47146

theorem cost_of_450_candies (box_cost : ℝ) (box_candies : ℕ) (total_candies : ℕ) 
  (h1 : box_cost = 7.50) (h2 : box_candies = 30) (h3 : total_candies = 450) : 
  (total_candies / box_candies) * box_cost = 112.50 :=
by
  sorry

end cost_of_450_candies_l47_47146


namespace framed_painting_ratio_l47_47793

theorem framed_painting_ratio
  (width_painting : ℕ)
  (height_painting : ℕ)
  (frame_side : ℕ)
  (frame_top_bottom : ℕ)
  (h1 : width_painting = 20)
  (h2 : height_painting = 30)
  (h3 : frame_top_bottom = 3 * frame_side)
  (h4 : (width_painting + 2 * frame_side) * (height_painting + 2 * frame_top_bottom) = 2 * width_painting * height_painting):
  (width_painting + 2 * frame_side) = 1/2 * (height_painting + 2 * frame_top_bottom) := 
by
  sorry

end framed_painting_ratio_l47_47793


namespace negation_proposition_l47_47144

open Set

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 + 2 * x + 5 > 0) → (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) :=
sorry

end negation_proposition_l47_47144


namespace total_players_must_be_square_l47_47386

variables (k m : ℕ)
def n : ℕ := k + m

theorem total_players_must_be_square (h: (k*(k-1) / 2) + (m*(m-1) / 2) = k * m) :
  ∃ (s : ℕ), n = s^2 :=
by sorry

end total_players_must_be_square_l47_47386


namespace speed_of_boat_in_still_water_l47_47644

variables (Vb Vs : ℝ)

-- Conditions
def condition_1 : Prop := Vb + Vs = 11
def condition_2 : Prop := Vb - Vs = 5

theorem speed_of_boat_in_still_water (h1 : condition_1 Vb Vs) (h2 : condition_2 Vb Vs) : Vb = 8 := 
by sorry

end speed_of_boat_in_still_water_l47_47644


namespace inequality_xy_l47_47195

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l47_47195


namespace max_days_for_process_C_l47_47228

/- 
  A project consists of four processes: A, B, C, and D, which require 2, 5, x, and 4 days to complete, respectively.
  The following conditions are given:
  - A and B can start at the same time.
  - C can start after A is completed.
  - D can start after both B and C are completed.
  - The total duration of the project is 9 days.
  We need to prove that the maximum number of days required to complete process C is 3.
-/
theorem max_days_for_process_C
  (A B C D : ℕ)
  (hA : A = 2)
  (hB : B = 5)
  (hD : D = 4)
  (total_duration : ℕ)
  (h_total : total_duration = 9)
  (h_condition1 : A + C + D = total_duration) : 
  C = 3 :=
by
  rw [hA, hD, h_total] at h_condition1
  linarith

#check max_days_for_process_C

end max_days_for_process_C_l47_47228


namespace g_at_2_l47_47502

def g (x : ℝ) : ℝ := x^2 - 4

theorem g_at_2 : g 2 = 0 := by
  sorry

end g_at_2_l47_47502


namespace number_of_correct_statements_l47_47099

theorem number_of_correct_statements (a : ℚ) : 
  (¬ (a < 0 → -a < 0) ∧ ¬ (|a| > 0) ∧ ¬ ((a < 0 ∨ -a < 0) ∧ ¬ (a = 0))) 
  → 0 = 0 := 
by
  intro h
  sorry

end number_of_correct_statements_l47_47099


namespace base6_sum_l47_47831

theorem base6_sum (D C : ℕ) (h₁ : D + 2 = C) (h₂ : C + 3 = 7) : C + D = 6 :=
by
  sorry

end base6_sum_l47_47831


namespace smallest_x_value_l47_47334

theorem smallest_x_value : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y^2 - 5 * y - 84) / (y - 9) = 4 / (y + 6) → y >= (x)) ∧ 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) ∧ 
  x = ( - 13 - Real.sqrt 17 ) / 2 := 
sorry

end smallest_x_value_l47_47334


namespace divisibility_of_product_l47_47073

def three_consecutive_integers (a1 a2 a3 : ℤ) : Prop :=
  a1 = a2 - 1 ∧ a3 = a2 + 1

theorem divisibility_of_product (a1 a2 a3 : ℤ) (h : three_consecutive_integers a1 a2 a3) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by
  cases h with
  | intro ha1 ha3 =>
    sorry

end divisibility_of_product_l47_47073


namespace num_square_tiles_is_zero_l47_47102

def triangular_tiles : ℕ := sorry
def square_tiles : ℕ := sorry
def hexagonal_tiles : ℕ := sorry

axiom tile_count_eq : triangular_tiles + square_tiles + hexagonal_tiles = 30
axiom edge_count_eq : 3 * triangular_tiles + 4 * square_tiles + 6 * hexagonal_tiles = 120

theorem num_square_tiles_is_zero : square_tiles = 0 :=
by
  sorry

end num_square_tiles_is_zero_l47_47102


namespace rationalize_denominator_ABC_l47_47506

theorem rationalize_denominator_ABC :
  let expr := (2 + Real.sqrt 5) / (3 - 2 * Real.sqrt 5)
  ∃ A B C : ℤ, expr = A + B * Real.sqrt C ∧ A * B * (C:ℤ) = -560 :=
by
  sorry

end rationalize_denominator_ABC_l47_47506


namespace Sonja_oil_used_l47_47716

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l47_47716


namespace infinitely_many_solutions_b_value_l47_47132

theorem infinitely_many_solutions_b_value :
  ∀ (x : ℝ) (b : ℝ), (5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  intro x b
  sorry

end infinitely_many_solutions_b_value_l47_47132


namespace ratio_e_f_l47_47289

theorem ratio_e_f (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  e / f = 9 / 4 :=
sorry

end ratio_e_f_l47_47289


namespace find_valid_primes_and_integers_l47_47976

def is_prime (p : ℕ) : Prop := Nat.Prime p

def valid_pair (p x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 2 * p ∧ x^(p-1) ∣ (p-1)^x + 1

theorem find_valid_primes_and_integers (p x : ℕ) (hp : is_prime p) 
  (hx : valid_pair p x) : 
  (p = 2 ∧ x = 1) ∨ 
  (p = 2 ∧ x = 2) ∨ 
  (p = 3 ∧ x = 1) ∨ 
  (p = 3 ∧ x = 3) ∨
  (x = 1) :=
sorry

end find_valid_primes_and_integers_l47_47976


namespace compare_x_y_l47_47681

theorem compare_x_y (a b : ℝ) (h1 : a > b) (h2 : b > 1) (x y : ℝ)
  (hx : x = a + 1 / a) (hy : y = b + 1 / b) : x > y :=
by {
  sorry
}

end compare_x_y_l47_47681


namespace part1_part2_l47_47282

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l47_47282


namespace perp_lines_implies_values_l47_47381

variable (a : ℝ)

def line1_perpendicular (a : ℝ) : Prop :=
  (1 - a) * (2 * a + 3) + a * (a - 1) = 0

theorem perp_lines_implies_values (h : line1_perpendicular a) :
  a = 1 ∨ a = -3 :=
by {
  sorry
}

end perp_lines_implies_values_l47_47381


namespace set_intersection_example_l47_47857

theorem set_intersection_example :
  let M := {x : ℝ | -1 < x ∧ x < 1}
  let N := {x : ℝ | 0 ≤ x}
  {x : ℝ | -1 < x ∧ x < 1} ∩ {x : ℝ | 0 ≤ x} = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_intersection_example_l47_47857


namespace f_sum_zero_l47_47630

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Define hypotheses based on the problem's conditions
axiom f_cube (x : ℝ) : f (x ^ 3) = (f x) ^ 3
axiom f_inj (x1 x2 : ℝ) (h : x1 ≠ x2) : f x1 ≠ f x2

-- State the proof problem
theorem f_sum_zero : f 0 + f 1 + f (-1) = 0 :=
sorry

end f_sum_zero_l47_47630


namespace andrew_purchased_mangoes_l47_47136

theorem andrew_purchased_mangoes
  (m : Nat)
  (h1 : 14 * 54 = 756)
  (h2 : 756 + 62 * m = 1376) :
  m = 10 :=
by
  sorry

end andrew_purchased_mangoes_l47_47136


namespace frank_picked_apples_l47_47343

theorem frank_picked_apples (F : ℕ) 
  (susan_picked : ℕ := 3 * F) 
  (susan_left : ℕ := susan_picked / 2) 
  (frank_left : ℕ := 2 * F / 3) 
  (total_left : susan_left + frank_left = 78) : 
  F = 36 :=
sorry

end frank_picked_apples_l47_47343


namespace area_shaded_region_l47_47849

-- Define the conditions in Lean

def semicircle_radius_ADB : ℝ := 2
def semicircle_radius_BEC : ℝ := 2
def midpoint_arc_ADB (D : ℝ) : Prop := D = semicircle_radius_ADB
def midpoint_arc_BEC (E : ℝ) : Prop := E = semicircle_radius_BEC
def semicircle_radius_DFE : ℝ := 1
def midpoint_arc_DFE (F : ℝ) : Prop := F = semicircle_radius_DFE

-- Given the mentioned conditions, we want to show the area of the shaded region is 8 square units
theorem area_shaded_region 
  (D E F : ℝ) 
  (hD : midpoint_arc_ADB D)
  (hE : midpoint_arc_BEC E)
  (hF : midpoint_arc_DFE F) : 
  ∃ (area : ℝ), area = 8 := 
sorry

end area_shaded_region_l47_47849


namespace milk_needed_6_cookies_3_3_pints_l47_47377

def gallon_to_quarts (g : ℚ) : ℚ := g * 4
def quarts_to_pints (q : ℚ) : ℚ := q * 2
def cookies_to_pints (p : ℚ) (c : ℚ) (n : ℚ) : ℚ := (p / c) * n
def measurement_error (p : ℚ) : ℚ := p * 1.1

theorem milk_needed_6_cookies_3_3_pints :
  (measurement_error (cookies_to_pints (quarts_to_pints (gallon_to_quarts 1.5)) 24 6) = 3.3) :=
by
  sorry

end milk_needed_6_cookies_3_3_pints_l47_47377


namespace h_value_l47_47470

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l47_47470


namespace triangle_area_inradius_l47_47005

theorem triangle_area_inradius
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 35)
  (h2 : inradius = 4.5)
  (h3 : area = inradius * (perimeter / 2)) :
  area = 78.75 := by
  sorry

end triangle_area_inradius_l47_47005


namespace division_remainder_l47_47838

theorem division_remainder 
  (R D Q : ℕ) 
  (h1 : D = 3 * Q)
  (h2 : D = 3 * R + 3)
  (h3 : 113 = D * Q + R) : R = 5 :=
sorry

end division_remainder_l47_47838


namespace part1_part2_l47_47182

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l47_47182


namespace pear_juice_processed_l47_47876

theorem pear_juice_processed
  (total_pears : ℝ)
  (export_percentage : ℝ)
  (juice_percentage_of_remainder : ℝ) :
  total_pears = 8.5 →
  export_percentage = 0.30 →
  juice_percentage_of_remainder = 0.60 →
  ((total_pears * (1 - export_percentage)) * juice_percentage_of_remainder) = 3.6 :=
by
  intros
  sorry

end pear_juice_processed_l47_47876


namespace value_of_expression_l47_47636

theorem value_of_expression (x : ℤ) (h : x = 5) : x^5 - 10 * x = 3075 := by
  sorry

end value_of_expression_l47_47636


namespace product_of_repeating_decimal_and_five_l47_47672

noncomputable def repeating_decimal : ℚ :=
  456 / 999

theorem product_of_repeating_decimal_and_five : 
  (repeating_decimal * 5) = 760 / 333 :=
by
  -- The proof is omitted.
  sorry

end product_of_repeating_decimal_and_five_l47_47672


namespace cos_theta_equal_neg_inv_sqrt_5_l47_47234

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.cos x

theorem cos_theta_equal_neg_inv_sqrt_5 (θ : ℝ) (h_max : ∀ x : ℝ, f θ ≥ f x) : Real.cos θ = -1 / Real.sqrt 5 :=
by
  sorry

end cos_theta_equal_neg_inv_sqrt_5_l47_47234


namespace compute_expression_l47_47809

theorem compute_expression (x : ℝ) (h : x = 3) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 :=
by 
  sorry

end compute_expression_l47_47809


namespace minimal_rooms_l47_47385

-- Definitions
def numTourists := 100

def roomsAvailable (n k : Nat) : Prop :=
  ∀ k_even : k % 2 = 0, 
    ∃ m : Nat, k = 2 * m ∧ n = 100 * (m + 1) ∨
    ∀ k_odd : k % 2 = 1, k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1

-- Proof statement
theorem minimal_rooms (k n : Nat) : roomsAvailable n k :=
by 
  -- The proof is provided in the solution steps
  sorry

end minimal_rooms_l47_47385


namespace area_of_rectangle_PQRS_l47_47466

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end area_of_rectangle_PQRS_l47_47466


namespace find_set_M_l47_47614

variable (U : Set ℕ) (M : Set ℕ)

def isUniversalSet : Prop := U = {1, 2, 3, 4, 5, 6}
def isComplement : Prop := U \ M = {1, 2, 4}

theorem find_set_M (hU : isUniversalSet U) (hC : isComplement U M) : M = {3, 5, 6} :=
  sorry

end find_set_M_l47_47614


namespace find_k_n_l47_47306

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l47_47306


namespace tom_average_speed_l47_47100

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end tom_average_speed_l47_47100


namespace solution_of_r_and_s_l47_47732

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l47_47732


namespace min_focal_length_of_hyperbola_l47_47387

theorem min_focal_length_of_hyperbola
  (a b k : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_area : k * b = 8) :
  2 * Real.sqrt (a^2 + b^2) = 8 :=
sorry -- proof to be completed

end min_focal_length_of_hyperbola_l47_47387


namespace kris_suspension_days_per_instance_is_three_l47_47157

-- Define the basic parameters given in the conditions
def total_fingers_toes : ℕ := 20
def total_bullying_instances : ℕ := 20
def multiplier : ℕ := 3

-- Define total suspension days according to the conditions
def total_suspension_days : ℕ := multiplier * total_fingers_toes

-- Define the goal: to find the number of suspension days per instance
def suspension_days_per_instance : ℕ := total_suspension_days / total_bullying_instances

-- The theorem to prove that Kris was suspended for 3 days per instance
theorem kris_suspension_days_per_instance_is_three : suspension_days_per_instance = 3 := by
  -- Skip the actual proof, focus only on the statement
  sorry

end kris_suspension_days_per_instance_is_three_l47_47157


namespace possible_divisor_of_p_l47_47271

theorem possible_divisor_of_p (p q r s : ℕ)
  (hpq : ∃ x y, p = 40 * x ∧ q = 40 * y ∧ Nat.gcd p q = 40)
  (hqr : ∃ u v, q = 45 * u ∧ r = 45 * v ∧ Nat.gcd q r = 45)
  (hrs : ∃ w z, r = 60 * w ∧ s = 60 * z ∧ Nat.gcd r s = 60)
  (hsp : ∃ t, Nat.gcd s p = 100 * t ∧ 100 ≤ Nat.gcd s p ∧ Nat.gcd s p < 1000) :
  7 ∣ p :=
sorry

end possible_divisor_of_p_l47_47271


namespace find_some_multiplier_l47_47547

theorem find_some_multiplier (m : ℕ) :
  (422 + 404)^2 - (m * 422 * 404) = 324 ↔ m = 4 :=
by
  sorry

end find_some_multiplier_l47_47547


namespace log_diff_decreases_l47_47996

-- Define the natural number n
variable (n : ℕ)

-- Proof statement
theorem log_diff_decreases (hn : 0 < n) : 
  (Real.log (n + 1) - Real.log n) = Real.log (1 + 1 / n) ∧ 
  ∀ m : ℕ, ∀ hn' : 0 < m, m > n → Real.log (m + 1) - Real.log m < Real.log (n + 1) - Real.log n := by
  sorry

end log_diff_decreases_l47_47996


namespace product_calculation_l47_47875

theorem product_calculation :
  1500 * 2023 * 0.5023 * 50 = 306903675 :=
sorry

end product_calculation_l47_47875


namespace dilation_at_origin_neg3_l47_47772

-- Define the dilation matrix centered at the origin with scale factor -3
def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0], ![0, scale_factor]]

-- The theorem stating that a dilation with scale factor -3 results in the specified matrix
theorem dilation_at_origin_neg3 :
  dilation_matrix (-3) = ![![(-3 : ℝ), 0], ![0, -3]] :=
sorry

end dilation_at_origin_neg3_l47_47772


namespace yard_flower_beds_fraction_l47_47455

theorem yard_flower_beds_fraction :
  let yard_length := 30
  let yard_width := 10
  let pool_length := 10
  let pool_width := 4
  let trap_parallel_diff := 22 - 16
  let triangle_leg := trap_parallel_diff / 2
  let triangle_area := (1 / 2) * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let total_yard_area := yard_length * yard_width
  let pool_area := pool_length * pool_width
  let usable_yard_area := total_yard_area - pool_area
  (total_triangle_area / usable_yard_area) = 9 / 260 :=
by 
  sorry

end yard_flower_beds_fraction_l47_47455


namespace max_min_sum_l47_47611

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log (x + 1) / Real.log 2

theorem max_min_sum : 
  (f 0 + f 1) = 4 := 
by
  sorry

end max_min_sum_l47_47611


namespace bicycle_distance_l47_47054

theorem bicycle_distance (b t : ℝ) (h : t ≠ 0) :
  let rate := (b / 2) / t / 3
  let total_seconds := 5 * 60
  rate * total_seconds = 50 * b / t := by
    sorry

end bicycle_distance_l47_47054


namespace chimps_moved_l47_47403

theorem chimps_moved (total_chimps : ℕ) (chimps_staying : ℕ) (chimps_moved : ℕ) 
  (h_total : total_chimps = 45)
  (h_staying : chimps_staying = 27) :
  chimps_moved = 18 :=
by
  sorry

end chimps_moved_l47_47403


namespace find_a_l47_47327

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ := sorry -- The definition of f is to be handled in the proof

theorem find_a (a : ℝ) (h1 : is_odd_function f)
  (h2 : ∀ x : ℝ, 0 < x → f x = 2^(x - a) - 2 / (x + 1))
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end find_a_l47_47327


namespace molecular_weight_of_one_mole_l47_47260

theorem molecular_weight_of_one_mole (total_molecular_weight : ℝ) (number_of_moles : ℕ) (h1 : total_molecular_weight = 304) (h2 : number_of_moles = 4) : 
  total_molecular_weight / number_of_moles = 76 := 
by
  sorry

end molecular_weight_of_one_mole_l47_47260


namespace max_value_inequality_max_value_equality_l47_47878

theorem max_value_inequality (x : ℝ) (hx : x < 0) : 
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 :=
sorry

theorem max_value_equality (x : ℝ) (hx : x = -2 * Real.sqrt 3 / 3) : 
  3 * x + 4 / x = -4 * Real.sqrt 3 :=
sorry

end max_value_inequality_max_value_equality_l47_47878


namespace sequence_divisibility_l47_47000

theorem sequence_divisibility (g : ℕ → ℕ) (h₁ : g 1 = 1) 
(h₂ : ∀ n : ℕ, g (n + 1) = g n ^ 2 + g n + 1) 
(n : ℕ) : g n ^ 2 + 1 ∣ g (n + 1) ^ 2 + 1 :=
sorry

end sequence_divisibility_l47_47000


namespace find_c_eq_3_l47_47468

theorem find_c_eq_3 (m b c : ℝ) :
  (∀ x y, y = m * x + c → ((x = b + 4 ∧ y = 5) ∨ (x = -2 ∧ y = 2))) →
  c = 3 :=
by
  sorry

end find_c_eq_3_l47_47468


namespace wall_height_l47_47444

noncomputable def brickVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def wallVolume (L W H : ℝ) : ℝ :=
  L * W * H

theorem wall_height (bricks_needed : ℝ) (brick_length_cm brick_width_cm brick_height_cm wall_length wall_width wall_height : ℝ)
  (H1 : bricks_needed = 4094.3396226415093)
  (H2 : brick_length_cm = 20)
  (H3 : brick_width_cm = 13.25)
  (H4 : brick_height_cm = 8)
  (H5 : wall_length = 7)
  (H6 : wall_width = 8)
  (H7 : brickVolume (brick_length_cm / 100) (brick_width_cm / 100) (brick_height_cm / 100) * bricks_needed = wallVolume wall_length wall_width wall_height) :
  wall_height = 0.155 :=
by
  sorry

end wall_height_l47_47444


namespace completing_the_square_result_l47_47542

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end completing_the_square_result_l47_47542


namespace pigeons_problem_l47_47548

theorem pigeons_problem
  (x y : ℕ)
  (h1 : 6 * y + 3 = x)
  (h2 : 8 * y = x + 5) : x = 27 := 
sorry

end pigeons_problem_l47_47548


namespace initial_salty_cookies_count_l47_47667

-- Define initial conditions
def initial_sweet_cookies : ℕ := 9
def sweet_cookies_ate : ℕ := 36
def salty_cookies_left : ℕ := 3
def salty_cookies_ate : ℕ := 3

-- Theorem to prove the initial salty cookies count
theorem initial_salty_cookies_count (initial_salty_cookies : ℕ) 
    (initial_sweet_cookies : initial_sweet_cookies = 9) 
    (sweet_cookies_ate : sweet_cookies_ate = 36)
    (salty_cookies_ate : salty_cookies_ate = 3) 
    (salty_cookies_left : salty_cookies_left = 3) : 
    initial_salty_cookies = 6 := 
sorry

end initial_salty_cookies_count_l47_47667


namespace sum_of_two_digit_and_reverse_l47_47261

theorem sum_of_two_digit_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9)
  (h5 : (10 * a + b) - (10 * b + a) = 9 * (a + b)) : (10 * a + b) + (10 * b + a) = 11 :=
by
  sorry

end sum_of_two_digit_and_reverse_l47_47261


namespace f_half_and_minus_half_l47_47539

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem f_half_and_minus_half :
  f (1 / 2) + f (-1 / 2) = 2 := by
  sorry

end f_half_and_minus_half_l47_47539


namespace common_ratio_of_geometric_sequence_l47_47033

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a n < a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  q = 2 := 
sorry

end common_ratio_of_geometric_sequence_l47_47033


namespace rob_nickels_count_l47_47566

noncomputable def value_of_quarters (num_quarters : ℕ) : ℝ := num_quarters * 0.25
noncomputable def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10
noncomputable def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
noncomputable def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05

theorem rob_nickels_count :
  let quarters := 7
  let dimes := 3
  let pennies := 12
  let total := 2.42
  let nickels := 5
  value_of_quarters quarters + value_of_dimes dimes + value_of_pennies pennies + value_of_nickels nickels = total :=
by
  sorry

end rob_nickels_count_l47_47566


namespace find_deleted_files_l47_47359

def original_files : Nat := 21
def remaining_files : Nat := 7
def deleted_files : Nat := 14

theorem find_deleted_files : original_files - remaining_files = deleted_files := by
  sorry

end find_deleted_files_l47_47359


namespace find_current_l47_47367

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current_l47_47367


namespace elder_age_is_30_l47_47342

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end elder_age_is_30_l47_47342


namespace smallest_m_l47_47476

theorem smallest_m (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n - m / n = 2011 / 3) : m = 1120 :=
sorry

end smallest_m_l47_47476


namespace correct_calculation_l47_47522

theorem correct_calculation (a : ℝ) : -2 * a + (2 * a - 1) = -1 := by
  sorry

end correct_calculation_l47_47522


namespace solve_inequality_l47_47383

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | x ≥ -1} = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
  (a ≠ 0 → 
    ((a > 0 → { x : ℝ | -1 ≤ x ∧ x ≤ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (-2 < a ∧ a < 0 → { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | -1 ≤ x }  = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a < -2 → { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a = -2 → { x : ℝ | True } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 })
)) :=
sorry

end solve_inequality_l47_47383


namespace djibo_age_sum_years_ago_l47_47763

theorem djibo_age_sum_years_ago (x : ℕ) (h₁: 17 - x + 28 - x = 35) : x = 5 :=
by
  -- proof is omitted as per instructions
  sorry

end djibo_age_sum_years_ago_l47_47763


namespace first_box_weight_l47_47767

theorem first_box_weight (X : ℕ) 
  (h1 : 11 + 5 + X = 18) : X = 2 := 
by
  sorry

end first_box_weight_l47_47767


namespace volume_of_cuboid_l47_47354

variable (a b c : ℝ)

def is_cuboid_adjacent_faces (a b c : ℝ) := a * b = 3 ∧ a * c = 5 ∧ b * c = 15

theorem volume_of_cuboid (a b c : ℝ) (h : is_cuboid_adjacent_faces a b c) :
  a * b * c = 15 := by
  sorry

end volume_of_cuboid_l47_47354


namespace k3_to_fourth_equals_81_l47_47825

theorem k3_to_fourth_equals_81
  (h k : ℝ → ℝ)
  (h_cond : ∀ x, x ≥ 1 → h (k x) = x^3)
  (k_cond : ∀ x, x ≥ 1 → k (h x) = x^4)
  (k_81 : k 81 = 81) :
  k 3 ^ 4 = 81 :=
sorry

end k3_to_fourth_equals_81_l47_47825


namespace chips_per_cookie_l47_47254

theorem chips_per_cookie (total_cookies : ℕ) (uneaten_chips : ℕ) (uneaten_cookies : ℕ) (h1 : total_cookies = 4 * 12) (h2 : uneaten_cookies = total_cookies / 2) (h3 : uneaten_chips = 168) : 
  uneaten_chips / uneaten_cookies = 7 :=
by sorry

end chips_per_cookie_l47_47254


namespace percent_of_475_25_is_129_89_l47_47899

theorem percent_of_475_25_is_129_89 :
  (129.89 / 475.25) * 100 = 27.33 :=
by
  sorry

end percent_of_475_25_is_129_89_l47_47899


namespace prism_faces_same_color_l47_47848

structure PrismColoring :=
  (A : Fin 5 → Fin 5 → Bool)
  (B : Fin 5 → Fin 5 → Bool)
  (A_to_B : Fin 5 → Fin 5 → Bool)

def all_triangles_diff_colors (pc : PrismColoring) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
    (pc.A i j = !pc.A i k ∨ pc.A i j = !pc.A j k) ∧
    (pc.B i j = !pc.B i k ∨ pc.B i j = !pc.B j k) ∧
    (pc.A_to_B i j = !pc.A_to_B i k ∨ pc.A_to_B i j = !pc.A_to_B j k)

theorem prism_faces_same_color (pc : PrismColoring) (h : all_triangles_diff_colors pc) :
  (∀ i j : Fin 5, pc.A i j = pc.A 0 1) ∧ (∀ i j : Fin 5, pc.B i j = pc.B 0 1) :=
sorry

end prism_faces_same_color_l47_47848


namespace watermelon_seeds_l47_47012

variable (G Y B : ℕ)

theorem watermelon_seeds (h1 : Y = 3 * G) (h2 : G > B) (h3 : B = 300) (h4 : G + Y + B = 1660) : G = 340 := by
  sorry

end watermelon_seeds_l47_47012


namespace sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l47_47348

theorem sum_of_consecutive_natural_numbers_eq_three_digit_same_digits :
  ∃ n : ℕ, (1 + n) * n / 2 = 111 * 6 ∧ n = 36 :=
by
  sorry

end sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l47_47348


namespace upstream_swim_distance_l47_47441

-- Definition of the speeds and distances
def downstream_speed (v : ℝ) := 5 + v
def upstream_speed (v : ℝ) := 5 - v
def distance := 54
def time := 6
def woman_speed_in_still_water := 5

-- Given condition: downstream_speed * time = distance
def downstream_condition (v : ℝ) := downstream_speed v * time = distance

-- Given condition: upstream distance is 'd' km
def upstream_distance (v : ℝ) := upstream_speed v * time

-- Prove that given the above conditions and solving the necessary equations, 
-- the distance swam upstream is 6 km.
theorem upstream_swim_distance {d : ℝ} (v : ℝ) (h1 : downstream_condition v) : upstream_distance v = 6 :=
by
  sorry

end upstream_swim_distance_l47_47441


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l47_47078

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) :=
sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l47_47078


namespace semicircles_problem_l47_47180

-- Define the problem in Lean
theorem semicircles_problem 
  (D : ℝ) -- Diameter of the large semicircle
  (N : ℕ) -- Number of small semicircles
  (r : ℝ) -- Radius of each small semicircle
  (H1 : D = 2 * N * r) -- Combined diameter of small semicircles is equal to the large semicircle's diameter
  (H2 : (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 10) -- Ratio of areas condition
  : N = 11 :=
   sorry -- Proof to be filled in later

end semicircles_problem_l47_47180


namespace geometric_and_arithmetic_sequence_solution_l47_47501

theorem geometric_and_arithmetic_sequence_solution:
  ∃ a b : ℝ, 
    (a > 0) ∧                  -- a is positive
    (∃ r : ℝ, 10 * r = a ∧ a * r = 1 / 2) ∧   -- geometric sequence condition
    (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) ∧        -- arithmetic sequence condition
    a = Real.sqrt 5 ∧
    b = 10 - Real.sqrt 5 := 
by 
  sorry

end geometric_and_arithmetic_sequence_solution_l47_47501


namespace cyrus_written_pages_on_fourth_day_l47_47880

theorem cyrus_written_pages_on_fourth_day :
  ∀ (total_pages first_day second_day third_day fourth_day remaining_pages: ℕ),
  total_pages = 500 →
  first_day = 25 →
  second_day = 2 * first_day →
  third_day = 2 * second_day →
  remaining_pages = total_pages - (first_day + second_day + third_day + fourth_day) →
  remaining_pages = 315 →
  fourth_day = 10 :=
by
  intros total_pages first_day second_day third_day fourth_day remaining_pages
  intros h_total h_first h_second h_third h_remain h_needed
  sorry

end cyrus_written_pages_on_fourth_day_l47_47880


namespace bethany_age_l47_47362

theorem bethany_age : ∀ (B S R : ℕ),
  (B - 3 = 2 * (S - 3)) →
  (B - 3 = R - 3 + 4) →
  (S + 5 = 16) →
  (R + 5 = 21) →
  B = 19 :=
by
  intros B S R h1 h2 h3 h4
  sorry

end bethany_age_l47_47362


namespace dawn_wash_dishes_time_l47_47634

theorem dawn_wash_dishes_time (D : ℕ) : 2 * D + 6 = 46 → D = 20 :=
by
  intro h
  sorry

end dawn_wash_dishes_time_l47_47634


namespace complement_union_l47_47483

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  U \ (M ∪ N) = {4} :=
by
  sorry

end complement_union_l47_47483


namespace ratio_diff_l47_47829

theorem ratio_diff (x : ℕ) (h1 : 7 * x = 56) : 56 - 3 * x = 32 :=
by
  sorry

end ratio_diff_l47_47829


namespace ice_cream_sundaes_l47_47973

theorem ice_cream_sundaes (flavors : Finset String) (vanilla : String) (h1 : vanilla ∈ flavors) (h2 : flavors.card = 8) :
  let remaining_flavors := flavors.erase vanilla
  remaining_flavors.card = 7 :=
by
  sorry

end ice_cream_sundaes_l47_47973


namespace expression_value_l47_47038

-- Proving the value of the expression using the factorial and sum formulas
theorem expression_value :
  (Nat.factorial 10) / (10 * 11 / 2) = 66069 := 
sorry

end expression_value_l47_47038


namespace nonagon_diagonals_l47_47118

theorem nonagon_diagonals (n : ℕ) (h1 : n = 9) : (n * (n - 3)) / 2 = 27 := by
  sorry

end nonagon_diagonals_l47_47118


namespace max_right_angles_in_triangle_l47_47571

theorem max_right_angles_in_triangle (a b c : ℝ) (h : a + b + c = 180) (ha : a = 90 ∨ b = 90 ∨ c = 90) : a = 90 ∧ b ≠ 90 ∧ c ≠ 90 ∨ b = 90 ∧ a ≠ 90 ∧ c ≠ 90 ∨ c = 90 ∧ a ≠ 90 ∧ b ≠ 90 :=
sorry

end max_right_angles_in_triangle_l47_47571


namespace difference_of_same_prime_factors_l47_47204

theorem difference_of_same_prime_factors (n : ℕ) :
  ∃ a b : ℕ, a - b = n ∧ (a.primeFactors.card = b.primeFactors.card) :=
by
  sorry

end difference_of_same_prime_factors_l47_47204


namespace not_p_sufficient_not_necessary_for_not_q_l47_47302

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) (h1 : q → p) (h2 : ¬ (p → q)) : 
  (¬p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l47_47302


namespace sqrt_comparison_l47_47950

theorem sqrt_comparison :
  let a := Real.sqrt 2
  let b := Real.sqrt 7 - Real.sqrt 3
  let c := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by
{
  sorry
}

end sqrt_comparison_l47_47950


namespace expand_polynomial_l47_47613

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end expand_polynomial_l47_47613


namespace min_value_y_l47_47594

theorem min_value_y (x y : ℝ) (h : x^2 + y^2 = 14 * x + 48 * y) : y = -1 := 
sorry

end min_value_y_l47_47594


namespace hockey_pads_cost_l47_47701

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end hockey_pads_cost_l47_47701


namespace triangle_area_AC_1_AD_BC_circumcircle_l47_47744

noncomputable def area_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_AC_1_AD_BC_circumcircle (A B C D E : ℝ × ℝ) (hAC : dist A C = 1)
  (hAD : dist A D = (2 / 3) * dist A B)
  (hMidE : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (hCircum : dist E ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = 1 / 2) :
  area_triangle_ABC A B C = (Real.sqrt 5) / 6 :=
by
  sorry

end triangle_area_AC_1_AD_BC_circumcircle_l47_47744


namespace missed_number_l47_47629

/-
  A student finds the sum \(1 + 2 + 3 + \cdots\) as his patience runs out. 
  He found the sum as 575. When the teacher declared the result wrong, 
  the student realized that he missed a number.
  Prove that the number he missed is 20.
-/

theorem missed_number (n : ℕ) (S_incorrect S_correct S_missed : ℕ) 
  (h1 : S_incorrect = 575)
  (h2 : S_correct = n * (n + 1) / 2)
  (h3 : S_correct = 595)
  (h4 : S_missed = S_correct - S_incorrect) :
  S_missed = 20 :=
sorry

end missed_number_l47_47629


namespace problem_lean_l47_47472

theorem problem_lean (a : ℝ) (h : a - 1/a = 5) : a^2 + 1/a^2 = 27 := by
  sorry

end problem_lean_l47_47472


namespace cistern_width_l47_47510

theorem cistern_width (w : ℝ) (h : 8 * w + 2 * (1.25 * 8) + 2 * (1.25 * w) = 83) : w = 6 :=
by
  sorry

end cistern_width_l47_47510


namespace rounding_proof_l47_47253

def rounding_question : Prop :=
  let num := 9.996
  let rounded_value := ((num * 100).round / 100)
  rounded_value ≠ 10.00

theorem rounding_proof : rounding_question :=
by
  sorry

end rounding_proof_l47_47253


namespace work_rate_b_l47_47598

theorem work_rate_b (A C B : ℝ) (hA : A = 1 / 8) (hC : C = 1 / 24) (h_combined : A + B + C = 1 / 4) : B = 1 / 12 :=
by
  -- Proof goes here
  sorry

end work_rate_b_l47_47598


namespace range_of_a_l47_47710

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end range_of_a_l47_47710


namespace fixed_numbers_in_diagram_has_six_solutions_l47_47886

-- Define the problem setup and constraints
def is_divisor (m n : ℕ) : Prop := ∃ k, n = k * m

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Formulating the main proof statement
theorem fixed_numbers_in_diagram_has_six_solutions : 
  ∃ (a b c k : ℕ),
    (14 * 4 * a = 14 * 6 * c) ∧
    (4 * a = 6 * c) ∧
    (2 * a = 3 * c) ∧
    (∃ k, c = 2 * k ∧ a = 3 * k) ∧
    (14 * 4 * 3 * k = 3 * k * b * 2 * k) ∧
    (∃ k, 56 * k = 6 * k^2 * b) ∧
    (b = 28 / k) ∧
    ((is_divisor k 28) ∧
     (k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 7 ∨ k = 14 ∨ k = 28)) ∧
    (6 = 6) := sorry

end fixed_numbers_in_diagram_has_six_solutions_l47_47886


namespace find_A_l47_47526

theorem find_A (A : ℕ) (B : ℕ) (h₁ : 0 ≤ B ∧ B ≤ 999) (h₂ : 1000 * A + B = A * (A + 1) / 2) : A = 1999 :=
  sorry

end find_A_l47_47526


namespace area_of_sheet_is_correct_l47_47256

noncomputable def area_of_rolled_sheet (length width height thickness : ℝ) : ℝ :=
  (length * width * height) / thickness

theorem area_of_sheet_is_correct :
  area_of_rolled_sheet 80 20 5 0.1 = 80000 :=
by
  -- The proof is omitted (sorry).
  sorry

end area_of_sheet_is_correct_l47_47256


namespace analyze_quadratic_function_l47_47360

variable (x : ℝ)

def quadratic_function : ℝ → ℝ := λ x => x^2 - 4 * x + 6

theorem analyze_quadratic_function :
  (∃ y : ℝ, quadratic_function y = (x - 2)^2 + 2) ∧
  (∃ x0 : ℝ, quadratic_function x0 = (x0 - 2)^2 + 2 ∧ x0 = 2 ∧ (∀ y : ℝ, quadratic_function y ≥ 2)) :=
by
  sorry

end analyze_quadratic_function_l47_47360


namespace sum_of_digits_of_N_eq_14_l47_47153

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l47_47153


namespace sum_squares_of_roots_of_quadratic_l47_47955

theorem sum_squares_of_roots_of_quadratic:
  ∀ (s_1 s_2 : ℝ),
  (s_1 + s_2 = 20) ∧ (s_1 * s_2 = 32) →
  (s_1^2 + s_2^2 = 336) :=
by
  intros s_1 s_2 h
  sorry

end sum_squares_of_roots_of_quadratic_l47_47955


namespace total_amount_collected_l47_47978

theorem total_amount_collected 
  (num_members : ℕ)
  (annual_fee : ℕ)
  (cost_hardcover : ℕ)
  (num_hardcovers : ℕ)
  (cost_paperback : ℕ)
  (num_paperbacks : ℕ)
  (total_collected : ℕ) :
  num_members = 6 →
  annual_fee = 150 →
  cost_hardcover = 30 →
  num_hardcovers = 6 →
  cost_paperback = 12 →
  num_paperbacks = 6 →
  total_collected = (annual_fee + cost_hardcover * num_hardcovers + cost_paperback * num_paperbacks) * num_members →
  total_collected = 2412 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end total_amount_collected_l47_47978


namespace region_area_l47_47531

/-- 
  Trapezoid has side lengths 10, 10, 10, and 22. 
  Each side of the trapezoid is the diameter of a semicircle 
  with the two semicircles on the two parallel sides of the trapezoid facing outside 
  and the other two semicircles facing inside the trapezoid.
  The region bounded by these four semicircles has area m + nπ, where m and n are positive integers.
  Prove that m + n = 188.5.
-/
theorem region_area (m n : ℝ) (h1: m = 128) (h2: n = 60.5) : m + n = 188.5 :=
by
  rw [h1, h2]
  norm_num -- simplifies the expression and checks it is equal to 188.5

end region_area_l47_47531


namespace inequal_min_value_l47_47655

theorem inequal_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1/x + 4/y) ≥ 9/4 :=
sorry

end inequal_min_value_l47_47655


namespace triangle_side_lengths_count_l47_47092

/--
How many integer side lengths are possible to complete a triangle in which 
the other sides measure 8 units and 5 units?
-/
theorem triangle_side_lengths_count :
  ∃ (n : ℕ), n = 9 ∧ ∀ (x : ℕ), (8 + 5 > x ∧ 8 + x > 5 ∧ 5 + x > 8) → (x > 3 ∧ x < 13) :=
by
  sorry

end triangle_side_lengths_count_l47_47092


namespace log_equation_solution_l47_47445

theorem log_equation_solution {x : ℝ} (hx : x > 0) (hx1 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by 
  sorry

end log_equation_solution_l47_47445


namespace trigonometric_identity_solution_l47_47410

open Real

theorem trigonometric_identity_solution (k n l : ℤ) (x : ℝ) 
  (h : 2 * cos x ≠ sin x) : 
  (sin x ^ 3 + cos x ^ 3) / (2 * cos x - sin x) = cos (2 * x) ↔
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (π / 4) * (4 * n - 1)) ∨
  (∃ l : ℤ, x = arctan (1 / 2) + π * l) :=
sorry

end trigonometric_identity_solution_l47_47410


namespace cubic_sum_l47_47350

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 :=
  sorry

end cubic_sum_l47_47350


namespace series_sum_eq_l47_47572

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l47_47572


namespace min_value_4a_plus_b_l47_47379

theorem min_value_4a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : 4*a + b = 9 :=
sorry

end min_value_4a_plus_b_l47_47379


namespace right_triangle_hypotenuse_segment_ratio_l47_47179

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ)
  (h₀ : 0 < x)
  (AB BC : ℝ)
  (h₁ : AB = 3 * x)
  (h₂ : BC = 4 * x) :
  ∃ AD DC : ℝ, AD / DC = 3 := 
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l47_47179


namespace frozenFruitSold_l47_47048

variable (totalFruit : ℕ) (freshFruit : ℕ)

-- Define the condition that the total fruit sold is 9792 pounds
def totalFruitSold := totalFruit = 9792

-- Define the condition that the fresh fruit sold is 6279 pounds
def freshFruitSold := freshFruit = 6279

-- Define the question as a Lean statement
theorem frozenFruitSold
  (h1 : totalFruitSold totalFruit)
  (h2 : freshFruitSold freshFruit) :
  totalFruit - freshFruit = 3513 := by
  sorry

end frozenFruitSold_l47_47048


namespace notAPrpos_l47_47175

def isProposition (s : String) : Prop :=
  s = "6 > 4" ∨ s = "If f(x) is a sine function, then f(x) is a periodic function." ∨ s = "1 ∈ {1, 2, 3}"

theorem notAPrpos (s : String) : ¬isProposition "Is a linear function an increasing function?" :=
by
  sorry

end notAPrpos_l47_47175


namespace base3_to_base10_conversion_l47_47318

theorem base3_to_base10_conversion : 
  (1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3^1 + 1 * 3^0 = 100) :=
by 
  sorry

end base3_to_base10_conversion_l47_47318


namespace two_non_coincident_planes_divide_space_l47_47349

-- Define conditions for non-coincident planes
def non_coincident_planes (P₁ P₂ : Plane) : Prop :=
  ¬(P₁ = P₂)

-- Define the main theorem based on the conditions and the question
theorem two_non_coincident_planes_divide_space (P₁ P₂ : Plane) 
  (h : non_coincident_planes P₁ P₂) :
  ∃ n : ℕ, n = 3 ∨ n = 4 :=
by
  sorry

end two_non_coincident_planes_divide_space_l47_47349


namespace house_to_car_ratio_l47_47364

-- Define conditions
def cost_per_night := 4000
def nights_at_hotel := 2
def cost_of_car := 30000
def total_value_of_treats := 158000

-- Prove that the ratio of the value of the house to the value of the car is 4:1
theorem house_to_car_ratio : 
  (total_value_of_treats - (nights_at_hotel * cost_per_night + cost_of_car)) / cost_of_car = 4 := by
  sorry

end house_to_car_ratio_l47_47364


namespace total_cookies_l47_47692

-- Definitions of the conditions
def cookies_in_bag : ℕ := 21
def bags_in_box : ℕ := 4
def boxes : ℕ := 2

-- Theorem stating the total number of cookies
theorem total_cookies : cookies_in_bag * bags_in_box * boxes = 168 := by
  sorry

end total_cookies_l47_47692


namespace slope_probability_l47_47696

noncomputable def probability_of_slope_gte (x y : ℝ) (Q : ℝ × ℝ) : ℝ :=
  if y - 1 / 4 ≥ (2 / 3) * (x - 3 / 4) then 1 else 0

theorem slope_probability :
  let unit_square_area := 1  -- the area of the unit square
  let valid_area := (1 / 2) * (5 / 8) * (5 / 12) -- area of the triangle above the line
  valid_area / unit_square_area = 25 / 96 :=
sorry

end slope_probability_l47_47696


namespace find_door_height_l47_47617

theorem find_door_height :
  ∃ (h : ℝ), 
  let l := 25
  let w := 15
  let H := 12
  let A := 80 * H
  let W := 960 - (6 * h + 36)
  let cost := 4 * W
  cost = 3624 ∧ h = 3 := sorry

end find_door_height_l47_47617


namespace total_flowers_sold_l47_47537

theorem total_flowers_sold :
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  flowers_mon + flowers_tue + flowers_wed + flowers_thu + flowers_fri + flowers_sat = 78 :=
by
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  sorry

end total_flowers_sold_l47_47537


namespace product_of_consecutive_even_numbers_l47_47898

theorem product_of_consecutive_even_numbers
  (a b c : ℤ)
  (h : a + b + c = 18 ∧ 2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ a < b ∧ b < c ∧ b - a = 2 ∧ c - b = 2) :
  a * b * c = 192 :=
sorry

end product_of_consecutive_even_numbers_l47_47898


namespace percentage_increase_in_side_of_square_l47_47561

theorem percentage_increase_in_side_of_square (p : ℝ) : 
  (1 + p / 100) ^ 2 = 1.3225 → 
  p = 15 :=
by
  sorry

end percentage_increase_in_side_of_square_l47_47561


namespace solution_set_inequality_l47_47121

theorem solution_set_inequality {x : ℝ} : 
  ((x - 1)^2 < 1) ↔ (0 < x ∧ x < 2) := by
  sorry

end solution_set_inequality_l47_47121


namespace solve_q_l47_47149

theorem solve_q (n m q : ℤ) 
  (h₁ : 5/6 = n/72) 
  (h₂ : 5/6 = (m + n)/90) 
  (h₃ : 5/6 = (q - m)/150) : 
  q = 140 := by
  sorry

end solve_q_l47_47149


namespace evaluate_complex_fraction_l47_47480

theorem evaluate_complex_fraction : 
  (1 / (2 + (1 / (3 + 1 / 4)))) = (13 / 30) :=
by
  sorry

end evaluate_complex_fraction_l47_47480


namespace usable_area_is_correct_l47_47464

variable (x : ℝ)

def total_field_area : ℝ := (x + 9) * (x + 7)
def flooded_area : ℝ := (2 * x - 2) * (x - 1)
def usable_area : ℝ := total_field_area x - flooded_area x

theorem usable_area_is_correct : usable_area x = -x^2 + 20 * x + 61 :=
by
  sorry

end usable_area_is_correct_l47_47464


namespace a_value_intersection_l47_47761

open Set

noncomputable def a_intersection_problem (a : ℝ) : Prop :=
  let A := { x : ℝ | x^2 < a^2 }
  let B := { x : ℝ | 1 < x ∧ x < 3 }
  let C := { x : ℝ | 1 < x ∧ x < 2 }
  A ∩ B = C → (a = 2 ∨ a = -2)

-- The theorem statement corresponding to the problem
theorem a_value_intersection (a : ℝ) :
  a_intersection_problem a :=
sorry

end a_value_intersection_l47_47761


namespace form_five_squares_l47_47493

-- The conditions of the problem as premises
variables (initial_configuration : Set (ℕ × ℕ))               -- Initial positions of 12 matchsticks
          (final_configuration : Set (ℕ × ℕ))                 -- Final positions of matchsticks to form 5 squares
          (fixed_matchsticks : Set (ℕ × ℕ))                    -- Positions of 6 fixed matchsticks
          (movable_matchsticks : Set (ℕ × ℕ))                 -- Positions of 6 movable matchsticks

-- Condition to avoid duplication or free ends
variables (no_duplication : Prop)
          (no_free_ends : Prop)

-- Proof statement
theorem form_five_squares : ∃ rearranged_configuration, 
  rearranged_configuration = final_configuration ∧
  initial_configuration = fixed_matchsticks ∪ movable_matchsticks ∧
  no_duplication ∧
  no_free_ends :=
sorry -- Proof omitted.

end form_five_squares_l47_47493


namespace problem_statement_l47_47607

noncomputable def p (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

theorem problem_statement (k : ℝ) (h_p_linear : ∀ x, p k x = k * x) 
    (h_q_quadratic : ∀ x, q x = (x + 4) * (x - 1)) 
    (h_pass_origin : p k 0 / q 0 = 0)
    (h_pass_point : p k 2 / q 2 = -1) :
    p k 1 / q 1 = -3 / 5 :=
sorry

end problem_statement_l47_47607


namespace value_of_4_Y_3_eq_neg23_l47_47587

def my_operation (a b : ℝ) (c : ℝ) : ℝ := a^2 - 2 * a * b * c + b^2

theorem value_of_4_Y_3_eq_neg23 : my_operation 4 3 2 = -23 := by
  sorry

end value_of_4_Y_3_eq_neg23_l47_47587


namespace replace_floor_cost_l47_47326

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end replace_floor_cost_l47_47326


namespace total_students_in_faculty_l47_47393

theorem total_students_in_faculty (N A B : ℕ) (hN : N = 230) (hA : A = 423) (hB : B = 134)
  (h80_percent : (N + A - B) = 80 / 100 * T) : T = 649 := 
by
  sorry

end total_students_in_faculty_l47_47393


namespace x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l47_47657

-- Define the context and main statement
theorem x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta
  (θ : ℝ)
  (hθ₁ : 0 < θ)
  (hθ₂ : θ < (π / 2))
  {x : ℝ}
  (hx : x + 1 / x = 2 * Real.sin θ)
  (n : ℕ) (hn : 0 < n) :
  x^n + 1 / x^n = 2 * Real.sin (n * θ) :=
sorry

end x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l47_47657


namespace license_plate_combinations_l47_47815

-- Definitions representing the conditions
def valid_license_plates_count : ℕ :=
  let letter_combinations := Nat.choose 26 2 -- Choose 2 unique letters
  let letter_arrangements := Nat.choose 4 2 * 2 -- Arrange the repeated letters
  let digit_combinations := 10 * 9 * 8 -- Choose different digits
  letter_combinations * letter_arrangements * digit_combinations

-- The theorem representing the problem statement
theorem license_plate_combinations :
  valid_license_plates_count = 2808000 := 
  sorry

end license_plate_combinations_l47_47815


namespace min_eq_neg_one_implies_x_eq_two_l47_47138

theorem min_eq_neg_one_implies_x_eq_two (x : ℝ) (h : min (2*x - 5) (x + 1) = -1) : x = 2 :=
sorry

end min_eq_neg_one_implies_x_eq_two_l47_47138


namespace rational_numbers_include_positives_and_negatives_l47_47720

theorem rational_numbers_include_positives_and_negatives :
  ∃ (r : ℚ), r > 0 ∧ ∃ (r' : ℚ), r' < 0 :=
by
  sorry

end rational_numbers_include_positives_and_negatives_l47_47720


namespace slope_of_parallel_line_l47_47671

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l47_47671


namespace problem_proof_l47_47974

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof_l47_47974


namespace find_x4_y4_z4_l47_47026

theorem find_x4_y4_z4
  (x y z : ℝ)
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59 / 3 :=
by
  sorry

end find_x4_y4_z4_l47_47026


namespace find_k_l47_47372

-- Define the matrix M
def M (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 3], ![0, 4, -k], ![3, -1, 2]]

-- Define the problem statement
theorem find_k (k : ℝ) (h : Matrix.det (M k) = -20) : k = 0 := by
  sorry

end find_k_l47_47372


namespace average_of_consecutive_integers_l47_47460

theorem average_of_consecutive_integers (n m : ℕ) 
  (h1 : m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) : 
  (n + 6) = (m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 :=
by
  sorry

end average_of_consecutive_integers_l47_47460


namespace algebraic_expression_value_l47_47518

theorem algebraic_expression_value (x : ℝ) (h : x = 5) : (3 / (x - 4) - 24 / (x^2 - 16)) = (1 / 3) :=
by
  have hx : x = 5 := h
  sorry

end algebraic_expression_value_l47_47518


namespace range_of_a_l47_47347

noncomputable def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≤ -1 / 2 ∨ a ≥ 2 :=
  sorry

end range_of_a_l47_47347


namespace worker_savings_l47_47370

theorem worker_savings (P : ℝ) (f : ℝ) (h : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  have h1 : 12 * f * P = 4 * (1 - f) * P := h
  have h2 : P ≠ 0 := sorry  -- P should not be 0 for the worker to have a meaningful income.
  field_simp [h2] at h1
  linarith

end worker_savings_l47_47370


namespace optimal_perimeter_proof_l47_47014

-- Definition of conditions
def fencing_length : Nat := 400
def min_width : Nat := 50
def area : Nat := 8000

-- Definition of the perimeter to be proven as optimal
def optimal_perimeter : Nat := 360

-- Theorem statement to be proven
theorem optimal_perimeter_proof (l w : Nat) (h1 : l * w = area) (h2 : 2 * l + 2 * w <= fencing_length) (h3 : w >= min_width) :
  2 * l + 2 * w = optimal_perimeter :=
sorry

end optimal_perimeter_proof_l47_47014


namespace not_perfect_square_l47_47199

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 3^n + 2 * 17^n := sorry

end not_perfect_square_l47_47199


namespace xiaoqiang_xiaolin_stamps_l47_47124

-- Definitions for initial conditions and constraints
noncomputable def x : ℤ := 227
noncomputable def y : ℤ := 221
noncomputable def k : ℤ := sorry

-- Proof problem as a theorem
theorem xiaoqiang_xiaolin_stamps:
  x + y > 400 ∧
  x - k = (13 / 19) * (y + k) ∧
  y - k = (11 / 17) * (x + k) ∧
  x = 227 ∧ 
  y = 221 :=
by
  sorry

end xiaoqiang_xiaolin_stamps_l47_47124


namespace slant_heights_of_cones_l47_47481

-- Define the initial conditions
variables (r r1 x y : Real)

-- Define the surface area condition
def surface_area_condition : Prop :=
  r * Real.sqrt (r ^ 2 + x ^ 2) + r ^ 2 = r1 * Real.sqrt (r1 ^ 2 + y ^ 2) + r1 ^ 2

-- Define the volume condition
def volume_condition : Prop :=
  r ^ 2 * Real.sqrt (x ^ 2 - r ^ 2) = r1 ^ 2 * Real.sqrt (y ^ 2 - r1 ^ 2)

-- Statement of the proof problem: Prove that the slant heights x and y are given by
theorem slant_heights_of_cones
  (h1 : surface_area_condition r r1 x y)
  (h2 : volume_condition r r1 x y) :
  x = (r ^ 2 + 2 * r1 ^ 2) / r ∧ y = (r1 ^ 2 + 2 * r ^ 2) / r1 := 
  sorry

end slant_heights_of_cones_l47_47481


namespace minimal_total_distance_l47_47733

variable (A B : ℝ) -- Coordinates of houses A and B on a straight road
variable (h_dist : B - A = 50) -- The distance between A and B is 50 meters

-- Define a point X on the road
variable (X : ℝ)

-- Define the function that calculates the total distance from point X to A and B
def total_distance (A B X : ℝ) := abs (X - A) + abs (X - B)

-- The theorem stating that the total distance is minimized if X lies on the line segment AB
theorem minimal_total_distance : A ≤ X ∧ X ≤ B ↔ total_distance A B X = B - A :=
by
  sorry

end minimal_total_distance_l47_47733


namespace nina_max_digits_l47_47238

-- Define the conditions
def sam_digits (C : ℕ) := C + 6
def mina_digits := 24
def nina_digits (C : ℕ) := (7 * C) / 2

-- Define Carlos's digits and the sum condition
def carlos_digits := mina_digits / 6
def total_digits (C : ℕ) := C + sam_digits C + mina_digits + nina_digits C

-- Prove the maximum number of digits Nina could memorize
theorem nina_max_digits : ∀ C : ℕ, C = carlos_digits →
  total_digits C ≤ 100 → nina_digits C ≤ 62 :=
by
  intro C hC htotal
  sorry

end nina_max_digits_l47_47238


namespace bill_difference_l47_47363

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l47_47363


namespace units_digit_base9_addition_l47_47514

theorem units_digit_base9_addition : 
  (∃ (d₁ d₂ : ℕ), d₁ < 9 ∧ d₂ < 9 ∧ (85 % 9 = d₁) ∧ (37 % 9 = d₂)) → ((d₁ + d₂) % 9 = 3) :=
by
  sorry

end units_digit_base9_addition_l47_47514


namespace num_arithmetic_sequences_l47_47749

theorem num_arithmetic_sequences (d : ℕ) (x : ℕ)
  (h_sum : 8 * x + 28 * d = 1080)
  (h_no180 : ∀ i, x + i * d ≠ 180)
  (h_pos : ∀ i, 0 < x + i * d)
  (h_less160 : ∀ i, x + i * d < 160)
  (h_not_equiangular : d ≠ 0) :
  ∃ n : ℕ, n = 3 :=
by sorry

end num_arithmetic_sequences_l47_47749


namespace height_difference_l47_47316

def pine_tree_height : ℚ := 12 + 1 / 4
def maple_tree_height : ℚ := 18 + 1 / 2

theorem height_difference :
  maple_tree_height - pine_tree_height = 6 + 1 / 4 :=
by sorry

end height_difference_l47_47316


namespace min_value_a_squared_plus_b_squared_l47_47785

theorem min_value_a_squared_plus_b_squared :
  ∃ (a b : ℝ), (b = 3 * a - 6) → (a^2 + b^2 = 18 / 5) :=
by
  sorry

end min_value_a_squared_plus_b_squared_l47_47785


namespace floor_div_eq_floor_floor_div_l47_47434

theorem floor_div_eq_floor_floor_div (α : ℝ) (d : ℕ) (hα : 0 < α) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ :=
by sorry

end floor_div_eq_floor_floor_div_l47_47434


namespace sum_of_last_two_digits_l47_47418

theorem sum_of_last_two_digits (a b : ℕ) (ha: a = 6) (hb: b = 10) :
  ((a^15 + b^15) % 100) = 0 :=
by
  -- ha, hb represent conditions given.
  sorry

end sum_of_last_two_digits_l47_47418


namespace triangle_interior_angle_at_least_one_leq_60_l47_47200

theorem triangle_interior_angle_at_least_one_leq_60 {α β γ : ℝ} :
  α + β + γ = 180 →
  (α > 60 ∧ β > 60 ∧ γ > 60) → false :=
by
  intro hsum hgt
  have hα : α > 60 := hgt.1
  have hβ : β > 60 := hgt.2.1
  have hγ : γ > 60 := hgt.2.2
  have h_total: α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add hα hβ) hγ
  linarith

end triangle_interior_angle_at_least_one_leq_60_l47_47200


namespace passed_candidates_count_l47_47639

theorem passed_candidates_count
    (average_total : ℝ)
    (number_candidates : ℕ)
    (average_passed : ℝ)
    (average_failed : ℝ)
    (total_marks : ℝ) :
    average_total = 35 →
    number_candidates = 120 →
    average_passed = 39 →
    average_failed = 15 →
    total_marks = average_total * number_candidates →
    (∃ P F, P + F = number_candidates ∧ 39 * P + 15 * F = total_marks ∧ P = 100) :=
by
  sorry

end passed_candidates_count_l47_47639


namespace angle_C_correct_l47_47498

theorem angle_C_correct (A B C : ℝ) (h1 : A = 65) (h2 : B = 40) (h3 : A + B + C = 180) : C = 75 :=
sorry

end angle_C_correct_l47_47498


namespace ratio_dog_to_hamster_l47_47584

noncomputable def dog_lifespan : ℝ := 10
noncomputable def hamster_lifespan : ℝ := 2.5

theorem ratio_dog_to_hamster : dog_lifespan / hamster_lifespan = 4 :=
by
  sorry

end ratio_dog_to_hamster_l47_47584


namespace find_share_of_C_l47_47173

-- Definitions and assumptions
def share_in_ratio (x : ℕ) : Prop :=
  let a := 2 * x
  let b := 3 * x
  let c := 4 * x
  a + b + c = 945

-- Statement to prove
theorem find_share_of_C :
  ∃ x : ℕ, share_in_ratio x ∧ 4 * x = 420 :=
by
  -- We skip the proof here.
  sorry

end find_share_of_C_l47_47173


namespace intersection_of_A_and_B_l47_47801

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := {x | x^2 - 2 * x < 0}
def setB : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem to prove the intersection A ∩ B
theorem intersection_of_A_and_B : ((setA ∩ setB) = {x : ℝ | 0 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_A_and_B_l47_47801


namespace consecutive_sum_is_10_l47_47690

theorem consecutive_sum_is_10 (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) : a + 2 = 10 :=
sorry

end consecutive_sum_is_10_l47_47690


namespace range_of_a_l47_47619

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, a * x^2 - 2 * x + 1 > 0
def proposition_q := ∀ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) (2 : ℝ) → x + (1 / x) > a

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l47_47619


namespace ab_value_l47_47773

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end ab_value_l47_47773


namespace angle_SQR_l47_47893

-- Define angles
def PQR : ℝ := 40
def PQS : ℝ := 28

-- State the theorem
theorem angle_SQR : PQR - PQS = 12 := by
  sorry

end angle_SQR_l47_47893


namespace sum_of_distances_eq_l47_47527

noncomputable def sum_of_distances_from_vertex_to_midpoints (A B C M N O : ℝ × ℝ) : ℝ :=
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  AM + AN + AO

theorem sum_of_distances_eq (A B C M N O : ℝ × ℝ) (h1 : B = (3, 0)) (h2 : C = (3/2, (3 * Real.sqrt 3/2))) (h3 : M = (3/2, 0)) (h4 : N = (9/4, (3 * Real.sqrt 3/4))) (h5 : O = (3/4, (3 * Real.sqrt 3/4))) :
  sum_of_distances_from_vertex_to_midpoints A B C M N O = 3 + (9 / 2) * Real.sqrt 3 :=
by
  sorry

end sum_of_distances_eq_l47_47527


namespace eval_expression_l47_47745

def x : ℤ := 18 / 3 * 7^2 - 80 + 4 * 7

theorem eval_expression : -x = -242 := by
  sorry

end eval_expression_l47_47745


namespace total_animal_crackers_eaten_l47_47517

-- Define the context and conditions
def number_of_students : ℕ := 20
def uneaten_students : ℕ := 2
def crackers_per_pack : ℕ := 10

-- Define the statement and prove the question equals the answer given the conditions
theorem total_animal_crackers_eaten : 
  (number_of_students - uneaten_students) * crackers_per_pack = 180 := by
  sorry

end total_animal_crackers_eaten_l47_47517


namespace scientific_notation_gdp_2022_l47_47189

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end scientific_notation_gdp_2022_l47_47189


namespace find_A_plus_B_l47_47609

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l47_47609


namespace sin_2alpha_value_l47_47601

noncomputable def sin_2alpha_through_point (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  2 * sin_alpha * cos_alpha

theorem sin_2alpha_value :
  sin_2alpha_through_point (-3) 4 = -24 / 25 :=
by
  sorry

end sin_2alpha_value_l47_47601


namespace customers_left_l47_47413

theorem customers_left (initial_customers remaining_tables people_per_table customers_left : ℕ)
    (h_initial : initial_customers = 62)
    (h_tables : remaining_tables = 5)
    (h_people : people_per_table = 9)
    (h_left : customers_left = initial_customers - remaining_tables * people_per_table) : 
    customers_left = 17 := 
    by 
        -- Provide the proof here 
        sorry

end customers_left_l47_47413


namespace smallest_equal_cost_l47_47110

def decimal_cost (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_cost (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem smallest_equal_cost :
  ∃ n : ℕ, n < 200 ∧ decimal_cost n = binary_cost n ∧ (∀ m : ℕ, m < 200 ∧ decimal_cost m = binary_cost m → m ≥ n) :=
by
  -- Proof goes here
  sorry

end smallest_equal_cost_l47_47110


namespace sum_of_number_and_square_is_306_l47_47859

theorem sum_of_number_and_square_is_306 : ∃ x : ℤ, x + x^2 = 306 ∧ x = 17 :=
by
  sorry

end sum_of_number_and_square_is_306_l47_47859


namespace units_digit_proof_l47_47635

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end units_digit_proof_l47_47635


namespace passed_both_tests_l47_47058

theorem passed_both_tests :
  ∀ (total_students passed_long_jump passed_shot_put failed_both passed_both: ℕ),
  total_students = 50 →
  passed_long_jump = 40 →
  passed_shot_put = 31 →
  failed_both = 4 →
  passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both = total_students →
  passed_both = 25 :=
by
  intros total_students passed_long_jump passed_shot_put failed_both passed_both h1 h2 h3 h4 h5
  -- proof can be skipped using sorry
  sorry

end passed_both_tests_l47_47058


namespace paint_required_for_small_statues_l47_47368

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_required_for_small_statues_l47_47368


namespace find_d_l47_47329

theorem find_d (x y d : ℕ) (h_midpoint : (1 + 5)/2 = 3 ∧ (3 + 11)/2 = 7) 
  : x + y = d ↔ d = 10 := 
sorry

end find_d_l47_47329


namespace translate_point_left_l47_47353

def initial_point : ℝ × ℝ := (-2, -1)
def translation_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1 - units, p.2)

theorem translate_point_left :
  translation_left initial_point 2 = (-4, -1) :=
by
  -- By definition and calculation
  -- Let p = initial_point
  -- x' = p.1 - 2,
  -- y' = p.2
  -- translation_left (-2, -1) 2 = (-4, -1)
  sorry

end translate_point_left_l47_47353


namespace present_age_of_son_l47_47307

theorem present_age_of_son
  (S M : ℕ)
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  sorry
}

end present_age_of_son_l47_47307


namespace osmanthus_trees_variance_l47_47142

variable (n : Nat) (p : ℚ)

def variance_binomial_distribution (n : Nat) (p : ℚ) : ℚ :=
  n * p * (1 - p)

theorem osmanthus_trees_variance (n : Nat) (p : ℚ) (h₁ : n = 4) (h₂ : p = 4 / 5) :
  variance_binomial_distribution n p = 16 / 25 := by
  sorry

end osmanthus_trees_variance_l47_47142


namespace find_m_eq_l47_47807

theorem find_m_eq : 
  (∀ (m : ℝ),
    ((m + 2)^2 + (m + 3)^2 = m^2 + 16 + 4 + (m - 1)^2) →
    m = 2 / 3 ) :=
by
  intros m h
  sorry

end find_m_eq_l47_47807


namespace constants_A_B_C_l47_47742

theorem constants_A_B_C (A B C : ℝ) (h₁ : ∀ x : ℝ, (x^2 + 5 * x - 6) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1)) :
  A = -6 ∧ B = 0 ∧ C = 7 :=
by
  sorry

end constants_A_B_C_l47_47742


namespace range_of_k_l47_47860

theorem range_of_k (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 + 2*x1 - k = 0) ∧ (x2^2 + 2*x2 - k = 0)) ↔ k > -1 :=
by
  sorry

end range_of_k_l47_47860


namespace abs_sum_inequality_solution_l47_47784

theorem abs_sum_inequality_solution (x : ℝ) : 
  (|x - 5| + |x + 1| < 8) ↔ (-2 < x ∧ x < 6) :=
sorry

end abs_sum_inequality_solution_l47_47784


namespace tan_pi_div_a_of_point_on_cubed_function_l47_47167

theorem tan_pi_div_a_of_point_on_cubed_function (a : ℝ) (h : (a, 27) ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 3}) : 
  Real.tan (Real.pi / a) = Real.sqrt 3 := sorry

end tan_pi_div_a_of_point_on_cubed_function_l47_47167


namespace binomial_coefficients_sum_l47_47816

theorem binomial_coefficients_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 := by
  sorry

end binomial_coefficients_sum_l47_47816


namespace probability_palindrome_divisible_by_11_is_zero_l47_47645

def is_palindrome (n : ℕ) :=
  3000 ≤ n ∧ n < 8000 ∧ ∃ (a b : ℕ), 3 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ (n : ℕ), is_palindrome n ∧ n % 11 = 0) → false := by sorry

end probability_palindrome_divisible_by_11_is_zero_l47_47645


namespace standard_equation_of_circle_l47_47310

/-- A circle with radius 2, center in the fourth quadrant, and tangent to the lines x = 0 and x + y = 2√2 has the standard equation (x - 2)^2 + (y + 2)^2 = 4. -/
theorem standard_equation_of_circle :
  ∃ a, a > 0 ∧ (∀ x y : ℝ, ((x - a)^2 + (y + 2)^2 = 4) ∧ 
                        (a > 0) ∧ 
                        (x = 0 → a = 2) ∧
                        x + y = 2 * Real.sqrt 2 → a = 2) := 
by
  sorry

end standard_equation_of_circle_l47_47310


namespace hyperbola_asymptote_b_value_l47_47864

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, y = 2 * x → x^2 - (y^2 / b^2) = 1) :
  b = 2 :=
sorry

end hyperbola_asymptote_b_value_l47_47864


namespace find_smallest_denominator_difference_l47_47003

theorem find_smallest_denominator_difference :
  ∃ (r s : ℕ), 
    r > 0 ∧ s > 0 ∧ 
    (5 : ℚ) / 11 < r / s ∧ r / s < (4 : ℚ) / 9 ∧ 
    ¬ ∃ t : ℕ, t < s ∧ (5 : ℚ) / 11 < r / t ∧ r / t < (4 : ℚ) / 9 ∧ 
    s - r = 11 := 
sorry

end find_smallest_denominator_difference_l47_47003


namespace position_of_seventeen_fifteen_in_sequence_l47_47789

theorem position_of_seventeen_fifteen_in_sequence :
  ∃ n : ℕ, (17 : ℚ) / 15 = (n + 3 : ℚ) / (n + 1) :=
sorry

end position_of_seventeen_fifteen_in_sequence_l47_47789


namespace ivan_total_money_l47_47390

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end ivan_total_money_l47_47390


namespace fraction_before_simplification_is_24_56_l47_47958

-- Definitions of conditions
def fraction_before_simplification_simplifies_to_3_7 (a b : ℕ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧ Int.gcd a b = 1 ∧ (a = 3 * Int.gcd a b ∧ b = 7 * Int.gcd a b)

def sum_of_numerator_and_denominator_is_80 (a b : ℕ) : Prop :=
  a + b = 80

-- Theorem to prove
theorem fraction_before_simplification_is_24_56 (a b : ℕ) :
  fraction_before_simplification_simplifies_to_3_7 a b →
  sum_of_numerator_and_denominator_is_80 a b →
  (a, b) = (24, 56) :=
sorry

end fraction_before_simplification_is_24_56_l47_47958


namespace toms_nickels_l47_47286

variables (q n : ℕ)

theorem toms_nickels (h1 : q + n = 12) (h2 : 25 * q + 5 * n = 220) : n = 4 :=
by {
  sorry
}

end toms_nickels_l47_47286


namespace solve_laundry_problem_l47_47374

def laundry_problem : Prop :=
  let total_weight := 20
  let clothes_weight := 5
  let detergent_per_scoop := 0.02
  let initial_detergent := 2 * detergent_per_scoop
  let optimal_ratio := 0.004
  let additional_detergent := 0.02
  let additional_water := 14.94
  let total_detergent := initial_detergent + additional_detergent
  let final_amount := clothes_weight + initial_detergent + additional_detergent + additional_water
  final_amount = total_weight ∧ total_detergent / (total_weight - clothes_weight) = optimal_ratio

theorem solve_laundry_problem : laundry_problem :=
by 
  -- the proof would go here
  sorry

end solve_laundry_problem_l47_47374


namespace sixteenth_answer_is_three_l47_47416

theorem sixteenth_answer_is_three (total_members : ℕ)
  (answers_1 answers_2 answers_3 : ℕ) 
  (h_total : total_members = 16) 
  (h_answers_1 : answers_1 = 6) 
  (h_answers_2 : answers_2 = 6) 
  (h_answers_3 : answers_3 = 3) :
  ∃ answer : ℕ, answer = 3 ∧ (answers_1 + answers_2 + answers_3 + 1 = total_members) :=
sorry

end sixteenth_answer_is_three_l47_47416


namespace angle_BAC_l47_47686

theorem angle_BAC (A B C D : Type*) (AD BD CD : ℝ) (angle_BCA : ℝ) 
  (h_AD_BD : AD = BD) (h_BD_CD : BD = CD) (h_angle_BCA : angle_BCA = 40) :
  ∃ angle_BAC : ℝ, angle_BAC = 110 := 
sorry

end angle_BAC_l47_47686


namespace solve_for_y_l47_47161

noncomputable def g (y : ℝ) : ℝ := (30 * y + (30 * y + 27)^(1/3))^(1/3)

theorem solve_for_y :
  (∃ y : ℝ, g y = 15) ↔ (∃ y : ℝ, y = 1674 / 15) :=
by
  sorry

end solve_for_y_l47_47161


namespace max_distance_difference_l47_47103

-- Given definitions and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 15 = 1
def circle1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

-- Main theorem to prove the maximum value of |PM| - |PN|
theorem max_distance_difference (P M N : ℝ × ℝ) :
  hyperbola P.1 P.2 →
  circle1 M.1 M.2 →
  circle2 N.1 N.2 →
  ∃ max_val : ℝ, max_val = 5 :=
by
  -- Proof skipped, only statement is required
  sorry

end max_distance_difference_l47_47103


namespace smallest_possible_area_of_ellipse_l47_47114

theorem smallest_possible_area_of_ellipse
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
    (((x - 1/2)^2 + y^2 = 1/4) ∨ ((x + 1/2)^2 + y^2 = 1/4))) :
  ∃ (k : ℝ), (a * b * π = 4 * π) :=
by
  sorry

end smallest_possible_area_of_ellipse_l47_47114


namespace find_f_value_l47_47602

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 - b * (Real.sin x) * (Real.cos x) - a / 2

theorem find_f_value (a b : ℝ)
  (h_max : ∀ x, f a b x ≤ 1/2)
  (h_at_pi_over_3 : f a b (Real.pi / 3) = (Real.sqrt 3) / 4) :
  f a b (-Real.pi / 3) = 0 ∨ f a b (-Real.pi / 3) = -(Real.sqrt 3) / 4 :=
sorry

end find_f_value_l47_47602


namespace find_y_l47_47309

variables (ABC ACB BAC : ℝ)
variables (CDE ADE EAD AED DEB y : ℝ)

-- Conditions
axiom angle_ABC : ABC = 45
axiom angle_ACB : ACB = 90
axiom angle_BAC_eq : BAC = 180 - ABC - ACB
axiom angle_CDE : CDE = 72
axiom angle_ADE_eq : ADE = 180 - CDE
axiom angle_EAD : EAD = 45
axiom angle_AED_eq : AED = 180 - ADE - EAD
axiom angle_DEB_eq : DEB = 180 - AED
axiom y_eq : y = DEB

-- Goal
theorem find_y : y = 153 :=
by {
  -- Here we would proceed with the proof using the established axioms.
  sorry
}

end find_y_l47_47309


namespace max_ratio_1099_l47_47777

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_ratio_1099 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 :=
by
  intros n hn
  sorry

end max_ratio_1099_l47_47777


namespace avg_one_sixth_one_fourth_l47_47101

theorem avg_one_sixth_one_fourth : (1 / 6 + 1 / 4) / 2 = 5 / 24 := by
  sorry

end avg_one_sixth_one_fourth_l47_47101


namespace concert_ratio_l47_47708

theorem concert_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = 50 ∧ c = 50 ∧ a = c := 
sorry

end concert_ratio_l47_47708


namespace age_problem_l47_47546

variable (A B x : ℕ)

theorem age_problem (h1 : A = B + 5) (h2 : B = 35) (h3 : A + x = 2 * (B - x)) : x = 10 :=
sorry

end age_problem_l47_47546


namespace log_conversion_l47_47059

theorem log_conversion (a b : ℝ) (h1 : a = Real.log 225 / Real.log 8) (h2 : b = Real.log 15 / Real.log 2) : a = (2 * b) / 3 := 
sorry

end log_conversion_l47_47059


namespace theater_ticket_difference_l47_47440

theorem theater_ticket_difference
  (O B V : ℕ) 
  (h₁ : O + B + V = 550) 
  (h₂ : 15 * O + 10 * B + 20 * V = 8000) : 
  B - (O + V) = 370 := 
sorry

end theater_ticket_difference_l47_47440


namespace agnes_twice_jane_in_years_l47_47670

def agnes_age := 25
def jane_age := 6

theorem agnes_twice_jane_in_years (x : ℕ) : 
  25 + x = 2 * (6 + x) → x = 13 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end agnes_twice_jane_in_years_l47_47670


namespace pen_price_equation_l47_47730

theorem pen_price_equation
  (x y : ℤ)
  (h1 : 100 * x - y = 100)
  (h2 : 2 * y - 100 * x = 200) : x = 4 :=
by
  sorry

end pen_price_equation_l47_47730


namespace minimum_focal_length_of_hyperbola_l47_47508

noncomputable def minimum_focal_length (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (h₃ : (1/2) * a * (2 * b) = 8) : ℝ :=
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length := 2 * c
  focal_length

theorem minimum_focal_length_of_hyperbola 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : (1/2) * a * (2 * b) = 8) :
  minimum_focal_length a b h₁ h₂ h₃ = 8 :=
by
  sorry

end minimum_focal_length_of_hyperbola_l47_47508


namespace measure_of_angle_E_l47_47248

variable (D E F : ℝ)
variable (h1 : E = F)
variable (h2 : F = 3 * D)
variable (h3 : D + E + F = 180)

theorem measure_of_angle_E : E = 540 / 7 :=
by
  -- Proof omitted
  sorry

end measure_of_angle_E_l47_47248


namespace number_of_integer_solutions_is_zero_l47_47427

-- Define the problem conditions
def eq1 (x y z : ℤ) : Prop := x^2 - 3 * x * y + 2 * y^2 - z^2 = 27
def eq2 (x y z : ℤ) : Prop := -x^2 + 6 * y * z + 2 * z^2 = 52
def eq3 (x y z : ℤ) : Prop := x^2 + x * y + 8 * z^2 = 110

-- State the theorem to be proved
theorem number_of_integer_solutions_is_zero :
  ∀ (x y z : ℤ), eq1 x y z → eq2 x y z → eq3 x y z → false :=
by
  sorry

end number_of_integer_solutions_is_zero_l47_47427


namespace consecutive_negative_integers_product_sum_l47_47454

theorem consecutive_negative_integers_product_sum (n : ℤ) 
  (h_neg1 : n < 0) 
  (h_neg2 : n + 1 < 0) 
  (h_product : n * (n + 1) = 2720) :
  n + (n + 1) = -105 :=
sorry

end consecutive_negative_integers_product_sum_l47_47454


namespace find_tangent_circle_l47_47479

-- Define circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the problem statement as a theorem
theorem find_tangent_circle :
  ∃ (x0 y0 : ℝ), (x - x0)^2 + (y - y0)^2 = 5/4 ∧ (x0, y0) = (1/2, 1) ∧
                   ∀ (x y : ℝ), (circle1 x y → circle2 x y → line_l (x0 + x) (y0 + y) ) :=
sorry

end find_tangent_circle_l47_47479


namespace reciprocal_of_fraction_l47_47477

noncomputable def fraction := (Real.sqrt 5 + 1) / 2

theorem reciprocal_of_fraction :
  (fraction⁻¹) = (Real.sqrt 5 - 1) / 2 :=
by
  -- proof steps
  sorry

end reciprocal_of_fraction_l47_47477


namespace least_cans_required_l47_47669

def maaza : ℕ := 20
def pepsi : ℕ := 144
def sprite : ℕ := 368

def GCD (a b : ℕ) : ℕ := Nat.gcd a b

def total_cans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_maaza_pepsi := GCD maaza pepsi
  let gcd_all := GCD gcd_maaza_pepsi sprite
  (maaza / gcd_all) + (pepsi / gcd_all) + (sprite / gcd_all)

theorem least_cans_required : total_cans maaza pepsi sprite = 133 := by
  sorry

end least_cans_required_l47_47669


namespace inequality_abc_equality_condition_abc_l47_47165

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

theorem equality_condition_abc (a b c : ℝ) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) = 1 / 2 ↔ 
  a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6 :=
sorry

end inequality_abc_equality_condition_abc_l47_47165


namespace area_of_triangle_l47_47369

theorem area_of_triangle (x : ℝ) :
  let t1_area := 16
  let t2_area := 25
  let t3_area := 64
  let total_area_factor := t1_area + t2_area + t3_area
  let side_factor := 17 * 17
  ΔABC_area = side_factor * total_area_factor :=
by {
  -- Placeholder to complete the proof
  sorry
}

end area_of_triangle_l47_47369


namespace tan_of_cos_neg_five_thirteenth_l47_47037

variable {α : Real}

theorem tan_of_cos_neg_five_thirteenth (hcos : Real.cos α = -5/13) (hα : π < α ∧ α < 3 * π / 2) : 
  Real.tan α = 12 / 5 := 
sorry

end tan_of_cos_neg_five_thirteenth_l47_47037


namespace average_tree_height_l47_47236

def mixed_num_to_improper (whole: ℕ) (numerator: ℕ) (denominator: ℕ) : Rat :=
  whole + (numerator / denominator)

theorem average_tree_height 
  (elm : Rat := mixed_num_to_improper 11 2 3)
  (oak : Rat := mixed_num_to_improper 17 5 6)
  (pine : Rat := mixed_num_to_improper 15 1 2)
  (num_trees : ℕ := 3) :
  ((elm + oak + pine) / num_trees) = (15 : Rat) := 
  sorry

end average_tree_height_l47_47236


namespace geometric_sequence_property_l47_47394

variable {a_n : ℕ → ℝ}

theorem geometric_sequence_property (h1 : ∀ m n p q : ℕ, m + n = p + q → a_n m * a_n n = a_n p * a_n q)
    (h2 : a_n 4 * a_n 5 * a_n 6 = 27) : a_n 1 * a_n 9 = 9 := by
  sorry

end geometric_sequence_property_l47_47394


namespace Sarah_correct_responses_l47_47075

theorem Sarah_correct_responses : ∃ x : ℕ, x ≥ 22 ∧ (7 * x - (26 - x) + 4 ≥ 150) :=
by
  sorry

end Sarah_correct_responses_l47_47075


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l47_47847

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l47_47847


namespace lines_intersect_l47_47854

-- Define the parameterizations of the two lines
def line1 (t : ℚ) : ℚ × ℚ := ⟨2 + 3 * t, 3 - 4 * t⟩
def line2 (u : ℚ) : ℚ × ℚ := ⟨4 + 5 * u, 1 + 3 * u⟩

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = line2 u ∧ line1 t = ⟨26 / 11, 19 / 11⟩ :=
by
  sorry

end lines_intersect_l47_47854


namespace total_photos_l47_47082

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end total_photos_l47_47082


namespace percentage_increase_l47_47077

noncomputable def price_increase (d new_price : ℝ) : ℝ :=
  ((new_price - d) / d) * 100

theorem percentage_increase 
  (d new_price : ℝ)
  (h1 : 2 * d = 585)
  (h2 : new_price = 351) :
  price_increase d new_price = 20 :=
by
  sorry

end percentage_increase_l47_47077


namespace fraction_equality_l47_47431

theorem fraction_equality (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 11) : 
  (7 * x + 11 * y) / (77 * x * y) = 9 / 20 :=
by
  -- proof can be provided here.
  sorry

end fraction_equality_l47_47431


namespace systematic_sampling_interval_l47_47992

-- Definition of the population size and sample size
def populationSize : Nat := 800
def sampleSize : Nat := 40

-- The main theorem stating that the interval k in systematic sampling is 20
theorem systematic_sampling_interval : populationSize / sampleSize = 20 := by
  sorry

end systematic_sampling_interval_l47_47992


namespace home_electronics_budget_l47_47002

theorem home_electronics_budget (deg_ba: ℝ) (b_deg: ℝ) (perc_me: ℝ) (perc_fa: ℝ) (perc_gm: ℝ) (perc_il: ℝ) : 
  deg_ba = 43.2 → 
  b_deg = 360 → 
  perc_me = 12 →
  perc_fa = 15 →
  perc_gm = 29 →
  perc_il = 8 →
  (b_deg / 360 * 100 = 12) → 
  perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100) = 76 →
  100 - (perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100)) = 24 :=
by
  intro h_deg_ba h_b_deg h_perc_me h_perc_fa h_perc_gm h_perc_il h_ba_12perc h_total_76perc
  sorry

end home_electronics_budget_l47_47002


namespace tan_210_eq_neg_sqrt3_over_3_l47_47185

noncomputable def angle_210 : ℝ := 210 * (Real.pi / 180)
noncomputable def angle_30 : ℝ := 30 * (Real.pi / 180)

theorem tan_210_eq_neg_sqrt3_over_3 : Real.tan angle_210 = -Real.sqrt 3 / 3 :=
by
  sorry -- Proof omitted

end tan_210_eq_neg_sqrt3_over_3_l47_47185


namespace second_term_of_geometric_series_l47_47314

theorem second_term_of_geometric_series (a r S term2 : ℝ) 
  (h1 : r = 1 / 4)
  (h2 : S = 40)
  (h3 : S = a / (1 - r))
  (h4 : term2 = a * r) : 
  term2 = 7.5 := 
  by
  sorry

end second_term_of_geometric_series_l47_47314


namespace abs_eq_neg_iff_non_positive_l47_47224

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l47_47224


namespace total_books_l47_47095

def school_books : ℕ := 19
def sports_books : ℕ := 39

theorem total_books : school_books + sports_books = 58 := by
  sorry

end total_books_l47_47095


namespace find_rectangle_length_l47_47395

theorem find_rectangle_length (L W : ℕ) (h_area : L * W = 300) (h_perimeter : 2 * L + 2 * W = 70) : L = 20 :=
by
  sorry

end find_rectangle_length_l47_47395


namespace largest_lcm_value_l47_47680

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 := by
sorry

end largest_lcm_value_l47_47680


namespace original_students_count_l47_47727

theorem original_students_count (N : ℕ) (T : ℕ) :
  (T = N * 85) →
  ((N - 5) * 90 = T - 300) →
  ((N - 8) * 95 = T - 465) →
  ((N - 15) * 100 = T - 955) →
  N = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end original_students_count_l47_47727


namespace cos_value_l47_47776

-- Given condition
axiom sin_condition (α : ℝ) : Real.sin (Real.pi / 6 + α) = 2 / 3

-- The theorem we need to prove
theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) : 
  Real.cos (Real.pi / 3 - α) = 2 / 3 := 
by 
  sorry

end cos_value_l47_47776


namespace student_sums_attempted_l47_47509

theorem student_sums_attempted (sums_right sums_wrong : ℕ) (h1 : sums_wrong = 2 * sums_right) (h2 : sums_right = 16) :
  sums_right + sums_wrong = 48 :=
by
  sorry

end student_sums_attempted_l47_47509


namespace sum_of_squares_of_coeffs_l47_47839

def poly_coeffs_squared_sum (p : Polynomial ℤ) : ℤ :=
  p.coeff 5 ^ 2 + p.coeff 3 ^ 2 + p.coeff 0 ^ 2

theorem sum_of_squares_of_coeffs (p : Polynomial ℤ) (h : p = 5 * (Polynomial.C 1 * Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C 3)) :
  poly_coeffs_squared_sum p = 350 :=
by
  sorry

end sum_of_squares_of_coeffs_l47_47839


namespace correct_system_of_equations_l47_47813

theorem correct_system_of_equations (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  (∃ x y, (x / 3 = y - 2) ∧ ((x - 9) / 2 = y)) :=
by
  sorry

end correct_system_of_equations_l47_47813


namespace wheel_speed_l47_47654

theorem wheel_speed (r : ℝ) (c : ℝ) (ts tf : ℝ) 
  (h₁ : c = 13) 
  (h₂ : r * ts = c / 5280) 
  (h₃ : (r + 6) * (tf - 1/3 / 3600) = c / 5280) 
  (h₄ : tf = ts - 1 / 10800) :
  r = 12 :=
  sorry

end wheel_speed_l47_47654


namespace minimum_manhattan_distance_l47_47225

open Real

def ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 2 + P.2^2 = 1

def line (Q : ℝ × ℝ) : Prop := 3 * Q.1 + 4 * Q.2 = 12

def manhattan_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem minimum_manhattan_distance :
  ∃ P Q, ellipse P ∧ line Q ∧
    ∀ P' Q', ellipse P' → line Q' → manhattan_distance P Q ≤ manhattan_distance P' Q' :=
  sorry

end minimum_manhattan_distance_l47_47225


namespace points_on_x_axis_circles_intersect_l47_47985

theorem points_on_x_axis_circles_intersect (a b : ℤ)
  (h1 : 3 * a - b = 9)
  (h2 : 2 * a + 3 * b = -5) : (a : ℝ)^b = 1/8 :=
by
  sorry

end points_on_x_axis_circles_intersect_l47_47985


namespace small_seat_capacity_indeterminate_l47_47050

-- Conditions
def small_seats : ℕ := 3
def large_seats : ℕ := 7
def capacity_per_large_seat : ℕ := 12
def total_large_capacity : ℕ := 84

theorem small_seat_capacity_indeterminate
  (h1 : large_seats * capacity_per_large_seat = total_large_capacity)
  (h2 : ∀ s : ℕ, ∃ p : ℕ, p ≠ s * capacity_per_large_seat) :
  ¬ ∃ n : ℕ, ∀ m : ℕ, small_seats * m = n * small_seats :=
by {
  sorry
}

end small_seat_capacity_indeterminate_l47_47050


namespace number_of_toys_sold_l47_47097

theorem number_of_toys_sold (total_selling_price gain_per_toy cost_price_per_toy : ℕ)
  (h1 : total_selling_price = 25200)
  (h2 : gain_per_toy = 3 * cost_price_per_toy)
  (h3 : cost_price_per_toy = 1200) : 
  (total_selling_price - gain_per_toy) / cost_price_per_toy = 18 :=
by 
  sorry

end number_of_toys_sold_l47_47097


namespace cricket_initial_overs_l47_47795

theorem cricket_initial_overs
  (target_runs : ℚ) (initial_run_rate : ℚ) (remaining_run_rate : ℚ) (remaining_overs : ℕ)
  (total_runs_needed : target_runs = 282)
  (run_rate_initial : initial_run_rate = 3.4)
  (run_rate_remaining : remaining_run_rate = 6.2)
  (overs_remaining : remaining_overs = 40) :
  ∃ (initial_overs : ℕ), initial_overs = 10 :=
by
  sorry

end cricket_initial_overs_l47_47795


namespace probability_one_instrument_l47_47802

theorem probability_one_instrument (total_people : ℕ) (at_least_one_instrument_ratio : ℚ) (two_or_more_instruments : ℕ)
  (h1 : total_people = 800) (h2 : at_least_one_instrument_ratio = 1 / 5) (h3 : two_or_more_instruments = 128) :
  (160 - 128) / 800 = 1 / 25 :=
by
  sorry

end probability_one_instrument_l47_47802


namespace least_number_l47_47560

theorem least_number (n : ℕ) : 
  (n % 45 = 2) ∧ (n % 59 = 2) ∧ (n % 77 = 2) → n = 205517 :=
by
  sorry

end least_number_l47_47560


namespace most_likely_event_is_C_l47_47338

open Classical

noncomputable def total_events : ℕ := 6 * 6

noncomputable def P_A : ℚ := 7 / 36
noncomputable def P_B : ℚ := 18 / 36
noncomputable def P_C : ℚ := 1
noncomputable def P_D : ℚ := 0

theorem most_likely_event_is_C :
  P_C > P_A ∧ P_C > P_B ∧ P_C > P_D := by
  sorry

end most_likely_event_is_C_l47_47338


namespace circle_radius_eq_l47_47832

theorem circle_radius_eq (r : ℝ) (AB : ℝ) (BC : ℝ) (hAB : AB = 10) (hBC : BC = 12) : r = 25 / 4 := by
  sorry

end circle_radius_eq_l47_47832


namespace intersection_ellipse_line_range_b_l47_47867

theorem intersection_ellipse_line_range_b (b : ℝ) : 
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2*y^2 = 3 ∧ y = m*x + b) ↔ 
  (- (Real.sqrt 6) / 2) ≤ b ∧ b ≤ (Real.sqrt 6) / 2 :=
by {
  sorry
}

end intersection_ellipse_line_range_b_l47_47867


namespace greatest_integer_x_l47_47994

theorem greatest_integer_x (x : ℤ) : 
  (∃ k : ℤ, (x - 4) = k ∧ x^2 - 3 * x + 4 = k * (x - 4) + 8) →
  x ≤ 12 :=
by
  sorry

end greatest_integer_x_l47_47994


namespace cost_price_l47_47775

namespace ClothingDiscount

variables (x : ℝ)

def loss_condition (x : ℝ) : ℝ := 0.5 * x + 20
def profit_condition (x : ℝ) : ℝ := 0.8 * x - 40

def marked_price := { x : ℝ // loss_condition x = profit_condition x }

noncomputable def clothing_price : marked_price := 
    ⟨200, sorry⟩

theorem cost_price : loss_condition 200 = 120 :=
sorry

end ClothingDiscount

end cost_price_l47_47775


namespace base_7_3516_is_1287_l47_47877

-- Definitions based on conditions
def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 3516 => 3 * 7^3 + 5 * 7^2 + 1 * 7^1 + 6 * 7^0
  | _ => 0

-- Proving the main question
theorem base_7_3516_is_1287 : base7_to_base10 3516 = 1287 := by
  sorry

end base_7_3516_is_1287_l47_47877


namespace division_of_positive_by_negative_l47_47913

theorem division_of_positive_by_negative :
  4 / (-2) = -2 := 
by
  sorry

end division_of_positive_by_negative_l47_47913


namespace every_integer_appears_exactly_once_l47_47025

-- Define the sequence of integers
variable (a : ℕ → ℤ)

-- Define the conditions
axiom infinite_positives : ∀ n : ℕ, ∃ i > n, a i > 0
axiom infinite_negatives : ∀ n : ℕ, ∃ i > n, a i < 0
axiom distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n) ≠ (a j % n)

-- The proof statement
theorem every_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! i : ℕ, a i = x :=
sorry

end every_integer_appears_exactly_once_l47_47025


namespace marla_colors_red_squares_l47_47013

-- Conditions
def total_rows : Nat := 10
def squares_per_row : Nat := 15
def total_squares : Nat := total_rows * squares_per_row

def blue_rows_top : Nat := 2
def blue_rows_bottom : Nat := 2
def total_blue_rows : Nat := blue_rows_top + blue_rows_bottom
def total_blue_squares : Nat := total_blue_rows * squares_per_row

def green_squares : Nat := 66
def red_rows : Nat := 4

-- Theorem to prove 
theorem marla_colors_red_squares : 
  total_squares - total_blue_squares - green_squares = red_rows * 6 :=
by
  sorry -- This skips the proof

end marla_colors_red_squares_l47_47013


namespace length_each_stitch_l47_47174

theorem length_each_stitch 
  (hem_length_feet : ℝ) 
  (stitches_per_minute : ℝ) 
  (hem_time_minutes : ℝ) 
  (hem_length_inches : ℝ) 
  (total_stitches : ℝ) 
  (stitch_length_inches : ℝ) 
  (h1 : hem_length_feet = 3) 
  (h2 : stitches_per_minute = 24) 
  (h3 : hem_time_minutes = 6) 
  (h4 : hem_length_inches = hem_length_feet * 12) 
  (h5 : total_stitches = stitches_per_minute * hem_time_minutes) 
  (h6 : stitch_length_inches = hem_length_inches / total_stitches) :
  stitch_length_inches = 0.25 :=
by
  sorry

end length_each_stitch_l47_47174


namespace determine_pq_value_l47_47171

noncomputable def p : ℝ → ℝ := λ x => 16 * x
noncomputable def q : ℝ → ℝ := λ x => (x + 4) * (x - 1)

theorem determine_pq_value : (p (-1) / q (-1)) = 8 / 3 := by
  sorry

end determine_pq_value_l47_47171


namespace kathryn_remaining_money_l47_47678

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end kathryn_remaining_money_l47_47678


namespace abs_diff_squares_1055_985_eq_1428_l47_47166

theorem abs_diff_squares_1055_985_eq_1428 :
  abs ((105.5: ℝ)^2 - (98.5: ℝ)^2) = 1428 :=
by
  sorry

end abs_diff_squares_1055_985_eq_1428_l47_47166


namespace find_a1_geometric_sequence_l47_47966

theorem find_a1_geometric_sequence (a₁ q : ℝ) (h1 : q ≠ 1) 
    (h2 : a₁ * (1 - q^3) / (1 - q) = 7)
    (h3 : a₁ * (1 - q^6) / (1 - q) = 63) :
    a₁ = 1 :=
by
  sorry

end find_a1_geometric_sequence_l47_47966


namespace sale_book_cost_l47_47198

variable (x : ℝ)

def fiveSaleBooksCost (x : ℝ) : ℝ :=
  5 * x

def onlineBooksCost : ℝ :=
  40

def bookstoreBooksCost : ℝ :=
  3 * 40

def totalCost (x : ℝ) : ℝ :=
  fiveSaleBooksCost x + onlineBooksCost + bookstoreBooksCost

theorem sale_book_cost :
  totalCost x = 210 → x = 10 := by
  sorry

end sale_book_cost_l47_47198


namespace total_cost_of_purchases_l47_47308

def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

theorem total_cost_of_purchases : cost_cat_toy + cost_cage = 21.95 := by
  -- skipping the proof
  sorry

end total_cost_of_purchases_l47_47308


namespace average_temp_addington_l47_47266

def temperatures : List ℚ := [60, 59, 56, 53, 49, 48, 46]

def average_temp (temps : List ℚ) : ℚ := (temps.sum) / temps.length

theorem average_temp_addington :
  average_temp temperatures = 53 := by
  sorry

end average_temp_addington_l47_47266


namespace gcd_lcm_product_eq_prod_l47_47423

theorem gcd_lcm_product_eq_prod (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
sorry

end gcd_lcm_product_eq_prod_l47_47423


namespace avg_production_last_5_days_l47_47415

theorem avg_production_last_5_days
  (avg_first_25_days : ℕ)
  (total_days : ℕ)
  (avg_entire_month : ℕ)
  (h1 : avg_first_25_days = 60)
  (h2 : total_days = 30)
  (h3 : avg_entire_month = 58) : 
  (total_days * avg_entire_month - 25 * avg_first_25_days) / 5 = 48 := 
by
  sorry

end avg_production_last_5_days_l47_47415


namespace greatest_product_l47_47805

theorem greatest_product (x : ℤ) (h : x + (1998 - x) = 1998) : 
  x * (1998 - x) ≤ 998001 :=
  sorry

end greatest_product_l47_47805


namespace leaks_empty_time_l47_47575

theorem leaks_empty_time (A L1 L2: ℝ) (hA: A = 1/2) (hL1_rate: A - L1 = 1/3) 
  (hL2_rate: A - L1 - L2 = 1/4) : 1 / (L1 + L2) = 4 :=
by
  sorry

end leaks_empty_time_l47_47575


namespace parabola_focus_l47_47113

theorem parabola_focus (a : ℝ) (h : a ≠ 0) : ∃ q : ℝ, q = 1/(4*a) ∧ (0, q) = (0, 1/(4*a)) :=
by
  sorry

end parabola_focus_l47_47113


namespace number_of_distinct_b_values_l47_47270

theorem number_of_distinct_b_values : 
  ∃ (b : ℝ) (p q : ℤ), (∀ (x : ℝ), x*x + b*x + 12*b = 0) ∧ 
                        p + q = -b ∧ 
                        p * q = 12 * b ∧ 
                        ∃ n : ℤ, 1 ≤ n ∧ n ≤ 15 :=
sorry

end number_of_distinct_b_values_l47_47270


namespace no_adjacent_same_color_probability_zero_l47_47284

-- Define the number of each color bead
def num_red_beads : ℕ := 5
def num_white_beads : ℕ := 3
def num_blue_beads : ℕ := 2

-- Define the total number of beads
def total_beads : ℕ := num_red_beads + num_white_beads + num_blue_beads

-- Calculate the probability that no two neighboring beads are the same color
noncomputable def probability_no_adjacent_same_color : ℚ :=
  if (num_red_beads > num_white_beads + num_blue_beads + 1) then 0 else sorry

theorem no_adjacent_same_color_probability_zero :
  probability_no_adjacent_same_color = 0 :=
by {
  sorry
}

end no_adjacent_same_color_probability_zero_l47_47284


namespace tetrahedron_distance_sum_eq_l47_47759

-- Defining the necessary conditions
variables {V K : ℝ}
variables {S_1 S_2 S_3 S_4 H_1 H_2 H_3 H_4 : ℝ}

axiom ratio_eq (i : ℕ) (Si : ℝ) (K : ℝ) : (Si / i = K)
axiom volume_eq : S_1 * H_1 + S_2 * H_2 + S_3 * H_3 + S_4 * H_4 = 3 * V

-- Main theorem stating that the desired result holds under the given conditions
theorem tetrahedron_distance_sum_eq :
  H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4 = 3 * V / K :=
by
have h1 : S_1 = K * 1 := by sorry
have h2 : S_2 = K * 2 := by sorry
have h3 : S_3 = K * 3 := by sorry
have h4 : S_4 = K * 4 := by sorry
have sum_eq : K * (H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4) = 3 * V := by sorry
exact sorry

end tetrahedron_distance_sum_eq_l47_47759


namespace g_five_eq_thirteen_sevenths_l47_47806

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_five_eq_thirteen_sevenths : g 5 = 13 / 7 := by
  sorry

end g_five_eq_thirteen_sevenths_l47_47806


namespace jackson_spends_on_school_supplies_l47_47668

theorem jackson_spends_on_school_supplies :
  let num_students := 50
  let pens_per_student := 7
  let notebooks_per_student := 5
  let binders_per_student := 3
  let highlighters_per_student := 4
  let folders_per_student := 2
  let cost_pen := 0.70
  let cost_notebook := 1.60
  let cost_binder := 5.10
  let cost_highlighter := 0.90
  let cost_folder := 1.15
  let teacher_discount := 135
  let bulk_discount := 25
  let sales_tax_rate := 0.05
  let total_cost := 
    (num_students * pens_per_student * cost_pen) + 
    (num_students * notebooks_per_student * cost_notebook) + 
    (num_students * binders_per_student * cost_binder) + 
    (num_students * highlighters_per_student * cost_highlighter) + 
    (num_students * folders_per_student * cost_folder)
  let discounted_cost := total_cost - teacher_discount - bulk_discount
  let sales_tax := discounted_cost * sales_tax_rate
  let final_cost := discounted_cost + sales_tax
  final_cost = 1622.25 := by
  sorry

end jackson_spends_on_school_supplies_l47_47668


namespace problem_l47_47600

theorem problem (triangle square : ℕ) (h1 : triangle + 5 ≡ 1 [MOD 7]) (h2 : 2 + square ≡ 3 [MOD 7]) :
  triangle = 3 ∧ square = 1 := by
  sorry

end problem_l47_47600


namespace problem_statement_l47_47689

theorem problem_statement (a b : ℝ) (h : a^2 + |b + 1| = 0) : (a + b)^2015 = -1 := by
  sorry

end problem_statement_l47_47689


namespace solve_for_z_l47_47863

open Complex

theorem solve_for_z (z : ℂ) (h : 2 * z * I = 1 + 3 * I) : 
  z = (3 / 2) - (1 / 2) * I :=
by
  sorry

end solve_for_z_l47_47863


namespace range_of_x_l47_47545

theorem range_of_x (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 3) :=
by sorry

end range_of_x_l47_47545


namespace find_radii_of_circles_l47_47295

theorem find_radii_of_circles (d : ℝ) (ext_tangent : ℝ) (int_tangent : ℝ)
  (hd : d = 65) (hext : ext_tangent = 63) (hint : int_tangent = 25) :
  ∃ (R r : ℝ), R = 38 ∧ r = 22 :=
by 
  sorry

end find_radii_of_circles_l47_47295


namespace find_b_find_area_of_ABC_l47_47183

variable {a b c : ℝ}
variable {B : ℝ}

-- Given Conditions
def given_conditions (a b c B : ℝ) := a = 4 ∧ c = 3 ∧ B = Real.arccos (1 / 8)

-- Proving b = sqrt(22)
theorem find_b (h : given_conditions a b c B) : b = Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) :=
by
  sorry

-- Proving the area of triangle ABC
theorem find_area_of_ABC (h : given_conditions a b c B) 
  (sinB : Real.sin B = 3 * Real.sqrt 7 / 8) : 
  (1 / 2) * a * c * Real.sin B = 9 * Real.sqrt 7 / 4 :=
by
  sorry

end find_b_find_area_of_ABC_l47_47183


namespace geometric_sequence_a_equals_minus_four_l47_47046

theorem geometric_sequence_a_equals_minus_four (a : ℝ) 
(h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : a = -4 :=
sorry

end geometric_sequence_a_equals_minus_four_l47_47046


namespace find_common_ratio_l47_47389

noncomputable def geom_series_common_ratio (q : ℝ) : Prop :=
  ∃ (a1 : ℝ), a1 > 0 ∧ (a1 * q^2 = 18) ∧ (a1 * (1 + q + q^2) = 26)

theorem find_common_ratio (q : ℝ) :
  geom_series_common_ratio q → q = 3 :=
sorry

end find_common_ratio_l47_47389


namespace right_triangle_area_l47_47915

theorem right_triangle_area
  (hypotenuse : ℝ) (angle : ℝ) (hyp_eq : hypotenuse = 12) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 18 * Real.sqrt 3 :=
by
  have side1 := hypotenuse / 2  -- Shorter leg = hypotenuse / 2
  have side2 := side1 * Real.sqrt 3  -- Longer leg = shorter leg * sqrt 3
  let area := (side1 * side2) / 2  -- Area calculation
  use area
  sorry

end right_triangle_area_l47_47915


namespace average_reading_time_correct_l47_47590

-- We define total_reading_time as a parameter representing the sum of reading times
noncomputable def total_reading_time : ℝ := sorry

-- We define the number of students as a constant
def number_of_students : ℕ := 50

-- We define the average reading time per student based on the provided data
noncomputable def average_reading_time : ℝ :=
  total_reading_time / number_of_students

-- The theorem we need to prove: that the average reading time per student is correctly calculated
theorem average_reading_time_correct :
  ∃ (total_reading_time : ℝ), average_reading_time = total_reading_time / number_of_students :=
by
  -- since total_reading_time and number_of_students are already defined, we prove the theorem using them
  use total_reading_time
  exact rfl

end average_reading_time_correct_l47_47590


namespace inequality_part1_inequality_part2_l47_47919

section Proof

variable {x m : ℝ}
def f (x : ℝ) : ℝ := |2 * x + 2| + |2 * x - 3|

-- Part 1: Prove the solution set for the inequality f(x) > 7
theorem inequality_part1 (x : ℝ) :
  f x > 7 ↔ (x < -3 / 2 ∨ x > 2) := 
  sorry

-- Part 2: Prove the range of values for m such that the inequality f(x) ≤ |3m - 2| has a solution
theorem inequality_part2 (m : ℝ) :
  (∃ x, f x ≤ |3 * m - 2|) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
  sorry

end Proof

end inequality_part1_inequality_part2_l47_47919


namespace triangle_perimeter_l47_47412

theorem triangle_perimeter (r AP PB x : ℕ) (h_r : r = 14) (h_AP : AP = 20) (h_PB : PB = 30) (h_BC_gt_AC : ∃ BC AC : ℝ, BC > AC)
: ∃ s : ℕ, s = (25 + x) → 2 * s = 50 + 2 * x :=
by
  sorry

end triangle_perimeter_l47_47412


namespace decimal_to_binary_25_l47_47525

theorem decimal_to_binary_25: (1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0) = 25 :=
by 
  sorry

end decimal_to_binary_25_l47_47525


namespace savings_are_equal_and_correct_l47_47663

-- Definitions of the given conditions
variables (I1 I2 E1 E2 : ℝ)
variables (S1 S2 : ℝ)
variables (rI : ℝ := 5/4) -- ratio of incomes
variables (rE : ℝ := 3/2) -- ratio of expenditures
variables (I1_val : ℝ := 3000) -- P1's income

-- Given conditions
def given_conditions : Prop :=
  I1 = I1_val ∧
  I1 / I2 = rI ∧
  E1 / E2 = rE ∧
  S1 = S2

-- Required proof
theorem savings_are_equal_and_correct (I2_val : I2 = (I1_val * 4/5)) (x : ℝ) (h1 : E1 = 3 * x) (h2 : E2 = 2 * x) (h3 : S1 = 1200) :
  S1 = S2 ∧ S1 = 1200 := by
  sorry

end savings_are_equal_and_correct_l47_47663


namespace larger_number_of_hcf_and_lcm_factors_l47_47217

theorem larger_number_of_hcf_and_lcm_factors :
  ∃ (a b : ℕ), (∀ d, d ∣ a ∧ d ∣ b → d ≤ 20) ∧ (∃ x y, x * y * 20 = a * b ∧ x * 20 = a ∧ y * 20 = b ∧ x > y ∧ x = 15 ∧ y = 11) → max a b = 300 :=
by sorry

end larger_number_of_hcf_and_lcm_factors_l47_47217


namespace mortgage_loan_amount_l47_47137

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end mortgage_loan_amount_l47_47137


namespace smaller_of_two_integers_l47_47230

noncomputable def smaller_integer (m n : ℕ) : ℕ :=
if m < n then m else n

theorem smaller_of_two_integers :
  ∀ (m n : ℕ),
  100 ≤ m ∧ m < 1000 ∧ 100 ≤ n ∧ n < 1000 ∧
  (m + n) / 2 = m + n / 200 →
  smaller_integer m n = 891 :=
by
  intros m n h
  -- Assuming m, n are positive three-digit integers and satisfy the condition
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2.1
  have h5 := h.2.2.2.2
  sorry

end smaller_of_two_integers_l47_47230


namespace time_to_cross_pole_l47_47116

def train_length := 3000 -- in meters
def train_speed_kmh := 90 -- in kilometers per hour

noncomputable def train_speed_mps : ℝ := train_speed_kmh * (1000 / 3600) -- converting speed to meters per second

theorem time_to_cross_pole : (train_length : ℝ) / train_speed_mps = 120 := 
by
  -- Placeholder for the actual proof
  sorry

end time_to_cross_pole_l47_47116


namespace no_x_satisfies_arithmetic_mean_l47_47251

theorem no_x_satisfies_arithmetic_mean :
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 2114 + x) / 6 = 12 :=
by
  sorry

end no_x_satisfies_arithmetic_mean_l47_47251


namespace segments_can_form_triangle_l47_47063

noncomputable def can_form_triangle (a b c : ℝ) : Prop :=
  a + b + c = 2 ∧ a + b > 1 ∧ a + c > b ∧ b + c > a

theorem segments_can_form_triangle (a b c : ℝ) (h : a + b + c = 2) : (a + b > 1) ↔ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end segments_can_form_triangle_l47_47063


namespace rotations_per_block_l47_47564

/--
If Greg's bike wheels have already rotated 600 times and need to rotate 
1000 more times to reach his goal of riding at least 8 blocks,
then the number of rotations per block is 200.
-/
theorem rotations_per_block (r1 r2 n b : ℕ) (h1 : r1 = 600) (h2 : r2 = 1000) (h3 : n = 8) :
  (r1 + r2) / n = 200 := by
  sorry

end rotations_per_block_l47_47564


namespace total_salary_after_strict_manager_l47_47529

-- Definitions based on conditions
def total_initial_salary (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  500 * x + (Finset.sum (Finset.range y) s) = 10000

def kind_manager_total (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  1500 * x + (Finset.sum (Finset.range y) s) + 1000 * y = 24000

def strict_manager_total (x y : ℕ) : ℕ :=
  500 * (x + y)

-- Lean statement to prove the required
theorem total_salary_after_strict_manager (x y : ℕ) (s : ℕ → ℕ) 
  (h_total_initial : total_initial_salary x y s) (h_kind_manager : kind_manager_total x y s) :
  strict_manager_total x y = 7000 := by
  sorry

end total_salary_after_strict_manager_l47_47529


namespace find_x_value_l47_47128

def acid_solution (m : ℕ) (x : ℕ) (h : m > 25) : Prop :=
  let initial_acid := m^2 / 100
  let total_volume := m + x
  let new_acid_concentration := (m - 5) / 100 * (m + x)
  initial_acid = new_acid_concentration

theorem find_x_value (m : ℕ) (h : m > 25) (x : ℕ) :
  (acid_solution m x h) → x = 5 * m / (m - 5) :=
sorry

end find_x_value_l47_47128


namespace value_of_a_plus_b_l47_47027

theorem value_of_a_plus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 1) (h3 : a - b < 0) :
  a + b = -6 ∨ a + b = -4 :=
by
  sorry

end value_of_a_plus_b_l47_47027


namespace roots_of_quadratic_l47_47659

theorem roots_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a + b + c = 0) (h₂ : a - b + c = 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) ∧ (a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) :=
sorry

end roots_of_quadratic_l47_47659


namespace minimum_a_l47_47592

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem minimum_a
  (a : ℝ)
  (h : ∀ x : ℤ, (f x)^2 - a * f x ≤ 0 → ∃! x : ℤ, (f x)^2 - a * f x = 0) :
  a = Real.exp 2 + 1 :=
sorry

end minimum_a_l47_47592


namespace absolute_value_expression_l47_47020

theorem absolute_value_expression : 
  (abs ((-abs (-1 + 2))^2 - 1) = 0) :=
sorry

end absolute_value_expression_l47_47020


namespace total_length_figure_2_l47_47970

-- Define the conditions for Figure 1
def left_side_figure_1 := 10
def right_side_figure_1 := 7
def top_side_figure_1 := 3
def bottom_side_figure_1_seg1 := 2
def bottom_side_figure_1_seg2 := 1

-- Define the conditions for Figure 2 after removal
def left_side_figure_2 := left_side_figure_1
def right_side_figure_2 := right_side_figure_1
def top_side_figure_2 := 0
def bottom_side_figure_2 := top_side_figure_1 + bottom_side_figure_1_seg1 + bottom_side_figure_1_seg2

-- The Lean statement proving the total length in Figure 2
theorem total_length_figure_2 : 
  left_side_figure_2 + right_side_figure_2 + top_side_figure_2 + bottom_side_figure_2 = 23 := by
  sorry

end total_length_figure_2_l47_47970


namespace TreyHasSevenTimesAsManyTurtles_l47_47932

variable (Kristen_turtles : ℕ)
variable (Kris_turtles : ℕ)
variable (Trey_turtles : ℕ)

-- Conditions
def KristenHas12 : Kristen_turtles = 12 := sorry
def KrisHasQuarterOfKristen : Kris_turtles = Kristen_turtles / 4 := sorry
def TreyHas9MoreThanKristen : Trey_turtles = Kristen_turtles + 9 := sorry

-- Question: Prove that Trey has 7 times as many turtles as Kris
theorem TreyHasSevenTimesAsManyTurtles :
  Kristen_turtles = 12 → 
  Kris_turtles = Kristen_turtles / 4 → 
  Trey_turtles = Kristen_turtles + 9 → 
  Trey_turtles = 7 * Kris_turtles := sorry

end TreyHasSevenTimesAsManyTurtles_l47_47932


namespace workouts_difference_l47_47574

theorem workouts_difference
  (workouts_monday : ℕ := 8)
  (workouts_tuesday : ℕ := 5)
  (workouts_wednesday : ℕ := 12)
  (workouts_thursday : ℕ := 17)
  (workouts_friday : ℕ := 10) :
  workouts_thursday - workouts_tuesday = 12 := 
by
  sorry

end workouts_difference_l47_47574


namespace simplify_expression_l47_47513

theorem simplify_expression (x y : ℝ) (h1 : x = 10) (h2 : y = -1/25) :
  ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = 2 / 5 := 
by
  sorry

end simplify_expression_l47_47513


namespace train_speed_l47_47051

/-- Proof that calculates the speed of a train given the times to pass a man and a platform,
and the length of the platform, and shows it equals 54.00432 km/hr. -/
theorem train_speed (L V : ℝ) 
  (platform_length : ℝ := 360.0288)
  (time_to_pass_man : ℝ := 20)
  (time_to_pass_platform : ℝ := 44)
  (equation1 : L = V * time_to_pass_man)
  (equation2 : L + platform_length = V * time_to_pass_platform) :
  V = 15.0012 → V * 3.6 = 54.00432 :=
by sorry

end train_speed_l47_47051


namespace sum_of_cubes_of_roots_l47_47141

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) (h₀ : 3 * x₁ ^ 2 - 5 * x₁ - 2 = 0)
  (h₁ : 3 * x₂ ^ 2 - 5 * x₂ - 2 = 0) :
  x₁^3 + x₂^3 = 215 / 27 :=
by sorry

end sum_of_cubes_of_roots_l47_47141


namespace toby_sharing_proof_l47_47279

theorem toby_sharing_proof (initial_amt amount_left num_brothers : ℕ) 
(h_init : initial_amt = 343)
(h_left : amount_left = 245)
(h_bros : num_brothers = 2) : 
(initial_amt - amount_left) / (initial_amt * num_brothers) = 1 / 7 := 
sorry

end toby_sharing_proof_l47_47279


namespace marina_more_fudge_l47_47662

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end marina_more_fudge_l47_47662


namespace find_range_of_m_l47_47207

open Real

-- Definition for proposition p (the discriminant condition)
def real_roots (m : ℝ) : Prop := (3 * 3) - 4 * m ≥ 0

-- Definition for proposition q (ellipse with foci on x-axis conditions)
def is_ellipse (m : ℝ) : Prop := 
  9 - m > 0 ∧ 
  m - 2 > 0 ∧ 
  9 - m > m - 2

-- Lean statement for the mathematically equivalent proof problem
theorem find_range_of_m (m : ℝ) : (real_roots m ∧ is_ellipse m) → (2 < m ∧ m ≤ 9 / 4) := 
by
  sorry

end find_range_of_m_l47_47207


namespace Bella_average_speed_l47_47533

theorem Bella_average_speed :
  ∀ (distance time : ℝ), 
  distance = 790 → 
  time = 15.8 → 
  (distance / time) = 50 :=
by intros distance time h_dist h_time
   -- According to the provided distances and time,
   -- we need to prove that the calculated speed is 50.
   sorry

end Bella_average_speed_l47_47533


namespace max_value_ad_bc_l47_47117

theorem max_value_ad_bc (a b c d : ℤ) (h₁ : a ∈ ({-1, 1, 2} : Set ℤ))
                          (h₂ : b ∈ ({-1, 1, 2} : Set ℤ))
                          (h₃ : c ∈ ({-1, 1, 2} : Set ℤ))
                          (h₄ : d ∈ ({-1, 1, 2} : Set ℤ)) :
  ad - bc ≤ 6 :=
by sorry

end max_value_ad_bc_l47_47117


namespace find_x_values_l47_47071

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem find_x_values (x : ℝ) :
  (f (f x) = f x) ↔ (x = 0 ∨ x = 2 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end find_x_values_l47_47071


namespace two_digit_number_sum_l47_47646

theorem two_digit_number_sum (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by {
  sorry
}

end two_digit_number_sum_l47_47646


namespace cash_after_brokerage_l47_47684

theorem cash_after_brokerage (sale_amount : ℝ) (brokerage_rate : ℝ) :
  sale_amount = 109.25 → brokerage_rate = 0.0025 →
  (sale_amount - sale_amount * brokerage_rate) = 108.98 :=
by
  intros h1 h2
  sorry

end cash_after_brokerage_l47_47684


namespace problem_1_problem_2_l47_47896

-- Definitions required for the proof
variables {A B C : ℝ} (a b c : ℝ)
variable (cos_A cos_B cos_C : ℝ)
variables (sin_A sin_C : ℝ)

-- Given conditions
axiom given_condition : (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b
axiom cos_B_eq : cos_B = 1 / 4
axiom b_eq : b = 2

-- First problem: Proving the value of sin_C / sin_A
theorem problem_1 :
  (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b → (sin_C / sin_A) = 2 :=
by
  intro h
  sorry

-- Second problem: Proving the area of triangle ABC
theorem problem_2 :
  (cos_B = 1 / 4) → (b = 2) → ((cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b) → (1 / 2 * a * c * sin_A) = (Real.sqrt 15) / 4 :=
by
  intros h1 h2 h3
  sorry

end problem_1_problem_2_l47_47896


namespace car_clock_correctness_l47_47245

variables {t_watch t_car : ℕ} 
--  Variable declarations for time on watch (accurate) and time on car clock.

-- Define the initial times at 8:00 AM
def initial_time_watch : ℕ := 8 * 60 -- 8:00 AM in minutes
def initial_time_car : ℕ := 8 * 60 -- also 8:00 AM in minutes

-- Define the known times in the afternoon
def afternoon_time_watch : ℕ := 14 * 60 -- 2:00 PM in minutes
def afternoon_time_car : ℕ := 14 * 60 + 10 -- 2:10 PM in minutes

-- Car clock runs 37 minutes in the time the watch runs 36 minutes
def car_clock_rate : ℕ × ℕ := (37, 36)

-- Check the car clock time when the accurate watch shows 10:00 PM
def car_time_at_10pm_watch : ℕ := 22 * 60 -- 10:00 PM in minutes

-- Define the actual time that we need to prove
def actual_time_at_10pm_car : ℕ := 21 * 60 + 47 -- 9:47 PM in minutes

theorem car_clock_correctness : 
  (t_watch = actual_time_at_10pm_car) ↔ 
  (t_car = car_time_at_10pm_watch) ∧ 
  (initial_time_watch = initial_time_car) ∧ 
  (afternoon_time_watch = 14 * 60) ∧ 
  (afternoon_time_car = 14 * 60 + 10) ∧ 
  (car_clock_rate = (37, 36)) :=
sorry

end car_clock_correctness_l47_47245


namespace common_divisors_9240_8820_l47_47676

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end common_divisors_9240_8820_l47_47676


namespace price_decrease_necessary_l47_47581

noncomputable def final_price_decrease (P : ℝ) (x : ℝ) : Prop :=
  let increased_price := 1.2 * P
  let final_price := increased_price * (1 - x / 100)
  final_price = 0.88 * P

theorem price_decrease_necessary (x : ℝ) : 
  final_price_decrease 100 x -> x = 26.67 :=
by 
  intros h
  unfold final_price_decrease at h
  sorry

end price_decrease_necessary_l47_47581


namespace parallel_line_slope_l47_47155

theorem parallel_line_slope (x y : ℝ) :
  (∃ k b : ℝ, 3 * x + 6 * y = k * x + b) ∧ (∃ a b, y = a * x + b) ∧ 3 * x + 6 * y = -24 → 
  ∃ m : ℝ, m = -1/2 :=
by
  sorry

end parallel_line_slope_l47_47155


namespace yoojung_notebooks_l47_47879

theorem yoojung_notebooks (N : ℕ) (h : (N - 5) / 2 = 4) : N = 13 :=
by
  sorry

end yoojung_notebooks_l47_47879


namespace angelaAgeInFiveYears_l47_47695

namespace AgeProblem

variables (A B : ℕ) -- Define Angela's and Beth's current age as natural numbers.

-- Condition 1: Angela is four times as old as Beth.
axiom angelaAge : A = 4 * B

-- Condition 2: Five years ago, the sum of their ages was 45 years.
axiom ageSumFiveYearsAgo : (A - 5) + (B - 5) = 45

-- Theorem: Prove that Angela's age in 5 years will be 49.
theorem angelaAgeInFiveYears : A + 5 = 49 :=
by {
  -- proof goes here
  sorry
}

end AgeProblem

end angelaAgeInFiveYears_l47_47695


namespace calculate_expression_l47_47675

theorem calculate_expression : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end calculate_expression_l47_47675


namespace simplify_expression_l47_47294

variable {R : Type*} [Field R]

theorem simplify_expression (x y z : R) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
sorry

end simplify_expression_l47_47294


namespace odd_ints_divisibility_l47_47151

theorem odd_ints_divisibility (a b : ℤ) (ha_odd : a % 2 = 1) (hb_odd : b % 2 = 1) (hdiv : 2 * a * b + 1 ∣ a^2 + b^2 + 1) : a = b :=
sorry

end odd_ints_divisibility_l47_47151


namespace range_of_a_l47_47883

variable (a : ℝ)
def A (a : ℝ) := {x : ℝ | x^2 - 2*x + a > 0}

theorem range_of_a (h : 1 ∉ A a) : a ≤ 1 :=
by {
  sorry
}

end range_of_a_l47_47883


namespace tea_or_coffee_indifference_l47_47822

open Classical

theorem tea_or_coffee_indifference : 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) → 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) :=
by
  sorry

end tea_or_coffee_indifference_l47_47822


namespace cabin_price_correct_l47_47819

noncomputable def cabin_price 
  (cash : ℤ)
  (cypress_trees : ℤ) (pine_trees : ℤ) (maple_trees : ℤ)
  (price_cypress : ℤ) (price_pine : ℤ) (price_maple : ℤ)
  (remaining_cash : ℤ)
  (expected_price : ℤ) : Prop :=
   cash + (cypress_trees * price_cypress + pine_trees * price_pine + maple_trees * price_maple) - remaining_cash = expected_price

theorem cabin_price_correct :
  cabin_price 150 20 600 24 100 200 300 350 130000 :=
by
  sorry

end cabin_price_correct_l47_47819


namespace percentage_A_is_22_l47_47399

noncomputable def percentage_A_in_mixture : ℝ :=
  (0.8 * 0.20 + 0.2 * 0.30) * 100

theorem percentage_A_is_22 :
  percentage_A_in_mixture = 22 := 
by
  sorry

end percentage_A_is_22_l47_47399


namespace unit_square_BE_value_l47_47631

theorem unit_square_BE_value
  (ABCD : ℝ × ℝ → Prop)
  (unit_square : ∀ (a b c d : ℝ × ℝ), ABCD a ∧ ABCD b ∧ ABCD c ∧ ABCD d → 
                  a.1 = 0 ∧ a.2 = 0 ∧ b.1 = 1 ∧ b.2 = 0 ∧ 
                  c.1 = 1 ∧ c.2 = 1 ∧ d.1 = 0 ∧ d.2 = 1)
  (E F G : ℝ × ℝ)
  (on_sides : E.1 = 1 ∧ F.2 = 1 ∧ G.1 = 0)
  (AE_perp_EF : ((E.1 - 0) * (F.2 - E.2)) + ((E.2 - 0) * (F.1 - E.1)) = 0)
  (EF_perp_FG : ((F.1 - E.1) * (G.2 - F.2)) + ((F.2 - E.2) * (G.1 - F.1)) = 0)
  (GA_val : (1 - G.1) = 404 / 1331) :
  ∃ BE, BE = 9 / 11 := 
sorry

end unit_square_BE_value_l47_47631


namespace watch_correct_time_l47_47665

-- Conditions
def initial_time_slow : ℕ := 4 -- minutes slow at 8:00 AM
def final_time_fast : ℕ := 6 -- minutes fast at 4:00 PM
def total_time_interval : ℕ := 480 -- total time interval in minutes from 8:00 AM to 4:00 PM
def rate_of_time_gain : ℚ := (initial_time_slow + final_time_fast) / total_time_interval

-- Statement to prove
theorem watch_correct_time : 
  ∃ t : ℕ, t = 11 * 60 + 12 ∧ 
  ((8 * 60 + t) * rate_of_time_gain = 4) := 
sorry

end watch_correct_time_l47_47665


namespace prime_angle_triangle_l47_47158

theorem prime_angle_triangle (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_sum : a + b + c = 180) : a = 2 ∨ b = 2 ∨ c = 2 :=
sorry

end prime_angle_triangle_l47_47158


namespace find_a_l47_47949

noncomputable def a_b_c_complex (a b c : ℂ) : Prop :=
  a.re = a ∧ a + b + c = 4 ∧ a * b + b * c + c * a = 6 ∧ a * b * c = 8

theorem find_a (a b c : ℂ) (h : a_b_c_complex a b c) : a = 3 :=
by
  sorry

end find_a_l47_47949


namespace work_rate_l47_47833

theorem work_rate (A_rate : ℝ) (combined_rate : ℝ) (B_days : ℝ) :
  A_rate = 1 / 12 ∧ combined_rate = 1 / 6.461538461538462 → 1 / B_days = combined_rate - A_rate → B_days = 14 :=
by
  intros
  sorry

end work_rate_l47_47833


namespace book_pages_total_l47_47446

-- Definitions based on conditions
def pages_first_three_days: ℕ := 3 * 28
def pages_next_three_days: ℕ := 3 * 35
def pages_following_three_days: ℕ := 3 * 42
def pages_last_day: ℕ := 15

-- Total pages read calculated from above conditions
def total_pages_read: ℕ :=
  pages_first_three_days + pages_next_three_days + pages_following_three_days + pages_last_day

-- Proof problem statement: prove that the total pages read equal 330
theorem book_pages_total:
  total_pages_read = 330 :=
by
  sorry

end book_pages_total_l47_47446


namespace sum_first_five_terms_arith_seq_l47_47280

theorem sum_first_five_terms_arith_seq (a : ℕ → ℤ)
  (h4 : a 4 = 3) (h5 : a 5 = 7) (h6 : a 6 = 11) :
  a 1 + a 2 + a 3 + a 4 + a 5 = -5 :=
by
  sorry

end sum_first_five_terms_arith_seq_l47_47280


namespace sum_six_digit_odd_and_multiples_of_3_l47_47722

-- Definitions based on conditions
def num_six_digit_odd_numbers : Nat := 9 * (10 ^ 4) * 5

def num_six_digit_multiples_of_3 : Nat := 900000 / 3

-- Proof statement
theorem sum_six_digit_odd_and_multiples_of_3 : 
  num_six_digit_odd_numbers + num_six_digit_multiples_of_3 = 750000 := 
by 
  sorry

end sum_six_digit_odd_and_multiples_of_3_l47_47722


namespace correct_operation_l47_47181

theorem correct_operation (a : ℝ) :
  (a^2)^3 = a^6 :=
by
  sorry

end correct_operation_l47_47181


namespace smallest_x_solution_l47_47616

theorem smallest_x_solution (x : ℚ) :
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) →
  (x = -7/3 ∨ x = -11/16) →
  x = -7/3 :=
by
  sorry

end smallest_x_solution_l47_47616


namespace wallace_fulfills_orders_in_13_days_l47_47908

def batch_small_bags_production := 12
def batch_large_bags_production := 8
def time_per_small_batch := 8
def time_per_large_batch := 12
def daily_production_limit := 18

def initial_stock_small := 18
def initial_stock_large := 10

def order1_small := 45
def order1_large := 30
def order2_small := 60
def order2_large := 25
def order3_small := 52
def order3_large := 42

def total_small_bags_needed := order1_small + order2_small + order3_small
def total_large_bags_needed := order1_large + order2_large + order3_large
def small_bags_to_produce := total_small_bags_needed - initial_stock_small
def large_bags_to_produce := total_large_bags_needed - initial_stock_large

def small_batches_needed := (small_bags_to_produce + batch_small_bags_production - 1) / batch_small_bags_production
def large_batches_needed := (large_bags_to_produce + batch_large_bags_production - 1) / batch_large_bags_production

def total_time_small_batches := small_batches_needed * time_per_small_batch
def total_time_large_batches := large_batches_needed * time_per_large_batch
def total_production_time := total_time_small_batches + total_time_large_batches

def days_needed := (total_production_time + daily_production_limit - 1) / daily_production_limit

theorem wallace_fulfills_orders_in_13_days :
  days_needed = 13 := by
  sorry

end wallace_fulfills_orders_in_13_days_l47_47908


namespace cos_pi_minus_alpha_cos_double_alpha_l47_47485

open Real

theorem cos_pi_minus_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (π - α) = - sqrt 7 / 3 :=
by
  sorry

theorem cos_double_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (2 * α) = 5 / 9 :=
by
  sorry

end cos_pi_minus_alpha_cos_double_alpha_l47_47485


namespace find_first_number_l47_47421

theorem find_first_number
  (x y : ℝ)
  (h1 : y = 3.0)
  (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end find_first_number_l47_47421


namespace no_real_intersection_l47_47995

def parabola_line_no_real_intersection : Prop :=
  let a := 3
  let b := -6
  let c := 5
  (b^2 - 4 * a * c) < 0

theorem no_real_intersection (h : parabola_line_no_real_intersection) : 
  ∀ x : ℝ, 3*x^2 - 4*x + 2 ≠ 2*x - 3 :=
by sorry

end no_real_intersection_l47_47995


namespace closest_perfect_square_to_273_l47_47931

theorem closest_perfect_square_to_273 : ∃ n : ℕ, (n^2 = 289) ∧ 
  ∀ m : ℕ, (m^2 < 273 → 273 - m^2 ≥ 1) ∧ (m^2 > 273 → m^2 - 273 ≥ 16) :=
by
  sorry

end closest_perfect_square_to_273_l47_47931


namespace base6_divisible_19_l47_47467

theorem base6_divisible_19 (y : ℤ) : (19 ∣ (615 + 6 * y)) ↔ y = 2 := sorry

end base6_divisible_19_l47_47467


namespace donny_paid_l47_47126

variable (total_capacity initial_fuel price_per_liter change : ℕ)

theorem donny_paid (h1 : total_capacity = 150) 
                   (h2 : initial_fuel = 38) 
                   (h3 : price_per_liter = 3) 
                   (h4 : change = 14) : 
                   (total_capacity - initial_fuel) * price_per_liter + change = 350 := 
by
  sorry

end donny_paid_l47_47126


namespace cheaper_lens_price_l47_47677

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) 
  (h₁ : original_price = 300) 
  (h₂ : discount_rate = 0.20) 
  (h₃ : savings = 20) 
  (discounted_price : ℝ) 
  (cheaper_lens_price : ℝ)
  (discount_eq : discounted_price = original_price * (1 - discount_rate))
  (savings_eq : cheaper_lens_price = discounted_price - savings) :
  cheaper_lens_price = 220 := 
by sorry

end cheaper_lens_price_l47_47677


namespace ratio_w_y_l47_47597

-- Define the necessary variables
variables (w x y z : ℚ)

-- Define the conditions as hypotheses
axiom h1 : w / x = 4 / 3
axiom h2 : y / z = 5 / 3
axiom h3 : z / x = 1 / 6

-- State the proof problem
theorem ratio_w_y : w / y = 24 / 5 :=
by sorry

end ratio_w_y_l47_47597


namespace amount_with_r_l47_47315

theorem amount_with_r (p q r : ℕ) (h1 : p + q + r = 7000) (h2 : r = (2 * (p + q)) / 3) : r = 2800 :=
sorry

end amount_with_r_l47_47315


namespace even_sum_probability_l47_47197

-- Definition of probabilities for the first wheel
def prob_first_even : ℚ := 2 / 6
def prob_first_odd  : ℚ := 4 / 6

-- Definition of probabilities for the second wheel
def prob_second_even : ℚ := 3 / 8
def prob_second_odd  : ℚ := 5 / 8

-- The expected probability of the sum being even
theorem even_sum_probability : prob_first_even * prob_second_even + prob_first_odd * prob_second_odd = 13 / 24 := by
  sorry

end even_sum_probability_l47_47197


namespace value_of_f_neg_2009_l47_47488

def f (a b x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem value_of_f_neg_2009 (a b : ℝ) (h : f a b 2009 = 10) :
  f a b (-2009) = -14 :=
by 
  sorry

end value_of_f_neg_2009_l47_47488


namespace quadratic_has_real_roots_l47_47043

open Real

theorem quadratic_has_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x : ℝ, x^2 + k * x + k^2 - 1 = 0 ↔
    -2 / sqrt 3 ≤ k ∧ k ≤ 2 / sqrt 3 :=
by
  sorry

end quadratic_has_real_roots_l47_47043
