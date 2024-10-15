import Mathlib

namespace NUMINAMATH_GPT_tamara_has_30_crackers_l193_19348

theorem tamara_has_30_crackers :
  ∀ (Tamara Nicholas Marcus Mona : ℕ),
    Tamara = 2 * Nicholas →
    Marcus = 3 * Mona →
    Nicholas = Mona + 6 →
    Marcus = 27 →
    Tamara = 30 :=
by
  intros Tamara Nicholas Marcus Mona h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tamara_has_30_crackers_l193_19348


namespace NUMINAMATH_GPT_length_of_train_l193_19325

variable (d_train d_bridge v t : ℝ)

theorem length_of_train
  (h1 : v = 12.5) 
  (h2 : t = 30) 
  (h3 : d_bridge = 255) 
  (h4 : v * t = d_train + d_bridge) : 
  d_train = 120 := 
by {
  -- We should infer from here that d_train = 120
  sorry
}

end NUMINAMATH_GPT_length_of_train_l193_19325


namespace NUMINAMATH_GPT_ratio_a_to_c_l193_19358

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 7) : 
  a / c = 105 / 16 :=
by sorry

end NUMINAMATH_GPT_ratio_a_to_c_l193_19358


namespace NUMINAMATH_GPT_main_world_population_transition_l193_19329

noncomputable def world_population_reproduction_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) : Prop :=
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional"

theorem main_world_population_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) :
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional" :=
by
  sorry

end NUMINAMATH_GPT_main_world_population_transition_l193_19329


namespace NUMINAMATH_GPT_sum_first_15_terms_l193_19302

noncomputable def sum_of_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

noncomputable def fourth_term (a d : ℝ) : ℝ := a + 3 * d
noncomputable def twelfth_term (a d : ℝ) : ℝ := a + 11 * d

theorem sum_first_15_terms (a d : ℝ) 
  (h : fourth_term a d + twelfth_term a d = 10) : sum_of_terms a d 15 = 75 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_15_terms_l193_19302


namespace NUMINAMATH_GPT_company_picnic_l193_19378

theorem company_picnic :
  (20 / 100 * (30 / 100 * 100) + 40 / 100 * (70 / 100 * 100)) / 100 * 100 = 34 := by
  sorry

end NUMINAMATH_GPT_company_picnic_l193_19378


namespace NUMINAMATH_GPT_area_code_length_l193_19318

theorem area_code_length (n : ℕ) (h : 224^n - 222^n = 888) : n = 2 :=
sorry

end NUMINAMATH_GPT_area_code_length_l193_19318


namespace NUMINAMATH_GPT_termites_count_l193_19354

theorem termites_count (total_workers monkeys : ℕ) (h1 : total_workers = 861) (h2 : monkeys = 239) : total_workers - monkeys = 622 :=
by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_termites_count_l193_19354


namespace NUMINAMATH_GPT_distance_to_pinedale_mall_l193_19392

-- Define the conditions given in the problem
def average_speed : ℕ := 60  -- km/h
def stops_interval : ℕ := 5   -- minutes
def number_of_stops : ℕ := 8

-- The distance from Yahya's house to Pinedale Mall
theorem distance_to_pinedale_mall : 
  (average_speed * (number_of_stops * stops_interval / 60) = 40) :=
by
  sorry

end NUMINAMATH_GPT_distance_to_pinedale_mall_l193_19392


namespace NUMINAMATH_GPT_find_a_l193_19372

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l193_19372


namespace NUMINAMATH_GPT_option_C_correct_l193_19398

variable {a b c d : ℝ}

theorem option_C_correct (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end NUMINAMATH_GPT_option_C_correct_l193_19398


namespace NUMINAMATH_GPT_no_solution_xn_yn_zn_l193_19341

theorem no_solution_xn_yn_zn (x y z n : ℕ) (h : n ≥ z) : ¬ (x^n + y^n = z^n) :=
sorry

end NUMINAMATH_GPT_no_solution_xn_yn_zn_l193_19341


namespace NUMINAMATH_GPT_cubing_identity_l193_19309

theorem cubing_identity (x : ℝ) (hx : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_GPT_cubing_identity_l193_19309


namespace NUMINAMATH_GPT_outliers_in_data_set_l193_19338

-- Define the data set
def dataSet : List ℕ := [6, 19, 33, 33, 39, 41, 41, 43, 51, 57]

-- Define the given quartiles
def Q1 : ℕ := 33
def Q3 : ℕ := 43

-- Define the interquartile range
def IQR : ℕ := Q3 - Q1

-- Define the outlier thresholds
def lowerOutlierThreshold : ℕ := Q1 - 3 / 2 * IQR
def upperOutlierThreshold : ℕ := Q3 + 3 / 2 * IQR

-- Define what it means to be an outlier
def isOutlier (x : ℕ) : Bool :=
  x < lowerOutlierThreshold ∨ x > upperOutlierThreshold

-- Count the number of outliers in the data set
def countOutliers (data : List ℕ) : ℕ :=
  (data.filter isOutlier).length

theorem outliers_in_data_set :
  countOutliers dataSet = 1 :=
by
  sorry

end NUMINAMATH_GPT_outliers_in_data_set_l193_19338


namespace NUMINAMATH_GPT_range_of_t_l193_19304

theorem range_of_t (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : x + y > 1) (h4 : x + 1 > y) (h5 : y + 1 > x) :
    1 ≤ max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) ∧
    max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) < (1 + Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_range_of_t_l193_19304


namespace NUMINAMATH_GPT_exists_x_divisible_by_3n_not_by_3np1_l193_19364

noncomputable def f (x : ℕ) : ℕ := x ^ 3 + 17

theorem exists_x_divisible_by_3n_not_by_3np1 (n : ℕ) (hn : 2 ≤ n) : 
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬ (3^(n+1) ∣ f x) :=
sorry

end NUMINAMATH_GPT_exists_x_divisible_by_3n_not_by_3np1_l193_19364


namespace NUMINAMATH_GPT_soccer_ball_problem_l193_19328

-- Definitions of conditions
def price_eqs (x y : ℕ) : Prop :=
  x + 2 * y = 800 ∧ 3 * x + 2 * y = 1200

def total_cost_constraint (m : ℕ) : Prop :=
  200 * m + 300 * (20 - m) ≤ 5000 ∧ 1 ≤ m ∧ m ≤ 19

def store_discounts (x y : ℕ) (m : ℕ) : Prop :=
  200 * m + (3 / 5) * 300 * (20 - m) = (200 * m + (3 / 5) * 300 * (20 - m))

-- Main problem statement
theorem soccer_ball_problem :
  ∃ (x y m : ℕ), price_eqs x y ∧ total_cost_constraint m ∧ store_discounts x y m :=
sorry

end NUMINAMATH_GPT_soccer_ball_problem_l193_19328


namespace NUMINAMATH_GPT_increasing_sum_sequence_l193_19359

theorem increasing_sum_sequence (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < a (n + 1))
  (hSn : ∀ n : ℕ, Sn (n + 1) = Sn n + a (n + 1)) :
  (∀ n : ℕ, Sn (n + 1) > Sn n)
  ∧ ¬ (∀ n : ℕ, Sn (n + 1) > Sn n → 0 < a (n + 1)) :=
sorry

end NUMINAMATH_GPT_increasing_sum_sequence_l193_19359


namespace NUMINAMATH_GPT_div_eq_four_l193_19355

theorem div_eq_four (x : ℝ) (h : 64 / x = 4) : x = 16 :=
sorry

end NUMINAMATH_GPT_div_eq_four_l193_19355


namespace NUMINAMATH_GPT_sequence_a_n_is_n_l193_19311

-- Definitions and statements based on the conditions
def sequence_cond (a : ℕ → ℕ) (n : ℕ) : ℕ := 
1 / 2 * (a n) ^ 2 + n / 2

theorem sequence_a_n_is_n :
  ∀ (a : ℕ → ℕ), (∀ n, n > 0 → ∃ (S_n : ℕ), S_n = sequence_cond a n) → 
  (∀ n, n > 0 → a n = n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_is_n_l193_19311


namespace NUMINAMATH_GPT_cylinder_surface_area_l193_19351

theorem cylinder_surface_area
  (r : ℝ) (V : ℝ) (h_radius : r = 1) (h_volume : V = 4 * Real.pi) :
  ∃ S : ℝ, S = 10 * Real.pi :=
by
  let l := V / (Real.pi * r^2)
  have h_l : l = 4 := sorry
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  have h_S : S = 10 * Real.pi := sorry
  exact ⟨S, h_S⟩

end NUMINAMATH_GPT_cylinder_surface_area_l193_19351


namespace NUMINAMATH_GPT_signs_of_x_and_y_l193_19391

theorem signs_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_signs_of_x_and_y_l193_19391


namespace NUMINAMATH_GPT_sum_of_cubes_minus_tripled_product_l193_19344

theorem sum_of_cubes_minus_tripled_product (a b c d : ℝ) 
  (h1 : a + b + c + d = 15)
  (h2 : ab + ac + ad + bc + bd + cd = 40) :
  a^3 + b^3 + c^3 + d^3 - 3 * a * b * c * d = 1695 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_minus_tripled_product_l193_19344


namespace NUMINAMATH_GPT_prime_constraint_unique_solution_l193_19390

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end NUMINAMATH_GPT_prime_constraint_unique_solution_l193_19390


namespace NUMINAMATH_GPT_min_time_to_cover_distance_l193_19389

variable (distance : ℝ := 3)
variable (vasya_speed_run : ℝ := 4)
variable (vasya_speed_skate : ℝ := 8)
variable (petya_speed_run : ℝ := 5)
variable (petya_speed_skate : ℝ := 10)

theorem min_time_to_cover_distance :
  ∃ (t : ℝ), t = 0.5 ∧
    ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ distance ∧ 
    (distance - x) / vasya_speed_run + x / vasya_speed_skate = t ∧
    x / petya_speed_run + (distance - x) / petya_speed_skate = t :=
by
  sorry

end NUMINAMATH_GPT_min_time_to_cover_distance_l193_19389


namespace NUMINAMATH_GPT_solve_inequality_l193_19339

variable (x : ℝ)

theorem solve_inequality : 3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2) → x ≥ 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l193_19339


namespace NUMINAMATH_GPT_find_z_value_l193_19353

noncomputable def y_varies_inversely_with_z (y z : ℝ) (k : ℝ) : Prop :=
  (y^4 * z^(1/4) = k)

theorem find_z_value (y z : ℝ) (k : ℝ) : 
  y_varies_inversely_with_z y z k → 
  y_varies_inversely_with_z 3 16 162 → 
  k = 162 →
  y = 6 → 
  z = 1 / 4096 := 
by 
  sorry

end NUMINAMATH_GPT_find_z_value_l193_19353


namespace NUMINAMATH_GPT_least_time_for_4_horses_sum_of_digits_S_is_6_l193_19385

-- Definition of horse run intervals
def horse_intervals : List Nat := List.range' 1 9 |>.map (λ k => 2 * k)

-- Function to compute LCM of a set of numbers
def lcm_set (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

-- Proving that 4 of the horse intervals have an LCM of 24
theorem least_time_for_4_horses : 
  ∃ S > 0, (S = 24 ∧ (lcm_set [2, 4, 6, 8] = S)) ∧
  (List.length (horse_intervals.filter (λ t => S % t = 0)) ≥ 4) := 
by
  sorry

-- Proving the sum of the digits of S (24) is 6
theorem sum_of_digits_S_is_6 : 
  let S := 24
  (S / 10 + S % 10 = 6) :=
by
  sorry

end NUMINAMATH_GPT_least_time_for_4_horses_sum_of_digits_S_is_6_l193_19385


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l193_19319

theorem first_term_of_geometric_sequence :
  ∀ (a b c : ℝ), 
    (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ 16 = b * r ∧ c = 16 * r ∧ 128 = c * r) →
    a = 1 / 4 :=
by
  intros a b c
  rintro ⟨r, hr0, hbr, h16r, hcr, h128r⟩
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l193_19319


namespace NUMINAMATH_GPT_problem_solution_l193_19314

def grid_side : ℕ := 4
def square_size : ℝ := 2
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

noncomputable def grid_area : ℝ :=
  (grid_side * grid_side) * (square_size * square_size)

noncomputable def circle_area : ℝ :=
  num_circles * (Real.pi * (circle_radius ^ 2))

noncomputable def ellipse_area : ℝ :=
  Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)

noncomputable def visible_shaded_area (A B : ℝ) : Prop :=
  grid_area = A - B * Real.pi

theorem problem_solution : ∃ A B, visible_shaded_area A B ∧ (A + B = 69) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l193_19314


namespace NUMINAMATH_GPT_simplify_expr_l193_19352

open Real

theorem simplify_expr (x : ℝ) (hx : 1 ≤ x) :
  sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 
  if x ≤ 2 then 2 else 2 * sqrt (x - 1) :=
by sorry

end NUMINAMATH_GPT_simplify_expr_l193_19352


namespace NUMINAMATH_GPT_Connie_needs_more_money_l193_19327

-- Definitions based on the given conditions
def Money_saved : ℝ := 39
def Cost_of_watch : ℝ := 55
def Cost_of_watch_strap : ℝ := 15
def Tax_rate : ℝ := 0.08

-- Lean 4 statement to prove the required amount of money
theorem Connie_needs_more_money : 
  let total_cost_before_tax := Cost_of_watch + Cost_of_watch_strap
  let tax_amount := total_cost_before_tax * Tax_rate
  let total_cost_including_tax := total_cost_before_tax + tax_amount
  Money_saved < total_cost_including_tax →
  total_cost_including_tax - Money_saved = 36.60 :=
by
  sorry

end NUMINAMATH_GPT_Connie_needs_more_money_l193_19327


namespace NUMINAMATH_GPT_candy_selection_l193_19342

theorem candy_selection (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m = 1) (h3 : n = 5) :
  m + n = 6 := by
  sorry

end NUMINAMATH_GPT_candy_selection_l193_19342


namespace NUMINAMATH_GPT_minimum_value_expression_l193_19381

theorem minimum_value_expression (x y : ℝ) : 
  ∃ m : ℝ, ∀ x y : ℝ, 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ m ∧ m = 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l193_19381


namespace NUMINAMATH_GPT_range_of_a_l193_19313

open Set

noncomputable def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1}

theorem range_of_a :
  (∀ x, x ∈ B a → x ∈ A) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l193_19313


namespace NUMINAMATH_GPT_tan_ratio_l193_19334

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end NUMINAMATH_GPT_tan_ratio_l193_19334


namespace NUMINAMATH_GPT_trig_identity_l193_19361

theorem trig_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_trig_identity_l193_19361


namespace NUMINAMATH_GPT_difference_ne_1998_l193_19366

-- Define the function f(n) = n^2 + 4n
def f (n : ℕ) : ℕ := n^2 + 4 * n

-- Statement: For all natural numbers n and m, the difference f(n) - f(m) is not 1998
theorem difference_ne_1998 (n m : ℕ) : f n - f m ≠ 1998 := 
by {
  sorry
}

end NUMINAMATH_GPT_difference_ne_1998_l193_19366


namespace NUMINAMATH_GPT_not_simplifiable_by_difference_of_squares_l193_19367

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end NUMINAMATH_GPT_not_simplifiable_by_difference_of_squares_l193_19367


namespace NUMINAMATH_GPT_sum_of_xyz_l193_19324

theorem sum_of_xyz (x y z : ℝ) (h : (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0) : x + y + z = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_xyz_l193_19324


namespace NUMINAMATH_GPT_sum_of_tens_and_ones_digit_of_7_pow_25_l193_19393

theorem sum_of_tens_and_ones_digit_of_7_pow_25 : 
  let n := 7 ^ 25 
  let ones_digit := n % 10 
  let tens_digit := (n / 10) % 10 
  ones_digit + tens_digit = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_tens_and_ones_digit_of_7_pow_25_l193_19393


namespace NUMINAMATH_GPT_birth_death_rate_interval_l193_19301

theorem birth_death_rate_interval
  (b_rate : ℕ) (d_rate : ℕ) (population_increase_one_day : ℕ) (seconds_in_one_day : ℕ)
  (net_increase_per_t_seconds : ℕ) (t : ℕ)
  (h1 : b_rate = 5)
  (h2 : d_rate = 3)
  (h3 : population_increase_one_day = 86400)
  (h4 : seconds_in_one_day = 86400)
  (h5 : net_increase_per_t_seconds = b_rate - d_rate)
  (h6 : population_increase_one_day = net_increase_per_t_seconds * (seconds_in_one_day / t)) :
  t = 2 :=
by
  sorry

end NUMINAMATH_GPT_birth_death_rate_interval_l193_19301


namespace NUMINAMATH_GPT_fraction_pow_zero_is_one_l193_19315

theorem fraction_pow_zero_is_one (a b : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_pow_zero_is_one_l193_19315


namespace NUMINAMATH_GPT_notebook_cost_l193_19345

theorem notebook_cost (n c : ℝ) (h1 : n + c = 2.50) (h2 : n = c + 2) : n = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_notebook_cost_l193_19345


namespace NUMINAMATH_GPT_sum_of_products_of_three_numbers_l193_19371

theorem sum_of_products_of_three_numbers
    (a b c : ℝ)
    (h1 : a^2 + b^2 + c^2 = 179)
    (h2 : a + b + c = 21) :
  ab + bc + ac = 131 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_products_of_three_numbers_l193_19371


namespace NUMINAMATH_GPT_perfect_cube_prime_l193_19336

theorem perfect_cube_prime (p : ℕ) (h_prime : Nat.Prime p) (h_cube : ∃ x : ℕ, 2 * p + 1 = x^3) : 
  2 * p + 1 = 27 ∧ p = 13 :=
by
  sorry

end NUMINAMATH_GPT_perfect_cube_prime_l193_19336


namespace NUMINAMATH_GPT_distance_between_5th_and_23rd_red_light_l193_19320

theorem distance_between_5th_and_23rd_red_light :
  let inch_to_feet (inches : ℕ) : ℝ := inches / 12.0
  let distance_in_inches := 40 * 8
  inch_to_feet distance_in_inches = 26.67 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_5th_and_23rd_red_light_l193_19320


namespace NUMINAMATH_GPT_cities_with_fewer_than_200000_residents_l193_19335

def percentage_of_cities_with_fewer_than_50000 : ℕ := 20
def percentage_of_cities_with_50000_to_199999 : ℕ := 65

theorem cities_with_fewer_than_200000_residents :
  percentage_of_cities_with_fewer_than_50000 + percentage_of_cities_with_50000_to_199999 = 85 :=
by
  sorry

end NUMINAMATH_GPT_cities_with_fewer_than_200000_residents_l193_19335


namespace NUMINAMATH_GPT_solve_dfrac_eq_l193_19317

theorem solve_dfrac_eq (x : ℝ) (h : (x / 5) / 3 = 3 / (x / 5)) : x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_GPT_solve_dfrac_eq_l193_19317


namespace NUMINAMATH_GPT_chocolate_bars_in_large_box_l193_19303

theorem chocolate_bars_in_large_box
  (small_box_count : ℕ) (chocolate_per_small_box : ℕ)
  (h1 : small_box_count = 20)
  (h2 : chocolate_per_small_box = 25) :
  (small_box_count * chocolate_per_small_box) = 500 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_in_large_box_l193_19303


namespace NUMINAMATH_GPT_minimum_value_of_y_at_l193_19368

def y (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem minimum_value_of_y_at (x : ℝ) :
  (∀ x : ℝ, y x ≥ 2) ∧ (y (-2) = 2) :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_at_l193_19368


namespace NUMINAMATH_GPT_total_is_twenty_l193_19307

def num_blue := 5
def num_red := 7
def prob_red_or_white : ℚ := 0.75

noncomputable def total_marbles (T : ℕ) (W : ℕ) :=
  5 + 7 + W = T ∧ (7 + W) / T = prob_red_or_white

theorem total_is_twenty : ∃ (T : ℕ) (W : ℕ), total_marbles T W ∧ T = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_is_twenty_l193_19307


namespace NUMINAMATH_GPT_multiplicative_inverse_modulo_2799_l193_19332

theorem multiplicative_inverse_modulo_2799 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 2799 ∧ (225 * n) % 2799 = 1 :=
by {
  -- conditions are expressed directly in the theorem assumption
  sorry
}

end NUMINAMATH_GPT_multiplicative_inverse_modulo_2799_l193_19332


namespace NUMINAMATH_GPT_abs_add_opposite_signs_l193_19331

theorem abs_add_opposite_signs (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a * b < 0) : |a + b| = 1 := 
sorry

end NUMINAMATH_GPT_abs_add_opposite_signs_l193_19331


namespace NUMINAMATH_GPT_find_radius_of_circle_l193_19323

variable (AB BC AC R : ℝ)

-- Conditions
def is_right_triangle (ABC : Type) (AB BC : ℝ) (AC : outParam ℝ) : Prop :=
  AC = Real.sqrt (AB^2 + BC^2)

def is_tangent (O : Type) (AB BC AC R : ℝ) : Prop :=
  ∃ (P Q : ℝ), P = R ∧ Q = R ∧ P < AC ∧ Q < AC

theorem find_radius_of_circle (h1 : is_right_triangle ABC 21 28 AC) (h2 : is_tangent O 21 28 AC R) : R = 12 :=
sorry

end NUMINAMATH_GPT_find_radius_of_circle_l193_19323


namespace NUMINAMATH_GPT_exists_four_integers_mod_5050_l193_19397

theorem exists_four_integers_mod_5050 (S : Finset ℕ) (hS_card : S.card = 101) (hS_bound : ∀ x ∈ S, x < 5050) : 
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b - c - d) % 5050 = 0 :=
sorry

end NUMINAMATH_GPT_exists_four_integers_mod_5050_l193_19397


namespace NUMINAMATH_GPT_number_of_item_B_l193_19308

theorem number_of_item_B
    (x y z : ℕ)
    (total_items total_cost : ℕ)
    (hx_price : 1 ≤ x ∧ x ≤ 100)
    (hy_price : 1 ≤ y ∧ y ≤ 100)
    (hz_price : 1 ≤ z ∧ z ≤ 100)
    (h_total_items : total_items = 100)
    (h_total_cost : total_cost = 100)
    (h_price_equation : (x / 8) + 10 * y = z)
    (h_item_equation : x + y + (total_items - (x + y)) = total_items)
    : total_items - (x + y) = 21 :=
sorry

end NUMINAMATH_GPT_number_of_item_B_l193_19308


namespace NUMINAMATH_GPT_transport_cost_expression_and_min_cost_l193_19384

noncomputable def total_transport_cost (x : ℕ) (a : ℕ) : ℕ :=
if 2 ≤ a ∧ a ≤ 6 then (5 - a) * x + 23200 else 0

theorem transport_cost_expression_and_min_cost :
  ∀ x : ℕ, ∀ a : ℕ,
  (100 ≤ x ∧ x ≤ 800) →
  (2 ≤ a ∧ a ≤ 6) →
  (total_transport_cost x a = 5 * x + 23200) ∧ 
  (a = 6 → total_transport_cost 800 a = 22400) :=
by
  intros
  -- Provide the detailed proof here.
  sorry

end NUMINAMATH_GPT_transport_cost_expression_and_min_cost_l193_19384


namespace NUMINAMATH_GPT_travel_time_in_minutes_l193_19357

def bird_speed : ℝ := 8 -- Speed of the bird in miles per hour
def distance_to_travel : ℝ := 3 -- Distance to be traveled in miles

theorem travel_time_in_minutes : (distance_to_travel / bird_speed) * 60 = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_in_minutes_l193_19357


namespace NUMINAMATH_GPT_fraction_of_married_men_l193_19383

theorem fraction_of_married_men (num_women : ℕ) (num_single_women : ℕ) (num_married_women : ℕ)
  (num_married_men : ℕ) (total_people : ℕ) 
  (h1 : num_single_women = num_women / 4) 
  (h2 : num_married_women = num_women - num_single_women)
  (h3 : num_married_men = num_married_women) 
  (h4 : total_people = num_women + num_married_men) :
  (num_married_men : ℚ) / (total_people : ℚ) = 3 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_married_men_l193_19383


namespace NUMINAMATH_GPT_product_of_cubes_91_l193_19399

theorem product_of_cubes_91 :
  ∃ (a b : ℤ), (a = 3 ∨ a = 4) ∧ (b = 3 ∨ b = 4) ∧ (a^3 + b^3 = 91) ∧ (a * b = 12) :=
by
  sorry

end NUMINAMATH_GPT_product_of_cubes_91_l193_19399


namespace NUMINAMATH_GPT_minimum_blocks_l193_19395

-- Assume we have the following conditions encoded:
-- 
-- 1) Each block is a cube with a snap on one side and receptacle holes on the other five sides.
-- 2) Blocks can connect on the sides, top, and bottom.
-- 3) All snaps must be covered by other blocks' receptacle holes.
-- 
-- Define a formal statement of this requirement.

def block : Type := sorry -- to model the block with snap and holes
def connects (b1 b2 : block) : Prop := sorry -- to model block connectivity

def snap_covered (b : block) : Prop := sorry -- True if and only if the snap is covered by another block’s receptacle hole

theorem minimum_blocks (blocks : List block) : 
  (∀ b ∈ blocks, snap_covered b) → blocks.length ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_blocks_l193_19395


namespace NUMINAMATH_GPT_sector_to_cone_volume_l193_19376

theorem sector_to_cone_volume (θ : ℝ) (A : ℝ) (V : ℝ) (l r h : ℝ) :
  θ = (2 * Real.pi / 3) →
  A = (3 * Real.pi) →
  A = (1 / 2 * l^2 * θ) →
  θ = (r / l * 2 * Real.pi) →
  h = Real.sqrt (l^2 - r^2) →
  V = (1 / 3 * Real.pi * r^2 * h) →
  V = (2 * Real.sqrt 2 * Real.pi / 3) :=
by
  intros hθ hA hAeq hθeq hh hVeq
  sorry

end NUMINAMATH_GPT_sector_to_cone_volume_l193_19376


namespace NUMINAMATH_GPT_production_volume_l193_19362

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end NUMINAMATH_GPT_production_volume_l193_19362


namespace NUMINAMATH_GPT_seventh_monomial_l193_19375

noncomputable def sequence_monomial (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * 2^(n-1) * x^(n-1)

theorem seventh_monomial (x : ℝ) : sequence_monomial 7 x = -64 * x^6 := by
  sorry

end NUMINAMATH_GPT_seventh_monomial_l193_19375


namespace NUMINAMATH_GPT_tony_belinda_combined_age_l193_19369

/-- Tony and Belinda have a combined age. Belinda is 8 more than twice Tony's age. 
Tony is 16 years old and Belinda is 40 years old. What is their combined age? -/
theorem tony_belinda_combined_age 
  (tonys_age : ℕ)
  (belindas_age : ℕ)
  (h1 : tonys_age = 16)
  (h2 : belindas_age = 40)
  (h3 : belindas_age = 2 * tonys_age + 8) :
  tonys_age + belindas_age = 56 :=
  by sorry

end NUMINAMATH_GPT_tony_belinda_combined_age_l193_19369


namespace NUMINAMATH_GPT_find_vector_v1_v2_l193_19387

noncomputable def point_on_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 5 + 2 * t)

noncomputable def point_on_line_m (s : ℝ) : ℝ × ℝ :=
  (3 + 5 * s, 7 + 2 * s)

noncomputable def P_foot_of_perpendicular (B : ℝ × ℝ) : ℝ × ℝ :=
  (4, 8)  -- As derived from the given solution

noncomputable def vector_AB (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def vector_PB (P B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - P.1, B.2 - P.2)

theorem find_vector_v1_v2 :
  ∃ (v1 v2 : ℝ), (v1 + v2 = 1) ∧ (vector_PB (P_foot_of_perpendicular (3,7)) (3,7) = (v1, v2)) :=
  sorry

end NUMINAMATH_GPT_find_vector_v1_v2_l193_19387


namespace NUMINAMATH_GPT_clock_hands_angle_120_between_7_and_8_l193_19382

theorem clock_hands_angle_120_between_7_and_8 :
  ∃ (t₁ t₂ : ℕ), (t₁ = 5) ∧ (t₂ = 16) ∧ 
  (∃ (h₀ m₀ : ℕ → ℝ), 
    h₀ 7 = 210 ∧ 
    m₀ 7 = 0 ∧
    (∀ t : ℕ, h₀ (7 + t / 60) = 210 + t * (30 / 60)) ∧
    (∀ t : ℕ, m₀ (7 + t / 60) = t * (360 / 60)) ∧
    ((h₀ (7 + t₁ / 60) - m₀ (7 + t₁ / 60)) % 360 = 120) ∧ 
    ((h₀ (7 + t₂ / 60) - m₀ (7 + t₂ / 60)) % 360 = 120)) := by
  sorry

end NUMINAMATH_GPT_clock_hands_angle_120_between_7_and_8_l193_19382


namespace NUMINAMATH_GPT_prime_roots_range_l193_19310

theorem prime_roots_range (p : ℕ) (hp : Prime p) (h : ∃ x₁ x₂ : ℤ, x₁ + x₂ = -p ∧ x₁ * x₂ = -444 * p) : 31 < p ∧ p ≤ 41 :=
by sorry

end NUMINAMATH_GPT_prime_roots_range_l193_19310


namespace NUMINAMATH_GPT_area_of_nonagon_on_other_cathetus_l193_19360

theorem area_of_nonagon_on_other_cathetus 
    (A₁ A₂ A₃ : ℝ) 
    (h1 : A₁ = 2019) 
    (h2 : A₂ = 1602) 
    (h3 : A₁ = A₂ + A₃) : 
    A₃ = 417 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_area_of_nonagon_on_other_cathetus_l193_19360


namespace NUMINAMATH_GPT_g_of_3_over_8_l193_19330

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end NUMINAMATH_GPT_g_of_3_over_8_l193_19330


namespace NUMINAMATH_GPT_length_of_platform_l193_19394

noncomputable def len_train : ℝ := 120
noncomputable def speed_train : ℝ := 60 * (1000 / 3600) -- kmph to m/s
noncomputable def time_cross : ℝ := 15

theorem length_of_platform (L_train : ℝ) (S_train : ℝ) (T_cross : ℝ) (H_train : L_train = len_train)
  (H_speed : S_train = speed_train) (H_time : T_cross = time_cross) : 
  ∃ (L_platform : ℝ), L_platform = (S_train * T_cross) - L_train ∧ L_platform = 130.05 :=
by
  rw [H_train, H_speed, H_time]
  sorry

end NUMINAMATH_GPT_length_of_platform_l193_19394


namespace NUMINAMATH_GPT_second_term_is_neg_12_l193_19322

-- Define the problem conditions
variables {a d : ℤ}
axiom tenth_term : a + 9 * d = 20
axiom eleventh_term : a + 10 * d = 24

-- Define the second term calculation
def second_term (a d : ℤ) := a + d

-- The problem statement: Prove that the second term is -12 given the conditions
theorem second_term_is_neg_12 : second_term a d = -12 :=
by sorry

end NUMINAMATH_GPT_second_term_is_neg_12_l193_19322


namespace NUMINAMATH_GPT_value_of_a4_l193_19380

variables {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers.

-- Conditions: The sequence is geometric, positive and satisfies the given product condition.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = (a n) * (a k)

-- Condition: All terms are positive.
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- Given product condition:
axiom a1_a7_product : a 1 * a 7 = 36

-- The theorem to prove:
theorem value_of_a4 (h_geo : is_geometric_sequence a) (h_pos : all_terms_positive a) : 
  a 4 = 6 :=
sorry

end NUMINAMATH_GPT_value_of_a4_l193_19380


namespace NUMINAMATH_GPT_emma_list_count_l193_19316

theorem emma_list_count : 
  let m1 := 900
  let m2 := 27000
  let d := 30
  (m1 / d <= m2 / d) → (m2 / d - m1 / d + 1 = 871) :=
by
  intros m1 m2 d h
  have h1 : m1 / d ≤ m2 / d := h
  have h2 : m2 / d - m1 / d + 1 = 871 := by sorry
  exact h2

end NUMINAMATH_GPT_emma_list_count_l193_19316


namespace NUMINAMATH_GPT_prop_C_prop_D_l193_19379

theorem prop_C (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

theorem prop_D (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end NUMINAMATH_GPT_prop_C_prop_D_l193_19379


namespace NUMINAMATH_GPT_polynomial_A_l193_19396

theorem polynomial_A (A a : ℝ) (h : A * (a + 1) = a^2 - 1) : A = a - 1 :=
sorry

end NUMINAMATH_GPT_polynomial_A_l193_19396


namespace NUMINAMATH_GPT_tank_capacity_l193_19337

theorem tank_capacity :
  let rateA := 40  -- Pipe A fills at 40 liters per minute
  let rateB := 30  -- Pipe B fills at 30 liters per minute
  let rateC := -20  -- Pipe C (drains) at 20 liters per minute, thus negative contribution
  let cycle_duration := 3  -- The cycle duration is 3 minutes
  let total_duration := 51  -- The tank gets full in 51 minutes
  let net_per_cycle := rateA + rateB + rateC  -- Net fill per cycle of 3 minutes
  let num_cycles := total_duration / cycle_duration  -- Number of complete cycles
  let tank_capacity := net_per_cycle * num_cycles  -- Tank capacity in liters
  tank_capacity = 850  -- Assertion that needs to be proven
:= by
  let rateA := 40
  let rateB := 30
  let rateC := -20
  let cycle_duration := 3
  let total_duration := 51
  let net_per_cycle := rateA + rateB + rateC
  let num_cycles := total_duration / cycle_duration
  let tank_capacity := net_per_cycle * num_cycles
  have : tank_capacity = 850 := by
    sorry
  assumption

end NUMINAMATH_GPT_tank_capacity_l193_19337


namespace NUMINAMATH_GPT_M_inter_N_eq_l193_19300

open Set

def M : Set ℝ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 < n ∧ n ≤ 3 }

theorem M_inter_N_eq : M ∩ (coe '' N) = {0, 1} :=
by sorry

end NUMINAMATH_GPT_M_inter_N_eq_l193_19300


namespace NUMINAMATH_GPT_percentage_of_mixture_X_is_13_333_l193_19326

variable (X Y : ℝ) (P : ℝ)

-- Conditions
def mixture_X_contains_40_percent_ryegrass : Prop := X = 0.40
def mixture_Y_contains_25_percent_ryegrass : Prop := Y = 0.25
def final_mixture_contains_27_percent_ryegrass : Prop := 0.4 * P + 0.25 * (100 - P) = 27

-- The goal
theorem percentage_of_mixture_X_is_13_333
    (h1 : mixture_X_contains_40_percent_ryegrass X)
    (h2 : mixture_Y_contains_25_percent_ryegrass Y)
    (h3 : final_mixture_contains_27_percent_ryegrass P) :
  P = 200 / 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_mixture_X_is_13_333_l193_19326


namespace NUMINAMATH_GPT_solution_exists_l193_19312

noncomputable def verify_triples (a b c : ℝ) : Prop :=
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ b = -2 * a ∧ c = 4 * a

theorem solution_exists (a b c : ℝ) : verify_triples a b c :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_l193_19312


namespace NUMINAMATH_GPT_complementary_event_l193_19363

-- Definitions based on the conditions
def EventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≥ 2

def complementEventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≤ 1

-- Theorem based on the question and correct answer
theorem complementary_event (products : List Bool) :
  complementEventA products ↔ ¬ EventA products :=
by sorry

end NUMINAMATH_GPT_complementary_event_l193_19363


namespace NUMINAMATH_GPT_travel_distance_correct_l193_19340

noncomputable def traveler_distance : ℝ :=
  let x1 : ℝ := -4
  let y1 : ℝ := 0
  let x2 : ℝ := x1 + 5 * Real.cos (-(Real.pi / 3))
  let y2 : ℝ := y1 + 5 * Real.sin (-(Real.pi / 3))
  let x3 : ℝ := x2 + 2
  let y3 : ℝ := y2
  Real.sqrt (x3^2 + y3^2)

theorem travel_distance_correct : traveler_distance = Real.sqrt 19 := by
  sorry

end NUMINAMATH_GPT_travel_distance_correct_l193_19340


namespace NUMINAMATH_GPT_distinct_values_of_products_l193_19321

theorem distinct_values_of_products (n : ℤ) (h : 1 ≤ n) :
  ¬ ∃ a b c d : ℤ, n^2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 ∧ ad = bc :=
sorry

end NUMINAMATH_GPT_distinct_values_of_products_l193_19321


namespace NUMINAMATH_GPT_quadratic_solution_downward_solution_minimum_solution_l193_19365

def is_quadratic (m : ℝ) : Prop :=
  m^2 + 3 * m - 2 = 2

def opens_downwards (m : ℝ) : Prop :=
  m + 3 < 0

def has_minimum (m : ℝ) : Prop :=
  m + 3 > 0

theorem quadratic_solution (m : ℝ) :
  is_quadratic m → (m = -4 ∨ m = 1) :=
sorry

theorem downward_solution (m : ℝ) :
  is_quadratic m → opens_downwards m → m = -4 :=
sorry

theorem minimum_solution (m : ℝ) :
  is_quadratic m → has_minimum m → m = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_downward_solution_minimum_solution_l193_19365


namespace NUMINAMATH_GPT_youseff_lives_6_blocks_from_office_l193_19350

-- Definitions
def blocks_youseff_lives_from_office (x : ℕ) : Prop :=
  ∃ t_walk t_bike : ℕ,
    t_walk = x ∧
    t_bike = (20 * x) / 60 ∧
    t_walk = t_bike + 4

-- Main theorem
theorem youseff_lives_6_blocks_from_office (x : ℕ) (h : blocks_youseff_lives_from_office x) : x = 6 :=
  sorry

end NUMINAMATH_GPT_youseff_lives_6_blocks_from_office_l193_19350


namespace NUMINAMATH_GPT_regular_polygon_with_12_degree_exterior_angle_has_30_sides_l193_19306

def regular_polygon_sides (e : ℤ) : ℤ :=
  360 / e

theorem regular_polygon_with_12_degree_exterior_angle_has_30_sides :
  regular_polygon_sides 12 = 30 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_regular_polygon_with_12_degree_exterior_angle_has_30_sides_l193_19306


namespace NUMINAMATH_GPT_together_work_days_l193_19374

/-- 
  X does the work in 10 days and Y does the same work in 15 days.
  Together, they will complete the work in 6 days.
 -/
theorem together_work_days (hx : ℝ) (hy : ℝ) : 
  (hx = 10) → (hy = 15) → (1 / (1 / hx + 1 / hy) = 6) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_together_work_days_l193_19374


namespace NUMINAMATH_GPT_front_view_heights_l193_19333

-- Define conditions
def column1 := [4, 2]
def column2 := [3, 0, 3]
def column3 := [1, 5]

-- Define a function to get the max height in each column
def max_height (col : List Nat) : Nat :=
  col.foldr Nat.max 0

-- Define the statement to prove the frontal view heights
theorem front_view_heights : 
  max_height column1 = 4 ∧ 
  max_height column2 = 3 ∧ 
  max_height column3 = 5 :=
by 
  sorry

end NUMINAMATH_GPT_front_view_heights_l193_19333


namespace NUMINAMATH_GPT_multiplication_correct_l193_19373

theorem multiplication_correct :
  23 * 195 = 4485 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_correct_l193_19373


namespace NUMINAMATH_GPT_no_a_for_empty_intersection_a_in_range_for_subset_union_l193_19347

open Set

def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4 * x - 5 > 0}

-- Problem 1: There is no a such that A ∩ B = ∅
theorem no_a_for_empty_intersection : ∀ a : ℝ, A a ∩ B = ∅ → False := by
  sorry

-- Problem 2: If A ∪ B = B, then a ∈ (-∞, -4) ∪ (5, ∞)
theorem a_in_range_for_subset_union (a : ℝ) : A a ∪ B = B → a ∈ Iio (-4) ∪ Ioi 5 := by
  sorry

end NUMINAMATH_GPT_no_a_for_empty_intersection_a_in_range_for_subset_union_l193_19347


namespace NUMINAMATH_GPT_molly_takes_180_minutes_more_l193_19349

noncomputable def xanthia_speed : ℕ := 120
noncomputable def molly_speed : ℕ := 60
noncomputable def first_book_pages : ℕ := 360

-- Time taken by Xanthia to read the first book in hours
noncomputable def xanthia_time_first_book : ℕ := first_book_pages / xanthia_speed

-- Time taken by Molly to read the first book in hours
noncomputable def molly_time_first_book : ℕ := first_book_pages / molly_speed

-- Difference in time taken to read the first book in minutes
noncomputable def time_diff_minutes : ℕ := (molly_time_first_book - xanthia_time_first_book) * 60

theorem molly_takes_180_minutes_more : time_diff_minutes = 180 := by
  sorry

end NUMINAMATH_GPT_molly_takes_180_minutes_more_l193_19349


namespace NUMINAMATH_GPT_rearrange_squares_into_one_square_l193_19388

theorem rearrange_squares_into_one_square 
  (a b : ℕ) (h_a : a = 3) (h_b : b = 1) 
  (parts : Finset (ℕ × ℕ)) 
  (h_parts1 : parts.card ≤ 3)
  (h_parts2 : ∀ p ∈ parts, p.1 * p.2 = a * a ∨ p.1 * p.2 = b * b)
  : ∃ c : ℕ, (c * c = (a * a) + (b * b)) :=
by
  sorry

end NUMINAMATH_GPT_rearrange_squares_into_one_square_l193_19388


namespace NUMINAMATH_GPT_translated_parabola_eq_new_equation_l193_19346

-- Definitions following directly from the condition
def original_parabola (x : ℝ) : ℝ := 2 * x^2
def new_vertex : (ℝ × ℝ) := (-2, -2)
def new_parabola (x : ℝ) : ℝ := 2 * (x + 2)^2 - 2

-- Statement to prove the equivalency of the translated parabola equation
theorem translated_parabola_eq_new_equation :
  (∀ (x : ℝ), (original_parabola x = new_parabola (x - 2))) :=
by
  sorry

end NUMINAMATH_GPT_translated_parabola_eq_new_equation_l193_19346


namespace NUMINAMATH_GPT_divisors_not_multiples_of_14_l193_19377

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 2
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 3
def is_perfect_fifth (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 5
def is_perfect_seventh (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 7

def n : ℕ := 2^2 * 3^3 * 5^5 * 7^7

theorem divisors_not_multiples_of_14 :
  is_perfect_square (n / 2) →
  is_perfect_cube (n / 3) →
  is_perfect_fifth (n / 5) →
  is_perfect_seventh (n / 7) →
  (∃ d : ℕ, d = 240) :=
by
  sorry

end NUMINAMATH_GPT_divisors_not_multiples_of_14_l193_19377


namespace NUMINAMATH_GPT_box_width_l193_19305

theorem box_width (W : ℕ) (h₁ : 15 * W * 13 = 3120) : W = 16 := by
  sorry

end NUMINAMATH_GPT_box_width_l193_19305


namespace NUMINAMATH_GPT_find_bottle_price_l193_19370

theorem find_bottle_price 
  (x : ℝ) 
  (promotion_free_bottles : ℝ := 3)
  (discount_per_bottle : ℝ := 0.6)
  (box_price : ℝ := 26)
  (box_bottles : ℝ := 4) :
  ∃ x : ℝ, (box_price / (x - discount_per_bottle)) - (box_price / x) = promotion_free_bottles :=
sorry

end NUMINAMATH_GPT_find_bottle_price_l193_19370


namespace NUMINAMATH_GPT_ice_cream_weekend_total_l193_19386

theorem ice_cream_weekend_total 
  (f : ℝ) (r : ℝ) (n : ℕ)
  (h_friday : f = 3.25)
  (h_saturday_reduction : r = 0.25)
  (h_num_people : n = 4)
  (h_saturday : (f - r * n) = 2.25)
  (h_sunday : 2 * ((f - r * n) / n) * n = 4.5) :
  f + (f - r * n) + (2 * ((f - r * n) / n) * n) = 10 := sorry

end NUMINAMATH_GPT_ice_cream_weekend_total_l193_19386


namespace NUMINAMATH_GPT_combinations_of_coins_l193_19343

noncomputable def count_combinations (target : ℕ) : ℕ :=
  (30 - 0*0) -- As it just returns 45 combinations

theorem combinations_of_coins : count_combinations 30 = 45 :=
  sorry

end NUMINAMATH_GPT_combinations_of_coins_l193_19343


namespace NUMINAMATH_GPT_inclination_angle_range_l193_19356

theorem inclination_angle_range (k : ℝ) (h : |k| ≤ 1) :
    ∃ α : ℝ, (k = Real.tan α) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_inclination_angle_range_l193_19356
