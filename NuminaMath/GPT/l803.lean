import Mathlib

namespace area_of_region_l803_80390

theorem area_of_region (x y : ℝ) :
  x ≤ 2 * y ∧ y ≤ 2 * x ∧ x + y ≤ 60 →
  ∃ (A : ℝ), A = 600 :=
by
  sorry

end area_of_region_l803_80390


namespace value_of_expression_when_x_is_neg2_l803_80399

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l803_80399


namespace parallelogram_height_l803_80378

theorem parallelogram_height (A B H : ℝ) (hA : A = 462) (hB : B = 22) (hArea : A = B * H) : H = 21 :=
by
  sorry

end parallelogram_height_l803_80378


namespace shares_difference_l803_80352

theorem shares_difference (x : ℝ) (hp : ℝ) (hq : ℝ) (hr : ℝ)
  (hx : hp = 3 * x) (hqx : hq = 7 * x) (hrx : hr = 12 * x) 
  (hqr_diff : hr - hq = 3500) : (hq - hp = 2800) :=
by
  -- The proof would be done here, but the problem statement requires only the theorem statement
  sorry

end shares_difference_l803_80352


namespace jasmine_max_stickers_l803_80347

-- Given conditions and data
def sticker_cost : ℝ := 0.75
def jasmine_budget : ℝ := 10.0

-- Proof statement
theorem jasmine_max_stickers : ∃ n : ℕ, (n : ℝ) * sticker_cost ≤ jasmine_budget ∧ (∀ m : ℕ, (m > n) → (m : ℝ) * sticker_cost > jasmine_budget) :=
sorry

end jasmine_max_stickers_l803_80347


namespace no_zero_sum_of_vectors_l803_80344

-- Definitions and conditions for the problem
variable {n : ℕ} (odd_n : n % 2 = 1) -- n is odd, representing the number of sides of the polygon

-- The statement of the proof problem
theorem no_zero_sum_of_vectors (odd_n : n % 2 = 1) : false :=
by
  sorry

end no_zero_sum_of_vectors_l803_80344


namespace vector_addition_correct_l803_80354

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, 2)

theorem vector_addition_correct : vector_a + vector_b = (1, 5) :=
by
  -- Assume a and b are vectors in 2D space
  have a := vector_a
  have b := vector_b
  -- By definition of vector addition
  sorry

end vector_addition_correct_l803_80354


namespace sum_of_numbers_with_six_zeros_and_56_divisors_l803_80375

theorem sum_of_numbers_with_six_zeros_and_56_divisors :
  ∃ N1 N2 : ℕ, (N1 % 10^6 = 0) ∧ (N2 % 10^6 = 0) ∧ (N1_divisors = 56) ∧ (N2_divisors = 56) ∧ (N1 + N2 = 7000000) :=
by
  sorry

end sum_of_numbers_with_six_zeros_and_56_divisors_l803_80375


namespace total_money_received_l803_80383

-- Define the conditions
def total_puppies : ℕ := 20
def fraction_sold : ℚ := 3 / 4
def price_per_puppy : ℕ := 200

-- Define the statement to prove
theorem total_money_received : fraction_sold * total_puppies * price_per_puppy = 3000 := by
  sorry

end total_money_received_l803_80383


namespace hyperbola_asymptotes_l803_80374

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 - (y^2 / 4) = 1) ↔ (y = 2 * x ∨ y = -2 * x) := by
  sorry

end hyperbola_asymptotes_l803_80374


namespace soccer_team_probability_l803_80381

theorem soccer_team_probability :
  let total_players := 12
  let forwards := 6
  let defenders := 6
  let total_ways := Nat.choose total_players 2
  let defender_ways := Nat.choose defenders 2
  ∃ p : ℚ, p = defender_ways / total_ways ∧ p = 5 / 22 :=
sorry

end soccer_team_probability_l803_80381


namespace find_expression_value_l803_80342

theorem find_expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end find_expression_value_l803_80342


namespace max_area_of_garden_l803_80323

theorem max_area_of_garden (L : ℝ) (hL : 0 ≤ L) :
  ∃ x y : ℝ, x + 2 * y = L ∧ x ≥ 0 ∧ y ≥ 0 ∧ x * y = L^2 / 8 :=
by
  sorry

end max_area_of_garden_l803_80323


namespace scientist_birth_day_is_wednesday_l803_80379

noncomputable def calculate_birth_day : String :=
  let years := 150
  let leap_years := 36
  let regular_years := years - leap_years
  let total_days_backward := regular_years + 2 * leap_years -- days to move back
  let days_mod := total_days_backward % 7
  let day_of_birth := (5 + 7 - days_mod) % 7 -- 5 is for backward days from Monday
  match day_of_birth with
  | 0 => "Monday"
  | 1 => "Sunday"
  | 2 => "Saturday"
  | 3 => "Friday"
  | 4 => "Thursday"
  | 5 => "Wednesday"
  | 6 => "Tuesday"
  | _ => "Error"

theorem scientist_birth_day_is_wednesday :
  calculate_birth_day = "Wednesday" :=
  by
    sorry

end scientist_birth_day_is_wednesday_l803_80379


namespace find_integer_N_l803_80358

theorem find_integer_N : ∃ N : ℤ, (N ^ 2 ≡ N [ZMOD 10000]) ∧ (N - 2 ≡ 0 [ZMOD 7]) :=
by
  sorry

end find_integer_N_l803_80358


namespace geometric_sequence_ratio_l803_80304

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (q : ℕ) (h1 : q = 2)
  (h2 : ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 4 / S 2 = 5 :=
by
  sorry

end geometric_sequence_ratio_l803_80304


namespace find_number_l803_80332

theorem find_number (x : ℝ) (h : x - (3/5) * x = 50) : x = 125 := by
  sorry

end find_number_l803_80332


namespace range_of_independent_variable_l803_80361

theorem range_of_independent_variable (x : ℝ) :
  (x + 2 >= 0) → (x - 1 ≠ 0) → (x ≥ -2 ∧ x ≠ 1) :=
by
  intros h₁ h₂
  sorry

end range_of_independent_variable_l803_80361


namespace bert_money_left_l803_80303

theorem bert_money_left (initial_money : ℕ) (spent_hardware : ℕ) (spent_cleaners : ℕ) (spent_grocery : ℕ) :
  initial_money = 52 →
  spent_hardware = initial_money * 1 / 4 →
  spent_cleaners = 9 →
  spent_grocery = (initial_money - spent_hardware - spent_cleaners) / 2 →
  initial_money - spent_hardware - spent_cleaners - spent_grocery = 15 := 
by
  intros h_initial h_hardware h_cleaners h_grocery
  rw [h_initial, h_hardware, h_cleaners, h_grocery]
  sorry

end bert_money_left_l803_80303


namespace count_valid_m_l803_80307

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_m (m : ℕ) : Prop :=
  m > 1 ∧ is_divisor m 480 ∧ (480 / m) > 1

theorem count_valid_m : (∃ m, valid_m m) → Nat.card {m // valid_m m} = 22 :=
by sorry

end count_valid_m_l803_80307


namespace boxes_needed_l803_80356

def initial_games : ℕ := 76
def games_sold : ℕ := 46
def games_per_box : ℕ := 5

theorem boxes_needed : (initial_games - games_sold) / games_per_box = 6 := by
  sorry

end boxes_needed_l803_80356


namespace no_primes_in_sequence_l803_80301

-- Definitions and conditions derived from the problem statement
variable (a : ℕ → ℕ) -- sequence of natural numbers
variable (increasing : ∀ n, a n < a (n + 1)) -- increasing sequence
variable (is_arith_or_geom : ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) ^ 2 = a n * a (n + 2))) -- arithmetic or geometric progression condition
variable (divisible_by_four : a 0 % 4 = 0 ∧ a 1 % 4 = 0) -- first two numbers divisible by 4

-- The statement to prove: no prime numbers exist in the sequence
theorem no_primes_in_sequence : ∀ n, ¬ (Nat.Prime (a n)) :=
by 
  sorry

end no_primes_in_sequence_l803_80301


namespace apple_ratio_simplest_form_l803_80387

theorem apple_ratio_simplest_form (sarah_apples brother_apples cousin_apples : ℕ) 
  (h1 : sarah_apples = 630)
  (h2 : brother_apples = 270)
  (h3 : cousin_apples = 540)
  (gcd_simplified : Nat.gcd (Nat.gcd sarah_apples brother_apples) cousin_apples = 90) :
  (sarah_apples / 90, brother_apples / 90, cousin_apples / 90) = (7, 3, 6) := 
by
  sorry

end apple_ratio_simplest_form_l803_80387


namespace custom_dollar_five_neg3_l803_80327

-- Define the custom operation
def custom_dollar (a b : Int) : Int :=
  a * (b - 1) + a * b

-- State the theorem
theorem custom_dollar_five_neg3 : custom_dollar 5 (-3) = -35 := by
  sorry

end custom_dollar_five_neg3_l803_80327


namespace difference_is_four_l803_80318

open Nat

-- Assume we have a 5x5x5 cube
def cube_side_length : ℕ := 5
def total_unit_cubes : ℕ := cube_side_length ^ 3

-- Define the two configurations
def painted_cubes_config1 : ℕ := 65  -- Two opposite faces and one additional face
def painted_cubes_config2 : ℕ := 61  -- Three adjacent faces

-- The difference in the number of unit cubes with at least one painted face
def painted_difference : ℕ := painted_cubes_config1 - painted_cubes_config2

theorem difference_is_four :
    painted_difference = 4 := by
  sorry

end difference_is_four_l803_80318


namespace proof_y_times_1_minus_g_eq_1_l803_80314
noncomputable def y : ℝ := (3 + Real.sqrt 8) ^ 100
noncomputable def m : ℤ := Int.floor y
noncomputable def g : ℝ := y - m

theorem proof_y_times_1_minus_g_eq_1 :
  y * (1 - g) = 1 := 
sorry

end proof_y_times_1_minus_g_eq_1_l803_80314


namespace min_value_expr_l803_80384

theorem min_value_expr (a d : ℝ) (b c : ℝ) (h_a : 0 ≤ a) (h_d : 0 ≤ d) (h_b : 0 < b) (h_c : 0 < c) (h : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expr_l803_80384


namespace villager4_truth_teller_l803_80397

def villager1_statement (liars : Finset ℕ) : Prop := liars = {0, 1, 2, 3}
def villager2_statement (liars : Finset ℕ) : Prop := liars.card = 1
def villager3_statement (liars : Finset ℕ) : Prop := liars.card = 2
def villager4_statement (liars : Finset ℕ) : Prop := 3 ∉ liars

theorem villager4_truth_teller (liars : Finset ℕ) :
  ¬ villager1_statement liars ∧
  ¬ villager2_statement liars ∧
  ¬ villager3_statement liars ∧
  villager4_statement liars ↔
  liars = {0, 1, 2} :=
by
  sorry

end villager4_truth_teller_l803_80397


namespace probability_of_six_being_largest_l803_80309

noncomputable def probability_six_is_largest : ℚ := sorry

theorem probability_of_six_being_largest (cards : Finset ℕ) (selected_cards : Finset ℕ) :
  cards = {1, 2, 3, 4, 5, 6, 7} →
  selected_cards ⊆ cards →
  selected_cards.card = 4 →
  (probability_six_is_largest = 2 / 7) := sorry

end probability_of_six_being_largest_l803_80309


namespace smallest_even_sum_equals_200_l803_80331

theorem smallest_even_sum_equals_200 :
  ∃ (x : ℤ), (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) ∧ (x = 36) :=
by
  sorry

end smallest_even_sum_equals_200_l803_80331


namespace ratio_of_money_given_l803_80365

theorem ratio_of_money_given
  (T : ℕ) (W : ℕ) (Th : ℕ) (m : ℕ)
  (h1 : T = 8) 
  (h2 : W = m * T) 
  (h3 : Th = W + 9)
  (h4 : Th = T + 41) : 
  W / T = 5 := 
sorry

end ratio_of_money_given_l803_80365


namespace dollar_eval_l803_80360

def dollar (a b : ℝ) : ℝ := (a^2 - b^2)^2

theorem dollar_eval (x : ℝ) : dollar (x^3 + x) (x - x^3) = 16 * x^8 :=
by
  sorry

end dollar_eval_l803_80360


namespace age_difference_l803_80310

theorem age_difference (M S : ℕ) (h1 : S = 16) (h2 : M + 2 = 2 * (S + 2)) : M - S = 18 :=
by
  sorry

end age_difference_l803_80310


namespace directrix_of_parabola_l803_80326

noncomputable def parabola_directrix (x : ℝ) : ℝ := 4 * x^2 + 4 * x + 1

theorem directrix_of_parabola :
  ∃ (y : ℝ) (x : ℝ), parabola_directrix x = y ∧ y = 4 * (x + 1/2)^2 + 3/4 ∧ y - 1/16 = 11/16 :=
by
  sorry

end directrix_of_parabola_l803_80326


namespace wicket_keeper_age_difference_l803_80376

def cricket_team_average_age : Nat := 24
def total_members : Nat := 11
def remaining_members : Nat := 9
def age_difference : Nat := 1

theorem wicket_keeper_age_difference :
  let total_age := cricket_team_average_age * total_members
  let remaining_average_age := cricket_team_average_age - age_difference
  let remaining_total_age := remaining_average_age * remaining_members
  let combined_age := total_age - remaining_total_age
  let average_age := cricket_team_average_age
  let wicket_keeper_age := combined_age - average_age
  wicket_keeper_age - average_age = 9 := 
by
  sorry

end wicket_keeper_age_difference_l803_80376


namespace total_payment_correct_l803_80346

-- Define the prices of different apples.
def price_small_apple : ℝ := 1.5
def price_medium_apple : ℝ := 2.0
def price_big_apple : ℝ := 3.0

-- Define the quantities of apples bought by Donny.
def quantity_small_apples : ℕ := 6
def quantity_medium_apples : ℕ := 6
def quantity_big_apples : ℕ := 8

-- Define the conditions.
def discount_medium_apples_threshold : ℕ := 5
def discount_medium_apples_rate : ℝ := 0.20
def tax_rate : ℝ := 0.10
def big_apple_special_offer_count : ℕ := 3
def big_apple_special_offer_discount_rate : ℝ := 0.50

-- Step function to calculate discount and total cost.
noncomputable def total_cost : ℝ :=
  let cost_small := quantity_small_apples * price_small_apple
  let cost_medium := quantity_medium_apples * price_medium_apple
  let discount_medium := if quantity_medium_apples > discount_medium_apples_threshold 
                         then cost_medium * discount_medium_apples_rate else 0
  let cost_medium_after_discount := cost_medium - discount_medium
  let cost_big := quantity_big_apples * price_big_apple
  let discount_big := (quantity_big_apples / big_apple_special_offer_count) * 
                       (price_big_apple * big_apple_special_offer_discount_rate)
  let cost_big_after_discount := cost_big - discount_big
  let total_cost_before_tax := cost_small + cost_medium_after_discount + cost_big_after_discount
  let tax := total_cost_before_tax * tax_rate
  total_cost_before_tax + tax

-- Define the expected total payment.
def expected_total_payment : ℝ := 43.56

-- The theorem statement: Prove that total_cost equals the expected total payment.
theorem total_payment_correct : total_cost = expected_total_payment := sorry

end total_payment_correct_l803_80346


namespace min_value_expression_l803_80385

theorem min_value_expression : ∃ x y : ℝ, (x = 2 ∧ y = -3/2) ∧ ∀ a b : ℝ, 2 * a^2 + 2 * b^2 - 8 * a + 6 * b + 28 ≥ 10.5 :=
sorry

end min_value_expression_l803_80385


namespace maximize_profit_marginal_profit_monotonic_decreasing_l803_80335

-- Definition of revenue function R
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Definition of cost function C
def C (x : ℕ) : ℤ := 460 * x + 500

-- Definition of profit function p
def p (x : ℕ) : ℤ := R x - C x

-- Lemma for the solution
theorem maximize_profit (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 20) : 
  p x = -10 * x^3 + 45 * x^2 + 3240 * x - 500 ∧ 
  (∀ y, 1 ≤ y ∧ y ≤ 20 → p y ≤ p 12) :=
by
  sorry

-- Definition of marginal profit function Mp
def Mp (x : ℕ) : ℤ := p (x + 1) - p x

-- Lemma showing Mp is monotonically decreasing
theorem marginal_profit_monotonic_decreasing (x : ℕ) (h2 : 1 ≤ x ∧ x ≤ 19) : 
  Mp x = -30 * x^2 + 60 * x + 3275 ∧ 
  ∀ y, 1 ≤ y ∧ y ≤ 19 → (Mp y ≥ Mp (y + 1)) :=
by
  sorry

end maximize_profit_marginal_profit_monotonic_decreasing_l803_80335


namespace ratio_of_supply_to_demand_l803_80340

theorem ratio_of_supply_to_demand (supply demand : ℕ)
  (hs : supply = 1800000)
  (hd : demand = 2400000) :
  supply / (Nat.gcd supply demand) = 3 ∧ demand / (Nat.gcd supply demand) = 4 :=
by
  sorry

end ratio_of_supply_to_demand_l803_80340


namespace number_is_0_point_5_l803_80324

theorem number_is_0_point_5 (x : ℝ) (h : x = 1/6 + 0.33333333333333337) : x = 0.5 := 
by
  -- The actual proof would go here.
  sorry

end number_is_0_point_5_l803_80324


namespace not_perfect_power_l803_80300

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ∀ m n : ℕ, m > 1 → n > 1 → 10^k - 1 ≠ m ^ n :=
by 
  sorry

end not_perfect_power_l803_80300


namespace gym_monthly_revenue_l803_80386

-- Defining the conditions
def charge_per_session : ℕ := 18
def sessions_per_month : ℕ := 2
def number_of_members : ℕ := 300

-- Defining the question as a theorem statement
theorem gym_monthly_revenue : 
  (number_of_members * (charge_per_session * sessions_per_month)) = 10800 := 
by 
  -- Skip the proof, verifying the statement only
  sorry

end gym_monthly_revenue_l803_80386


namespace rational_solution_exists_l803_80382

theorem rational_solution_exists :
  ∃ (a b : ℚ), (a + b) / a + a / (a + b) = b :=
by
  sorry

end rational_solution_exists_l803_80382


namespace comparison_of_abc_l803_80338

noncomputable def a : ℝ := 24 / 7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7 / Real.exp 1) / Real.log 3 + 1

theorem comparison_of_abc :
  (a = 24 / 7) →
  (b * Real.exp b = 7 * Real.log 7) →
  (3 ^ (c - 1) = 7 / Real.exp 1) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  sorry

end comparison_of_abc_l803_80338


namespace difference_place_values_l803_80349

def place_value (digit : Char) (position : String) : Real :=
  match digit, position with
  | '1', "hundreds" => 100
  | '1', "tenths" => 0.1
  | _, _ => 0 -- for any other cases (not required in this problem)

theorem difference_place_values :
  (place_value '1' "hundreds" - place_value '1' "tenths" = 99.9) :=
by
  sorry

end difference_place_values_l803_80349


namespace cone_height_l803_80328

theorem cone_height (R : ℝ) (r h l : ℝ)
  (volume_sphere : ∀ R,  V_sphere = (4 / 3) * π * R^3)
  (volume_cone : ∀ r h,  V_cone = (1 / 3) * π * r^2 * h)
  (lateral_surface_area : ∀ r l, A_lateral = π * r * l)
  (area_base : ∀ r, A_base = π * r^2)
  (vol_eq : (1/3) * π * r^2 * h = (4/3) * π * R^3)
  (lat_eq : π * r * l = 3 * π * r^2) 
  (pyth_rel : l^2 = r^2 + h^2) :
  h = 4 * R * Real.sqrt 2 := 
sorry

end cone_height_l803_80328


namespace arith_seq_s14_gt_0_l803_80317

variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of an arithmetic sequence
variable {a : ℕ → ℝ} -- a_n is the nth term of the arithmetic sequence
variable {d : ℝ} -- d is the common difference of the arithmetic sequence

-- Conditions
variable (a_7_lt_0 : a 7 < 0)
variable (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0)

-- Assertion
theorem arith_seq_s14_gt_0 (a_7_lt_0 : a 7 < 0) (a_5_plus_a_10_gt_0 : a 5 + a 10 > 0) : S 14 > 0 := by
  sorry

end arith_seq_s14_gt_0_l803_80317


namespace point_below_parabola_l803_80362

theorem point_below_parabola (a b c : ℝ) (h : 2 < a + b + c) : 
  2 < c + b + a :=
by
  sorry

end point_below_parabola_l803_80362


namespace afternoon_snack_calories_l803_80364

def ellen_daily_calories : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def dinner_remaining_calories : ℕ := 832

theorem afternoon_snack_calories :
  ellen_daily_calories - (breakfast_calories + lunch_calories + dinner_remaining_calories) = 130 :=
by sorry

end afternoon_snack_calories_l803_80364


namespace centroid_distance_l803_80371

-- Define the given conditions and final goal
theorem centroid_distance (a b c p q r : ℝ) 
  (ha : a ≠ 0)  (hb : b ≠ 0)  (hc : c ≠ 0)
  (centroid : p = a / 3 ∧ q = b / 3 ∧ r = c / 3) 
  (plane_distance : (1 / (1 / a^2 + 1 / b^2 + 1 / c^2).sqrt) = 2) :
  (1 / p^2 + 1 / q^2 + 1 / r^2) = 2.25 := 
by 
  -- Start proof here
  sorry

end centroid_distance_l803_80371


namespace eleven_pow_2023_mod_eight_l803_80359

theorem eleven_pow_2023_mod_eight (h11 : 11 % 8 = 3) (h3 : 3^2 % 8 = 1) : 11^2023 % 8 = 3 :=
by
  sorry

end eleven_pow_2023_mod_eight_l803_80359


namespace minimum_value_l803_80388

noncomputable def min_value_b_plus_4_over_a (a : ℝ) (b : ℝ) :=
  b + 4 / a

theorem minimum_value (a : ℝ) (b : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x : ℝ, x > 0 → (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  min_value_b_plus_4_over_a a b = 2 * Real.sqrt 5 :=
sorry

end minimum_value_l803_80388


namespace no_infinite_subdivision_exists_l803_80394

theorem no_infinite_subdivision_exists : ¬ ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ n : ℕ,
    ∃ (ai bi : ℝ), ai > bi ∧ bi > 0 ∧ ai * bi = a * b ∧
    (ai / bi = a / b ∨ bi / ai = a / b)) :=
sorry

end no_infinite_subdivision_exists_l803_80394


namespace profit_percentage_l803_80313

theorem profit_percentage (SP CP : ℝ) (h₁ : SP = 300) (h₂ : CP = 250) : ((SP - CP) / CP) * 100 = 20 := by
  sorry

end profit_percentage_l803_80313


namespace book_transaction_difference_l803_80337

def number_of_books : ℕ := 15
def cost_per_book : ℕ := 11
def selling_price_per_book : ℕ := 25

theorem book_transaction_difference :
  number_of_books * selling_price_per_book - number_of_books * cost_per_book = 210 :=
by
  sorry

end book_transaction_difference_l803_80337


namespace sweatshirt_cost_l803_80380

/--
Hannah bought 3 sweatshirts and 2 T-shirts.
Each T-shirt cost $10.
Hannah spent $65 in total.
Prove that the cost of each sweatshirt is $15.
-/
theorem sweatshirt_cost (S : ℝ) (h1 : 3 * S + 2 * 10 = 65) : S = 15 :=
by
  sorry

end sweatshirt_cost_l803_80380


namespace num_pos_multiples_of_six_is_150_l803_80391

theorem num_pos_multiples_of_six_is_150 : 
  ∃ (n : ℕ), (∀ k, (n = 150) ↔ (102 + (k - 1) * 6 = 996 ∧ 102 ≤ 6 * k ∧ 6 * k ≤ 996)) :=
sorry

end num_pos_multiples_of_six_is_150_l803_80391


namespace find_trapezoid_bases_l803_80398

-- Define the conditions of the isosceles trapezoid
variables {AD BC : ℝ}
variables (h1 : ∀ (A B C D : ℝ), is_isosceles_trapezoid A B C D ∧ intersects_at_right_angle A B C D)
variables (h2 : ∀ {A B C D : ℝ}, trapezoid_area A B C D = 12)
variables (h3 : ∀ {A B C D : ℝ}, trapezoid_height A B C D = 2)

-- Prove the bases AD and BC are 8 and 4 respectively under the given conditions
theorem find_trapezoid_bases (AD BC : ℝ) : 
  AD = 8 ∧ BC = 4 :=
  sorry

end find_trapezoid_bases_l803_80398


namespace value_of_y_l803_80395

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l803_80395


namespace inequality_f_n_l803_80330

theorem inequality_f_n {f : ℕ → ℕ} {k : ℕ} (strict_mono_f : ∀ {a b : ℕ}, a < b → f a < f b)
  (h_f : ∀ n : ℕ, f (f n) = k * n) : ∀ n : ℕ, 
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  sorry

end inequality_f_n_l803_80330


namespace employee_pays_216_l803_80373

def retail_price (wholesale_cost : ℝ) (markup_percentage : ℝ) : ℝ :=
    wholesale_cost + markup_percentage * wholesale_cost

def employee_payment (retail_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    retail_price - discount_percentage * retail_price

theorem employee_pays_216 (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
    wholesale_cost = 200 ∧ markup_percentage = 0.20 ∧ discount_percentage = 0.10 →
    employee_payment (retail_price wholesale_cost markup_percentage) discount_percentage = 216 :=
by
  intro h
  rcases h with ⟨h_wholesale, h_markup, h_discount⟩
  rw [h_wholesale, h_markup, h_discount]
  -- Now we have to prove the final statement: employee_payment (retail_price 200 0.20) 0.10 = 216
  -- This follows directly by computation, so we leave it as a sorry for now
  sorry

end employee_pays_216_l803_80373


namespace regression_prediction_l803_80348

-- Define the linear regression model as a function
def linear_regression (x : ℝ) : ℝ :=
  7.19 * x + 73.93

-- State that using this model, the predicted height at age 10 is approximately 145.83
theorem regression_prediction :
  abs (linear_regression 10 - 145.83) < 0.01 :=
by 
  sorry

end regression_prediction_l803_80348


namespace find_x_squared_plus_y_squared_l803_80363

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = -8) : x^2 + y^2 = 33 := 
by 
  sorry

end find_x_squared_plus_y_squared_l803_80363


namespace inequality_am_gm_l803_80396

theorem inequality_am_gm (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h : a^2 + b^2 + c^2 = 12) :
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := 
by
  sorry

end inequality_am_gm_l803_80396


namespace count_ones_digits_of_numbers_divisible_by_4_and_3_l803_80345

theorem count_ones_digits_of_numbers_divisible_by_4_and_3 :
  let eligible_numbers := { n : ℕ | n < 100 ∧ n % 4 = 0 ∧ n % 3 = 0 }
  ∃ (digits : Finset ℕ), 
    (∀ n ∈ eligible_numbers, n % 10 ∈ digits) ∧
    digits.card = 5 :=
by
  sorry

end count_ones_digits_of_numbers_divisible_by_4_and_3_l803_80345


namespace unknown_number_is_three_or_twenty_seven_l803_80306

theorem unknown_number_is_three_or_twenty_seven
    (x y : ℝ)
    (h1 : y - 3 = x - y)
    (h2 : (y - 6) / 3 = x / (y - 6)) :
    x = 3 ∨ x = 27 :=
by
  sorry

end unknown_number_is_three_or_twenty_seven_l803_80306


namespace donna_soda_crates_l803_80311

def soda_crates (bridge_limit : ℕ) (truck_empty : ℕ) (crate_weight : ℕ) (dryer_weight : ℕ) (num_dryers : ℕ) (truck_loaded : ℕ) (produce_ratio : ℕ) : ℕ :=
  sorry

theorem donna_soda_crates :
  soda_crates 20000 12000 50 3000 3 24000 2 = 20 :=
sorry

end donna_soda_crates_l803_80311


namespace vector_dot_product_l803_80343

open Real

variables (a b : ℝ × ℝ)

def condition1 : Prop := (a.1 + b.1 = 1 ∧ a.2 + b.2 = -3)
def condition2 : Prop := (a.1 - b.1 = 3 ∧ a.2 - b.2 = 7)
def dot_product : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product :
  condition1 a b ∧ condition2 a b → dot_product a b = -12 := by
  sorry

end vector_dot_product_l803_80343


namespace total_charge_correct_l803_80322

def boxwoodTrimCost (numBoxwoods : Nat) (trimCost : Nat) : Nat :=
  numBoxwoods * trimCost

def boxwoodShapeCost (numBoxwoods : Nat) (shapeCost : Nat) : Nat :=
  numBoxwoods * shapeCost

theorem total_charge_correct :
  let numBoxwoodsTrimmed := 30
  let trimCost := 5
  let numBoxwoodsShaped := 4
  let shapeCost := 15
  let totalTrimCost := boxwoodTrimCost numBoxwoodsTrimmed trimCost
  let totalShapeCost := boxwoodShapeCost numBoxwoodsShaped shapeCost
  let totalCharge := totalTrimCost + totalShapeCost
  totalCharge = 210 :=
by sorry

end total_charge_correct_l803_80322


namespace fare_ratio_l803_80308

theorem fare_ratio (F1 F2 : ℕ) (h1 : F1 = 96000) (h2 : F1 + F2 = 224000) : F1 / (Nat.gcd F1 F2) = 3 ∧ F2 / (Nat.gcd F1 F2) = 4 :=
by
  sorry

end fare_ratio_l803_80308


namespace abs_neg_five_l803_80341

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l803_80341


namespace simplify_expression_l803_80319

-- Define the fractions involved
def frac1 : ℚ := 1 / 2
def frac2 : ℚ := 1 / 3
def frac3 : ℚ := 1 / 5
def frac4 : ℚ := 1 / 7

-- Define the expression to be simplified
def expr : ℚ := (frac1 - frac2 + frac3) / (frac2 - frac1 + frac4)

-- The goal is to show that the expression simplifies to -77 / 5
theorem simplify_expression : expr = -77 / 5 := by
  sorry

end simplify_expression_l803_80319


namespace parallelogram_sides_l803_80325

theorem parallelogram_sides (a b : ℕ): 
  (a = 3 * b) ∧ (2 * a + 2 * b = 24) → (a = 9) ∧ (b = 3) :=
by
  sorry

end parallelogram_sides_l803_80325


namespace zero_of_f_l803_80351

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ↔ x = 1 :=
by
  sorry

end zero_of_f_l803_80351


namespace focus_of_parabola_l803_80357

theorem focus_of_parabola (x y : ℝ) : 
  (∃ x y : ℝ, x^2 = -2 * y) → (0, -1/2) = (0, -1/2) :=
sorry

end focus_of_parabola_l803_80357


namespace sum_of_reciprocals_of_transformed_roots_l803_80316

theorem sum_of_reciprocals_of_transformed_roots :
  ∀ (a b c : ℂ), (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -2) :=
by
  intros a b c ha hb hc habc
  sorry

end sum_of_reciprocals_of_transformed_roots_l803_80316


namespace ratio_chloe_to_max_l803_80368

/-- Chloe’s wins and Max’s wins -/
def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

/-- The ratio of Chloe's wins to Max's wins is 8:3 -/
theorem ratio_chloe_to_max : (chloe_wins / Nat.gcd chloe_wins max_wins) = 8 ∧ (max_wins / Nat.gcd chloe_wins max_wins) = 3 := by
  sorry

end ratio_chloe_to_max_l803_80368


namespace range_of_m_for_false_proposition_l803_80370

theorem range_of_m_for_false_proposition :
  ¬ (∃ x : ℝ, x^2 - m * x - m ≤ 0) → m ∈ Set.Ioo (-4 : ℝ) 0 :=
sorry

end range_of_m_for_false_proposition_l803_80370


namespace intersection_P_Q_l803_80350

def P : Set ℤ := {-4, -2, 0, 2, 4}
def Q : Set ℤ := {x : ℤ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end intersection_P_Q_l803_80350


namespace fraction_difference_l803_80355

variable (a b : ℝ)

theorem fraction_difference (h : 1/a - 1/b = 1/(a + b)) : 
  1/a^2 - 1/b^2 = 1/(a * b) := 
  sorry

end fraction_difference_l803_80355


namespace distance_AC_l803_80353

theorem distance_AC (south_dist : ℕ) (west_dist : ℕ) (north_dist : ℕ) (east_dist : ℕ) :
  south_dist = 50 → west_dist = 70 → north_dist = 30 → east_dist = 40 →
  Real.sqrt ((south_dist - north_dist)^2 + (west_dist - east_dist)^2) = 36.06 :=
by
  intros h_south h_west h_north h_east
  rw [h_south, h_west, h_north, h_east]
  simp
  norm_num
  sorry

end distance_AC_l803_80353


namespace expression_evaluation_l803_80305

theorem expression_evaluation (x y : ℝ) (h₁ : x > y) (h₂ : y > 0) : 
    (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x / y)^(y - x) :=
by
  sorry

end expression_evaluation_l803_80305


namespace hyperbola_condition_l803_80372

theorem hyperbola_condition (m : ℝ) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → m < 0 ↔ x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end hyperbola_condition_l803_80372


namespace simplify_expression_l803_80392

theorem simplify_expression (x : ℝ) : 
  x - 2 * (1 + x) + 3 * (1 - x) - 4 * (1 + 2 * x) = -12 * x - 3 := 
by 
  -- Proof goes here
  sorry

end simplify_expression_l803_80392


namespace sin_1320_eq_neg_sqrt_3_div_2_l803_80367

theorem sin_1320_eq_neg_sqrt_3_div_2 : Real.sin (1320 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_eq_neg_sqrt_3_div_2_l803_80367


namespace geometric_sequence_b_mn_theorem_l803_80315

noncomputable def geometric_sequence_b_mn (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ) 
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2) 
  (h_nm_pos : m > 0 ∧ n > 0): Prop :=
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m))

-- We skip the proof using sorry.
theorem geometric_sequence_b_mn_theorem 
  (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ)
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2)
  (h_nm_pos : m > 0 ∧ n > 0) : 
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m)) :=
sorry

end geometric_sequence_b_mn_theorem_l803_80315


namespace smallest_b_factors_l803_80334

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l803_80334


namespace problem_a_problem_b_l803_80369

-- Problem (a): Prove that (1 + 1/x)(1 + 1/y) ≥ 9 given x > 0, y > 0, and x + y = 1
theorem problem_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 := sorry

-- Problem (b): Prove that 0 < u + v - uv < 1 given 0 < u < 1 and 0 < v < 1
theorem problem_b (u v : ℝ) (hu : 0 < u) (hu1 : u < 1) (hv : 0 < v) (hv1 : v < 1) : 
  0 < u + v - u * v ∧ u + v - u * v < 1 := sorry

end problem_a_problem_b_l803_80369


namespace gymnastics_average_people_per_team_l803_80393

def average_people_per_team (boys girls teams : ℕ) : ℕ :=
  (boys + girls) / teams

theorem gymnastics_average_people_per_team:
  average_people_per_team 83 77 4 = 40 :=
by
  sorry

end gymnastics_average_people_per_team_l803_80393


namespace polynomial_value_l803_80329

theorem polynomial_value (a b : ℝ) (h₁ : a * b = 7) (h₂ : a + b = 2) : a^2 * b + a * b^2 - 20 = -6 :=
by {
  sorry
}

end polynomial_value_l803_80329


namespace max_distance_circle_to_point_A_l803_80366

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (-1, 3)

noncomputable def max_distance (d : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ d = Real.sqrt ((2 + 1)^2 + (0 - 3)^2) + Real.sqrt 2 

theorem max_distance_circle_to_point_A : max_distance (4 * Real.sqrt 2) :=
sorry

end max_distance_circle_to_point_A_l803_80366


namespace find_real_roots_l803_80336

theorem find_real_roots : 
  {x : ℝ | x^9 + (9 / 8) * x^6 + (27 / 64) * x^3 - x + (219 / 512) = 0} =
  {1 / 2, (-1 + Real.sqrt 13) / 4, (-1 - Real.sqrt 13) / 4} :=
by
  sorry

end find_real_roots_l803_80336


namespace fg_of_3_l803_80339

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement to prove f(g(3)) = 2199
theorem fg_of_3 : f (g 3) = 2199 :=
by
  sorry

end fg_of_3_l803_80339


namespace jasmine_percentage_new_solution_l803_80321

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_jasmine_percent : ℝ := 0.10
def added_jasmine : ℝ := 5
def added_water : ℝ := 15

-- Define the correct answer
theorem jasmine_percentage_new_solution :
  let initial_jasmine := initial_jasmine_percent * initial_volume
  let new_jasmine := initial_jasmine + added_jasmine
  let total_new_volume := initial_volume + added_jasmine + added_water
  (new_jasmine / total_new_volume) * 100 = 13 := 
by 
  sorry

end jasmine_percentage_new_solution_l803_80321


namespace middle_managers_sample_count_l803_80389

def employees_total : ℕ := 1000
def managers_middle_total : ℕ := 150
def sample_total : ℕ := 200

theorem middle_managers_sample_count :
  sample_total * managers_middle_total / employees_total = 30 := by
  sorry

end middle_managers_sample_count_l803_80389


namespace a3_eq_5_l803_80320

variable {a_n : ℕ → Real} (S : ℕ → Real)
variable (a1 d : Real)

-- Define arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → Real) (a1 d : Real) : Prop :=
  ∀ n : ℕ, n > 0 → a_n n = a1 + (n - 1) * d

-- Define sum of first n terms
def sum_of_arithmetic (S : ℕ → Real) (a_n : ℕ → Real) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (a_n 1 + a_n n)

-- Given conditions: S_5 = 25
def S_5_eq_25 (S : ℕ → Real) : Prop :=
  S 5 = 25

-- Goal: prove a_3 = 5
theorem a3_eq_5 (h_arith : is_arithmetic_sequence a_n a1 d)
                (h_sum : sum_of_arithmetic S a_n)
                (h_S5 : S_5_eq_25 S) : a_n 3 = 5 :=
  sorry

end a3_eq_5_l803_80320


namespace find_D_coordinates_l803_80312

theorem find_D_coordinates:
  ∀ (A B C : (ℝ × ℝ)), 
  A = (-2, 5) ∧ C = (3, 7) ∧ B = (-3, 0) →
  ∃ D : (ℝ × ℝ), D = (2, 2) :=
by
  sorry

end find_D_coordinates_l803_80312


namespace min_value_3x_plus_4y_l803_80333

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l803_80333


namespace num_passengers_on_second_plane_l803_80302

theorem num_passengers_on_second_plane :
  ∃ x : ℕ, 600 - (2 * 50) + 600 - (2 * x) + 600 - (2 * 40) = 1500 →
  x = 60 :=
by
  sorry

end num_passengers_on_second_plane_l803_80302


namespace number_add_thrice_number_eq_twenty_l803_80377

theorem number_add_thrice_number_eq_twenty (x : ℝ) (h : x + 3 * x = 20) : x = 5 :=
sorry

end number_add_thrice_number_eq_twenty_l803_80377
