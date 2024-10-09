import Mathlib

namespace juice_cost_l2295_229525

theorem juice_cost (J : ℝ) (h1 : 15 * 3 + 25 * 1 + 12 * J = 88) : J = 1.5 :=
by
  sorry

end juice_cost_l2295_229525


namespace tan_alpha_l2295_229599

theorem tan_alpha {α : ℝ} (h : 3 * Real.sin α + 4 * Real.cos α = 5) : Real.tan α = 3 / 4 :=
by
  -- Proof goes here
  sorry

end tan_alpha_l2295_229599


namespace students_second_scenario_l2295_229539

def total_students (R : ℕ) : ℕ := 5 * R + 6
def effective_students (R : ℕ) : ℕ := 6 * (R - 3)
def filled_rows (R : ℕ) : ℕ := R - 3
def students_per_row := 6

theorem students_second_scenario:
  ∀ (R : ℕ), R = 24 → total_students R = effective_students R → students_per_row = 6
:= by
  intro R h_eq h_total_eq_effective
  -- Insert proof steps here
  sorry

end students_second_scenario_l2295_229539


namespace platform_length_correct_l2295_229554

noncomputable def platform_length (train_speed_kmph : ℝ) (crossing_time_s : ℝ) (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * crossing_time_s
  distance_covered - train_length_m

theorem platform_length_correct :
  platform_length 72 26 260.0416 = 259.9584 :=
by
  sorry

end platform_length_correct_l2295_229554


namespace determine_a_b_l2295_229587

theorem determine_a_b (a b : ℤ) :
  (∀ x : ℤ, x^2 + a * x + b = (x - 1) * (x + 4)) → (a = 3 ∧ b = -4) :=
by
  intro h
  sorry

end determine_a_b_l2295_229587


namespace power_function_monotonic_incr_l2295_229598

theorem power_function_monotonic_incr (m : ℝ) (h₁ : m^2 - 5 * m + 7 = 1) (h₂ : m^2 - 6 > 0) : m = 3 := 
by
  sorry

end power_function_monotonic_incr_l2295_229598


namespace divisor_is_36_l2295_229576

theorem divisor_is_36
  (Dividend Quotient Remainder : ℕ)
  (h1 : Dividend = 690)
  (h2 : Quotient = 19)
  (h3 : Remainder = 6)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Divisor = 36 :=
sorry

end divisor_is_36_l2295_229576


namespace min_value_x2_y2_l2295_229566

theorem min_value_x2_y2 (x y : ℝ) (h : 2 * x + y + 5 = 0) : x^2 + y^2 ≥ 5 :=
by
  sorry

end min_value_x2_y2_l2295_229566


namespace selling_price_41_l2295_229568

-- Purchase price per item
def purchase_price : ℝ := 30

-- Government restriction on pice increase: selling price cannot be more than 40% increase of the purchase price
def price_increase_restriction (a : ℝ) : Prop :=
  a <= purchase_price * 1.4

-- Profit condition equation
def profit_condition (a : ℝ) : Prop :=
  (a - purchase_price) * (112 - 2 * a) = 330

-- The selling price of each item that satisfies all conditions is 41 yuan  
theorem selling_price_41 (a : ℝ) (h1 : profit_condition a) (h2 : price_increase_restriction a) :
  a = 41 := sorry

end selling_price_41_l2295_229568


namespace find_valid_tax_range_l2295_229548

noncomputable def valid_tax_range (t : ℝ) : Prop :=
  let initial_consumption := 200000
  let price_per_cubic_meter := 240
  let consumption_reduction := 2.5 * t * 10^4
  let tax_revenue := (initial_consumption - consumption_reduction) * price_per_cubic_meter * (t / 100)
  tax_revenue >= 900000

theorem find_valid_tax_range (t : ℝ) : 3 ≤ t ∧ t ≤ 5 ↔ valid_tax_range t :=
sorry

end find_valid_tax_range_l2295_229548


namespace pauls_score_is_91_l2295_229595

theorem pauls_score_is_91 (q s c w : ℕ) 
  (h1 : q = 35)
  (h2 : s = 35 + 5 * c - 2 * w)
  (h3 : s > 90)
  (h4 : c + w ≤ 35)
  (h5 : ∀ s', 90 < s' ∧ s' < s → ¬ (∃ c' w', s' = 35 + 5 * c' - 2 * w' ∧ c' + w' ≤ 35 ∧ c' ≠ c)) : 
  s = 91 := 
sorry

end pauls_score_is_91_l2295_229595


namespace find_a_l2295_229515

theorem find_a (a : ℝ) (h : 3 ∈ ({1, a, a - 2} : Set ℝ)) : a = 5 :=
sorry

end find_a_l2295_229515


namespace pay_nineteen_rubles_l2295_229547

/-- 
Given a purchase cost of 19 rubles, a customer with only three-ruble bills, 
and a cashier with only five-ruble bills, both having 15 bills each,
prove that it is possible for the customer to pay exactly 19 rubles.
-/
theorem pay_nineteen_rubles (purchase_cost : ℕ) (customer_bills cashier_bills : ℕ) 
  (customer_denomination cashier_denomination : ℕ) (customer_count cashier_count : ℕ) :
  purchase_cost = 19 →
  customer_denomination = 3 →
  cashier_denomination = 5 →
  customer_count = 15 →
  cashier_count = 15 →
  (∃ m n : ℕ, m * customer_denomination - n * cashier_denomination = purchase_cost 
  ∧ m ≤ customer_count ∧ n ≤ cashier_count) :=
by
  intros
  sorry

end pay_nineteen_rubles_l2295_229547


namespace printer_time_ratio_l2295_229529

theorem printer_time_ratio
  (X_time : ℝ) (Y_time : ℝ) (Z_time : ℝ)
  (hX : X_time = 15)
  (hY : Y_time = 10)
  (hZ : Z_time = 20) :
  (X_time / (Y_time * Z_time / (Y_time + Z_time))) = 9 / 4 :=
by
  sorry

end printer_time_ratio_l2295_229529


namespace karlson_word_count_l2295_229520

def single_word_count : Nat := 9
def ten_to_nineteen_count : Nat := 10
def two_word_count (num_tens_units : Nat) : Nat := 2 * num_tens_units

def count_words_1_to_99 : Nat :=
  let single_word := single_word_count + ten_to_nineteen_count
  let two_word := two_word_count (99 - (single_word_count + ten_to_nineteen_count))
  single_word + two_word

def prefix_hundred (count_1_to_99 : Nat) : Nat := 9 * count_1_to_99
def extra_prefix (num_two_word_transformed : Nat) : Nat := 9 * num_two_word_transformed

def total_words : Nat :=
  let first_99 := count_words_1_to_99
  let nine_hundreds := prefix_hundred count_words_1_to_99 + extra_prefix 72
  first_99 + nine_hundreds + 37

theorem karlson_word_count : total_words = 2611 :=
  by
    sorry

end karlson_word_count_l2295_229520


namespace time_to_run_up_and_down_l2295_229594

/-- Problem statement: Prove that the time it takes Vasya to run up and down a moving escalator 
which moves upwards is 468 seconds, given these conditions:
1. Vasya runs down twice as fast as he runs up.
2. When the escalator is not working, it takes Vasya 6 minutes to run up and down.
3. When the escalator is moving down, it takes Vasya 13.5 minutes to run up and down.
--/
theorem time_to_run_up_and_down (up_speed down_speed : ℝ) (escalator_speed : ℝ) 
  (h1 : down_speed = 2 * up_speed) 
  (h2 : (1 / up_speed + 1 / down_speed) = 6) 
  (h3 : (1 / (up_speed + escalator_speed) + 1 / (down_speed - escalator_speed)) = 13.5) : 
  (1 / (up_speed - escalator_speed) + 1 / (down_speed + escalator_speed)) * 60 = 468 := 
sorry

end time_to_run_up_and_down_l2295_229594


namespace triangle_side_identity_l2295_229519

theorem triangle_side_identity
  (a b c : ℝ)
  (alpha beta gamma : ℝ)
  (h1 : alpha = 60)
  (h2 : a^2 = b^2 + c^2 - b * c) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := 
by
  sorry

end triangle_side_identity_l2295_229519


namespace initial_concentration_l2295_229555

theorem initial_concentration (f : ℚ) (C : ℚ) (h₀ : f = 0.7142857142857143) (h₁ : (1 - f) * C + f * 0.25 = 0.35) : C = 0.6 :=
by
  rw [h₀] at h₁
  -- The proof will follow the steps to solve for C
  sorry

end initial_concentration_l2295_229555


namespace circle_center_coordinates_l2295_229514

open Real

noncomputable def circle_center (x y : Real) : Prop := 
  x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center_coordinates :
  ∃ (a b : Real), circle_center a b ∧ a = 2 ∧ b = -3 :=
by
  use 2, -3
  sorry

end circle_center_coordinates_l2295_229514


namespace tennis_racket_price_l2295_229560

theorem tennis_racket_price (P : ℝ) : 
    (0.8 * P + 515) * 1.10 + 20 = 800 → 
    P = 242.61 :=
by
  sorry

end tennis_racket_price_l2295_229560


namespace trig_problem_1_trig_problem_2_l2295_229505

-- Problem (1)
theorem trig_problem_1 (α : ℝ) (h1 : Real.tan (π + α) = -4 / 3) (h2 : 3 * Real.sin α / 4 = -Real.cos α)
  : Real.sin α = -4 / 5 ∧ Real.cos α = 3 / 5 := by
  sorry

-- Problem (2)
theorem trig_problem_2 : Real.sin (25 * π / 6) + Real.cos (26 * π / 3) + Real.tan (-25 * π / 4) = -1 := by
  sorry

end trig_problem_1_trig_problem_2_l2295_229505


namespace number_of_red_candies_is_4_l2295_229500

-- Define the parameters as given in the conditions
def number_of_green_candies : ℕ := 5
def number_of_blue_candies : ℕ := 3
def likelihood_of_blue_candy : ℚ := 25 / 100

-- Define the total number of candies
def total_number_of_candies (number_of_red_candies : ℕ) : ℕ :=
  number_of_green_candies + number_of_blue_candies + number_of_red_candies

-- Define the proof statement
theorem number_of_red_candies_is_4 (R : ℕ) :
  (3 / total_number_of_candies R = 25 / 100) → R = 4 :=
sorry

end number_of_red_candies_is_4_l2295_229500


namespace line_through_point_parallel_l2295_229540

/-
Given the point P(2, 0) and a line x - 2y + 3 = 0,
prove that the equation of the line passing through 
P and parallel to the given line is 2y - x + 2 = 0.
-/
theorem line_through_point_parallel
  (P : ℝ × ℝ)
  (x y : ℝ)
  (line_eq : x - 2*y + 3 = 0)
  (P_eq : P = (2, 0)) :
  ∃ (a b c : ℝ), a * y - b * x + c = 0 :=
sorry

end line_through_point_parallel_l2295_229540


namespace denise_travel_l2295_229502

theorem denise_travel (a b c : ℕ) (h₀ : a ≥ 1) (h₁ : a + b + c = 8) (h₂ : 90 * (b - a) % 48 = 0) : a^2 + b^2 + c^2 = 26 :=
sorry

end denise_travel_l2295_229502


namespace manicure_cost_l2295_229549

noncomputable def cost_of_manicure : ℝ := 30

theorem manicure_cost
    (cost_hair_updo : ℝ)
    (total_cost_with_tips : ℝ)
    (tip_rate : ℝ)
    (M : ℝ) :
  cost_hair_updo = 50 →
  total_cost_with_tips = 96 →
  tip_rate = 0.20 →
  (cost_hair_updo + M + tip_rate * cost_hair_updo + tip_rate * M = total_cost_with_tips) →
  M = cost_of_manicure :=
by
  intros h1 h2 h3 h4
  sorry

end manicure_cost_l2295_229549


namespace simple_interest_double_in_4_years_interest_25_percent_l2295_229567

theorem simple_interest_double_in_4_years_interest_25_percent :
  ∀ {P : ℕ} (h : P > 0), ∃ (R : ℕ), R = 25 ∧ P + P * R * 4 / 100 = 2 * P :=
by
  sorry

end simple_interest_double_in_4_years_interest_25_percent_l2295_229567


namespace marginal_cost_proof_l2295_229565

theorem marginal_cost_proof (fixed_cost : ℕ) (total_cost : ℕ) (n : ℕ) (MC : ℕ)
  (h1 : fixed_cost = 12000)
  (h2 : total_cost = 16000)
  (h3 : n = 20)
  (h4 : total_cost = fixed_cost + MC * n) :
  MC = 200 :=
  sorry

end marginal_cost_proof_l2295_229565


namespace proof_equivalent_problem_l2295_229573

-- Definition of conditions
def cost_condition_1 (x y : ℚ) : Prop := 500 * x + 40 * y = 1250
def cost_condition_2 (x y : ℚ) : Prop := 1000 * x + 20 * y = 1000
def budget_condition (a b : ℕ) (total_masks : ℕ) (budget : ℕ) : Prop := 2 * a + (total_masks - a) / 2 + 25 * b = budget

-- Main theorem
theorem proof_equivalent_problem : 
  ∃ (x y : ℚ) (a b : ℕ), 
    cost_condition_1 x y ∧
    cost_condition_2 x y ∧
    (x = 1 / 2) ∧ 
    (y = 25) ∧
    (budget_condition a b 200 400) ∧
    ((a = 150 ∧ b = 3) ∨
     (a = 100 ∧ b = 6) ∨
     (a = 50 ∧ b = 9)) :=
by {
  sorry -- The proof steps are not required
}

end proof_equivalent_problem_l2295_229573


namespace color_preference_l2295_229521

-- Define the conditions
def total_students := 50
def girls := 30
def boys := 20

def girls_pref_pink := girls / 3
def girls_pref_purple := 2 * girls / 5
def girls_pref_blue := girls - girls_pref_pink - girls_pref_purple

def boys_pref_red := 2 * boys / 5
def boys_pref_green := 3 * boys / 10
def boys_pref_orange := boys - boys_pref_red - boys_pref_green

-- Proof statement
theorem color_preference :
  girls_pref_pink = 10 ∧
  girls_pref_purple = 12 ∧
  girls_pref_blue = 8 ∧
  boys_pref_red = 8 ∧
  boys_pref_green = 6 ∧
  boys_pref_orange = 6 :=
by
  sorry

end color_preference_l2295_229521


namespace tree_growth_per_two_weeks_l2295_229569

-- Definitions based on conditions
def initial_height_meters : ℕ := 2
def initial_height_centimeters : ℕ := initial_height_meters * 100
def final_height_centimeters : ℕ := 600
def total_growth : ℕ := final_height_centimeters - initial_height_centimeters
def weeks_in_4_months : ℕ := 16
def number_of_two_week_periods : ℕ := weeks_in_4_months / 2

-- Objective: Prove that the growth every two weeks is 50 centimeters
theorem tree_growth_per_two_weeks :
  (total_growth / number_of_two_week_periods) = 50 :=
  by
  sorry

end tree_growth_per_two_weeks_l2295_229569


namespace christine_wander_time_l2295_229516

-- Definitions based on conditions
def distance : ℝ := 50.0
def speed : ℝ := 6.0

-- The statement to prove
theorem christine_wander_time : (distance / speed) = 8 + 20/60 :=
by
  sorry

end christine_wander_time_l2295_229516


namespace find_least_number_l2295_229517

theorem find_least_number (x : ℕ) :
  (∀ k, 24 ∣ k + 7 → 32 ∣ k + 7 → 36 ∣ k + 7 → 54 ∣ k + 7 → x = k) → 
  x + 7 = Nat.lcm (Nat.lcm (Nat.lcm 24 32) 36) 54 → x = 857 :=
by
  sorry

end find_least_number_l2295_229517


namespace stewart_farm_sheep_l2295_229542

theorem stewart_farm_sheep (S H : ℕ)
  (h1 : S / H = 2 / 7)
  (h2 : H * 230 = 12880) :
  S = 16 :=
by sorry

end stewart_farm_sheep_l2295_229542


namespace find_k_l2295_229510

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end find_k_l2295_229510


namespace grayson_travels_further_l2295_229541

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l2295_229541


namespace toilet_paper_packs_needed_l2295_229591

-- Definitions based on conditions
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def weeks : ℕ := 4
def rolls_per_pack : ℕ := 12
def daily_stock : ℕ := 1

-- The main theorem statement
theorem toilet_paper_packs_needed : 
  (bathrooms * days_per_week * weeks) / rolls_per_pack = 14 := by
sorry

end toilet_paper_packs_needed_l2295_229591


namespace ratio_Pat_Mark_l2295_229504

-- Total hours charged by all three
def total_hours (P K M : ℕ) : Prop :=
  P + K + M = 144

-- Pat charged twice as much time as Kate
def pat_hours (P K : ℕ) : Prop :=
  P = 2 * K

-- Mark charged 80 hours more than Kate
def mark_hours (M K : ℕ) : Prop :=
  M = K + 80

-- The ratio of Pat's hours to Mark's hours
def ratio (P M : ℕ) : ℚ :=
  (P : ℚ) / (M : ℚ)

theorem ratio_Pat_Mark (P K M : ℕ)
  (h1 : total_hours P K M)
  (h2 : pat_hours P K)
  (h3 : mark_hours M K) :
  ratio P M = (1 : ℚ) / (3 : ℚ) :=
by
  sorry

end ratio_Pat_Mark_l2295_229504


namespace find_a_l2295_229596

def new_operation (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (b : ℝ) (h : b = 4) (h2 : new_operation a b = 10) : a = 14 := by
  have h' : new_operation a 4 = 10 := by rw [h] at h2; exact h2
  unfold new_operation at h'
  linarith

end find_a_l2295_229596


namespace negation_of_p_l2295_229544

def f (a x : ℝ) : ℝ := a * x - x - a

theorem negation_of_p :
  (¬ ∀ a > 0, a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔ (∃ a > 0, a ≠ 1 ∧ ¬ ∃ x : ℝ, f a x = 0) :=
by {
  sorry
}

end negation_of_p_l2295_229544


namespace texts_sent_total_l2295_229570

def texts_sent_on_monday_to_allison_and_brittney : Nat := 5 + 5
def texts_sent_on_tuesday_to_allison_and_brittney : Nat := 15 + 15

def total_texts_sent (texts_monday : Nat) (texts_tuesday : Nat) : Nat := texts_monday + texts_tuesday

theorem texts_sent_total :
  total_texts_sent texts_sent_on_monday_to_allison_and_brittney texts_sent_on_tuesday_to_allison_and_brittney = 40 :=
by
  sorry

end texts_sent_total_l2295_229570


namespace maximum_value_a3_b3_c3_d3_l2295_229535

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem maximum_value_a3_b3_c3_d3
  (a b c d : ℝ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 20)
  (h2 : a + b + c + d = 10) :
  max_value a b c d ≤ 500 :=
sorry

end maximum_value_a3_b3_c3_d3_l2295_229535


namespace heartsuit_fraction_l2295_229574

def heartsuit (n m : ℕ) : ℕ := n ^ 4 * m ^ 3

theorem heartsuit_fraction :
  (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 :=
by
  sorry

end heartsuit_fraction_l2295_229574


namespace circle_not_pass_second_quadrant_l2295_229530

theorem circle_not_pass_second_quadrant (a : ℝ) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ (x - a)^2 + y^2 = 4) → a ≥ 2 :=
by
  intro h
  by_contra
  sorry

end circle_not_pass_second_quadrant_l2295_229530


namespace sum_of_coefficients_evaluated_l2295_229588

theorem sum_of_coefficients_evaluated 
  (x y : ℤ) (h1 : x = 2) (h2 : y = -1)
  : (3 * x + 4 * y)^9 + (2 * x - 5 * y)^9 = 387420501 := 
by
  rw [h1, h2]
  sorry

end sum_of_coefficients_evaluated_l2295_229588


namespace man_work_days_l2295_229581

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days_l2295_229581


namespace smallest_possible_sum_l2295_229582

-- Defining the conditions for x and y.
variables (x y : ℕ)

-- We need a theorem to formalize our question with the given conditions.
theorem smallest_possible_sum (hx : x > 0) (hy : y > 0) (hne : x ≠ y) (hxy : 1/x + 1/y = 1/24) : x + y = 100 :=
by
  sorry

end smallest_possible_sum_l2295_229582


namespace rotated_line_x_intercept_l2295_229550

theorem rotated_line_x_intercept (x y : ℝ) :
  (∃ (k : ℝ), y = (3 * Real.sqrt 3 + 5) / (2 * Real.sqrt 3) * x) →
  (∃ y : ℝ, 3 * x - 5 * y + 40 = 0) →
  (∃ (x_intercept : ℝ), x_intercept = 0) := 
by
  sorry

end rotated_line_x_intercept_l2295_229550


namespace sum_possible_values_l2295_229503

theorem sum_possible_values (x : ℤ) (h : ∃ y : ℤ, y = (3 * x + 13) / (x + 6)) :
  ∃ s : ℤ, s = -2 + 8 + 2 + 4 :=
sorry

end sum_possible_values_l2295_229503


namespace project_total_hours_l2295_229556

def pat_time (k : ℕ) : ℕ := 2 * k
def mark_time (k : ℕ) : ℕ := k + 120

theorem project_total_hours (k : ℕ) (H1 : 3 * 2 * k = k + 120) :
  k + pat_time k + mark_time k = 216 :=
by
  sorry

end project_total_hours_l2295_229556


namespace large_seat_capacity_l2295_229527

-- Definition of conditions
def num_large_seats : ℕ := 7
def total_capacity_large_seats : ℕ := 84

-- Theorem to prove
theorem large_seat_capacity : total_capacity_large_seats / num_large_seats = 12 :=
by
  sorry

end large_seat_capacity_l2295_229527


namespace smallest_value_l2295_229545

theorem smallest_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (v : ℝ), (∀ x y : ℝ, 0 < x → 0 < y → v ≤ (16 / x + 108 / y + x * y)) ∧ v = 36 :=
sorry

end smallest_value_l2295_229545


namespace john_less_than_anna_l2295_229534

theorem john_less_than_anna (J A L T : ℕ) (h1 : A = 50) (h2: L = 3) (h3: T = 82) (h4: T + L = A + J) : A - J = 15 :=
by
  sorry

end john_less_than_anna_l2295_229534


namespace exists_indices_l2295_229579

open Nat List

theorem exists_indices (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (h1 : ∀ i : Fin m, a i ≤ n) (h2 : ∀ i j : Fin m, i ≤ j → a i ≤ a j)
  (h3 : ∀ j : Fin n, b j ≤ m) (h4 : ∀ i j : Fin n, i ≤ j → b i ≤ b j) :
  ∃ i : Fin m, ∃ j : Fin n, a i + i.val + 1 = b j + j.val + 1 := by
  sorry

end exists_indices_l2295_229579


namespace max_sum_abs_values_l2295_229589

-- Define the main problem in Lean
theorem max_sum_abs_values (a b c : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |a| + |b| + |c| ≤ 3 :=
by
  intros h
  sorry

end max_sum_abs_values_l2295_229589


namespace cylindrical_to_rectangular_conversion_l2295_229592

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 6 (5 * Real.pi / 4) (-3) = (-3 * Real.sqrt 2, -3 * Real.sqrt 2, -3) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l2295_229592


namespace ratio_third_second_l2295_229564

theorem ratio_third_second (k : ℝ) (x y z : ℝ) (h1 : y = 4 * x) (h2 : x = 18) (h3 : z = k * y) (h4 : (x + y + z) / 3 = 78) :
  z = 2 * y :=
by
  sorry

end ratio_third_second_l2295_229564


namespace first_complete_row_cover_l2295_229526

def is_shaded_square (n : ℕ) : ℕ := n ^ 2

def row_number (square_number : ℕ) : ℕ :=
  (square_number + 9) / 10 -- ceiling of square_number / 10

theorem first_complete_row_cover : ∃ n, ∀ r : ℕ, 1 ≤ r ∧ r ≤ 10 → ∃ k : ℕ, is_shaded_square k ≤ n ∧ row_number (is_shaded_square k) = r :=
by
  use 100
  intros r h
  sorry

end first_complete_row_cover_l2295_229526


namespace triangles_in_figure_l2295_229571

-- Definitions for the figure
def number_of_triangles : ℕ :=
  -- The number of triangles in a figure composed of a rectangle with three vertical lines and two horizontal lines
  50

-- The theorem we want to prove
theorem triangles_in_figure : number_of_triangles = 50 :=
by
  sorry

end triangles_in_figure_l2295_229571


namespace distance_is_3_l2295_229597

-- define the distance between Masha's and Misha's homes
def distance_between_homes (d : ℝ) : Prop :=
  -- Masha and Misha meet 1 kilometer from Masha's home in the first occasion
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / v_m = (d - 1) / v_i) ∧

  -- On the second occasion, Masha walked at twice her original speed,
  -- and Misha walked at half his original speed, and they met 1 kilometer away from Misha's home.
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / (2 * v_m) = 2 * (d - 1) / (0.5 * v_i))

-- The theorem to prove the distance is 3
theorem distance_is_3 : distance_between_homes 3 :=
  sorry

end distance_is_3_l2295_229597


namespace total_amount_l2295_229583

noncomputable def x_share : ℝ := 60
noncomputable def y_share : ℝ := 27
noncomputable def z_share : ℝ := 0.30 * x_share

theorem total_amount (hx : y_share = 0.45 * x_share) : x_share + y_share + z_share = 105 :=
by
  have hx_val : x_share = 27 / 0.45 := by
  { -- Proof that x_share is indeed 60 as per the given problem
    sorry }
  sorry

end total_amount_l2295_229583


namespace third_offense_percentage_increase_l2295_229593

theorem third_offense_percentage_increase 
    (base_per_5000 : ℕ)
    (goods_stolen : ℕ)
    (additional_years : ℕ)
    (total_sentence : ℕ) :
    base_per_5000 = 1 →
    goods_stolen = 40000 →
    additional_years = 2 →
    total_sentence = 12 →
    100 * (total_sentence - additional_years - goods_stolen / 5000) / (goods_stolen / 5000) = 25 :=
by
  intros h_base h_goods h_additional h_total
  sorry

end third_offense_percentage_increase_l2295_229593


namespace total_company_pay_monthly_l2295_229533

-- Define the given conditions
def hours_josh_works_daily : ℕ := 8
def days_josh_works_weekly : ℕ := 5
def weeks_josh_works_monthly : ℕ := 4
def hourly_rate_josh : ℕ := 9

-- Define Carl's working hours and rate based on the conditions
def hours_carl_works_daily : ℕ := hours_josh_works_daily - 2
def hourly_rate_carl : ℕ := hourly_rate_josh / 2

-- Calculate total hours worked monthly by Josh and Carl
def total_hours_josh_monthly : ℕ := hours_josh_works_daily * days_josh_works_weekly * weeks_josh_works_monthly
def total_hours_carl_monthly : ℕ := hours_carl_works_daily * days_josh_works_weekly * weeks_josh_works_monthly

-- Calculate monthly pay for Josh and Carl
def monthly_pay_josh : ℕ := total_hours_josh_monthly * hourly_rate_josh
def monthly_pay_carl : ℕ := total_hours_carl_monthly * hourly_rate_carl

-- Theorem to prove the total pay for both Josh and Carl in one month
theorem total_company_pay_monthly : monthly_pay_josh + monthly_pay_carl = 1980 := by
  sorry

end total_company_pay_monthly_l2295_229533


namespace arrangement_count_l2295_229584

-- Definitions corresponding to the conditions in a)
def num_students : ℕ := 8
def max_per_activity : ℕ := 5

-- Lean statement reflecting the target theorem in c)
theorem arrangement_count (n : ℕ) (max : ℕ) 
  (h1 : n = num_students)
  (h2 : max = max_per_activity) :
  ∃ total : ℕ, total = 182 :=
sorry

end arrangement_count_l2295_229584


namespace value_of_transformed_product_of_roots_l2295_229552

theorem value_of_transformed_product_of_roots 
  (a b : ℚ)
  (h1 : 3 * a^2 + 4 * a - 7 = 0)
  (h2 : 3 * b^2 + 4 * b - 7 = 0)
  (h3 : a ≠ b) : 
  (a - 2) * (b - 2) = 13 / 3 :=
by
  -- The exact proof would be completed here.
  sorry

end value_of_transformed_product_of_roots_l2295_229552


namespace simple_interest_rate_problem_l2295_229563

noncomputable def simple_interest_rate (P : ℝ) (T : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - P) * 100 / (P * T)

theorem simple_interest_rate_problem
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : T = 2)
  (h2 : final_amount = (7 / 6) * P)
  (h3 : simple_interest_rate P T final_amount = R) : 
  R = 100 / 12 := sorry

end simple_interest_rate_problem_l2295_229563


namespace find_a_l2295_229513

theorem find_a (α β : ℝ) (h1 : α + β = 10) (h2 : α * β = 20) : (1 / α + 1 / β) = 1 / 2 :=
sorry

end find_a_l2295_229513


namespace max_clouds_crossed_by_plane_l2295_229507

-- Define the conditions
def plane_region_divide (num_planes : ℕ) : ℕ :=
  num_planes + 1

-- Hypotheses/Conditions
variable (num_planes : ℕ)
variable (initial_region_clouds : ℕ)
variable (max_crosses : ℕ)

-- The primary statement to be proved
theorem max_clouds_crossed_by_plane : 
  num_planes = 10 → initial_region_clouds = 1 → max_crosses = num_planes + initial_region_clouds →
  max_crosses = 11 := 
by
  -- Placeholder for the actual proof
  intros
  sorry

end max_clouds_crossed_by_plane_l2295_229507


namespace opposite_face_A_is_E_l2295_229575

-- Axiomatically defining the basic conditions from the problem statement.

-- We have six labels for the faces of a net
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def adjacent (x y : Face) : Prop :=
  (x = A ∧ y = B) ∨ (x = A ∧ y = D) ∨ (x = B ∧ y = A) ∨ (x = D ∧ y = A)

-- Define the "not directly attached" relationship
def not_adjacent (x y : Face) : Prop :=
  ¬adjacent x y

-- Given the conditions in the problem statement
axiom condition1 : adjacent A B
axiom condition2 : adjacent A D
axiom condition3 : not_adjacent A E

-- The proof objective is to show that E is the face opposite to A
theorem opposite_face_A_is_E : ∃ (F : Face), 
  (∀ x : Face, adjacent A x ∨ not_adjacent A x) → (∀ y : Face, adjacent A y ↔ y ≠ E) → E = F :=
sorry

end opposite_face_A_is_E_l2295_229575


namespace M_gt_N_l2295_229553

-- Define the variables and conditions
variables (a : ℝ)
def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

-- Statement to prove
theorem M_gt_N : M a > N a := by
  -- Placeholder for the actual proof
  sorry

end M_gt_N_l2295_229553


namespace blue_chairs_fewer_than_yellow_l2295_229531

theorem blue_chairs_fewer_than_yellow :
  ∀ (red_chairs yellow_chairs chairs_left total_chairs blue_chairs : ℕ),
    red_chairs = 4 →
    yellow_chairs = 2 * red_chairs →
    chairs_left = 15 →
    total_chairs = chairs_left + 3 →
    blue_chairs = total_chairs - (red_chairs + yellow_chairs) →
    yellow_chairs - blue_chairs = 2 :=
by sorry

end blue_chairs_fewer_than_yellow_l2295_229531


namespace distance_to_base_is_42_l2295_229580

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42_l2295_229580


namespace find_a_l2295_229506

noncomputable def A (a : ℝ) : Set ℝ := {1, 2, a}
noncomputable def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem find_a (a : ℝ) : A a ⊇ B a → a = -1 ∨ a = 0 :=
by
  sorry

end find_a_l2295_229506


namespace cos_three_theta_l2295_229585

open Complex

theorem cos_three_theta (θ : ℝ) (h : cos θ = 1 / 2) : cos (3 * θ) = -1 / 2 :=
by
  sorry

end cos_three_theta_l2295_229585


namespace find_A_l2295_229524

variable (A B x : ℝ)
variable (hB : B ≠ 0)
variable (h : f (g 2) = 0)
def f := λ x => A * x^3 - B
def g := λ x => B * x^2

theorem find_A (hB : B ≠ 0) (h : (λ x => A * x^3 - B) ((λ x => B * x^2) 2) = 0) : 
  A = 1 / (64 * B^2) :=
  sorry

end find_A_l2295_229524


namespace sum_of_corners_10x10_l2295_229546

theorem sum_of_corners_10x10 : 
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  (top_left + top_right + bottom_left + bottom_right) = 202 :=
by
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  show top_left + top_right + bottom_left + bottom_right = 202
  sorry

end sum_of_corners_10x10_l2295_229546


namespace contractor_total_engaged_days_l2295_229557

-- Definitions based on conditions
def earnings_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_earnings : ℝ := 425
def days_absent : ℝ := 10

-- The proof problem statement
theorem contractor_total_engaged_days :
  ∃ (x y : ℝ), y = days_absent ∧ total_earnings = earnings_per_work_day * x - fine_per_absent_day * y ∧ x + y = 30 :=
by
  -- let x be the number of working days
  -- let y be the number of absent days
  -- y is given as 10
  -- total_earnings = 25 * x - 7.5 * 10
  -- solve for x and sum x and y to get 30
  sorry

end contractor_total_engaged_days_l2295_229557


namespace least_divisor_for_perfect_square_l2295_229523

theorem least_divisor_for_perfect_square : 
  ∃ d : ℕ, (∀ n : ℕ, n > 0 → 16800 / d = n * n) ∧ d = 21 := 
sorry

end least_divisor_for_perfect_square_l2295_229523


namespace total_amount_in_account_after_two_years_l2295_229511

-- Initial definitions based on conditions in the problem
def initial_investment : ℝ := 76800
def annual_interest_rate : ℝ := 0.125
def annual_contribution : ℝ := 5000

-- Function to calculate amount after n years with annual contributions
def total_amount_after_years (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) : ℝ :=
  let rec helper (P : ℝ) (n : ℕ) :=
    if n = 0 then P
    else 
      let previous_amount := helper P (n - 1)
      (previous_amount * (1 + r) + A)
  helper P n

-- Theorem to prove the final total amount after 2 years
theorem total_amount_in_account_after_two_years :
  total_amount_after_years initial_investment annual_interest_rate annual_contribution 2 = 107825 :=
  by 
  -- proof goes here
  sorry

end total_amount_in_account_after_two_years_l2295_229511


namespace part1_part2_l2295_229532

-- Definitions
def p (t : ℝ) := ∀ x : ℝ, x^2 + 2 * x + 2 * t - 4 ≠ 0
def q (t : ℝ) := (4 - t > 0) ∧ (t - 2 > 0)

-- Theorem statements
theorem part1 (t : ℝ) (hp : p t) : t > 5 / 2 := sorry

theorem part2 (t : ℝ) (h : p t ∨ q t) (h_and : ¬ (p t ∧ q t)) : (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) := sorry

end part1_part2_l2295_229532


namespace a2017_value_l2295_229590

def seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n / (a n + 1)

theorem a2017_value :
  ∃ (a : ℕ → ℝ),
  seq a ∧ a 1 = 1 / 2 ∧ a 2017 = 1 / 2018 :=
by
  sorry

end a2017_value_l2295_229590


namespace peasant_initial_money_l2295_229562

theorem peasant_initial_money :
  ∃ (x1 x2 x3 : ℕ), 
    (x1 / 2 + 1 = x2) ∧ 
    (x2 / 2 + 2 = x3) ∧ 
    (x3 / 2 + 1 = 0) ∧ 
    x1 = 18 := 
by
  sorry

end peasant_initial_money_l2295_229562


namespace sequence_properties_l2295_229518

open BigOperators

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) := ∃ q > 0, ∀ n, a (n + 1) = a n * q
def sequence_a (n : ℕ) : ℝ := 2^(n - 1)

-- Definitions for b_n and S_n
def sequence_b (n : ℕ) : ℕ := n - 1
def sequence_c (n : ℕ) : ℝ := sequence_a n * (sequence_b n) -- c_n = a_n * b_n

-- Statement of the problem
theorem sequence_properties (a : ℕ → ℝ) (hgeo : is_geometric_sequence a) (h1 : a 1 = 1) (h2 : a 2 * a 4 = 16) : 
 (∀ n, sequence_b n = n - 1 ) ∧ S_n = ∑ i in Finset.range n, sequence_c (i + 1) := sorry

end sequence_properties_l2295_229518


namespace diameter_of_circular_ground_l2295_229512

noncomputable def radius_of_garden_condition (area_garden : ℝ) (broad_garden : ℝ) : ℝ :=
  let pi_val := Real.pi
  (area_garden / pi_val - broad_garden * broad_garden) / (2 * broad_garden)

-- Given conditions
variable (area_garden : ℝ := 226.19467105846502)
variable (broad_garden : ℝ := 2)

-- Goal to prove: diameter of the circular ground is 34 metres
theorem diameter_of_circular_ground : 2 * radius_of_garden_condition area_garden broad_garden = 34 :=
  sorry

end diameter_of_circular_ground_l2295_229512


namespace gcd_840_1764_l2295_229508

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2295_229508


namespace simplify_expression_l2295_229577

theorem simplify_expression : 
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = 1 / 120 := 
by 
  sorry

end simplify_expression_l2295_229577


namespace inequality_proof_l2295_229538

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + 
  c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 2 / 2) :=
by
  sorry

end inequality_proof_l2295_229538


namespace sum_of_first_3n_terms_l2295_229522

def arithmetic_geometric_sequence (n : ℕ) (s : ℕ → ℕ) :=
  (s n = 10) ∧ (s (2 * n) = 30)

theorem sum_of_first_3n_terms (n : ℕ) (s : ℕ → ℕ) :
  arithmetic_geometric_sequence n s → s (3 * n) = 70 :=
by
  intro h
  sorry

end sum_of_first_3n_terms_l2295_229522


namespace inequality_proof_l2295_229561

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem inequality_proof :
  (1 + a) / (1 - a) + (1 + b) / (1 - a) + (1 + c) / (1 - c) ≤ 2 * ((b / a) + (c / b) + (a / c)) :=
by sorry

end inequality_proof_l2295_229561


namespace solution_set_correct_l2295_229528

theorem solution_set_correct (a b c : ℝ) (h : a < 0) (h1 : ∀ x, (ax^2 + bx + c < 0) ↔ ((x < 1) ∨ (x > 3))) :
  ∀ x, (cx^2 + bx + a > 0) ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end solution_set_correct_l2295_229528


namespace teamAPointDifferenceTeamB_l2295_229536

-- Definitions for players' scores and penalties
structure Player where
  name : String
  points : ℕ
  penalties : List ℕ

def TeamA : List Player := [
  { name := "Beth", points := 12, penalties := [1, 2] },
  { name := "Jan", points := 18, penalties := [1, 2, 3] },
  { name := "Mike", points := 5, penalties := [] },
  { name := "Kim", points := 7, penalties := [1, 2] },
  { name := "Chris", points := 6, penalties := [1] }
]

def TeamB : List Player := [
  { name := "Judy", points := 10, penalties := [1, 2] },
  { name := "Angel", points := 9, penalties := [1] },
  { name := "Nick", points := 12, penalties := [] },
  { name := "Steve", points := 8, penalties := [1, 2, 3] },
  { name := "Mary", points := 5, penalties := [1, 2] },
  { name := "Vera", points := 4, penalties := [1] }
]

-- Helper function to calculate total points for a player considering penalties
def Player.totalPoints (p : Player) : ℕ :=
  p.points - p.penalties.sum

-- Helper function to calculate total points for a team
def totalTeamPoints (team : List Player) : ℕ :=
  team.foldr (λ p acc => acc + p.totalPoints) 0

def teamAPoints : ℕ := totalTeamPoints TeamA
def teamBPoints : ℕ := totalTeamPoints TeamB

theorem teamAPointDifferenceTeamB :
  teamAPoints - teamBPoints = 1 :=
  sorry

end teamAPointDifferenceTeamB_l2295_229536


namespace find_power_of_7_l2295_229578

theorem find_power_of_7 :
  (7^(1/4)) / (7^(1/6)) = 7^(1/12) :=
by
  sorry

end find_power_of_7_l2295_229578


namespace kate_average_speed_correct_l2295_229586

noncomputable def kate_average_speed : ℝ :=
  let biking_time_hours := 20 / 60
  let walking_time_hours := 60 / 60
  let jogging_time_hours := 40 / 60
  let biking_distance := 20 * biking_time_hours
  let walking_distance := 4 * walking_time_hours
  let jogging_distance := 6 * jogging_time_hours
  let total_distance := biking_distance + walking_distance + jogging_distance
  let total_time_hours := biking_time_hours + walking_time_hours + jogging_time_hours
  total_distance / total_time_hours

theorem kate_average_speed_correct : kate_average_speed = 9 :=
by
  sorry

end kate_average_speed_correct_l2295_229586


namespace burn_time_for_structure_l2295_229509

noncomputable def time_to_burn_structure (total_toothpicks : ℕ) (burn_time_per_toothpick : ℕ) (adjacent_corners : Bool) : ℕ :=
  if total_toothpicks = 38 ∧ burn_time_per_toothpick = 10 ∧ adjacent_corners = true then 65 else 0

theorem burn_time_for_structure :
  time_to_burn_structure 38 10 true = 65 :=
sorry

end burn_time_for_structure_l2295_229509


namespace total_books_l2295_229558

-- Definitions for the conditions
def SandyBooks : Nat := 10
def BennyBooks : Nat := 24
def TimBooks : Nat := 33

-- Stating the theorem we need to prove
theorem total_books : SandyBooks + BennyBooks + TimBooks = 67 := by
  sorry

end total_books_l2295_229558


namespace sum_of_roots_l2295_229572

theorem sum_of_roots :
  let a := 1
  let b := 10
  let c := -25
  let sum_of_roots := -b / a
  (∀ x, 25 - 10 * x - x ^ 2 = 0 ↔ x ^ 2 + 10 * x - 25 = 0) →
  sum_of_roots = -10 :=
by
  intros
  sorry

end sum_of_roots_l2295_229572


namespace magnitude_difference_l2295_229501

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 2) (norm_b : ‖b‖ = 1) (norm_a_plus_b : ‖a + b‖ = Real.sqrt 3)

theorem magnitude_difference :
  ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end magnitude_difference_l2295_229501


namespace sufficient_but_not_necessary_decreasing_l2295_229543

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f y ≤ f x

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6 * m * x + 6

theorem sufficient_but_not_necessary_decreasing (m : ℝ) :
  m = 1 → is_decreasing_on (f m) (Set.Iic 3) :=
by
  intros h
  rw [h]
  sorry

end sufficient_but_not_necessary_decreasing_l2295_229543


namespace domain_of_f_l2295_229551

theorem domain_of_f : 
  ∀ x, (2 - x ≥ 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x ≤ 2) := by
  sorry

end domain_of_f_l2295_229551


namespace tissue_pallets_ratio_l2295_229537

-- Define the total number of pallets received
def total_pallets : ℕ := 20

-- Define the number of pallets of each type
def paper_towels_pallets : ℕ := total_pallets / 2
def paper_plates_pallets : ℕ := total_pallets / 5
def paper_cups_pallets : ℕ := 1

-- Calculate the number of pallets of tissues
def tissues_pallets : ℕ := total_pallets - (paper_towels_pallets + paper_plates_pallets + paper_cups_pallets)

-- Prove the ratio of pallets of tissues to total pallets is 1/4
theorem tissue_pallets_ratio : (tissues_pallets : ℚ) / total_pallets = 1 / 4 :=
by
  -- Proof goes here
  sorry

end tissue_pallets_ratio_l2295_229537


namespace determine_k_a_l2295_229559

theorem determine_k_a (k a : ℝ) (h : k - a ≠ 0) : (k = 0 ∧ a = 1 / 2) ↔ 
  (∀ x : ℝ, (x + 2) / (kx - ax - 1) = x → x = -2) :=
by
  sorry

end determine_k_a_l2295_229559
