import Mathlib

namespace NUMINAMATH_GPT_root_quadratic_sum_product_l169_16963

theorem root_quadratic_sum_product (x1 x2 : ℝ) (h1 : x1^2 - 2 * x1 - 5 = 0) (h2 : x2^2 - 2 * x2 - 5 = 0) 
  (h3 : x1 ≠ x2) : (x1 + x2 + 3 * (x1 * x2)) = -13 := 
by 
  sorry

end NUMINAMATH_GPT_root_quadratic_sum_product_l169_16963


namespace NUMINAMATH_GPT_division_result_l169_16987

theorem division_result :
  3486 / 189 = 18.444444444444443 := by
  sorry

end NUMINAMATH_GPT_division_result_l169_16987


namespace NUMINAMATH_GPT_problem1_problem2_l169_16949

variable {A B C : ℝ} {AC BC : ℝ}

-- Condition: BC = 2AC
def condition1 (AC BC : ℝ) : Prop := BC = 2 * AC

-- Problem 1: Prove 4cos^2(B) - cos^2(A) = 3
theorem problem1 (h : condition1 AC BC) :
  4 * Real.cos B ^ 2 - Real.cos A ^ 2 = 3 :=
sorry

-- Problem 2: Prove the maximum value of (sin(A) / (2cos(B) + cos(A))) is 2/3 for A ∈ (0, π)
theorem problem2 (h : condition1 AC BC) (hA : 0 < A ∧ A < Real.pi) :
  ∃ t : ℝ, (t = Real.sin A / (2 * Real.cos B + Real.cos A) ∧ t ≤ 2/3) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l169_16949


namespace NUMINAMATH_GPT_lottery_probability_prizes_l169_16994

theorem lottery_probability_prizes :
  let total_tickets := 3
  let first_prize_tickets := 1
  let second_prize_tickets := 1
  let non_prize_tickets := 1
  let person_a_wins_first := (2 / 3 : ℝ)
  let person_b_wins_from_remaining := (1 / 2 : ℝ)
  (2 / 3 * 1 / 2) = (1 / 3 : ℝ) := sorry

end NUMINAMATH_GPT_lottery_probability_prizes_l169_16994


namespace NUMINAMATH_GPT_range_of_smallest_nonprime_with_condition_l169_16900

def smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 : ℕ :=
121

theorem range_of_smallest_nonprime_with_condition :
  120 < smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ∧ 
  smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ≤ 130 :=
by
  unfold smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_range_of_smallest_nonprime_with_condition_l169_16900


namespace NUMINAMATH_GPT_num_digits_sum_l169_16990

theorem num_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let num1 := 9643
  let num2 := A * 10 ^ 2 + 7 * 10 + 5
  let num3 := 5 * 10 ^ 2 + B * 10 + 2
  let sum := num1 + num2 + num3
  10^4 ≤ sum ∧ sum < 10^5 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_digits_sum_l169_16990


namespace NUMINAMATH_GPT_union_of_A_and_B_l169_16916

namespace SetUnionProof

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | x ≤ 2 }
def C : Set ℝ := { x | x ≤ 2 }

theorem union_of_A_and_B : A ∪ B = C := by
  -- proof goes here
  sorry

end SetUnionProof

end NUMINAMATH_GPT_union_of_A_and_B_l169_16916


namespace NUMINAMATH_GPT_evaluate_expression_l169_16962

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) :
  a^2 * b^3 * c = 5 / 256 :=
by
  rw [ha, hb, hc]
  norm_num

end NUMINAMATH_GPT_evaluate_expression_l169_16962


namespace NUMINAMATH_GPT_smallest_integer_proof_l169_16980

def smallest_integer_condition (n : ℤ) : Prop := n^2 - 15 * n + 56 ≤ 0

theorem smallest_integer_proof :
  ∃ n : ℤ, smallest_integer_condition n ∧ ∀ m : ℤ, smallest_integer_condition m → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_integer_proof_l169_16980


namespace NUMINAMATH_GPT_boys_count_l169_16937

-- Define the number of girls
def girls : ℕ := 635

-- Define the number of boys as being 510 more than the number of girls
def boys : ℕ := girls + 510

-- Prove that the number of boys in the school is 1145
theorem boys_count : boys = 1145 := by
  sorry

end NUMINAMATH_GPT_boys_count_l169_16937


namespace NUMINAMATH_GPT_combination_8_choose_2_l169_16998

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end NUMINAMATH_GPT_combination_8_choose_2_l169_16998


namespace NUMINAMATH_GPT_value_of_x_l169_16923

/-
Given the following conditions:
  x = a + 7,
  a = b + 9,
  b = c + 15,
  c = d + 25,
  d = 60,
Prove that x = 116.
-/

theorem value_of_x (a b c d x : ℤ) 
    (h1 : x = a + 7)
    (h2 : a = b + 9)
    (h3 : b = c + 15)
    (h4 : c = d + 25)
    (h5 : d = 60) : x = 116 := 
  sorry

end NUMINAMATH_GPT_value_of_x_l169_16923


namespace NUMINAMATH_GPT_sales_volume_conditions_l169_16959

noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
else if 3 < x ∧ x ≤ 5 then k * x + 7
else 0

theorem sales_volume_conditions (a k : ℝ) :
  (sales_volume 3 a k = 4) ∧ (sales_volume 5 a k = 2) ∧
  ((∃ x, 1 < x ∧ x ≤ 3 ∧ sales_volume x a k = 10) ∨ 
   (∃ x, 3 < x ∧ x ≤ 5 ∧ sales_volume x a k = 9)) :=
sorry

end NUMINAMATH_GPT_sales_volume_conditions_l169_16959


namespace NUMINAMATH_GPT_find_cubic_polynomial_l169_16909

theorem find_cubic_polynomial (a b c d : ℚ) :
  (a + b + c + d = -5) →
  (8 * a + 4 * b + 2 * c + d = -8) →
  (27 * a + 9 * b + 3 * c + d = -17) →
  (64 * a + 16 * b + 4 * c + d = -34) →
  a = -1/3 ∧ b = -1 ∧ c = -2/3 ∧ d = -3 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_cubic_polynomial_l169_16909


namespace NUMINAMATH_GPT_scaled_multiplication_l169_16992

theorem scaled_multiplication (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 :=
by
  sorry

end NUMINAMATH_GPT_scaled_multiplication_l169_16992


namespace NUMINAMATH_GPT_nancy_picked_l169_16952

variable (total_picked : ℕ) (alyssa_picked : ℕ)

-- Assuming the conditions given in the problem
def conditions := total_picked = 59 ∧ alyssa_picked = 42

-- Proving that Nancy picked 17 pears
theorem nancy_picked : conditions total_picked alyssa_picked → total_picked - alyssa_picked = 17 := by
  sorry

end NUMINAMATH_GPT_nancy_picked_l169_16952


namespace NUMINAMATH_GPT_A_left_after_3_days_l169_16966

def work_done_by_A_and_B_together (x : ℕ) : ℚ :=
  (1 / 21) * x + (1 / 28) * x

def work_done_by_B_alone (days : ℕ) : ℚ :=
  (1 / 28) * days

def total_work_done (x days_b_alone : ℕ) : ℚ :=
  work_done_by_A_and_B_together x + work_done_by_B_alone days_b_alone

theorem A_left_after_3_days :
  ∀ (x : ℕ), total_work_done x 21 = 1 ↔ x = 3 := by
  sorry

end NUMINAMATH_GPT_A_left_after_3_days_l169_16966


namespace NUMINAMATH_GPT_total_money_in_wallet_l169_16961

-- Definitions of conditions
def initial_five_dollar_bills := 7
def initial_ten_dollar_bills := 1
def initial_twenty_dollar_bills := 3
def initial_fifty_dollar_bills := 1
def initial_one_dollar_coins := 8

def spent_groceries := 65
def paid_fifty_dollar_bill := 1
def paid_twenty_dollar_bill := 1
def received_five_dollar_bill_change := 1
def received_one_dollar_coin_change := 5

def received_twenty_dollar_bills_from_friend := 2
def received_one_dollar_bills_from_friend := 2

-- Proving total amount of money
theorem total_money_in_wallet : 
  initial_five_dollar_bills * 5 + 
  initial_ten_dollar_bills * 10 + 
  initial_twenty_dollar_bills * 20 + 
  initial_fifty_dollar_bills * 50 + 
  initial_one_dollar_coins * 1 - 
  spent_groceries + 
  received_five_dollar_bill_change * 5 + 
  received_one_dollar_coin_change * 1 + 
  received_twenty_dollar_bills_from_friend * 20 + 
  received_one_dollar_bills_from_friend * 1 
  = 150 := 
by
  -- This is where the proof would be located
  sorry

end NUMINAMATH_GPT_total_money_in_wallet_l169_16961


namespace NUMINAMATH_GPT_sum_divisible_by_4_l169_16947

theorem sum_divisible_by_4 (a b c d x : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9) : 4 ∣ (a + b + c + d) :=
by
  sorry

end NUMINAMATH_GPT_sum_divisible_by_4_l169_16947


namespace NUMINAMATH_GPT_total_value_of_item_l169_16942

theorem total_value_of_item (V : ℝ) 
  (h1 : ∃ V > 1000, 
              0.07 * (V - 1000) + 
              (if 55 > 50 then (55 - 50) * 0.15 else 0) + 
              0.05 * V = 112.70) :
  V = 1524.58 :=
by 
  sorry

end NUMINAMATH_GPT_total_value_of_item_l169_16942


namespace NUMINAMATH_GPT_roots_in_interval_l169_16984

theorem roots_in_interval (P : Polynomial ℝ) (h : ∀ i, P.coeff i = 1 ∨ P.coeff i = 0 ∨ P.coeff i = -1) : 
  ∀ x : ℝ, P.eval x = 0 → -2 ≤ x ∧ x ≤ 2 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_roots_in_interval_l169_16984


namespace NUMINAMATH_GPT_find_smaller_number_l169_16910

theorem find_smaller_number (a b : ℕ) (h1 : b = 2 * a - 3) (h2 : a + b = 39) : a = 14 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_smaller_number_l169_16910


namespace NUMINAMATH_GPT_processing_plant_growth_eq_l169_16958

-- Definition of the conditions given in the problem
def initial_amount : ℝ := 10
def november_amount : ℝ := 13
def growth_rate (x : ℝ) : ℝ := initial_amount * (1 + x)^2

-- Lean theorem statement to prove the equation
theorem processing_plant_growth_eq (x : ℝ) : 
  growth_rate x = november_amount ↔ initial_amount * (1 + x)^2 = 13 := 
by
  sorry

end NUMINAMATH_GPT_processing_plant_growth_eq_l169_16958


namespace NUMINAMATH_GPT_ellipse_eq_find_k_l169_16903

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eq_find_k_l169_16903


namespace NUMINAMATH_GPT_num_accompanying_year_2022_l169_16997

theorem num_accompanying_year_2022 : 
  ∃ N : ℤ, (N = 2) ∧ 
    (∀ n : ℤ, (100 * n + 22) % n = 0 ∧ 10 ≤ n ∧ n < 100 → n = 11 ∨ n = 22) :=
by 
  sorry

end NUMINAMATH_GPT_num_accompanying_year_2022_l169_16997


namespace NUMINAMATH_GPT_fraction_division_l169_16985

theorem fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l169_16985


namespace NUMINAMATH_GPT_total_food_per_day_l169_16934

def num_dogs : ℝ := 2
def food_per_dog_per_day : ℝ := 0.12

theorem total_food_per_day : (num_dogs * food_per_dog_per_day) = 0.24 :=
by sorry

end NUMINAMATH_GPT_total_food_per_day_l169_16934


namespace NUMINAMATH_GPT_ribbon_cost_l169_16988

variable (c_g c_m s : ℝ)

theorem ribbon_cost (h1 : 5 * c_g + s = 295) (h2 : 7 * c_m + s = 295) (h3 : 2 * c_m + c_g = 102) : s = 85 :=
sorry

end NUMINAMATH_GPT_ribbon_cost_l169_16988


namespace NUMINAMATH_GPT_bob_should_give_l169_16977

theorem bob_should_give (alice_paid bob_paid charlie_paid : ℕ)
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_charlie : charlie_paid = 180) :
  bob_paid - (120 + 150 + 180) / 3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_bob_should_give_l169_16977


namespace NUMINAMATH_GPT_wickets_before_last_match_l169_16914

theorem wickets_before_last_match (W : ℕ) (avg_before : ℝ) (wickets_taken : ℕ) (runs_conceded : ℝ) (avg_drop : ℝ) :
  avg_before = 12.4 → wickets_taken = 4 → runs_conceded = 26 → avg_drop = 0.4 →
  (avg_before - avg_drop) * (W + wickets_taken) = avg_before * W + runs_conceded →
  W = 55 :=
by
  intros
  sorry

end NUMINAMATH_GPT_wickets_before_last_match_l169_16914


namespace NUMINAMATH_GPT_complex_fraction_simplification_l169_16911

open Complex

theorem complex_fraction_simplification : (1 + 2 * I) / (1 - I)^2 = 1 - 1 / 2 * I :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l169_16911


namespace NUMINAMATH_GPT_parameterized_line_solution_l169_16927

theorem parameterized_line_solution :
  ∃ (s l : ℚ), 
  (∀ t : ℚ, 
    ∃ x y : ℚ, 
      x = -3 + t * l ∧ 
      y = s + t * (-7) ∧ 
      y = 3 * x + 2
  ) ∧
  s = -7 ∧ l = -7 / 3 := 
sorry

end NUMINAMATH_GPT_parameterized_line_solution_l169_16927


namespace NUMINAMATH_GPT_fraction_eval_l169_16920

theorem fraction_eval : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = (84 / 35) :=
by
  sorry

end NUMINAMATH_GPT_fraction_eval_l169_16920


namespace NUMINAMATH_GPT_bacon_vs_tomatoes_l169_16933

theorem bacon_vs_tomatoes :
  let (n_b : ℕ) := 337
  let (n_t : ℕ) := 23
  n_b - n_t = 314 := by
  let n_b := 337
  let n_t := 23
  have h1 : n_b = 337 := rfl
  have h2 : n_t = 23 := rfl
  sorry

end NUMINAMATH_GPT_bacon_vs_tomatoes_l169_16933


namespace NUMINAMATH_GPT_boat_speed_upstream_l169_16925

noncomputable def V_b : ℝ := 11
noncomputable def V_down : ℝ := 15
noncomputable def V_s : ℝ := V_down - V_b
noncomputable def V_up : ℝ := V_b - V_s

theorem boat_speed_upstream :
  V_up = 7 := by
  sorry

end NUMINAMATH_GPT_boat_speed_upstream_l169_16925


namespace NUMINAMATH_GPT_number_of_occupied_cars_l169_16940

theorem number_of_occupied_cars (k : ℕ) (x y : ℕ) :
  18 * k / 9 = 2 * k → 
  3 * x + 2 * y = 12 → 
  x + y ≤ 18 → 
  18 - x - y = 13 :=
by sorry

end NUMINAMATH_GPT_number_of_occupied_cars_l169_16940


namespace NUMINAMATH_GPT_sum_of_angles_l169_16975

theorem sum_of_angles (x u v : ℝ) (h1 : u = Real.sin x) (h2 : v = Real.cos x)
  (h3 : 0 ≤ x ∧ x ≤ 2 * Real.pi) 
  (h4 : Real.sin x ^ 4 - Real.cos x ^ 4 = (u - v) / (u * v)) 
  : x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 → (Real.pi / 4 + 5 * Real.pi / 4) = 3 * Real.pi / 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_angles_l169_16975


namespace NUMINAMATH_GPT_freeze_alcohol_time_l169_16951

theorem freeze_alcohol_time :
  ∀ (init_temp freeze_temp : ℝ)
    (cooling_rate : ℝ), 
    init_temp = 12 → 
    freeze_temp = -117 → 
    cooling_rate = 1.5 →
    (freeze_temp - init_temp) / cooling_rate = -129 / cooling_rate :=
by
  intros init_temp freeze_temp cooling_rate h1 h2 h3
  rw [h2, h1, h3]
  exact sorry

end NUMINAMATH_GPT_freeze_alcohol_time_l169_16951


namespace NUMINAMATH_GPT_kate_needs_more_money_for_trip_l169_16918

theorem kate_needs_more_money_for_trip:
  let kate_money_base6 := 3 * 6^3 + 2 * 6^2 + 4 * 6^1 + 2 * 6^0
  let ticket_cost := 1000
  kate_money_base6 - ticket_cost = -254 :=
by
  -- Proving the theorem, steps will go here.
  sorry

end NUMINAMATH_GPT_kate_needs_more_money_for_trip_l169_16918


namespace NUMINAMATH_GPT_compute_sqrt_fraction_l169_16953

theorem compute_sqrt_fraction :
  (Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35))) = (256 / Real.sqrt 2049) :=
sorry

end NUMINAMATH_GPT_compute_sqrt_fraction_l169_16953


namespace NUMINAMATH_GPT_kristy_initial_cookies_l169_16930

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_kristy_initial_cookies_l169_16930


namespace NUMINAMATH_GPT_calc_angle_CAB_l169_16921

theorem calc_angle_CAB (α β γ ε : ℝ) (hα : α = 79) (hβ : β = 63) (hγ : γ = 131) (hε : ε = 123.5) : 
  ∃ φ : ℝ, φ = 24 + 52 / 60 :=
by
  sorry

end NUMINAMATH_GPT_calc_angle_CAB_l169_16921


namespace NUMINAMATH_GPT_sufficient_condition_for_P_l169_16915

noncomputable def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem sufficient_condition_for_P (f : ℝ → ℝ) (t : ℝ) 
  (h_inc : increasing f) (h_val1 : f (-1) = -4) (h_val2 : f 2 = 2) :
  (∀ x, (x ∈ {x | -1 - t < x ∧ x < 2 - t}) → x < -1) → t ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_P_l169_16915


namespace NUMINAMATH_GPT_map_distance_l169_16946

theorem map_distance
  (s d_m : ℝ) (d_r : ℝ)
  (h1 : s = 0.4)
  (h2 : d_r = 5.3)
  (h3 : d_m = 64) :
  (d_m * d_r / s) = 848 := by
  sorry

end NUMINAMATH_GPT_map_distance_l169_16946


namespace NUMINAMATH_GPT_total_cost_l169_16924

-- Definition: Cost of first 100 notebooks
def cost_first_100_notebooks : ℕ := 230

-- Definition: Cost per notebook beyond the first 100 notebooks
def cost_additional_notebooks (n : ℕ) : ℕ := n * 2

-- Theorem: Total cost given a > 100 notebooks
theorem total_cost (a : ℕ) (h : a > 100) : (cost_first_100_notebooks + cost_additional_notebooks (a - 100) = 2 * a + 30) := by
  sorry

end NUMINAMATH_GPT_total_cost_l169_16924


namespace NUMINAMATH_GPT_observed_wheels_l169_16902

theorem observed_wheels (num_cars wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_GPT_observed_wheels_l169_16902


namespace NUMINAMATH_GPT_well_rate_correct_l169_16917

noncomputable def well_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  total_cost / volume

theorem well_rate_correct :
  well_rate 14 3 1583.3626974092558 = 15.993 :=
by
  sorry

end NUMINAMATH_GPT_well_rate_correct_l169_16917


namespace NUMINAMATH_GPT_royalty_amount_l169_16922

-- Define the conditions and the question proof.
theorem royalty_amount (x : ℝ) :
  (800 ≤ x ∧ x ≤ 4000 → (x - 800) * 0.14 = 420) ∧
  (x > 4000 → x * 0.11 = 420) ∧
  420 = 420 →
  x = 3800 :=
by
  sorry

end NUMINAMATH_GPT_royalty_amount_l169_16922


namespace NUMINAMATH_GPT_evaluate_expression_l169_16928

theorem evaluate_expression : ((2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l169_16928


namespace NUMINAMATH_GPT_Randy_trip_distance_l169_16945

noncomputable def total_distance (x : ℝ) :=
  (x / 4) + 40 + 10 + (x / 6)

theorem Randy_trip_distance (x : ℝ) (h : total_distance x = x) : x = 600 / 7 :=
by
  sorry

end NUMINAMATH_GPT_Randy_trip_distance_l169_16945


namespace NUMINAMATH_GPT_starting_number_divisible_by_3_l169_16964

theorem starting_number_divisible_by_3 (x : ℕ) (h₁ : ∀ n, 1 ≤ n → n < 14 → ∃ k, x + (n - 1) * 3 = 3 * k ∧ x + (n - 1) * 3 ≤ 50) :
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_starting_number_divisible_by_3_l169_16964


namespace NUMINAMATH_GPT_john_lift_total_weight_l169_16983

-- Define the conditions as constants
def initial_weight : ℝ := 135
def weight_increase : ℝ := 265
def bracer_factor : ℝ := 6

-- Define a theorem to prove the total weight John can lift
theorem john_lift_total_weight : initial_weight + weight_increase + (initial_weight + weight_increase) * bracer_factor = 2800 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_john_lift_total_weight_l169_16983


namespace NUMINAMATH_GPT_basket_ratio_l169_16968

variable (S A H : ℕ)

theorem basket_ratio 
  (alex_baskets : A = 8) 
  (hector_baskets : H = 2 * S) 
  (total_baskets : A + S + H = 80) : 
  (S : ℚ) / (A : ℚ) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_basket_ratio_l169_16968


namespace NUMINAMATH_GPT_parametric_line_eq_l169_16979

theorem parametric_line_eq (t : ℝ) : 
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x = 3 * t + 6 → y = 5 * t - 8 → y = m * x + b)) ∧ m = 5 / 3 ∧ b = -18 :=
sorry

end NUMINAMATH_GPT_parametric_line_eq_l169_16979


namespace NUMINAMATH_GPT_total_whales_observed_l169_16955

-- Define the conditions
def trip1_male_whales : ℕ := 28
def trip1_female_whales : ℕ := 2 * trip1_male_whales
def trip1_total_whales : ℕ := trip1_male_whales + trip1_female_whales

def baby_whales_trip2 : ℕ := 8
def adult_whales_trip2 : ℕ := 2 * baby_whales_trip2
def trip2_total_whales : ℕ := baby_whales_trip2 + adult_whales_trip2

def trip3_male_whales : ℕ := trip1_male_whales / 2
def trip3_female_whales : ℕ := trip1_female_whales
def trip3_total_whales : ℕ := trip3_male_whales + trip3_female_whales

-- Prove the total number of whales observed
theorem total_whales_observed : trip1_total_whales + trip2_total_whales + trip3_total_whales = 178 := by
  -- Assuming all intermediate steps are correct
  sorry

end NUMINAMATH_GPT_total_whales_observed_l169_16955


namespace NUMINAMATH_GPT_maximum_value_40_l169_16907

theorem maximum_value_40 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2 ≤ 40 :=
sorry

end NUMINAMATH_GPT_maximum_value_40_l169_16907


namespace NUMINAMATH_GPT_julia_tag_kids_monday_l169_16919

-- Definitions based on conditions
def total_tag_kids (M T : ℕ) : Prop := M + T = 20
def tag_kids_Tuesday := 13

-- Problem statement
theorem julia_tag_kids_monday (M : ℕ) : total_tag_kids M tag_kids_Tuesday → M = 7 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_julia_tag_kids_monday_l169_16919


namespace NUMINAMATH_GPT_total_wages_of_12_men_l169_16931

variable {M W B x y : Nat}
variable {total_wages : Nat}

-- Condition 1: 12 men do the work equivalent to W women
axiom work_equivalent_1 : 12 * M = W

-- Condition 2: 12 men do the work equivalent to 20 boys
axiom work_equivalent_2 : 12 * M = 20 * B

-- Condition 3: All together earn Rs. 450
axiom total_earnings : (12 * M) + (x * (12 * M / W)) + (y * (12 * M / (20 * B))) = 450

-- The theorem to prove
theorem total_wages_of_12_men : total_wages = 12 * M → false :=
by sorry

end NUMINAMATH_GPT_total_wages_of_12_men_l169_16931


namespace NUMINAMATH_GPT_find_mean_l169_16956

noncomputable def mean_of_normal_distribution (σ : ℝ) (value : ℝ) (std_devs : ℝ) : ℝ :=
value + std_devs * σ

theorem find_mean
  (σ : ℝ := 1.5)
  (value : ℝ := 12)
  (std_devs : ℝ := 2)
  (h : value = mean_of_normal_distribution σ (value - std_devs * σ) std_devs) :
  mean_of_normal_distribution σ value std_devs = 15 :=
sorry

end NUMINAMATH_GPT_find_mean_l169_16956


namespace NUMINAMATH_GPT_continuous_stripe_probability_l169_16957

-- Definitions based on conditions from a)
def total_possible_combinations : ℕ := 4^6

def favorable_outcomes : ℕ := 12

def probability_of_continuous_stripe : ℚ := favorable_outcomes / total_possible_combinations

-- The theorem equivalent to prove the given problem
theorem continuous_stripe_probability :
  probability_of_continuous_stripe = 3 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l169_16957


namespace NUMINAMATH_GPT_jill_average_number_of_stickers_l169_16989

def average_stickers (packs : List ℕ) : ℚ :=
  (packs.sum : ℚ) / packs.length

theorem jill_average_number_of_stickers :
  average_stickers [5, 7, 9, 9, 11, 15, 15, 17, 19, 21] = 12.8 :=
by
  sorry

end NUMINAMATH_GPT_jill_average_number_of_stickers_l169_16989


namespace NUMINAMATH_GPT_certain_number_unique_l169_16972

-- Define the necessary conditions and statement
def is_certain_number (n : ℕ) : Prop :=
  (∃ k : ℕ, 25 * k = n) ∧ (∃ k : ℕ, 35 * k = n) ∧ 
  (n > 0) ∧ (∃ a b c : ℕ, 1 ≤ a * n ∧ a * n ≤ 1050 ∧ 1 ≤ b * n ∧ b * n ≤ 1050 ∧ 1 ≤ c * n ∧ c * n ≤ 1050 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem certain_number_unique :
  ∃ n : ℕ, is_certain_number n ∧ n = 350 :=
by 
  sorry

end NUMINAMATH_GPT_certain_number_unique_l169_16972


namespace NUMINAMATH_GPT_evaluate_expression_l169_16913

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l169_16913


namespace NUMINAMATH_GPT_find_k_l169_16969

noncomputable def a : ℚ := sorry -- Represents positive rational number a
noncomputable def b : ℚ := sorry -- Represents positive rational number b

def minimal_period (x : ℚ) : ℕ := sorry -- Function to determine minimal period of a rational number

-- Conditions as definitions
axiom h1 : minimal_period a = 30
axiom h2 : minimal_period b = 30
axiom h3 : minimal_period (a - b) = 15

-- Statement to prove smallest natural number k such that minimal period of (a + k * b) is 15
theorem find_k : ∃ k : ℕ, minimal_period (a + k * b) = 15 ∧ ∀ n < k, minimal_period (a + n * b) ≠ 15 :=
sorry

end NUMINAMATH_GPT_find_k_l169_16969


namespace NUMINAMATH_GPT_range_of_a_l169_16978

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 4) : -3 ≤ a ∧ a ≤ 5 := 
sorry

end NUMINAMATH_GPT_range_of_a_l169_16978


namespace NUMINAMATH_GPT_common_ratio_is_2_l169_16950

noncomputable def common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : ℝ :=
(a1 + 2 * d) / a1

theorem common_ratio_is_2 (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : 
    common_ratio a1 d h1 h2 = 2 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_common_ratio_is_2_l169_16950


namespace NUMINAMATH_GPT_ratio_of_discretionary_income_l169_16982

theorem ratio_of_discretionary_income 
  (salary : ℝ) (D : ℝ)
  (h_salary : salary = 3500)
  (h_discretionary : 0.15 * D = 105) :
  D / salary = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_discretionary_income_l169_16982


namespace NUMINAMATH_GPT_bert_fraction_spent_l169_16971

theorem bert_fraction_spent (f : ℝ) :
  let initial := 52
  let after_hardware := initial - initial * f
  let after_cleaners := after_hardware - 9
  let after_grocery := after_cleaners / 2
  let final := 15
  after_grocery = final → f = 1/4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bert_fraction_spent_l169_16971


namespace NUMINAMATH_GPT_ellipse_fence_cost_is_correct_l169_16995

noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

noncomputable def fence_cost_per_meter (rate : ℝ) (a b : ℝ) : ℝ :=
  rate * ellipse_perimeter a b

theorem ellipse_fence_cost_is_correct :
  fence_cost_per_meter 3 16 12 = 265.32 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_fence_cost_is_correct_l169_16995


namespace NUMINAMATH_GPT_a2_equals_3_l169_16906

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem a2_equals_3 (a : ℕ → ℕ) (S3 : ℕ) (h1 : a 1 = 1) (h2 : a 1 + a 2 + a 3 = 9) : a 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_a2_equals_3_l169_16906


namespace NUMINAMATH_GPT_remainder_of_sum_l169_16986

theorem remainder_of_sum (k j : ℤ) (a b : ℤ) (h₁ : a = 60 * k + 53) (h₂ : b = 45 * j + 17) : ((a + b) % 15) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l169_16986


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l169_16954

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 + a 8 = 15) 
  (h2 : a 3 * a 7 = 36) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  (a 19 / a 13 = 4) ∨ (a 19 / a 13 = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l169_16954


namespace NUMINAMATH_GPT_smallest_term_abs_l169_16973

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem smallest_term_abs {a : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 > 0)
  (hS12 : (12 / 2) * (2 * a 1 + 11 * (a 2 - a 1)) > 0)
  (hS13 : (13 / 2) * (2 * a 1 + 12 * (a 2 - a 1)) < 0) :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 13 → n ≠ 7 → abs (a 6) > abs (a 1 + 6 * (a 2 - a 1)) :=
sorry

end NUMINAMATH_GPT_smallest_term_abs_l169_16973


namespace NUMINAMATH_GPT_bamboo_tube_middle_capacity_l169_16991

-- Definitions and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem bamboo_tube_middle_capacity:
  ∃ a d, (arithmetic_sequence a d 0 + arithmetic_sequence a d 1 + arithmetic_sequence a d 2 = 3.9) ∧
         (arithmetic_sequence a d 5 + arithmetic_sequence a d 6 + arithmetic_sequence a d 7 + arithmetic_sequence a d 8 = 3) ∧
         (arithmetic_sequence a d 4 = 1) :=
sorry

end NUMINAMATH_GPT_bamboo_tube_middle_capacity_l169_16991


namespace NUMINAMATH_GPT_taeyeon_height_proof_l169_16929

noncomputable def seonghee_height : ℝ := 134.5
noncomputable def taeyeon_height : ℝ := seonghee_height * 1.06

theorem taeyeon_height_proof : taeyeon_height = 142.57 := 
by
  sorry

end NUMINAMATH_GPT_taeyeon_height_proof_l169_16929


namespace NUMINAMATH_GPT_farm_horse_food_needed_l169_16912

-- Definitions given in the problem
def sheep_count : ℕ := 16
def sheep_to_horse_ratio : ℕ × ℕ := (2, 7)
def food_per_horse_per_day : ℕ := 230

-- The statement we want to prove
theorem farm_horse_food_needed : 
  ∃ H : ℕ, (sheep_count * sheep_to_horse_ratio.2 = sheep_to_horse_ratio.1 * H) ∧ 
           (H * food_per_horse_per_day = 12880) :=
sorry

end NUMINAMATH_GPT_farm_horse_food_needed_l169_16912


namespace NUMINAMATH_GPT_kurt_less_marbles_than_dennis_l169_16943

theorem kurt_less_marbles_than_dennis
  (Laurie_marbles : ℕ)
  (Kurt_marbles : ℕ)
  (Dennis_marbles : ℕ)
  (h1 : Laurie_marbles = 37)
  (h2 : Laurie_marbles = Kurt_marbles + 12)
  (h3 : Dennis_marbles = 70) :
  Dennis_marbles - Kurt_marbles = 45 := by
  sorry

end NUMINAMATH_GPT_kurt_less_marbles_than_dennis_l169_16943


namespace NUMINAMATH_GPT_volume_of_regular_quadrilateral_pyramid_l169_16948

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  let x := 1 -- A placeholder to outline the structure
  let PM := (6 * a) / 5
  let V := (2 * a^3) / 5
  V

theorem volume_of_regular_quadrilateral_pyramid
  (a PM : ℝ)
  (h1 : PM = (6 * a) / 5)
  [InstReal : Nonempty (Real)] :
  volume_of_pyramid a = (2 * a^3) / 5 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_regular_quadrilateral_pyramid_l169_16948


namespace NUMINAMATH_GPT_car_travel_distance_l169_16941

noncomputable def distance_traveled (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := pi * diameter
  circumference * revolutions / 12 / 5280

theorem car_travel_distance
  (diameter : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 13)
  (h_revolutions : revolutions = 775.5724667489372) :
  distance_traveled diameter revolutions = 0.5 :=
by
  simp [distance_traveled, h_diameter, h_revolutions, Real.pi]
  sorry

end NUMINAMATH_GPT_car_travel_distance_l169_16941


namespace NUMINAMATH_GPT_runners_meetings_on_track_l169_16999

def number_of_meetings (speed1 speed2 laps : ℕ) : ℕ := ((speed1 + speed2) * laps) / (2 * (speed2 - speed1))

theorem runners_meetings_on_track 
  (speed1 speed2 : ℕ) 
  (start_laps : ℕ)
  (speed1_spec : speed1 = 4) 
  (speed2_spec : speed2 = 10) 
  (laps_spec : start_laps = 28) : 
  number_of_meetings speed1 speed2 start_laps = 77 := 
by
  rw [speed1_spec, speed2_spec, laps_spec]
  -- Add further necessary steps or lemmas if required to reach the final proving statement
  sorry

end NUMINAMATH_GPT_runners_meetings_on_track_l169_16999


namespace NUMINAMATH_GPT_loan_interest_rate_l169_16932

theorem loan_interest_rate (P SI T R : ℕ) (h1 : P = 900) (h2 : SI = 729) (h3 : T = R) :
  (SI = (P * R * T) / 100) -> R = 9 :=
by
  sorry

end NUMINAMATH_GPT_loan_interest_rate_l169_16932


namespace NUMINAMATH_GPT_h_h_3_eq_2915_l169_16981

def h (x : ℕ) : ℕ := 3 * x^2 + x + 1

theorem h_h_3_eq_2915 : h (h 3) = 2915 := by
  sorry

end NUMINAMATH_GPT_h_h_3_eq_2915_l169_16981


namespace NUMINAMATH_GPT_inequality_sum_l169_16904

theorem inequality_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
  (a / (a ^ 3 + b * c) + b / (b ^ 3 + c * a) + c / (c ^ 3 + a * b)) > 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_sum_l169_16904


namespace NUMINAMATH_GPT_power_is_seventeen_l169_16967

theorem power_is_seventeen (x : ℕ) : (1000^7 : ℝ) / (10^x) = (10000 : ℝ) ↔ x = 17 := by
  sorry

end NUMINAMATH_GPT_power_is_seventeen_l169_16967


namespace NUMINAMATH_GPT_third_speed_correct_l169_16936

variable (total_time : ℝ := 11)
variable (total_distance : ℝ := 900)
variable (speed1_km_hr : ℝ := 3)
variable (speed2_km_hr : ℝ := 9)

noncomputable def convert_speed_km_hr_to_m_min (speed: ℝ) : ℝ := speed * 1000 / 60

noncomputable def equal_distance : ℝ := total_distance / 3

noncomputable def third_speed_m_min : ℝ :=
  let speed1_m_min := convert_speed_km_hr_to_m_min speed1_km_hr
  let speed2_m_min := convert_speed_km_hr_to_m_min speed2_km_hr
  let d := equal_distance
  300 / (total_time - (d / speed1_m_min + d / speed2_m_min))

noncomputable def third_speed_km_hr : ℝ := third_speed_m_min * 60 / 1000

theorem third_speed_correct : third_speed_km_hr = 6 := by
  sorry

end NUMINAMATH_GPT_third_speed_correct_l169_16936


namespace NUMINAMATH_GPT_tan_alpha_value_l169_16908

variable (α : Real)
variable (h1 : Real.sin α = 4/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_value : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_value_l169_16908


namespace NUMINAMATH_GPT_potato_cost_l169_16944

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end NUMINAMATH_GPT_potato_cost_l169_16944


namespace NUMINAMATH_GPT_determine_c_l169_16901

theorem determine_c (c : ℚ) : (∀ x : ℝ, (x + 7) * (x^2 * c * x + 19 * x^2 - c * x - 49) = 0) → c = 21 / 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_c_l169_16901


namespace NUMINAMATH_GPT_total_marbles_l169_16993

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3
def Peter_marbles : ℕ := 7

theorem total_marbles : Mary_marbles + Joan_marbles + Peter_marbles = 19 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l169_16993


namespace NUMINAMATH_GPT_min_a_for_decreasing_f_l169_16965

theorem min_a_for_decreasing_f {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 - a / (2 * Real.sqrt x) ≤ 0) →
  a ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_a_for_decreasing_f_l169_16965


namespace NUMINAMATH_GPT_abs_x_lt_2_sufficient_but_not_necessary_l169_16905

theorem abs_x_lt_2_sufficient_but_not_necessary (x : ℝ) :
  (|x| < 2) → (x ^ 2 - x - 6 < 0) ∧ ¬ ((x ^ 2 - x - 6 < 0) → (|x| < 2)) := by
  sorry

end NUMINAMATH_GPT_abs_x_lt_2_sufficient_but_not_necessary_l169_16905


namespace NUMINAMATH_GPT_condition_inequality_l169_16996

theorem condition_inequality (x y : ℝ) :
  (¬ (x ≤ y → |x| ≤ |y|)) ∧ (¬ (|x| ≤ |y| → x ≤ y)) :=
by
  sorry

end NUMINAMATH_GPT_condition_inequality_l169_16996


namespace NUMINAMATH_GPT_geometric_sequence_S5_eq_11_l169_16939

theorem geometric_sequence_S5_eq_11 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (q : ℤ)
  (h1 : a 1 = 1)
  (h4 : a 4 = -8)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_S : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 5 = 11 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_geometric_sequence_S5_eq_11_l169_16939


namespace NUMINAMATH_GPT_complement_of_M_in_U_l169_16974

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U : U \ M = {2, 3, 5} := by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l169_16974


namespace NUMINAMATH_GPT_original_treadmill_price_l169_16976

-- Given conditions in Lean definitions
def discount_rate : ℝ := 0.30
def plate_cost : ℝ := 50
def num_plates : ℕ := 2
def total_paid : ℝ := 1045

noncomputable def treadmill_price :=
  let plate_total := num_plates * plate_cost
  let treadmill_discount := (1 - discount_rate)
  (total_paid - plate_total) / treadmill_discount

theorem original_treadmill_price :
  treadmill_price = 1350 := by
  sorry

end NUMINAMATH_GPT_original_treadmill_price_l169_16976


namespace NUMINAMATH_GPT_sample_size_is_59_l169_16960

def totalStudents : Nat := 295
def samplingRatio : Nat := 5

theorem sample_size_is_59 : totalStudents / samplingRatio = 59 := 
by
  sorry

end NUMINAMATH_GPT_sample_size_is_59_l169_16960


namespace NUMINAMATH_GPT_ordered_pairs_count_l169_16970

theorem ordered_pairs_count : ∃ (count : ℕ), count = 4 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≥ n → m^2 - n^2 = 144 → (∃ (i : ℕ), i < count) := by
  sorry

end NUMINAMATH_GPT_ordered_pairs_count_l169_16970


namespace NUMINAMATH_GPT_part1_part2_l169_16935

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem part1 (x : ℝ) : (∀ x, f x 2 ≤ x + 4 → (1 / 2 ≤ x ∧ x ≤ 7 / 2)) :=
by sorry

theorem part2 (x : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -5 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l169_16935


namespace NUMINAMATH_GPT_find_f_of_2_l169_16926

theorem find_f_of_2 (f g : ℝ → ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) 
                    (h₂ : ∀ x : ℝ, g x = f x + 9) (h₃ : g (-2) = 3) :
                    f 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l169_16926


namespace NUMINAMATH_GPT_markup_rate_l169_16938

variable (S : ℝ) (C : ℝ)
variable (profit_percent : ℝ := 0.12) (expense_percent : ℝ := 0.18)
variable (selling_price : ℝ := 8.00)

theorem markup_rate (h1 : C + profit_percent * S + expense_percent * S = S)
                    (h2 : S = selling_price) :
  ((S - C) / C) * 100 = 42.86 := by
  sorry

end NUMINAMATH_GPT_markup_rate_l169_16938
