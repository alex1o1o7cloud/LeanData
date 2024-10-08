import Mathlib

namespace correct_operations_l129_129628

theorem correct_operations : 6 * 3 + 4 + 2 = 24 := by
  -- Proof goes here
  sorry

end correct_operations_l129_129628


namespace abs_sum_of_factors_of_quadratic_l129_129291

variable (h b c d : ℤ)

theorem abs_sum_of_factors_of_quadratic :
  (∀ x : ℤ, 6 * x * x + x - 12 = (h * x + b) * (c * x + d)) →
  (|h| + |b| + |c| + |d| = 12) :=
by
  sorry

end abs_sum_of_factors_of_quadratic_l129_129291


namespace a_gt_b_l129_129975

variable (n : ℕ) (a b : ℝ)
variable (n_pos : n > 1) (a_pos : 0 < a) (b_pos : 0 < b)
variable (a_eqn : a^n = a + 1)
variable (b_eqn : b^{2 * n} = b + 3 * a)

theorem a_gt_b : a > b :=
by {
  -- Proof is needed here
  sorry
}

end a_gt_b_l129_129975


namespace tank_emptying_time_l129_129513

theorem tank_emptying_time
  (initial_volume : ℝ)
  (filling_rate : ℝ)
  (emptying_rate : ℝ)
  (initial_fraction_full : initial_volume = 1 / 5)
  (pipe_a_rate : filling_rate = 1 / 10)
  (pipe_b_rate : emptying_rate = 1 / 6) :
  (initial_volume / (filling_rate - emptying_rate) = 3) :=
by
  sorry

end tank_emptying_time_l129_129513


namespace combined_time_in_pool_l129_129968

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end combined_time_in_pool_l129_129968


namespace students_not_enrolled_l129_129419

-- Declare the conditions
def total_students : Nat := 79
def students_french : Nat := 41
def students_german : Nat := 22
def students_both : Nat := 9

-- Define the problem statement
theorem students_not_enrolled : total_students - (students_french + students_german - students_both) = 25 := by
  sorry

end students_not_enrolled_l129_129419


namespace titu_andreescu_inequality_l129_129028

theorem titu_andreescu_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
sorry

end titu_andreescu_inequality_l129_129028


namespace no_nonneg_rational_sol_for_equation_l129_129976

theorem no_nonneg_rational_sol_for_equation :
  ¬ ∃ (x y z : ℚ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^5 + 2 * y^5 + 5 * z^5 = 11 :=
by
  sorry

end no_nonneg_rational_sol_for_equation_l129_129976


namespace parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l129_129286

theorem parts_from_blanks_9 : ∀ (produced_parts : ℕ), produced_parts = 13 :=
by
  sorry

theorem parts_from_blanks_14 : ∀ (produced_parts : ℕ), produced_parts = 20 :=
by
  sorry

theorem blanks_needed_for_40_parts : ∀ (required_blanks : ℕ), required_blanks = 27 :=
by
  sorry

end parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l129_129286


namespace find_m_value_l129_129304

theorem find_m_value
  (y_squared_4x : ∀ x y : ℝ, y^2 = 4 * x)
  (Focus_F : ℝ × ℝ)
  (M N : ℝ × ℝ)
  (E : ℝ)
  (P Q : ℝ × ℝ)
  (k1 k2 : ℝ)
  (MN_slope : k1 = (N.snd - M.snd) / (N.fst - M.fst))
  (PQ_slope : k2 = (Q.snd - P.snd) / (Q.fst - P.fst))
  (slope_condition : k1 = 3 * k2) :
  E = 3 := 
sorry

end find_m_value_l129_129304


namespace complement_intersection_range_of_a_l129_129747

open Set

variable {α : Type*} [TopologicalSpace α]

def U : Set ℝ := univ

def A : Set ℝ := { x | -1 < x ∧ x < 1 }

def B : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 3/2 }

def C (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x ≤ 2 * a - 7 }

-- Question 1
theorem complement_intersection (x : ℝ) :
  x ∈ (U \ A) ∩ B ↔ 1 ≤ x ∧ x ≤ 3 / 2 := sorry

-- Question 2
theorem range_of_a {a : ℝ} (h : A ∩ C a = C a) : a < 4 := sorry

end complement_intersection_range_of_a_l129_129747


namespace discount_percentage_l129_129924

theorem discount_percentage (CP MP SP D : ℝ) (cp_value : CP = 100) 
(markup : MP = CP + 0.5 * CP) (profit : SP = CP + 0.35 * CP) 
(discount : D = MP - SP) : (D / MP) * 100 = 10 := 
by 
  sorry

end discount_percentage_l129_129924


namespace complement_M_intersect_N_l129_129694

def M : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}
def complement_M : Set ℤ := {m | -3 < m ∧ m < 2} 

theorem complement_M_intersect_N : (complement_M ∩ N) = {-1, 0, 1} := by
  sorry

end complement_M_intersect_N_l129_129694


namespace sqrt_factorial_product_squared_l129_129712

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l129_129712


namespace man_speed_in_still_water_l129_129412

theorem man_speed_in_still_water (c_speed : ℝ) (distance_m : ℝ) (time_sec : ℝ) (downstream_distance_km : ℝ) (downstream_time_hr : ℝ) :
    c_speed = 3 →
    distance_m = 15 →
    time_sec = 2.9997600191984644 →
    downstream_distance_km = distance_m / 1000 →
    downstream_time_hr = time_sec / 3600 →
    (downstream_distance_km / downstream_time_hr) - c_speed = 15 :=
by
  intros hc hd ht hdownstream_distance hdownstream_time 
  sorry

end man_speed_in_still_water_l129_129412


namespace percentage_second_division_l129_129389

theorem percentage_second_division (total_students : ℕ) 
                                  (first_division_percentage : ℝ) 
                                  (just_passed : ℕ) 
                                  (all_students_passed : total_students = 300) 
                                  (percentage_first_division : first_division_percentage = 26) 
                                  (students_just_passed : just_passed = 60) : 
  (26 / 100 * 300 + (total_students - (26 / 100 * 300 + 60)) + 60) = 300 → 
  ((total_students - (26 / 100 * 300 + 60)) / total_students * 100) = 54 := 
by 
  sorry

end percentage_second_division_l129_129389


namespace smallest_x_multiple_of_53_l129_129965

theorem smallest_x_multiple_of_53 :
  ∃ (x : ℕ), (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
sorry

end smallest_x_multiple_of_53_l129_129965


namespace sum_reciprocal_of_roots_l129_129632

variables {m n : ℝ}

-- Conditions: m and n are real roots of the quadratic equation x^2 + 4x - 1 = 0
def is_root (a : ℝ) : Prop := a^2 + 4 * a - 1 = 0

theorem sum_reciprocal_of_roots (hm : is_root m) (hn : is_root n) : 
  (1 / m) + (1 / n) = 4 :=
by sorry

end sum_reciprocal_of_roots_l129_129632


namespace cos_45_deg_l129_129842

theorem cos_45_deg : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_45_deg_l129_129842


namespace combined_tax_rate_correct_l129_129338

noncomputable def combined_tax_rate (income_john income_ingrid tax_rate_john tax_rate_ingrid : ℝ) : ℝ :=
  let tax_john := tax_rate_john * income_john
  let tax_ingrid := tax_rate_ingrid * income_ingrid
  let total_tax := tax_john + tax_ingrid
  let combined_income := income_john + income_ingrid
  total_tax / combined_income * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 56000 74000 0.30 0.40 = 35.69 := by
  sorry

end combined_tax_rate_correct_l129_129338


namespace infinite_prime_set_exists_l129_129197

noncomputable def P : Set Nat := {p | Prime p ∧ ∃ m : Nat, p ∣ m^2 + 1}

theorem infinite_prime_set_exists :
  ∃ (P : Set Nat), (∀ p ∈ P, Prime p) ∧ (Set.Infinite P) ∧ 
  (∀ (p : Nat) (hp : p ∈ P) (k : ℕ),
    ∃ (m : Nat), p^k ∣ m^2 + 1 ∧ ¬(p^(k+1) ∣ m^2 + 1)) :=
sorry

end infinite_prime_set_exists_l129_129197


namespace sufficient_not_necessary_condition_l129_129420

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 + t * x - t

-- The proof statement about the condition for roots
theorem sufficient_not_necessary_condition (t : ℝ) :
  (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) :=
sorry

end sufficient_not_necessary_condition_l129_129420


namespace alice_bob_meet_l129_129109

theorem alice_bob_meet :
  ∃ k : ℕ, (4 * k - 4 * (k / 5) ≡ 8 * k [MOD 15]) ∧ (k = 5) :=
by
  sorry

end alice_bob_meet_l129_129109


namespace sum_of_ages_of_cousins_l129_129307

noncomputable def is_valid_age_group (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (1 ≤ a) ∧ (a ≤ 9) ∧ (1 ≤ b) ∧ (b ≤ 9) ∧ (1 ≤ c) ∧ (c ≤ 9) ∧ (1 ≤ d) ∧ (d ≤ 9)

theorem sum_of_ages_of_cousins :
  ∃ (a b c d : ℕ), is_valid_age_group a b c d ∧ (a * b = 40) ∧ (c * d = 36) ∧ (a + b + c + d = 26) := 
sorry

end sum_of_ages_of_cousins_l129_129307


namespace difference_mean_median_l129_129865

theorem difference_mean_median :
  let percentage_scored_60 : ℚ := 0.20
  let percentage_scored_70 : ℚ := 0.30
  let percentage_scored_85 : ℚ := 0.25
  let percentage_scored_95 : ℚ := 1 - (percentage_scored_60 + percentage_scored_70 + percentage_scored_85)
  let score_60 : ℚ := 60
  let score_70 : ℚ := 70
  let score_85 : ℚ := 85
  let score_95 : ℚ := 95
  let mean : ℚ := percentage_scored_60 * score_60 + percentage_scored_70 * score_70 + percentage_scored_85 * score_85 + percentage_scored_95 * score_95
  let median : ℚ := 85
  (median - mean) = 7 := 
by 
  sorry

end difference_mean_median_l129_129865


namespace MrC_loses_240_after_transactions_l129_129060

theorem MrC_loses_240_after_transactions :
  let house_initial_value := 12000
  let first_transaction_loss_percent := 0.15
  let second_transaction_gain_percent := 0.20
  let house_value_after_first_transaction :=
    house_initial_value * (1 - first_transaction_loss_percent)
  let house_value_after_second_transaction :=
    house_value_after_first_transaction * (1 + second_transaction_gain_percent)
  house_value_after_second_transaction - house_initial_value = 240 :=
by
  sorry

end MrC_loses_240_after_transactions_l129_129060


namespace tangent_line_slope_l129_129815

theorem tangent_line_slope (m : ℝ) :
  (∀ x y, (x^2 + y^2 - 4*x + 2 = 0) → (y = m * x)) → (m = 1 ∨ m = -1) := 
by
  intro h
  sorry

end tangent_line_slope_l129_129815


namespace infinite_triangular_pairs_l129_129255

theorem infinite_triangular_pairs : ∃ (a_i b_i : ℕ → ℕ), (∀ m : ℕ, ∃ n : ℕ, m = n * (n + 1) / 2 ↔ ∃ k : ℕ, a_i k * m + b_i k = k * (k + 1) / 2) ∧ ∀ j : ℕ, ∃ k : ℕ, k > j :=
by {
  sorry
}

end infinite_triangular_pairs_l129_129255


namespace average_bmi_is_correct_l129_129901

-- Define Rachel's parameters
def rachel_weight : ℕ := 75
def rachel_height : ℕ := 60  -- in inches

-- Define Jimmy's parameters based on the conditions
def jimmy_weight : ℕ := rachel_weight + 6
def jimmy_height : ℕ := rachel_height + 3

-- Define Adam's parameters based on the conditions
def adam_weight : ℕ := rachel_weight - 15
def adam_height : ℕ := rachel_height - 2

-- Define the BMI formula
def bmi (weight : ℕ) (height : ℕ) : ℚ := (weight * 703 : ℚ) / (height * height)

-- Rachel's, Jimmy's, and Adam's BMIs
def rachel_bmi : ℚ := bmi rachel_weight rachel_height
def jimmy_bmi : ℚ := bmi jimmy_weight jimmy_height
def adam_bmi : ℚ := bmi adam_weight adam_height

-- Proving the average BMI
theorem average_bmi_is_correct : 
  (rachel_bmi + jimmy_bmi + adam_bmi) / 3 = 13.85 := 
by
  sorry

end average_bmi_is_correct_l129_129901


namespace Jeremy_payment_total_l129_129360

theorem Jeremy_payment_total :
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  total_payment = (553 : ℚ) / 40 :=
by {
  -- Definitions
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  
  -- Main goal
  sorry
}

end Jeremy_payment_total_l129_129360


namespace total_profit_calculation_l129_129140

variables {I_B T_B : ℝ}

-- Conditions as definitions
def investment_A (I_B : ℝ) : ℝ := 3 * I_B
def period_A (T_B : ℝ) : ℝ := 2 * T_B
def profit_B (I_B T_B : ℝ) : ℝ := I_B * T_B
def total_profit (I_B T_B : ℝ) : ℝ := 7 * I_B * T_B

-- To prove
theorem total_profit_calculation
  (h1 : investment_A I_B = 3 * I_B)
  (h2 : period_A T_B = 2 * T_B)
  (h3 : profit_B I_B T_B = 4000)
  : total_profit I_B T_B = 28000 := by
  sorry

end total_profit_calculation_l129_129140


namespace drinking_problem_solution_l129_129372

def drinking_rate (name : String) (hours : ℕ) (total_liters : ℕ) : ℚ :=
  total_liters / hours

def total_wine_consumed_in_x_hours (x : ℚ) :=
  x * (
  drinking_rate "assistant1" 12 40 +
  drinking_rate "assistant2" 10 40 +
  drinking_rate "assistant3" 8 40
  )

theorem drinking_problem_solution : 
  (∃ x : ℚ, total_wine_consumed_in_x_hours x = 40) →
  ∃ x : ℚ, x = 120 / 37 :=
by 
  sorry

end drinking_problem_solution_l129_129372


namespace necessary_and_sufficient_condition_perpendicular_lines_l129_129347

def are_perpendicular (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x + y = 0) → (x - a * y = 0) → x = 0 ∧ y = 0

theorem necessary_and_sufficient_condition_perpendicular_lines :
  ∀ (a : ℝ), are_perpendicular a → a = 1 :=
sorry

end necessary_and_sufficient_condition_perpendicular_lines_l129_129347


namespace part_one_part_two_l129_129223

-- Defining the sequence {a_n} with the sum of the first n terms.
def S (n : ℕ) : ℕ := 3 * n ^ 2 + 10 * n

-- Defining a_n in terms of the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Defining the arithmetic sequence {b_n}
def b (n : ℕ) : ℕ := 3 * n + 2

-- Defining the sequence {c_n}
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n

-- Defining the sum of the first n terms of {c_n}
def T (n : ℕ) : ℕ :=
  (3 * n + 1) * 2^(n + 2) - 4

-- Theorem to prove general term formula for {b_n}
theorem part_one : ∀ n : ℕ, b n = 3 * n + 2 := 
by sorry

-- Theorem to prove the sum of the first n terms of {c_n}
theorem part_two (n : ℕ) : ∀ n : ℕ, T n = (3 * n + 1) * 2^(n + 2) - 4 :=
by sorry

end part_one_part_two_l129_129223


namespace quadratic_inequality_solution_l129_129226

theorem quadratic_inequality_solution (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c) * x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} :=
sorry

end quadratic_inequality_solution_l129_129226


namespace ellipse_equation_standard_form_l129_129538

theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ) (h k : ℝ), 
    a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧ 
    b = Real.sqrt ((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9 ∧ 
    h = 1 ∧ 
    k = 4 ∧ 
    (∀ x y : ℝ, (x, y) = (12, -4) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
  sorry

end ellipse_equation_standard_form_l129_129538


namespace find_principal_l129_129908

theorem find_principal (R : ℝ) (P : ℝ) (h : ((P * (R + 5) * 10) / 100) = ((P * R * 10) / 100 + 600)) : P = 1200 :=
by
  sorry

end find_principal_l129_129908


namespace common_terms_sequence_l129_129231

-- Definitions of sequences
def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℤ := 2 ^ n
def c (n : ℕ) : ℤ := 2 ^ (2 * n - 1)

-- Theorem stating the conjecture
theorem common_terms_sequence :
  ∀ n : ℕ, ∃ m : ℕ, a m = b (2 * n - 1) :=
by
  sorry

end common_terms_sequence_l129_129231


namespace min_value_of_a_l129_129807

theorem min_value_of_a (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2004) (h5 : a^2 - b^2 + c^2 - d^2 = 2004) : a = 503 :=
sorry

end min_value_of_a_l129_129807


namespace value_of_g_neg2_l129_129872

-- Define the function g as given in the conditions
def g (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Statement of the problem: Prove that g(-2) = 11
theorem value_of_g_neg2 : g (-2) = 11 := by
  sorry

end value_of_g_neg2_l129_129872


namespace grain_storage_bins_total_l129_129820

theorem grain_storage_bins_total
  (b20 : ℕ) (b20_tonnage : ℕ) (b15_tonnage : ℕ) (total_capacity : ℕ) (b20_count : ℕ)
  (h_b20_capacity : b20_count * b20_tonnage = b20)
  (h_total_capacity : b20 + (total_capacity - b20) = total_capacity)
  (h_b20_given : b20_count = 12)
  (h_b20_tonnage : b20_tonnage = 20)
  (h_b15_tonnage : b15_tonnage = 15)
  (h_total_capacity_given : total_capacity = 510) :
  ∃ b_total : ℕ, b_total = b20_count + ((total_capacity - (b20_count * b20_tonnage)) / b15_tonnage) ∧ b_total = 30 :=
by
  sorry

end grain_storage_bins_total_l129_129820


namespace gift_cost_calc_l129_129910

theorem gift_cost_calc (C N : ℕ) (hN : N = 12)
    (h : C / (N - 4) = C / N + 10) : C = 240 := by
  sorry

end gift_cost_calc_l129_129910


namespace correct_statements_l129_129495

-- Definitions
noncomputable def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statements
theorem correct_statements (b c : ℝ) :
  (b > 0 → ∀ x y : ℝ, x ≤ y → f x b c ≤ f y b c) ∧
  (b < 0 → ¬ (∀ x : ℝ, ∃ m : ℝ, f x b c = m)) ∧
  (b = 0 → ∀ x : ℝ, f (x) b c = f (-x) b c) ∧
  (∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0) :=
sorry

end correct_statements_l129_129495


namespace algebraic_expression_value_l129_129424

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 6 - Real.sqrt 2) : 2 * x^2 + 4 * Real.sqrt 2 * x = 8 :=
sorry

end algebraic_expression_value_l129_129424


namespace multiples_33_between_1_and_300_l129_129956

theorem multiples_33_between_1_and_300 : ∃ (x : ℕ), (∀ n : ℕ, n ≤ 300 → n % x = 0 → n / x ≤ 33) ∧ x = 9 :=
by
  sorry

end multiples_33_between_1_and_300_l129_129956


namespace expand_and_simplify_l129_129218

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end expand_and_simplify_l129_129218


namespace red_car_count_l129_129076

-- Define the ratio and the given number of black cars
def ratio_red_to_black (R B : ℕ) : Prop := R * 8 = B * 3

-- Define the given number of black cars
def black_cars : ℕ := 75

-- State the theorem we want to prove
theorem red_car_count : ∃ R : ℕ, ratio_red_to_black R black_cars ∧ R = 28 :=
by
  sorry

end red_car_count_l129_129076


namespace volume_pyramid_ABC_l129_129850

structure Point where
  x : ℝ
  y : ℝ

def triangle_volume (A B C : Point) : ℝ :=
  -- The implementation would calculate the volume of the pyramid formed
  -- by folding along the midpoint sides.
  sorry

theorem volume_pyramid_ABC :
  let A := Point.mk 0 0
  let B := Point.mk 30 0
  let C := Point.mk 20 15
  triangle_volume A B C = 900 :=
by
  -- To be filled with the proof
  sorry

end volume_pyramid_ABC_l129_129850


namespace solution_correct_l129_129852

noncomputable def solution_set : Set ℝ :=
  {x | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)}

theorem solution_correct (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2))) < 1 / 4 ↔ (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x) :=
by sorry

end solution_correct_l129_129852


namespace fencing_cost_proof_l129_129532

noncomputable def totalCostOfFencing (length : ℕ) (breadth : ℕ) (costPerMeter : ℚ) : ℚ :=
  2 * (length + breadth) * costPerMeter

theorem fencing_cost_proof : totalCostOfFencing 56 (56 - 12) 26.50 = 5300 := by
  sorry

end fencing_cost_proof_l129_129532


namespace condition_iff_odd_function_l129_129166

theorem condition_iff_odd_function (f : ℝ → ℝ) :
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
by
  sorry

end condition_iff_odd_function_l129_129166


namespace find_positive_number_l129_129874

theorem find_positive_number (x : ℝ) (hx : 0 < x) (h : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := by
  sorry

end find_positive_number_l129_129874


namespace clock_four_different_digits_l129_129734

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l129_129734


namespace find_integer_x_l129_129435

theorem find_integer_x (x y : ℕ) (h_gt : x > y) (h_gt_zero : y > 0) (h_eq : x + y + x * y = 99) : x = 49 :=
sorry

end find_integer_x_l129_129435


namespace problem1_problem2_problem3_problem4_l129_129554

-- Problem 1
theorem problem1 (x : ℤ) (h : 4 * x = 20) : x = 5 :=
sorry

-- Problem 2
theorem problem2 (x : ℤ) (h : x - 18 = 40) : x = 58 :=
sorry

-- Problem 3
theorem problem3 (x : ℤ) (h : x / 7 = 12) : x = 84 :=
sorry

-- Problem 4
theorem problem4 (n : ℚ) (h : 8 * n / 2 = 15) : n = 15 / 4 :=
sorry

end problem1_problem2_problem3_problem4_l129_129554


namespace initial_friends_count_l129_129474

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count_l129_129474


namespace find_number_l129_129907

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l129_129907


namespace zip_code_relationship_l129_129251

theorem zip_code_relationship (A B C D E : ℕ) 
(h1 : A + B + C + D + E = 10) 
(h2 : C = 0) 
(h3 : D = 2 * A) 
(h4 : D + E = 8) : 
A + B = 2 :=
sorry

end zip_code_relationship_l129_129251


namespace problem_l129_129758

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end problem_l129_129758


namespace simplify_fraction_multiplication_l129_129702

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l129_129702


namespace problem_part_1_problem_part_2_l129_129136

def f (x m : ℝ) := 2 * x^2 + (2 - m) * x - m
def g (x m : ℝ) := x^2 - x + 2 * m

theorem problem_part_1 (x : ℝ) : f x 1 > 0 ↔ (x > 1/2 ∨ x < -1) :=
by sorry

theorem problem_part_2 {m x : ℝ} (hm : 0 < m) : f x m ≤ g x m ↔ (-3 ≤ x ∧ x ≤ m) :=
by sorry

end problem_part_1_problem_part_2_l129_129136


namespace dogwood_trees_proof_l129_129311

def dogwood_trees_left (a b c : Float) : Float :=
  a + b - c

theorem dogwood_trees_proof : dogwood_trees_left 5.0 4.0 7.0 = 2.0 :=
by
  -- The proof itself is left out intentionally as per the instructions
  sorry

end dogwood_trees_proof_l129_129311


namespace exists_n_gt_1958_l129_129456

noncomputable def polyline_path (n : ℕ) : ℝ := sorry
noncomputable def distance_to_origin (n : ℕ) : ℝ := sorry 
noncomputable def sum_lengths (n : ℕ) : ℝ := sorry

theorem exists_n_gt_1958 :
  ∃ (n : ℕ), n > 1958 ∧ (sum_lengths n) / (distance_to_origin n) > 1958 := 
sorry

end exists_n_gt_1958_l129_129456


namespace domain_of_sqrt_l129_129661

theorem domain_of_sqrt (x : ℝ) : (x - 1 ≥ 0) → (x ≥ 1) :=
by
  sorry

end domain_of_sqrt_l129_129661


namespace floor_sqrt_18_squared_eq_16_l129_129407

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end floor_sqrt_18_squared_eq_16_l129_129407


namespace find_a_for_inverse_proportion_l129_129906

theorem find_a_for_inverse_proportion (a : ℝ)
  (h_A : ∃ k : ℝ, 4 = k / (-1))
  (h_B : ∃ k : ℝ, 2 = k / a) :
  a = -2 :=
sorry

end find_a_for_inverse_proportion_l129_129906


namespace merchant_profit_percentage_l129_129591

-- Given
def initial_cost_price : ℝ := 100
def marked_price : ℝ := initial_cost_price + 0.50 * initial_cost_price
def discount_percentage : ℝ := 0.20
def discount : ℝ := discount_percentage * marked_price
def selling_price : ℝ := marked_price - discount

-- Prove
theorem merchant_profit_percentage :
  ((selling_price - initial_cost_price) / initial_cost_price) * 100 = 20 :=
by
  sorry

end merchant_profit_percentage_l129_129591


namespace value_range_f_in_0_to_4_l129_129858

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem value_range_f_in_0_to_4 :
  ∀ (x : ℝ), (0 < x ∧ x ≤ 4) → (1 ≤ f x ∧ f x ≤ 10) :=
sorry

end value_range_f_in_0_to_4_l129_129858


namespace no_pre_period_decimal_representation_l129_129610

theorem no_pre_period_decimal_representation (m : ℕ) (h : Nat.gcd m 10 = 1) : ¬∃ k : ℕ, ∃ a : ℕ, 0 < a ∧ 10^a < m ∧ (10^a - 1) % m = k ∧ k ≠ 0 :=
sorry

end no_pre_period_decimal_representation_l129_129610


namespace line_cannot_pass_through_third_quadrant_l129_129215

theorem line_cannot_pass_through_third_quadrant :
  ∀ (x y : ℝ), x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end line_cannot_pass_through_third_quadrant_l129_129215


namespace consecutive_integers_equality_l129_129523

theorem consecutive_integers_equality (n : ℕ) (h_eq : (n - 3) + (n - 2) + (n - 1) + n = (n + 1) + (n + 2) + (n + 3)) : n = 12 :=
by {
  sorry
}

end consecutive_integers_equality_l129_129523


namespace fraction_used_first_day_l129_129378

theorem fraction_used_first_day (x : ℝ) :
  let initial_supplies := 400
  let supplies_remaining_after_first_day := initial_supplies * (1 - x)
  let supplies_remaining_after_three_days := (2/5 : ℝ) * supplies_remaining_after_first_day
  supplies_remaining_after_three_days = 96 → 
  x = (2/5 : ℝ) :=
by
  intros
  sorry

end fraction_used_first_day_l129_129378


namespace sara_basketball_loss_l129_129939

theorem sara_basketball_loss (total_games : ℕ) (games_won : ℕ) (games_lost : ℕ) 
  (h1 : total_games = 16) 
  (h2 : games_won = 12) 
  (h3 : games_lost = total_games - games_won) : 
  games_lost = 4 :=
by
  sorry

end sara_basketball_loss_l129_129939


namespace neg_of_if_pos_then_real_roots_l129_129193

variable (m : ℝ)

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b * x + c = 0

theorem neg_of_if_pos_then_real_roots :
  (∀ m : ℝ, m > 0 → has_real_roots 1 1 (-m) )
  → ( ∀ m : ℝ, m ≤ 0 → ¬ has_real_roots 1 1 (-m) ) := 
sorry

end neg_of_if_pos_then_real_roots_l129_129193


namespace robin_uploaded_pics_from_camera_l129_129876

-- Definitions of the conditions
def pics_from_phone := 35
def albums := 5
def pics_per_album := 8

-- The statement we want to prove
theorem robin_uploaded_pics_from_camera : (albums * pics_per_album) - pics_from_phone = 5 :=
by
  -- Proof goes here
  sorry

end robin_uploaded_pics_from_camera_l129_129876


namespace paint_total_gallons_l129_129604

theorem paint_total_gallons
  (white_paint_gallons : ℕ)
  (blue_paint_gallons : ℕ)
  (h_wp : white_paint_gallons = 660)
  (h_bp : blue_paint_gallons = 6029) :
  white_paint_gallons + blue_paint_gallons = 6689 := 
by
  sorry

end paint_total_gallons_l129_129604


namespace simplify_eq_neg_one_l129_129730

variable (a b c : ℝ)

noncomputable def simplify_expression := 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_eq_neg_one 
  (a_ne_zero : a ≠ 0) 
  (b_ne_zero : b ≠ 0) 
  (c_ne_zero : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) 
  : simplify_expression a b c = -1 :=
by sorry

end simplify_eq_neg_one_l129_129730


namespace deepak_age_l129_129075

theorem deepak_age : ∀ (R D : ℕ), (R / D = 4 / 3) ∧ (R + 6 = 18) → D = 9 :=
by
  sorry

end deepak_age_l129_129075


namespace sqrt_sixteen_l129_129297

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l129_129297


namespace ted_and_mike_seeds_l129_129810

noncomputable def ted_morning_seeds (T : ℕ) (mike_morning_seeds : ℕ) (mike_afternoon_seeds : ℕ) (total_seeds : ℕ) : Prop :=
  mike_morning_seeds = 50 ∧
  mike_afternoon_seeds = 60 ∧
  total_seeds = 250 ∧
  T + (mike_afternoon_seeds - 20) + (mike_morning_seeds + mike_afternoon_seeds) = total_seeds ∧
  2 * mike_morning_seeds = T

theorem ted_and_mike_seeds :
  ∃ T : ℕ, ted_morning_seeds T 50 60 250 :=
by {
  sorry
}

end ted_and_mike_seeds_l129_129810


namespace cost_for_33_people_employees_for_14000_cost_l129_129886

-- Define the conditions for pricing
def price_per_ticket (x : Nat) : Int :=
  if x ≤ 30 then 400
  else max 280 (400 - 5 * (x - 30))

def total_cost (x : Nat) : Int :=
  x * price_per_ticket x

-- Problem Part 1: Proving the total cost for 33 people
theorem cost_for_33_people :
  total_cost 33 = 12705 :=
by
  sorry

-- Problem Part 2: Given a total cost of 14000, finding the number of employees
theorem employees_for_14000_cost :
  ∃ x : Nat, total_cost x = 14000 ∧ price_per_ticket x ≥ 280 :=
by
  sorry

end cost_for_33_people_employees_for_14000_cost_l129_129886


namespace no_real_b_for_inequality_l129_129216

theorem no_real_b_for_inequality (b : ℝ) :
  (∃ x : ℝ, |x^2 + 3*b*x + 4*b| ≤ 5 ∧ (∀ y : ℝ, |y^2 + 3*b*y + 4*b| ≤ 5 → y = x)) → false :=
by
  sorry

end no_real_b_for_inequality_l129_129216


namespace number_of_divisors_of_8_fact_l129_129833

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l129_129833


namespace price_of_first_doughnut_l129_129503

theorem price_of_first_doughnut 
  (P : ℕ)  -- Price of the first doughnut
  (total_doughnuts : ℕ := 48)  -- Total number of doughnuts
  (price_per_dozen : ℕ := 6)  -- Price per dozen of additional doughnuts
  (total_cost : ℕ := 24)  -- Total cost spent
  (doughnuts_left : ℕ := total_doughnuts - 1)  -- Doughnuts left after the first one
  (dozens : ℕ := doughnuts_left / 12)  -- Number of whole dozens
  (cost_of_dozens : ℕ := dozens * price_per_dozen)  -- Cost of the dozens of doughnuts
  (cost_after_first : ℕ := total_cost - cost_of_dozens)  -- Remaining cost after dozens
  : P = 6 := 
by
  -- Proof to be filled in
  sorry

end price_of_first_doughnut_l129_129503


namespace sum_of_xyz_l129_129170

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem sum_of_xyz :
  ∃ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 ∧
  x + y + z = 932 :=
by
  sorry

end sum_of_xyz_l129_129170


namespace convex_polygon_with_tiles_l129_129864

variable (n : ℕ)

def canFormConvexPolygon (n : ℕ) : Prop :=
  3 ≤ n ∧ n ≤ 12

theorem convex_polygon_with_tiles (n : ℕ) 
  (square_internal_angle : ℕ := 90) 
  (equilateral_triangle_internal_angle : ℕ := 60)
  (external_angle_step : ℕ := 30)
  (total_external_angle : ℕ := 360) :
  canFormConvexPolygon n :=
by 
  sorry

end convex_polygon_with_tiles_l129_129864


namespace abs_ab_cd_leq_one_fourth_l129_129068

theorem abs_ab_cd_leq_one_fourth (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) (h3 : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  |a * b - c * d| ≤ 1 / 4 :=
sorry

end abs_ab_cd_leq_one_fourth_l129_129068


namespace altitude_from_A_to_BC_l129_129229

theorem altitude_from_A_to_BC (x y : ℝ) : 
  (3 * x + 4 * y + 12 = 0) ∧ 
  (4 * x - 3 * y + 16 = 0) ∧ 
  (2 * x + y - 2 = 0) → 
  (∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1 / 2) ∧ (b = 2 - 8)) :=
by 
  sorry

end altitude_from_A_to_BC_l129_129229


namespace max_of_inverse_power_sums_l129_129836

theorem max_of_inverse_power_sums (s p r1 r2 : ℝ) 
  (h_eq_roots : r1 + r2 = s ∧ r1 * r2 = p)
  (h_eq_powers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2023 → r1^n + r2^n = s) :
  1 / r1^(2024:ℕ) + 1 / r2^(2024:ℕ) ≤ 2 :=
sorry

end max_of_inverse_power_sums_l129_129836


namespace complex_conjugate_x_l129_129397

theorem complex_conjugate_x (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3 * x + 2 : ℂ) * Complex.I = 4 + 20 * Complex.I) : x = -3 := sorry

end complex_conjugate_x_l129_129397


namespace compute_expression_l129_129106

theorem compute_expression (x y z : ℝ) (h₀ : x ≠ y) (h₁ : y ≠ z) (h₂ : z ≠ x) (h₃ : x + y + z = 3) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 9 / (2 * (x^2 + y^2 + z^2)) - 1 / 2 :=
by
  sorry

end compute_expression_l129_129106


namespace fraction_not_integer_l129_129705

def containsExactlyTwoOccurrences (d : List ℕ) : Prop :=
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7], d.count n = 2

theorem fraction_not_integer
  (k m : ℕ)
  (hk : 14 = (List.length (Nat.digits 10 k)))
  (hm : 14 = (List.length (Nat.digits 10 m)))
  (hkd : containsExactlyTwoOccurrences (Nat.digits 10 k))
  (hmd : containsExactlyTwoOccurrences (Nat.digits 10 m))
  (hkm : k ≠ m) :
  ¬ ∃ d : ℕ, k = m * d := 
sorry

end fraction_not_integer_l129_129705


namespace average_fruits_per_basket_is_correct_l129_129249

noncomputable def average_fruits_per_basket : ℕ :=
  let basket_A := 15
  let basket_B := 30
  let basket_C := 20
  let basket_D := 25
  let basket_E := 35
  let total_fruits := basket_A + basket_B + basket_C + basket_D + basket_E
  let number_of_baskets := 5
  total_fruits / number_of_baskets

theorem average_fruits_per_basket_is_correct : average_fruits_per_basket = 25 := by
  unfold average_fruits_per_basket
  rfl

end average_fruits_per_basket_is_correct_l129_129249


namespace rectangle_perimeter_l129_129509

theorem rectangle_perimeter {a b c width : ℕ} (h₁: a = 15) (h₂: b = 20) (h₃: c = 25) (w : ℕ) (h₄: w = 5) :
  let area_triangle := (a * b) / 2
  let length := area_triangle / w
  let perimeter := 2 * (length + w)
  perimeter = 70 :=
by
  sorry

end rectangle_perimeter_l129_129509


namespace oa_dot_ob_eq_neg2_l129_129105

/-!
# Problem Statement
Given AB as the diameter of the smallest radius circle centered at C(0,1) that intersects 
the graph of y = 1 / (|x| - 1), where O is the origin. Prove that the dot product 
\overrightarrow{OA} · \overrightarrow{OB} equals -2.
-/

noncomputable def smallest_radius_circle_eqn (x : ℝ) : ℝ :=
  x^2 + ((1 / (|x| - 1)) - 1)^2

noncomputable def radius_of_circle (x : ℝ) : ℝ :=
  Real.sqrt (smallest_radius_circle_eqn x)

noncomputable def OA (x : ℝ) : ℝ × ℝ :=
  (x, (1 / (|x| - 1)) + 1)

noncomputable def OB (x : ℝ) : ℝ × ℝ :=
  (-x, 1 - (1 / (|x| - 1)))

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem oa_dot_ob_eq_neg2 (x : ℝ) (hx : |x| > 1) :
  let a := OA x
  let b := OB x
  dot_product a b = -2 :=
by
  sorry

end oa_dot_ob_eq_neg2_l129_129105


namespace smallest_n_for_terminating_decimal_l129_129504

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n+53) → (∃ a b : ℕ, n + 53 = 2^a * 5^b) → n = 11 :=
by
  sorry

end smallest_n_for_terminating_decimal_l129_129504


namespace problem_statement_l129_129515

variable (a : ℕ → ℝ)

-- Defining sequences {b_n} and {c_n}
def b (n : ℕ) := a n - a (n + 2)
def c (n : ℕ) := a n + 2 * a (n + 1) + 3 * a (n + 2)

-- Defining that a sequence is arithmetic
def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

-- Problem statement
theorem problem_statement :
  is_arithmetic a ↔ (is_arithmetic (c a) ∧ ∀ n, b a n ≤ b a (n + 1)) :=
sorry

end problem_statement_l129_129515


namespace floor_length_is_twelve_l129_129290

-- Definitions based on the conditions
def floor_width := 10
def strip_width := 3
def rug_area := 24

-- Problem statement
theorem floor_length_is_twelve (L : ℕ) 
  (h1 : rug_area = (L - 2 * strip_width) * (floor_width - 2 * strip_width)) :
  L = 12 := 
sorry

end floor_length_is_twelve_l129_129290


namespace area_of_trapezium_eq_336_l129_129026

-- Define the lengths of the parallel sides and the distance between them
def a := 30 -- length of one parallel side in cm
def b := 12 -- length of the other parallel side in cm
def h := 16 -- distance between the parallel sides (height) in cm

-- Define the expected area
def expectedArea := 336 -- area in square cm

-- State the theorem to prove
theorem area_of_trapezium_eq_336 : (1/2 : ℝ) * (a + b) * h = expectedArea := 
by 
  -- The proof is omitted
  sorry

end area_of_trapezium_eq_336_l129_129026


namespace range_of_f_on_nonneg_reals_l129_129646

theorem range_of_f_on_nonneg_reals (k : ℕ) (h_even : k % 2 = 0) (h_pos : 0 < k) :
    ∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, 0 ≤ x ∧ x^k = y :=
by
  sorry

end range_of_f_on_nonneg_reals_l129_129646


namespace isosceles_triangle_height_ratio_l129_129524

theorem isosceles_triangle_height_ratio (a b : ℝ) (h₁ : b = (4 / 3) * a) :
  ∃ m n : ℝ, b / 2 = m + n ∧ m = (2 / 3) * a ∧ n = (1 / 3) * a ∧ (m / n) = 2 :=
by
  sorry

end isosceles_triangle_height_ratio_l129_129524


namespace correct_union_l129_129473

universe u

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2}
def C_I (A : Set ℕ) : Set ℕ := {x ∈ I | x ∉ A}

-- Theorem statement
theorem correct_union : B ∪ C_I A = {2, 4, 5} :=
by
  sorry

end correct_union_l129_129473


namespace sequence_general_formula_l129_129306

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (rec : ∀ n : ℕ, n > 0 → a n = n * (a (n + 1) - a n)) : 
  ∀ n, a n = n := 
by 
  sorry

end sequence_general_formula_l129_129306


namespace maximum_enclosed_area_l129_129427

theorem maximum_enclosed_area (P : ℝ) (A : ℝ) : 
  P = 100 → (∃ l w : ℝ, P = 2 * l + 2 * w ∧ A = l * w) → A ≤ 625 :=
by
  sorry

end maximum_enclosed_area_l129_129427


namespace total_number_of_items_l129_129530

-- Definitions based on the problem conditions
def number_of_notebooks : ℕ := 40
def pens_more_than_notebooks : ℕ := 80
def pencils_more_than_notebooks : ℕ := 45

-- Total items calculation based on the conditions
def number_of_pens : ℕ := number_of_notebooks + pens_more_than_notebooks
def number_of_pencils : ℕ := number_of_notebooks + pencils_more_than_notebooks
def total_items : ℕ := number_of_notebooks + number_of_pens + number_of_pencils

-- Statement to be proved
theorem total_number_of_items : total_items = 245 := 
by 
  sorry

end total_number_of_items_l129_129530


namespace theater_total_seats_l129_129938

theorem theater_total_seats
  (occupied_seats : ℕ) (empty_seats : ℕ) 
  (h1 : occupied_seats = 532) (h2 : empty_seats = 218) :
  occupied_seats + empty_seats = 750 := 
by
  -- This is the placeholder for the proof
  sorry

end theater_total_seats_l129_129938


namespace triangle_solid_revolution_correct_l129_129757

noncomputable def triangle_solid_revolution (t : ℝ) (alpha beta gamma : ℝ) (longest_side : string) : ℝ × ℝ :=
  let pi := Real.pi;
  let sin := Real.sin;
  let cos := Real.cos;
  let sqrt := Real.sqrt;
  let to_rad (x : ℝ) : ℝ := x * pi / 180;
  let alpha_rad := to_rad alpha;
  let beta_rad := to_rad beta;
  let gamma_rad := to_rad gamma;
  let a := sqrt (2 * t * sin alpha_rad / (sin beta_rad * sin gamma_rad));
  let b := sqrt (2 * t * sin beta_rad / (sin gamma_rad * sin alpha_rad));
  let m_c := sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  let F := 2 * pi * t * cos ((alpha_rad - beta_rad) / 2) / sin (gamma_rad / 2);
  let K := 2 * pi / 3 * t * sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  (F, K)

theorem triangle_solid_revolution_correct :
  triangle_solid_revolution 80.362 (39 + 34/60 + 30/3600) (60 : ℝ) (80 + 25/60 + 30/3600) "c" = (769.3, 1595.3) :=
sorry

end triangle_solid_revolution_correct_l129_129757


namespace question1_question2_l129_129838

-- Define the sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Question 1: When a = 2, find the set A ∩ B
theorem question1 : A ∩ B = { x | x > 3 } :=
  sorry

-- Question 2: If A ∩ (complement of B) = ∅, find the range of a
theorem question2 : A ∩ (U \ B) = ∅ → a ≤ -6 :=
  sorry

end question1_question2_l129_129838


namespace temperature_difference_l129_129710

theorem temperature_difference (T_south T_north : ℤ) (h1 : T_south = -7) (h2 : T_north = -15) :
  T_south - T_north = 8 :=
by
  sorry

end temperature_difference_l129_129710


namespace longest_segment_cylinder_l129_129086

theorem longest_segment_cylinder (r h : ℤ) (c : ℝ) (hr : r = 4) (hh : h = 9) : 
  c = Real.sqrt (2 * r * r + h * h) ↔ c = Real.sqrt 145 :=
by
  sorry

end longest_segment_cylinder_l129_129086


namespace probability_of_selecting_green_ball_l129_129469

-- Declare the probability of selecting each container
def prob_of_selecting_container := (1 : ℚ) / 4

-- Declare the number of balls in each container
def balls_in_container_A := 10
def balls_in_container_B := 14
def balls_in_container_C := 14
def balls_in_container_D := 10

-- Declare the number of green balls in each container
def green_balls_in_A := 6
def green_balls_in_B := 6
def green_balls_in_C := 6
def green_balls_in_D := 7

-- Calculate the probability of drawing a green ball from each container
def prob_green_from_A := (green_balls_in_A : ℚ) / balls_in_container_A
def prob_green_from_B := (green_balls_in_B : ℚ) / balls_in_container_B
def prob_green_from_C := (green_balls_in_C : ℚ) / balls_in_container_C
def prob_green_from_D := (green_balls_in_D : ℚ) / balls_in_container_D

-- Calculate the total probability of drawing a green ball
def total_prob_green :=
  prob_of_selecting_container * prob_green_from_A +
  prob_of_selecting_container * prob_green_from_B +
  prob_of_selecting_container * prob_green_from_C +
  prob_of_selecting_container * prob_green_from_D

theorem probability_of_selecting_green_ball : total_prob_green = 13 / 28 :=
by sorry

end probability_of_selecting_green_ball_l129_129469


namespace circle_center_and_radius_l129_129955

theorem circle_center_and_radius :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ 
    (x - C.1)^2 + (y - C.2)^2 = r^2) ∧ C = (1, -2) ∧ r = Real.sqrt 2 :=
by 
  sorry

end circle_center_and_radius_l129_129955


namespace survey_method_correct_l129_129500

/-- Definitions to represent the options in the survey method problem. -/
inductive SurveyMethod
| A
| B
| C
| D

/-- The function to determine the correct survey method. -/
def appropriate_survey_method : SurveyMethod :=
  SurveyMethod.C

/-- The theorem stating that the appropriate survey method is indeed option C. -/
theorem survey_method_correct : appropriate_survey_method = SurveyMethod.C :=
by
  /- The actual proof is omitted as per instruction. -/
  sorry

end survey_method_correct_l129_129500


namespace compute_XY_l129_129300

theorem compute_XY (BC AC AB : ℝ) (hBC : BC = 30) (hAC : AC = 50) (hAB : AB = 60) :
  let XA := (BC * AB) / AC 
  let AY := (BC * AC) / AB
  let XY := XA + AY
  XY = 61 :=
by
  sorry

end compute_XY_l129_129300


namespace complex_expression_eq_l129_129403

open Real

theorem complex_expression_eq (p q : ℝ) (hpq : p ≠ q) :
  (sqrt ((p^4 + q^4)/(p^4 - p^2 * q^2) + (2 * q^2)/(p^2 - q^2)) * (p^3 - p * q^2) - 2 * q * sqrt p) /
  (sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)) = 
  sqrt (p^2 - q^2) / sqrt p := 
sorry

end complex_expression_eq_l129_129403


namespace evaluate_difference_of_squares_l129_129631

theorem evaluate_difference_of_squares :
  (50^2 - 30^2 = 1600) :=
by sorry

end evaluate_difference_of_squares_l129_129631


namespace math_problem_solution_l129_129402

theorem math_problem_solution (pA : ℚ) (pB : ℚ)
  (hA : pA = 1/2) (hB : pB = 1/3) :
  let pNoSolve := (1 - pA) * (1 - pB)
  let pSolve := 1 - pNoSolve
  pNoSolve = 1/3 ∧ pSolve = 2/3 :=
by
  sorry

end math_problem_solution_l129_129402


namespace area_of_annulus_l129_129368

variable {b c h : ℝ}
variable (hb : b > c)
variable (h2 : h^2 = b^2 - 2 * c^2)

theorem area_of_annulus (hb : b > c) (h2 : h^2 = b^2 - 2 * c^2) :
    π * (b^2 - c^2) = π * h^2 := by
  sorry

end area_of_annulus_l129_129368


namespace exp_function_not_increasing_l129_129647

open Real

theorem exp_function_not_increasing (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a < 1) :
  ¬(∀ x₁ x₂ : ℝ, x₁ < x₂ → a^x₁ < a^x₂) := by
  sorry

end exp_function_not_increasing_l129_129647


namespace reciprocal_roots_l129_129366

theorem reciprocal_roots (a b : ℝ) (h : a ≠ 0) :
  ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + a = 0) ∧ (a * x2^2 + b * x2 + a = 0) → x1 = 1 / x2 ∧ x2 = 1 / x1 :=
by
  intros x1 x2 hroots
  have hsum : x1 + x2 = -b / a := by sorry
  have hprod : x1 * x2 = 1 := by sorry
  sorry

end reciprocal_roots_l129_129366


namespace problem_statement_l129_129148

noncomputable def a : ℝ := 6 * Real.sqrt 2
noncomputable def b : ℝ := 18 * Real.sqrt 2
noncomputable def c : ℝ := 6 * Real.sqrt 21
noncomputable def d : ℝ := 24 * Real.sqrt 2
noncomputable def e : ℝ := 48 * Real.sqrt 2
noncomputable def N : ℝ := 756 * Real.sqrt 10

axiom condition_a : a^2 + b^2 + c^2 + d^2 + e^2 = 504
axiom positive_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

theorem problem_statement : N + a + b + c + d + e = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by
  -- We'll insert the proof here later
  sorry

end problem_statement_l129_129148


namespace value_of_a_plus_b_is_zero_l129_129797

noncomputable def sum_geometric_sequence (a b : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + b

theorem value_of_a_plus_b_is_zero (a b : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = sum_geometric_sequence a b n) :
  a + b = 0 := 
sorry

end value_of_a_plus_b_is_zero_l129_129797


namespace exist_two_quadrilaterals_l129_129553

-- Define the structure of a quadrilateral with four sides and two diagonals
structure Quadrilateral :=
  (s1 : ℝ) -- side 1
  (s2 : ℝ) -- side 2
  (s3 : ℝ) -- side 3
  (s4 : ℝ) -- side 4
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2

-- The theorem stating the existence of two quadrilaterals satisfying the given conditions
theorem exist_two_quadrilaterals :
  ∃ (quad1 quad2 : Quadrilateral),
  quad1.s1 < quad2.s1 ∧ quad1.s2 < quad2.s2 ∧ quad1.s3 < quad2.s3 ∧ quad1.s4 < quad2.s4 ∧
  quad1.d1 > quad2.d1 ∧ quad1.d2 > quad2.d2 :=
by
  sorry

end exist_two_quadrilaterals_l129_129553


namespace total_revenue_l129_129674

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end total_revenue_l129_129674


namespace p_implies_q_l129_129957

theorem p_implies_q (x : ℝ) (h : |5 * x - 1| > 4) : x^2 - (3/2) * x + (1/2) > 0 := sorry

end p_implies_q_l129_129957


namespace Tim_pencils_value_l129_129011

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l129_129011


namespace maximize_S_n_decreasing_arithmetic_sequence_l129_129374

theorem maximize_S_n_decreasing_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d < 0)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (h4 : S 5 = S 10) :
  S 7 = S 8 :=
sorry

end maximize_S_n_decreasing_arithmetic_sequence_l129_129374


namespace annual_income_A_l129_129743

variable (A B C : ℝ)
variable (monthly_income_C : C = 17000)
variable (monthly_income_B : B = C + 0.12 * C)
variable (ratio_A_to_B : A / B = 5 / 2)

theorem annual_income_A (A B C : ℝ) 
    (hC : C = 17000) 
    (hB : B = C + 0.12 * C) 
    (hR : A / B = 5 / 2) : 
    A * 12 = 571200 :=
by
  sorry

end annual_income_A_l129_129743


namespace solve_for_y_l129_129322

theorem solve_for_y (y : ℝ) (h : (1 / 4) - (1 / 6) = 2 / y) : y = 24 :=
sorry

end solve_for_y_l129_129322


namespace q_value_l129_129410

noncomputable def prove_q (a b m p q : Real) :=
  (a * b = 5) → 
  (b + 1/a) * (a + 1/b) = q →
  q = 36/5

theorem q_value (a b : ℝ) (h_roots : a * b = 5) : (b + 1/a) * (a + 1/b) = 36 / 5 :=
by 
  sorry

end q_value_l129_129410


namespace find_point_B_l129_129630

structure Point where
  x : ℝ
  y : ℝ

def vec_scalar_mult (c : ℝ) (v : Point) : Point :=
  ⟨c * v.x, c * v.y⟩

def vec_add (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem find_point_B :
  let A := Point.mk 1 (-3)
  let a := Point.mk 3 4
  let B := vec_add A (vec_scalar_mult 2 a)
  B = Point.mk 7 5 :=
by {
  sorry
}

end find_point_B_l129_129630


namespace marcus_baseball_cards_l129_129697

/-- 
Marcus initially has 210.0 baseball cards.
Carter gives Marcus 58.0 more baseball cards.
Prove that Marcus now has 268.0 baseball cards.
-/
theorem marcus_baseball_cards (initial_cards : ℝ) (additional_cards : ℝ) 
  (h_initial : initial_cards = 210.0) (h_additional : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 :=
  by
    sorry

end marcus_baseball_cards_l129_129697


namespace area_quadrilateral_is_60_l129_129638

-- Definitions of the lengths of the quadrilateral sides and the ratio condition
def AB : ℝ := 8
def BC : ℝ := 5
def CD : ℝ := 17
def DA : ℝ := 10

-- Function representing the area of the quadrilateral ABCD
def area_ABCD (AB BC CD DA : ℝ) (ratio: ℝ) : ℝ :=
  -- Here we define the function to calculate the area, incorporating the given ratio
  sorry

-- The theorem to show that the area of quadrilateral ABCD is 60
theorem area_quadrilateral_is_60 : 
  area_ABCD AB BC CD DA (1/2) = 60 :=
by
  sorry

end area_quadrilateral_is_60_l129_129638


namespace product_of_solutions_l129_129794

theorem product_of_solutions (x : ℝ) (h : |(18 / x) - 6| = 3) : 2 * 6 = 12 :=
by
  sorry

end product_of_solutions_l129_129794


namespace arithmetic_geometric_mean_inequality_l129_129978

open BigOperators

noncomputable def A (a : Fin n → ℝ) : ℝ := (Finset.univ.sum a) / n

noncomputable def G (a : Fin n → ℝ) : ℝ := (Finset.univ.prod a) ^ (1 / n)

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : A a ≥ G a :=
  sorry

end arithmetic_geometric_mean_inequality_l129_129978


namespace initial_positions_2048_l129_129518

noncomputable def number_of_initial_positions (n : ℕ) : ℤ :=
  2 ^ n - 2

theorem initial_positions_2048 : number_of_initial_positions 2048 = 2 ^ 2048 - 2 :=
by
  sorry

end initial_positions_2048_l129_129518


namespace distance_Q_to_EH_l129_129707

noncomputable def N : ℝ × ℝ := (3, 0)
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 9
noncomputable def EH_line (y : ℝ) : Prop := y = 6

theorem distance_Q_to_EH :
  ∃ (Q : ℝ × ℝ), circle1 Q.1 Q.2 ∧ circle2 Q.1 Q.2 ∧ Q ≠ (0, 0) ∧ abs (Q.2 - 6) = 19 / 3 := sorry

end distance_Q_to_EH_l129_129707


namespace ellipse_equation_l129_129124

theorem ellipse_equation (a b c c1 : ℝ)
  (h_hyperbola_eq : ∀ x y, (y^2 / 4 - x^2 / 12 = 1))
  (h_sum_eccentricities : (c / a) + (c1 / 2) = 13 / 5)
  (h_foci_x_axis : c1 = 4) :
  (a = 5 ∧ b = 4 ∧ c = 3) → 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l129_129124


namespace find_starting_number_l129_129042

theorem find_starting_number :
  ∃ startnum : ℕ, startnum % 5 = 0 ∧ (∀ k : ℕ, 0 ≤ k ∧ k < 20 → startnum + 5 * k ≤ 100) ∧ startnum = 10 :=
sorry

end find_starting_number_l129_129042


namespace salary_of_N_l129_129135

theorem salary_of_N (total_salary : ℝ) (percent_M_from_N : ℝ) (N_salary : ℝ) : 
  (percent_M_from_N * N_salary + N_salary = total_salary) → (N_salary = 280) :=
by
  sorry

end salary_of_N_l129_129135


namespace golden_ratio_problem_l129_129321

theorem golden_ratio_problem
  (m n : ℝ) (sin cos : ℝ → ℝ)
  (h1 : m = 2 * sin (Real.pi / 10))
  (h2 : m ^ 2 + n = 4)
  (sin63 : sin (7 * Real.pi / 18) ≠ 0) :
  (m + Real.sqrt n) / (sin (7 * Real.pi / 18)) = 2 * Real.sqrt 2 := by
  sorry

end golden_ratio_problem_l129_129321


namespace ellie_oil_needs_l129_129454

def oil_per_wheel : ℕ := 10
def number_of_wheels : ℕ := 2
def oil_for_rest : ℕ := 5
def total_oil_needed : ℕ := oil_per_wheel * number_of_wheels + oil_for_rest

theorem ellie_oil_needs : total_oil_needed = 25 := by
  sorry

end ellie_oil_needs_l129_129454


namespace mans_rate_in_still_water_l129_129406

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water_l129_129406


namespace ratio_enlarged_by_nine_l129_129964

theorem ratio_enlarged_by_nine (a b : ℕ) (h : b ≠ 0) :
  (3 * a) / (b / 3) = 9 * (a / b) :=
by
  have h1 : b / 3 ≠ 0 := by sorry
  have h2 : a * 3 ≠ 0 := by sorry
  sorry

end ratio_enlarged_by_nine_l129_129964


namespace root_situation_l129_129162

theorem root_situation (a b : ℝ) : 
  ∃ (m n : ℝ), 
    (x - a) * (x - (a + b)) = 1 → 
    (m < a ∧ a < n) ∨ (n < a ∧ a < m) :=
sorry

end root_situation_l129_129162


namespace prime_factors_of_69_l129_129128

theorem prime_factors_of_69 
  (prime : ℕ → Prop)
  (is_prime : ∀ n, prime n ↔ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ 
                        n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23)
  (x y : ℕ)
  (h1 : 15 < 69)
  (h2 : 69 < 70)
  (h3 : prime y)
  (h4 : 13 < y)
  (h5 : y < 25)
  (h6 : 69 = x * y)
  : prime x ∧ x = 3 := 
sorry

end prime_factors_of_69_l129_129128


namespace log_expression_value_l129_129542

theorem log_expression_value : 
  let log4_3 := (Real.log 3) / (Real.log 4)
  let log8_3 := (Real.log 3) / (Real.log 8)
  let log3_2 := (Real.log 2) / (Real.log 3)
  let log9_2 := (Real.log 2) / (Real.log 9)
  (log4_3 + log8_3) * (log3_2 + log9_2) = 5 / 4 := 
by
  sorry

end log_expression_value_l129_129542


namespace good_set_exists_l129_129273

def is_good_set (A : List ℕ) : Prop :=
  ∀ i ∈ A, i > 0 ∧ ∀ j ∈ A, i ≠ j → i ^ 2015 % (List.prod (A.erase i)) = 0

theorem good_set_exists (n : ℕ) (h : 3 ≤ n ∧ n ≤ 2015) : 
  ∃ A : List ℕ, A.length = n ∧ ∀ (a : ℕ), a ∈ A → a > 0 ∧ is_good_set A :=
sorry

end good_set_exists_l129_129273


namespace geometric_sequence_a5_l129_129531

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n, a n + a (n + 1) = 3 * (1 / 2) ^ n)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 1 / 2) :
  a 5 = 1 / 16 :=
sorry

end geometric_sequence_a5_l129_129531


namespace ratio_of_friday_to_thursday_l129_129012

theorem ratio_of_friday_to_thursday
  (wednesday_copies : ℕ)
  (total_copies : ℕ)
  (ratio : ℚ)
  (h1 : wednesday_copies = 15)
  (h2 : total_copies = 69)
  (h3 : ratio = 1 / 5) :
  (total_copies - wednesday_copies - 3 * wednesday_copies) / (3 * wednesday_copies) = ratio :=
by
  -- proof goes here
  sorry

end ratio_of_friday_to_thursday_l129_129012


namespace number_of_valid_sequences_l129_129659

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end number_of_valid_sequences_l129_129659


namespace max_area_of_sector_l129_129691

theorem max_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 12) : 
  (1 / 2) * l * r ≤ 9 :=
by sorry

end max_area_of_sector_l129_129691


namespace area_of_largest_circle_l129_129010

theorem area_of_largest_circle (side_length : ℝ) (h : side_length = 2) : 
  (Real.pi * (side_length / 2)^2 = 3.14) :=
by
  sorry

end area_of_largest_circle_l129_129010


namespace find_some_number_l129_129788

theorem find_some_number (some_number : ℕ) : 
  ( ∃ n:ℕ, n = 54 ∧ (n / 18) * (n / some_number) = 1 ) ∧ some_number = 162 :=
by {
  sorry
}

end find_some_number_l129_129788


namespace smallest_value_36k_minus_5l_l129_129609

theorem smallest_value_36k_minus_5l (k l : ℕ) :
  ∃ k l, 0 < 36^k - 5^l ∧ (∀ k' l', (0 < 36^k' - 5^l' → 36^k - 5^l ≤ 36^k' - 5^l')) ∧ 36^k - 5^l = 11 :=
by sorry

end smallest_value_36k_minus_5l_l129_129609


namespace max_xy_value_l129_129827

theorem max_xy_value {x y : ℝ} (h : 2 * x + y = 1) : ∃ z, z = x * y ∧ z = 1 / 8 :=
by sorry

end max_xy_value_l129_129827


namespace minutes_before_noon_l129_129778

theorem minutes_before_noon
    (x : ℕ)
    (h1 : 20 <= x)
    (h2 : 180 - (x - 20) = 3 * (x - 20)) :
    x = 65 := by
  sorry

end minutes_before_noon_l129_129778


namespace ratio_passengers_i_to_ii_l129_129888

-- Definitions: Conditions from the problem
variables (total_fare : ℕ) (fare_ii_class : ℕ) (fare_i_class_ratio_to_ii : ℕ)

-- Given conditions
axiom total_fare_collected : total_fare = 1325
axiom fare_collected_from_ii_class : fare_ii_class = 1250
axiom i_to_ii_fare_ratio : fare_i_class_ratio_to_ii = 3

-- Define the fare for I class and II class passengers
def fare_i_class := 3 * (fare_ii_class / fare_i_class_ratio_to_ii)

-- Statement of the proof problem translating the question, conditions, and answer
theorem ratio_passengers_i_to_ii (x y : ℕ) (h1 : 3 * fare_i_class * x = total_fare - fare_ii_class)
    (h2 : (fare_ii_class / fare_i_class_ratio_to_ii) * y = fare_ii_class) : x = y / 50 :=
by
  sorry

end ratio_passengers_i_to_ii_l129_129888


namespace Karen_has_fewer_nail_polishes_than_Kim_l129_129003

theorem Karen_has_fewer_nail_polishes_than_Kim :
  ∀ (Kim Heidi Karen : ℕ), Kim = 12 → Heidi = Kim + 5 → Karen + Heidi = 25 → (Kim - Karen) = 4 :=
by
  intros Kim Heidi Karen hK hH hKH
  sorry

end Karen_has_fewer_nail_polishes_than_Kim_l129_129003


namespace shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l129_129234

noncomputable def k : ℝ := (1 / 20) * Real.log (1 / 4)
noncomputable def b : ℝ := Real.log 160
noncomputable def y (x : ℝ) : ℝ := Real.exp (k * x + b)

theorem shelf_life_at_30_degrees : y 30 = 20 := sorry

theorem temperature_condition_for_shelf_life (x : ℝ) : y x ≥ 80 → x ≤ 10 := sorry

end shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l129_129234


namespace gcf_120_180_240_l129_129655

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l129_129655


namespace initial_carrots_l129_129355

theorem initial_carrots (n : ℕ) 
    (h1: 3640 = 180 * (n - 4) + 760) 
    (h2: 180 * (n - 4) < 3640) 
    (h3: 4 * 190 = 760) : 
    n = 20 :=
by
  sorry

end initial_carrots_l129_129355


namespace train_speed_l129_129185

/-
Problem Statement:
Prove that the speed of a train is 26.67 meters per second given:
  1. The length of the train is 320 meters.
  2. The time taken to cross the telegraph post is 12 seconds.
-/

theorem train_speed (distance time : ℝ) (h1 : distance = 320) (h2 : time = 12) :
  (distance / time) = 26.67 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l129_129185


namespace inequality_abc_l129_129883

theorem inequality_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1/a + 1/(b * c)) * (1/b + 1/(c * a)) * (1/c + 1/(a * b)) ≥ 1728 :=
by sorry

end inequality_abc_l129_129883


namespace line_pass_through_point_l129_129642

theorem line_pass_through_point (k b : ℝ) (x1 x2 : ℝ) (h1: b ≠ 0) (h2: x1^2 - k*x1 - b = 0) (h3: x2^2 - k*x2 - b = 0)
(h4: x1 + x2 = k) (h5: x1 * x2 = -b) 
(h6: (k^2 * (-b) + k * b * k + b^2 = b^2) = true) : 
  ∃ (x y : ℝ), (y = k * x + 1) ∧ (x, y) = (0, 1) :=
by
  sorry

end line_pass_through_point_l129_129642


namespace rectangle_area_formula_l129_129648

-- Define the given conditions: perimeter is 20, one side length is x
def rectangle_perimeter (P x : ℝ) (w : ℝ) : Prop := P = 2 * (x + w)
def rectangle_area (x w : ℝ) : ℝ := x * w

-- The theorem to prove
theorem rectangle_area_formula (x : ℝ) (h_perimeter : rectangle_perimeter 20 x (10 - x)) : 
  rectangle_area x (10 - x) = x * (10 - x) := 
by 
  sorry

end rectangle_area_formula_l129_129648


namespace ramesh_share_correct_l129_129361

-- Define basic conditions
def suresh_investment := 24000
def ramesh_investment := 40000
def total_profit := 19000

-- Define Ramesh's share calculation
def ramesh_share : ℤ :=
  let ratio_ramesh := ramesh_investment / (suresh_investment + ramesh_investment)
  ratio_ramesh * total_profit

-- Proof statement
theorem ramesh_share_correct : ramesh_share = 11875 := by
  sorry

end ramesh_share_correct_l129_129361


namespace base_area_cone_l129_129549

theorem base_area_cone (V h : ℝ) (s_cylinder s_cone : ℝ) 
  (cylinder_volume : V = s_cylinder * h) 
  (cone_volume : V = (1 / 3) * s_cone * h) 
  (s_cylinder_val : s_cylinder = 15) : s_cone = 45 := 
by 
  sorry

end base_area_cone_l129_129549


namespace problem1_problem2_l129_129731

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l129_129731


namespace range_of_p_l129_129276

noncomputable def f (x p : ℝ) : ℝ := x - p/x + p/2

theorem range_of_p (p : ℝ) :
  (∀ x : ℝ, 1 < x → (1 + p / x^2) > 0) → p ≥ -1 :=
by
  intro h
  sorry

end range_of_p_l129_129276


namespace square_diagonal_l129_129399

theorem square_diagonal (p : ℤ) (h : p = 28) : ∃ d : ℝ, d = 7 * Real.sqrt 2 :=
by
  sorry

end square_diagonal_l129_129399


namespace number_of_friends_l129_129701

-- Define the conditions
def initial_apples := 55
def apples_given_to_father := 10
def apples_per_person := 9

-- Define the formula to calculate the number of friends
def friends (initial_apples apples_given_to_father apples_per_person : ℕ) : ℕ :=
  (initial_apples - apples_given_to_father - apples_per_person) / apples_per_person

-- State the Lean theorem
theorem number_of_friends :
  friends initial_apples apples_given_to_father apples_per_person = 4 :=
by
  sorry

end number_of_friends_l129_129701


namespace maximum_x_plus_2y_l129_129593

theorem maximum_x_plus_2y 
  (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^2 + 8 * y^2 + x * y = 2) :
  x + 2 * y ≤ 4 / 3 :=
sorry

end maximum_x_plus_2y_l129_129593


namespace value_of_k_l129_129751

theorem value_of_k : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := 
by 
  sorry

end value_of_k_l129_129751


namespace g_seven_l129_129305

def g (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem g_seven : g 7 = 17 / 23 := by
  sorry

end g_seven_l129_129305


namespace max_value_A_l129_129246

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem max_value_A (A : ℝ) (hA : A = Real.pi / 6) : 
  ∀ x : ℝ, f x ≤ f A :=
sorry

end max_value_A_l129_129246


namespace abs_eq_condition_l129_129582

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + |x - 5| = 4) : 1 ≤ x ∧ x ≤ 5 :=
by 
  sorry

end abs_eq_condition_l129_129582


namespace terms_before_one_l129_129954

-- Define the sequence parameters
def a : ℤ := 100
def d : ℤ := -7
def nth_term (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the target term we are interested in
def target_term : ℤ := 1

-- Define the main theorem
theorem terms_before_one : ∃ n : ℕ, nth_term n = target_term ∧ (n - 1) = 14 := by
  sorry

end terms_before_one_l129_129954


namespace find_m_for_min_value_l129_129818

theorem find_m_for_min_value :
  ∃ (m : ℝ), ( ∀ x : ℝ, (y : ℝ) = m * x^2 - 4 * x + 1 → (∃ x_min : ℝ, (∀ x : ℝ, (m * x_min^2 - 4 * x_min + 1 ≤ m * x^2 - 4 * x + 1) → y = -3))) :=
sorry

end find_m_for_min_value_l129_129818


namespace find_setC_l129_129008

def setA := {x : ℝ | x^2 - 3 * x + 2 = 0}
def setB (a : ℝ) := {x : ℝ | a * x - 2 = 0}
def union_condition (a : ℝ) : Prop := (setA ∪ setB a) = setA
def setC := {a : ℝ | union_condition a}

theorem find_setC : setC = {0, 1, 2} :=
by
  sorry

end find_setC_l129_129008


namespace exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l129_129363

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_arithmetic_progression_with_11_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 11 → j < 11 → i < j → a + i * d < a + j * d ∧ 
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem exists_arithmetic_progression_with_10000_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 10000 → j < 10000 → i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem not_exists_infinite_arithmetic_progression :
  ¬ (∃ a d : ℕ, ∀ i j : ℕ, i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d)) := by
  sorry

end exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l129_129363


namespace find_slope_l129_129854

theorem find_slope 
  (k : ℝ)
  (y : ℝ -> ℝ)
  (P : ℝ × ℝ)
  (l : ℝ -> ℝ -> Prop)
  (A B F : ℝ × ℝ)
  (C : ℝ × ℝ -> Prop)
  (d : ℝ × ℝ -> ℝ × ℝ -> ℝ)
  (k_pos : P = (3, 0))
  (k_slope : ∀ x, y x = k * (x - 3))
  (k_int_hyperbola_A : C A)
  (k_int_hyperbola_B : C B)
  (k_focus : F = (2, 0))
  (k_sum_dist : d A F + d B F = 16) :
  k = 1 ∨ k = -1 :=
sorry

end find_slope_l129_129854


namespace multiple_of_four_and_six_prime_sum_even_l129_129019

theorem multiple_of_four_and_six_prime_sum_even {a b : ℤ} 
  (h_a : ∃ m : ℤ, a = 4 * m) 
  (h_b1 : ∃ n : ℤ, b = 6 * n) 
  (h_b2 : Prime b) : 
  Even (a + b) := 
  by sorry

end multiple_of_four_and_six_prime_sum_even_l129_129019


namespace area_of_shaded_region_l129_129369

theorem area_of_shaded_region :
  let v1 := (0, 0)
  let v2 := (15, 0)
  let v3 := (45, 30)
  let v4 := (45, 45)
  let v5 := (30, 45)
  let v6 := (0, 15)
  let area_large_rectangle := 45 * 45
  let area_triangle1 := 1 / 2 * 15 * 15
  let area_triangle2 := 1 / 2 * 15 * 15
  let shaded_area := area_large_rectangle - (area_triangle1 + area_triangle2)
  shaded_area = 1800 :=
by
  sorry

end area_of_shaded_region_l129_129369


namespace count_valid_propositions_is_zero_l129_129699

theorem count_valid_propositions_is_zero :
  (∀ (a b : ℝ), (a > b → a^2 > b^2) = false) ∧
  (∀ (a b : ℝ), (a^2 > b^2 → a > b) = false) ∧
  (∀ (a b : ℝ), (a > b → b / a < 1) = false) ∧
  (∀ (a b : ℝ), (a > b → 1 / a < 1 / b) = false) :=
by
  sorry

end count_valid_propositions_is_zero_l129_129699


namespace factors_of_12_factors_of_18_l129_129331

def is_factor (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem factors_of_12 : 
  {k : ℕ | is_factor 12 k} = {1, 12, 2, 6, 3, 4} :=
by
  sorry

theorem factors_of_18 : 
  {k : ℕ | is_factor 18 k} = {1, 18, 2, 9, 3, 6} :=
by
  sorry

end factors_of_12_factors_of_18_l129_129331


namespace money_left_l129_129950

noncomputable def olivia_money : ℕ := 112
noncomputable def nigel_money : ℕ := 139
noncomputable def ticket_cost : ℕ := 28
noncomputable def num_tickets : ℕ := 6

theorem money_left : (olivia_money + nigel_money - ticket_cost * num_tickets) = 83 :=
by
  sorry

end money_left_l129_129950


namespace proof_stmt_l129_129724

variable (a x y : ℝ)
variable (ha : a > 0) (hneq : a ≠ 1)

noncomputable def S (x : ℝ) := a^x - a^(-x)
noncomputable def C (x : ℝ) := a^x + a^(-x)

theorem proof_stmt :
  2 * S a (x + y) = S a x * C a y + C a x * S a y ∧
  2 * S a (x - y) = S a x * C a y - C a x * S a y :=
by sorry

end proof_stmt_l129_129724


namespace largest_cyclic_decimal_l129_129274

def digits_on_circle := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def max_cyclic_decimal : ℕ := sorry

theorem largest_cyclic_decimal :
  max_cyclic_decimal = 957913 :=
sorry

end largest_cyclic_decimal_l129_129274


namespace residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l129_129391

noncomputable def phi (n : ℕ) : ℕ := Nat.totient n

theorem residues_exponent (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d ∧ ∀ x ∈ S, x^d % p = 1 :=
by sorry

theorem residues_divides_p_minus_one (p : ℕ) (d : ℕ) [hp : Fact (Nat.Prime p)] (hd : d ∣ p - 1) : 
  ∃ (S : Finset ℕ), S.card = phi d :=
by sorry
  
theorem primitive_roots_phi (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ∃ (S : Finset ℕ), S.card = phi (p-1) ∧ ∀ g ∈ S, IsPrimitiveRoot g p :=
by sorry

end residues_exponent_residues_divides_p_minus_one_primitive_roots_phi_l129_129391


namespace value_of_x_l129_129152

theorem value_of_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 :=
by
  sorry

end value_of_x_l129_129152


namespace product_of_integers_l129_129440

theorem product_of_integers (a b : ℤ) (h_lcm : Int.lcm a b = 45) (h_gcd : Int.gcd a b = 9) : a * b = 405 :=
by
  sorry

end product_of_integers_l129_129440


namespace original_number_is_28_l129_129123

theorem original_number_is_28 (N : ℤ) :
  (∃ k : ℤ, N - 11 = 17 * k) → N = 28 :=
by
  intro h
  obtain ⟨k, h₁⟩ := h
  have h₂: N = 17 * k + 11 := by linarith
  have h₃: k = 1 := sorry
  linarith [h₃]
 
end original_number_is_28_l129_129123


namespace parallelogram_side_lengths_l129_129737

theorem parallelogram_side_lengths (x y : ℚ) 
  (h1 : 12 * x - 2 = 10) 
  (h2 : 5 * y + 5 = 4) : 
  x + y = 4 / 5 := 
by 
  sorry

end parallelogram_side_lengths_l129_129737


namespace equivalent_single_reduction_l129_129491

theorem equivalent_single_reduction :
  ∀ (P : ℝ), P * (1 - 0.25) * (1 - 0.20) = P * (1 - 0.40) :=
by
  intros P
  -- Proof will be skipped
  sorry

end equivalent_single_reduction_l129_129491


namespace total_invested_amount_l129_129264

theorem total_invested_amount :
  ∃ (A B : ℝ), (A = 3000 ∧ B = 5000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000)
  ∨ 
  (A = 5000 ∧ B = 3000 ∧ 
  0.085 * A + 0.064 * B = 575 ∧ A + B = 8000) :=
sorry

end total_invested_amount_l129_129264


namespace problem_statement_l129_129484

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem problem_statement :
  let l := { p : ℝ × ℝ | p.1 - p.2 - 2 = 0 }
  let C := { p : ℝ × ℝ | ∃ θ : ℝ, p = (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ) }
  let A := (-4, -6)
  let B := (4, 2)
  let P := (-2 * Real.sqrt 3, 2)
  let d := (|2 * Real.sqrt 3 * Real.cos (5 * Real.pi / 6) - 2|) / Real.sqrt 2
  distance A B = 8 * Real.sqrt 2 ∧ d = 3 * Real.sqrt 2 ∧
  let max_area := 1 / 2 * 8 * Real.sqrt 2 * 3 * Real.sqrt 2
  P ∈ C ∧ max_area = 24 := by
sorry

end problem_statement_l129_129484


namespace payal_book_length_l129_129177

theorem payal_book_length (P : ℕ) 
  (h1 : (2/3 : ℚ) * P = (1/3 : ℚ) * P + 20) : P = 60 :=
sorry

end payal_book_length_l129_129177


namespace equilateral_triangle_perimeter_twice_side_area_l129_129382

noncomputable def triangle_side_length (s : ℝ) :=
  s * s * Real.sqrt 3 / 4 = 2 * s

noncomputable def triangle_perimeter (s : ℝ) := 3 * s

theorem equilateral_triangle_perimeter_twice_side_area (s : ℝ) (h : triangle_side_length s) : 
  triangle_perimeter s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_twice_side_area_l129_129382


namespace flower_bed_can_fit_l129_129592

noncomputable def flower_bed_fits_in_yard : Prop :=
  let yard_side := 70
  let yard_area := yard_side ^ 2
  let building1 := (20 * 10)
  let building2 := (25 * 15)
  let building3 := (30 * 30)
  let tank_radius := 10 / 2
  let tank_area := Real.pi * tank_radius^2
  let total_occupied_area := building1 + building2 + building3 + 2*tank_area
  let available_area := yard_area - total_occupied_area
  let flower_bed_radius := 10 / 2
  let flower_bed_area := Real.pi * flower_bed_radius^2
  let buffer_area := (yard_side - 2 * flower_bed_radius)^2
  available_area >= flower_bed_area ∧ buffer_area >= flower_bed_area

theorem flower_bed_can_fit : flower_bed_fits_in_yard := 
  sorry

end flower_bed_can_fit_l129_129592


namespace survived_trees_difference_l129_129721

theorem survived_trees_difference {original_trees died_trees survived_trees: ℕ} 
  (h1 : original_trees = 13) 
  (h2 : died_trees = 6)
  (h3 : survived_trees = original_trees - died_trees) :
  survived_trees - died_trees = 1 :=
by
  sorry

end survived_trees_difference_l129_129721


namespace number_of_donut_selections_l129_129789

-- Definitions for the problem
def g : ℕ := sorry
def c : ℕ := sorry
def p : ℕ := sorry

-- Condition: Pat wants to buy four donuts from three types
def equation : Prop := g + c + p = 4

-- Question: Prove the number of different selections possible
theorem number_of_donut_selections : (∃ n, n = 15) := 
by 
  -- Use combinatorial method to establish this
  sorry

end number_of_donut_selections_l129_129789


namespace relationship_depends_on_b_l129_129154

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b > a - b ∨ a + b < a - b ∨ a + b = a - b) ↔ (b > 0 ∨ b < 0 ∨ b = 0) :=
by
  sorry

end relationship_depends_on_b_l129_129154


namespace secondChapterPages_is_18_l129_129909

-- Define conditions as variables and constants
def thirdChapterPages : ℕ := 3
def additionalPages : ℕ := 15

-- The main statement to prove
theorem secondChapterPages_is_18 : (thirdChapterPages + additionalPages) = 18 := by
  -- Proof would go here, but we skip it with sorry
  sorry

end secondChapterPages_is_18_l129_129909


namespace train_pass_bridge_in_36_seconds_l129_129160

def train_length : ℝ := 360 -- meters
def bridge_length : ℝ := 140 -- meters
def train_speed_kmh : ℝ := 50 -- km/h

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- m/s
noncomputable def total_distance : ℝ := train_length + bridge_length -- meters
noncomputable def passing_time : ℝ := total_distance / train_speed_ms -- seconds

theorem train_pass_bridge_in_36_seconds :
  passing_time = 36 := 
sorry

end train_pass_bridge_in_36_seconds_l129_129160


namespace rate_per_kg_first_batch_l129_129528

/-- This theorem proves the rate per kg of the first batch of wheat. -/
theorem rate_per_kg_first_batch (x : ℝ) 
  (h1 : 30 * x + 20 * 14.25 = 285 + 30 * x) 
  (h2 : (30 * x + 285) * 1.3 = 819) : 
  x = 11.5 := 
sorry

end rate_per_kg_first_batch_l129_129528


namespace range_of_m_l129_129344

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
sorry

end range_of_m_l129_129344


namespace mrs_hilt_candy_l129_129795

theorem mrs_hilt_candy : 2 * 9 + 3 * 9 + 1 * 9 = 54 :=
by
  sorry

end mrs_hilt_candy_l129_129795


namespace find_a_in_triangle_l129_129603

theorem find_a_in_triangle (C : ℝ) (b c : ℝ) (hC : C = 60) (hb : b = 1) (hc : c = Real.sqrt 3) :
  ∃ (a : ℝ), a = 2 := 
by
  sorry

end find_a_in_triangle_l129_129603


namespace statues_at_end_of_fourth_year_l129_129039

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end statues_at_end_of_fourth_year_l129_129039


namespace racers_in_final_segment_l129_129463

def initial_racers := 200

def racers_after_segment_1 (initial: ℕ) := initial - 10
def racers_after_segment_2 (after_segment_1: ℕ) := after_segment_1 - after_segment_1 / 3
def racers_after_segment_3 (after_segment_2: ℕ) := after_segment_2 - after_segment_2 / 4
def racers_after_segment_4 (after_segment_3: ℕ) := after_segment_3 - after_segment_3 / 3
def racers_after_segment_5 (after_segment_4: ℕ) := after_segment_4 - after_segment_4 / 2
def racers_after_segment_6 (after_segment_5: ℕ) := after_segment_5 - (3 * after_segment_5 / 4)

theorem racers_in_final_segment : racers_after_segment_6 (racers_after_segment_5 (racers_after_segment_4 (racers_after_segment_3 (racers_after_segment_2 (racers_after_segment_1 initial_racers))))) = 8 :=
  by
  sorry

end racers_in_final_segment_l129_129463


namespace parallelogram_base_l129_129622

theorem parallelogram_base (A h b : ℝ) (hA : A = 375) (hh : h = 15) : b = 25 :=
by
  sorry

end parallelogram_base_l129_129622


namespace trader_sold_80_meters_l129_129937

variable (x : ℕ)
variable (selling_price_per_meter profit_per_meter cost_price_per_meter total_selling_price : ℕ)

theorem trader_sold_80_meters
  (h_cost_price : cost_price_per_meter = 118)
  (h_profit : profit_per_meter = 7)
  (h_selling_price : selling_price_per_meter = cost_price_per_meter + profit_per_meter)
  (h_total_selling_price : total_selling_price = 10000)
  (h_eq : selling_price_per_meter * x = total_selling_price) :
  x = 80 := by
    sorry

end trader_sold_80_meters_l129_129937


namespace cost_per_gumball_l129_129722

theorem cost_per_gumball (total_money : ℕ) (num_gumballs : ℕ) (cost_each : ℕ) 
  (h1 : total_money = 32) (h2 : num_gumballs = 4) : cost_each = 8 :=
by
  sorry -- Proof omitted

end cost_per_gumball_l129_129722


namespace domain_of_log_function_l129_129268

theorem domain_of_log_function (x : ℝ) :
  (5 - x > 0) ∧ (x - 2 > 0) ∧ (x - 2 ≠ 1) ↔ (2 < x ∧ x < 3) ∨ (3 < x ∧ x < 5) :=
by
  sorry

end domain_of_log_function_l129_129268


namespace find_b_l129_129150

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 315 * b) : b = 7 :=
by
  -- The actual proof would go here
  sorry

end find_b_l129_129150


namespace sum_of_roots_l129_129457

theorem sum_of_roots (z1 z2 : ℂ) (h : z1^2 + 5*z1 - 14 = 0 ∧ z2^2 + 5*z2 - 14 = 0) :
  z1 + z2 = -5 :=
sorry

end sum_of_roots_l129_129457


namespace larger_number_hcf_lcm_l129_129619

theorem larger_number_hcf_lcm (a b : ℕ) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : a = b / 4) : max a b = 84 :=
by
  sorry

end larger_number_hcf_lcm_l129_129619


namespace tree_height_at_3_years_l129_129913

-- Define the conditions as Lean definitions
def tree_height (years : ℕ) : ℕ :=
  2 ^ years

-- State the theorem using the defined conditions
theorem tree_height_at_3_years : tree_height 6 = 32 → tree_height 3 = 4 := by
  intro h
  sorry

end tree_height_at_3_years_l129_129913


namespace visited_both_countries_l129_129544

theorem visited_both_countries (total_people visited_Iceland visited_Norway visited_neither : ℕ) 
(h_total: total_people = 60)
(h_visited_Iceland: visited_Iceland = 35)
(h_visited_Norway: visited_Norway = 23)
(h_visited_neither: visited_neither = 33) : 
total_people - visited_neither = visited_Iceland + visited_Norway - (visited_Iceland + visited_Norway - (total_people - visited_neither)) :=
by sorry

end visited_both_countries_l129_129544


namespace angle_C_in_triangle_l129_129791

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l129_129791


namespace OddPrimeDivisorCondition_l129_129522

theorem OddPrimeDivisorCondition (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) : 
  ∃ p : ℕ, Prime p ∧ n = p ∧ ¬ Even p :=
sorry

end OddPrimeDivisorCondition_l129_129522


namespace min_value_quadratic_l129_129236

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l129_129236


namespace probability_of_same_color_l129_129314

noncomputable def prob_same_color (P_A P_B : ℚ) : ℚ :=
  P_A + P_B

theorem probability_of_same_color :
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  prob_same_color P_A P_B = 17 / 35 := 
by 
  -- Definition of P_A and P_B
  let P_A := (1 : ℚ) / 7
  let P_B := (12 : ℚ) / 35
  -- Use the definition of prob_same_color
  let result := prob_same_color P_A P_B
  -- Now we are supposed to prove that result = 17 / 35
  have : result = (5 : ℚ) / 35 + (12 : ℚ) / 35 := by
    -- Simplifying the fractions individually can be done at this intermediate step
    sorry
  sorry

end probability_of_same_color_l129_129314


namespace max_min_page_difference_l129_129108

-- Define the number of pages in each book
variables (Poetry Documents Rites Changes SpringAndAutumn : ℤ)

-- Define the conditions as given in the problem
axiom h1 : abs (Poetry - Documents) = 24
axiom h2 : abs (Documents - Rites) = 17
axiom h3 : abs (Rites - Changes) = 27
axiom h4 : abs (Changes - SpringAndAutumn) = 19
axiom h5 : abs (SpringAndAutumn - Poetry) = 15

-- Assertion to prove
theorem max_min_page_difference : 
  ∃ a b c d e : ℤ, a = Poetry ∧ b = Documents ∧ c = Rites ∧ d = Changes ∧ e = SpringAndAutumn ∧ 
  abs (a - b) = 24 ∧ abs (b - c) = 17 ∧ abs (c - d) = 27 ∧ abs (d - e) = 19 ∧ abs (e - a) = 15 ∧ 
  (max a (max b (max c (max d e))) - min a (min b (min c (min d e)))) = 34 :=
by {
  sorry
}

end max_min_page_difference_l129_129108


namespace trees_in_garden_l129_129451

theorem trees_in_garden (yard_length distance_between_trees : ℕ) (h1 : yard_length = 800) (h2 : distance_between_trees = 32) :
  ∃ n : ℕ, n = (yard_length / distance_between_trees) + 1 ∧ n = 26 :=
by
  sorry

end trees_in_garden_l129_129451


namespace perpendicular_slope_l129_129972

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end perpendicular_slope_l129_129972


namespace maximize_sector_area_l129_129486

noncomputable def sector_radius_angle (r l α : ℝ) : Prop :=
  2 * r + l = 40 ∧ α = l / r

theorem maximize_sector_area :
  ∃ r α : ℝ, sector_radius_angle r 20 α ∧ r = 10 ∧ α = 2 :=
by
  sorry

end maximize_sector_area_l129_129486


namespace no_sum_of_three_squares_l129_129335

theorem no_sum_of_three_squares (a k : ℕ) : 
  ¬ ∃ x y z : ℤ, 4^a * (8*k + 7) = x^2 + y^2 + z^2 :=
by
  sorry

end no_sum_of_three_squares_l129_129335


namespace doubled_volume_l129_129680

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l129_129680


namespace polynomial_value_l129_129101

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l129_129101


namespace min_value_expression_l129_129299

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) >= 1 / 4) ∧ (x = 1/3 ∧ y = 1/3 ∧ z = 1/3 → x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1 / 4) :=
sorry

end min_value_expression_l129_129299


namespace magnitude_of_complex_l129_129633

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l129_129633


namespace min_value_of_function_product_inequality_l129_129227

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality_l129_129227


namespace lunks_needed_for_12_apples_l129_129684

/-- 
  Given:
  1. 7 lunks can be traded for 4 kunks.
  2. 3 kunks will buy 5 apples.

  Prove that the number of lunks needed to purchase one dozen (12) apples is equal to 14.
-/
theorem lunks_needed_for_12_apples (L K : ℕ)
  (h1 : 7 * L = 4 * K)
  (h2 : 3 * K = 5) :
  (8 * K = 14 * L) :=
by
  sorry

end lunks_needed_for_12_apples_l129_129684


namespace A_pow_five_eq_rA_add_sI_l129_129660

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 1; 4, 3]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem A_pow_five_eq_rA_add_sI :
  ∃ (r s : ℚ), (A^5) = r • A + s • I :=
sorry

end A_pow_five_eq_rA_add_sI_l129_129660


namespace prove_y_eq_x_l129_129688

theorem prove_y_eq_x
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y)
  (h2 : y = 2 + 1 / x) : y = x :=
sorry

end prove_y_eq_x_l129_129688


namespace maximum_absolute_sum_l129_129557

theorem maximum_absolute_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : |x| + |y| + |z| ≤ 2 :=
sorry

end maximum_absolute_sum_l129_129557


namespace halfway_fraction_l129_129056

theorem halfway_fraction (a b : ℚ) (h1 : a = 1/5) (h2 : b = 1/3) : (a + b) / 2 = 4 / 15 :=
by 
  rw [h1, h2]
  norm_num

end halfway_fraction_l129_129056


namespace trajectory_C_find_m_l129_129608

noncomputable def trajectory_C_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 7

theorem trajectory_C (x y : ℝ) (hx : trajectory_C_eq x y) :
  (x - 3)^2 + y^2 = 7 := by
  sorry

theorem find_m (m : ℝ) : (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = 3 + m ∧ x1 * x2 + (1/(2:ℝ)) * ((m^2 + 2)/(2:ℝ)) = 0 ∧ x1 * x2 + (x1 - m) * (x2 - m) = 0) → m = 1 ∨ m = 2 := by
  sorry

end trajectory_C_find_m_l129_129608


namespace solve_system_l129_129493

open Real

theorem solve_system :
  (∃ x y : ℝ, (sin x) ^ 2 + (cos y) ^ 2 = y ^ 4 ∧ (sin y) ^ 2 + (cos x) ^ 2 = x ^ 2) → 
  (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) := by
  sorry

end solve_system_l129_129493


namespace exp_increasing_a_lt_zero_l129_129265

theorem exp_increasing_a_lt_zero (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (1 - a) ^ x1 < (1 - a) ^ x2) : a < 0 := 
sorry

end exp_increasing_a_lt_zero_l129_129265


namespace second_divisor_l129_129413

theorem second_divisor (x : ℕ) (k q : ℤ) : 
  (197 % 13 = 2) → 
  (x > 13) → 
  (197 % x = 5) → 
  x = 16 :=
by sorry

end second_divisor_l129_129413


namespace total_capacity_l129_129971

def eight_liters : ℝ := 8
def percentage : ℝ := 0.20
def num_containers : ℕ := 40

theorem total_capacity (h : eight_liters = percentage * capacity) :
  40 * (eight_liters / percentage) = 1600 := sorry

end total_capacity_l129_129971


namespace all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l129_129735

def coin_values : Set ℤ := {1, 5, 10, 25}

theorem all_values_achievable (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 30) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_1 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 40) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_2 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 50) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_3 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 60) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_4 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 70) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

end all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l129_129735


namespace subset_problem_l129_129629

theorem subset_problem (a : ℝ) (P S : Set ℝ) :
  P = { x | x^2 - 2 * x - 3 = 0 } →
  S = { x | a * x + 2 = 0 } →
  (S ⊆ P) →
  (a = 0 ∨ a = 2 ∨ a = -2 / 3) :=
by
  intro hP hS hSubset
  sorry

end subset_problem_l129_129629


namespace value_of_fraction_l129_129612

theorem value_of_fraction (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 :=
by
  sorry

end value_of_fraction_l129_129612


namespace cannot_determine_orange_groups_l129_129784

-- Definitions of the conditions
def oranges := 87
def bananas := 290
def bananaGroups := 2
def bananasPerGroup := 145

-- Lean statement asserting that the number of groups of oranges 
-- cannot be determined from the given conditions
theorem cannot_determine_orange_groups:
  ∀ (number_of_oranges_per_group : ℕ), 
  (bananasPerGroup * bananaGroups = bananas) ∧ (oranges = 87) → 
  ¬(∃ (number_of_orange_groups : ℕ), oranges = number_of_oranges_per_group * number_of_orange_groups) :=
by
  sorry -- Since we are not required to provide the proof here

end cannot_determine_orange_groups_l129_129784


namespace find_angle_measure_l129_129024

def complement_more_condition (x : ℝ) : Prop :=
  90 - x = (1 / 7) * x + 26

theorem find_angle_measure (x : ℝ) (h : complement_more_condition x) : x = 56 :=
sorry

end find_angle_measure_l129_129024


namespace negated_proposition_l129_129568

theorem negated_proposition : ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0 := by
  sorry

end negated_proposition_l129_129568


namespace discriminant_quadratic_eqn_l129_129187

def a := 1
def b := 1
def c := -2
def Δ : ℤ := b^2 - 4 * a * c

theorem discriminant_quadratic_eqn : Δ = 9 := by
  sorry

end discriminant_quadratic_eqn_l129_129187


namespace minimum_value_frac_sum_l129_129020

theorem minimum_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 / y = 3) :
  (2 / x + y) ≥ 8 / 3 :=
sorry

end minimum_value_frac_sum_l129_129020


namespace inequality_amgm_l129_129386

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l129_129386


namespace valentino_chickens_l129_129912

variable (C : ℕ) -- Number of chickens
variable (D : ℕ) -- Number of ducks
variable (T : ℕ) -- Number of turkeys
variable (total_birds : ℕ) -- Total number of birds on the farm

theorem valentino_chickens (h1 : D = 2 * C) 
                            (h2 : T = 3 * D)
                            (h3 : total_birds = C + D + T)
                            (h4 : total_birds = 1800) :
  C = 200 := by
  sorry

end valentino_chickens_l129_129912


namespace parabola_focus_coordinates_l129_129085

theorem parabola_focus_coordinates (h : ∀ y, y^2 = 4 * x) : ∃ x, x = 1 ∧ y = 0 := 
sorry

end parabola_focus_coordinates_l129_129085


namespace cost_of_chicken_l129_129214

theorem cost_of_chicken (cost_beef_per_pound : ℝ) (quantity_beef : ℝ) (cost_oil : ℝ) (total_grocery_cost : ℝ) (contribution_each : ℝ) :
  cost_beef_per_pound = 4 →
  quantity_beef = 3 →
  cost_oil = 1 →
  total_grocery_cost = 16 →
  contribution_each = 1 →
  ∃ (cost_chicken : ℝ), cost_chicken = 3 :=
by
  intros h1 h2 h3 h4 h5
  -- This line is required to help Lean handle any math operations
  have h6 := h1
  have h7 := h2
  have h8 := h3
  have h9 := h4
  have h10 := h5
  sorry

end cost_of_chicken_l129_129214


namespace christen_peeled_potatoes_l129_129830

open Nat

theorem christen_peeled_potatoes :
  ∀ (total_potatoes homer_rate homer_time christen_rate : ℕ) (combined_rate : ℕ),
    total_potatoes = 60 →
    homer_rate = 4 →
    homer_time = 6 →
    christen_rate = 6 →
    combined_rate = homer_rate + christen_rate →
    Nat.ceil ((total_potatoes - (homer_rate * homer_time)) / combined_rate * christen_rate) = 21 :=
by
  intros total_potatoes homer_rate homer_time christen_rate combined_rate
  intros htp hr ht cr cr_def
  rw [htp, hr, ht, cr, cr_def]
  sorry

end christen_peeled_potatoes_l129_129830


namespace simplify_log_expression_l129_129433

theorem simplify_log_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 
   1 / (Real.log 2 / Real.log 8 + 1) + 
   1 / (Real.log 3 / Real.log 9 + 1)) = 
  (5 * Real.log 2 + 2 * Real.log 3) / (4 * Real.log 2 + 3 * Real.log 3) :=
by sorry

end simplify_log_expression_l129_129433


namespace inequality_proof_l129_129250

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
    (b^2 / a + a^2 / b) ≥ (a + b) := 
    sorry

end inequality_proof_l129_129250


namespace misha_card_numbers_l129_129452

-- Define the context for digits
def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

-- Define conditions
def proper_fraction (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a < b

-- Original problem statement rewritten for Lean
theorem misha_card_numbers (L O M N S B : ℕ) :
  is_digit L → is_digit O → is_digit M → is_digit N → is_digit S → is_digit B →
  proper_fraction O M → proper_fraction O S →
  L + O / M + O + N + O / S = 10 + B :=
sorry

end misha_card_numbers_l129_129452


namespace target_expression_l129_129602

variable (a b : ℤ)

-- Definitions based on problem conditions
def op1 (x y : ℤ) : ℤ := x + y  -- "!" could be addition
def op2 (x y : ℤ) : ℤ := x - y  -- "?" could be subtraction in one order

-- Using these operations to create expressions
def exp1 (a b : ℤ) := op1 (op2 a b) (op2 b a)

def exp2 (x y : ℤ) := op2 (op2 x 0) (op2 0 y)

-- The final expression we need to check
def final_exp (a b : ℤ) := exp1 (20 * a) (18 * b)

-- Theorem proving the final expression equals target
theorem target_expression : final_exp a b = 20 * a - 18 * b :=
sorry

end target_expression_l129_129602


namespace mr_brown_selling_price_l129_129599

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end mr_brown_selling_price_l129_129599


namespace train_pass_time_is_38_seconds_l129_129587

noncomputable def speed_of_jogger_kmhr : ℝ := 9
noncomputable def speed_of_train_kmhr : ℝ := 45
noncomputable def lead_distance_m : ℝ := 260
noncomputable def train_length_m : ℝ := 120

noncomputable def speed_of_jogger_ms : ℝ := speed_of_jogger_kmhr * (1000 / 3600)
noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmhr * (1000 / 3600)

noncomputable def relative_speed_ms : ℝ := speed_of_train_ms - speed_of_jogger_ms
noncomputable def total_distance_m : ℝ := lead_distance_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

theorem train_pass_time_is_38_seconds :
  time_to_pass_jogger_s = 38 := 
sorry

end train_pass_time_is_38_seconds_l129_129587


namespace coffee_shop_sales_l129_129802

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end coffee_shop_sales_l129_129802


namespace sum_of_probability_fractions_l129_129715

def total_tree_count := 15
def non_birch_count := 9
def birch_count := 6
def total_arrangements := Nat.choose 15 6
def non_adjacent_birch_arrangements := Nat.choose 10 6
def birch_probability := non_adjacent_birch_arrangements / total_arrangements
def simplified_probability_numerator := 6
def simplified_probability_denominator := 143
def answer := simplified_probability_numerator + simplified_probability_denominator

theorem sum_of_probability_fractions :
  answer = 149 := by
  sorry

end sum_of_probability_fractions_l129_129715


namespace translate_quadratic_l129_129334

-- Define the original quadratic function
def original_quadratic (x : ℝ) : ℝ := (x - 2)^2 - 4

-- Define the translation of the graph one unit to the left and two units up
def translated_quadratic (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Statement to be proved
theorem translate_quadratic :
  ∀ x : ℝ, translated_quadratic x = original_quadratic (x-1) + 2 :=
by
  intro x
  unfold translated_quadratic original_quadratic
  sorry

end translate_quadratic_l129_129334


namespace fraction_ordering_l129_129558

theorem fraction_ordering :
  (6:ℚ)/29 < (8:ℚ)/25 ∧ (8:ℚ)/25 < (10:ℚ)/31 :=
by
  sorry

end fraction_ordering_l129_129558


namespace other_root_of_quadratic_l129_129180

theorem other_root_of_quadratic (m : ℝ) (h : ∃ α : ℝ, α = 1 ∧ (3 * α^2 + m * α = 5)) :
  ∃ β : ℝ, β = -5 / 3 :=
by
  sorry

end other_root_of_quadratic_l129_129180


namespace count_neither_multiples_of_2_nor_3_l129_129656

theorem count_neither_multiples_of_2_nor_3 : 
  let count_multiples (k n : ℕ) : ℕ := n / k
  let total_numbers := 100
  let multiples_of_2 := count_multiples 2 total_numbers
  let multiples_of_3 := count_multiples 3 total_numbers
  let multiples_of_6 := count_multiples 6 total_numbers
  let multiples_of_2_or_3 := multiples_of_2 + multiples_of_3 - multiples_of_6
  total_numbers - multiples_of_2_or_3 = 33 :=
by 
  sorry

end count_neither_multiples_of_2_nor_3_l129_129656


namespace evaluate_expression_l129_129442

theorem evaluate_expression : (4 * 4 + 4) / (2 * 2 - 2) = 10 := by
  sorry

end evaluate_expression_l129_129442


namespace integral_percentage_l129_129059

variable (a b : ℝ)

theorem integral_percentage (h : ∀ x, x^2 > 0) :
  (∫ x in a..b, (1 / 20 * x^2 + 3 / 10 * x^2)) = 0.35 * (∫ x in a..b, x^2) :=
by
  sorry

end integral_percentage_l129_129059


namespace solve_diophantine_equation_l129_129766

def is_solution (m n : ℕ) : Prop := 2^m - 3^n = 1

theorem solve_diophantine_equation : 
  { (m, n) : ℕ × ℕ | is_solution m n } = { (1, 0), (2, 1) } :=
by
  sorry

end solve_diophantine_equation_l129_129766


namespace friends_pets_ratio_l129_129041

theorem friends_pets_ratio (pets_total : ℕ) (pets_taylor : ℕ) (pets_friend4 : ℕ) (pets_friend5 : ℕ)
  (pets_first3_total : ℕ) : pets_total = 32 → pets_taylor = 4 → pets_friend4 = 2 → pets_friend5 = 2 →
  pets_first3_total = pets_total - pets_taylor - pets_friend4 - pets_friend5 →
  (pets_first3_total : ℚ) / pets_taylor = 6 :=
by
  sorry

end friends_pets_ratio_l129_129041


namespace inequality_solution_l129_129434

theorem inequality_solution (x : ℚ) : (3 * x - 5 ≥ 9 - 2 * x) → (x ≥ 14 / 5) :=
by
  sorry

end inequality_solution_l129_129434


namespace calculate_expression_l129_129079

theorem calculate_expression : 12 * (1 / (2 / 3 - 1 / 4 + 1 / 6)) = 144 / 7 :=
by
  sorry

end calculate_expression_l129_129079


namespace cans_increment_l129_129595

/--
If there are 9 rows of cans in a triangular display, where each successive row increases 
by a certain number of cans \( x \) compared to the row above it, with the seventh row having 
19 cans, and the total number of cans being fewer than 120, then 
each row has 4 more cans than the row above it.
-/
theorem cans_increment (x : ℕ) : 
  9 * 19 - 16 * x < 120 → x > 51 / 16 → x = 4 :=
by
  intros h1 h2
  sorry

end cans_increment_l129_129595


namespace good_numbers_product_sum_digits_not_equal_l129_129683

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l129_129683


namespace cone_curved_surface_area_at_5_seconds_l129_129792

theorem cone_curved_surface_area_at_5_seconds :
  let l := λ t : ℝ => 10 + 2 * t
  let r := λ t : ℝ => 5 + 1 * t
  let CSA := λ t : ℝ => Real.pi * r t * l t
  CSA 5 = 160 * Real.pi :=
by
  -- Definitions and calculations in the problem ensure this statement
  sorry

end cone_curved_surface_area_at_5_seconds_l129_129792


namespace gregory_current_age_l129_129811

-- Given conditions
variables (D G y : ℕ)
axiom dm_is_three_times_greg_was (x : ℕ) : D = 3 * y
axiom future_age_sum : D + (3 * y) = 49
axiom greg_age_difference x y : D - (3 * y) = (3 * y) - x

-- Prove statement: Gregory's current age is 14
theorem gregory_current_age : G = 14 := by
  sorry

end gregory_current_age_l129_129811


namespace mouse_lives_count_l129_129614

-- Define the basic conditions
def catLives : ℕ := 9
def dogLives : ℕ := catLives - 3
def mouseLives : ℕ := dogLives + 7

-- The main theorem to prove
theorem mouse_lives_count : mouseLives = 13 :=
by
  -- proof steps go here
  sorry

end mouse_lives_count_l129_129614


namespace thirteenth_term_geometric_sequence_l129_129708

theorem thirteenth_term_geometric_sequence 
  (a : ℕ → ℕ) 
  (r : ℝ)
  (h₁ : a 7 = 7) 
  (h₂ : a 10 = 21)
  (h₃ : ∀ (n : ℕ), a (n + 1) = a n * r) : 
  a 13 = 63 := 
by
  -- proof needed
  sorry

end thirteenth_term_geometric_sequence_l129_129708


namespace area_of_region_l129_129340

noncomputable def area : ℝ :=
  ∫ x in Set.Icc (-2 : ℝ) 0, (2 - (x + 1)^2 / 4) +
  ∫ x in Set.Icc (0 : ℝ) 2, (2 - x - (x + 1)^2 / 4)

theorem area_of_region : area = 5 / 3 := 
sorry

end area_of_region_l129_129340


namespace trigonometric_identity_l129_129258

theorem trigonometric_identity (α x : ℝ) (h₁ : 5 * Real.cos α = x) (h₂ : x ^ 2 + 16 = 25) (h₃ : α > Real.pi / 2 ∧ α < Real.pi):
  x = -3 ∧ Real.tan α = -4 / 3 :=
by
  sorry

end trigonometric_identity_l129_129258


namespace number_is_eight_l129_129188

theorem number_is_eight (x : ℤ) (h : x - 2 = 6) : x = 8 := 
sorry

end number_is_eight_l129_129188


namespace range_of_expression_l129_129767

theorem range_of_expression (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 ≤ β ∧ β ≤ π / 2) :
    -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
by
  sorry

end range_of_expression_l129_129767


namespace sin_600_eq_l129_129581

theorem sin_600_eq : Real.sin (600 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_l129_129581


namespace overlapping_region_area_l129_129417

noncomputable def radius : ℝ := 15
noncomputable def central_angle_radians : ℝ := Real.pi / 2
noncomputable def area_of_sector : ℝ := (1 / 4) * Real.pi * (radius^2)
noncomputable def side_length_equilateral_triangle : ℝ := radius
noncomputable def area_of_equilateral_triangle : ℝ := (Real.sqrt 3 / 4) * (side_length_equilateral_triangle^2)
noncomputable def overlapping_area : ℝ := 2 * area_of_sector - area_of_equilateral_triangle

theorem overlapping_region_area :
  overlapping_area = 112.5 * Real.pi - 56.25 * Real.sqrt 3 :=
by
  sorry
 
end overlapping_region_area_l129_129417


namespace exists_disjoint_A_B_l129_129323

def S (C : Finset ℕ) := C.sum id

theorem exists_disjoint_A_B : 
  ∃ (A B : Finset ℕ), 
  A ≠ ∅ ∧ B ≠ ∅ ∧ 
  A ∩ B = ∅ ∧ 
  A ∪ B = (Finset.range (2021 + 1)).erase 0 ∧ 
  ∃ k : ℕ, S A * S B = k^2 :=
by 
  sorry

end exists_disjoint_A_B_l129_129323


namespace question_condition_l129_129640

def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
  (1 - 2 * x) * (x + 1) < 0 → x > 1 / 2 ∨ x < -1

theorem question_condition
(x : ℝ) : sufficient_but_not_necessary_condition x := sorry

end question_condition_l129_129640


namespace olivia_cookies_total_l129_129271

def cookies_total (baggie_cookie_count : ℝ) (chocolate_chip_cookies : ℝ) 
                  (baggies_oatmeal_cookies : ℝ) (total_cookies : ℝ) : Prop :=
  let oatmeal_cookies := baggies_oatmeal_cookies * baggie_cookie_count
  oatmeal_cookies + chocolate_chip_cookies = total_cookies

theorem olivia_cookies_total :
  cookies_total 9.0 13.0 3.111111111 41.0 :=
by
  -- Proof goes here
  sorry

end olivia_cookies_total_l129_129271


namespace map_distance_representation_l129_129115

theorem map_distance_representation
  (cm_to_km_ratio : 15 = 90)
  (km_to_m_ratio : 1000 = 1000) :
  20 * (90 / 15) * 1000 = 120000 := by
  sorry

end map_distance_representation_l129_129115


namespace jacob_age_proof_l129_129165

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l129_129165


namespace original_cost_prices_l129_129803

variable (COST_A COST_B COST_C : ℝ)

theorem original_cost_prices :
  (COST_A * 0.8 + 100 = COST_A * 1.05) →
  (COST_B * 1.1 - 80 = COST_B * 0.92) →
  (COST_C * 0.85 + 120 = COST_C * 1.07) →
  COST_A = 400 ∧
  COST_B = 4000 / 9 ∧
  COST_C = 6000 / 11 := by
  intro h1 h2 h3
  sorry

end original_cost_prices_l129_129803


namespace multiplier_eq_l129_129037

-- Definitions of the given conditions
def length (w : ℝ) (m : ℝ) : ℝ := m * w + 2
def perimeter (l : ℝ) (w : ℝ) : ℝ := 2 * l + 2 * w

-- Condition definitions
def l : ℝ := 38
def P : ℝ := 100

-- Proof statement
theorem multiplier_eq (m w : ℝ) (h1 : length w m = l) (h2 : perimeter l w = P) : m = 3 :=
by
  sorry

end multiplier_eq_l129_129037


namespace valid_digit_for_multiple_of_5_l129_129362

theorem valid_digit_for_multiple_of_5 (d : ℕ) (h : d < 10) : (45670 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 :=
by
  sorry

end valid_digit_for_multiple_of_5_l129_129362


namespace length_of_place_mat_l129_129048

noncomputable def radius : ℝ := 6
noncomputable def width : ℝ := 1.5
def inner_corner_touch (n : ℕ) : Prop := n = 6

theorem length_of_place_mat (y : ℝ) (h1 : radius = 6) (h2 : width = 1.5) (h3 : inner_corner_touch 6) :
  y = (Real.sqrt 141.75 + 1.5) / 2 :=
sorry

end length_of_place_mat_l129_129048


namespace lucy_sales_is_43_l129_129233

def total_packs : Nat := 98
def robyn_packs : Nat := 55
def lucy_packs : Nat := total_packs - robyn_packs

theorem lucy_sales_is_43 : lucy_packs = 43 :=
by
  sorry

end lucy_sales_is_43_l129_129233


namespace arithmetic_sequence_120th_term_l129_129083

theorem arithmetic_sequence_120th_term :
  let a1 := 6
  let d := 6
  let n := 120
  let a_n := a1 + (n - 1) * d
  a_n = 720 := by
  sorry

end arithmetic_sequence_120th_term_l129_129083


namespace puppy_price_l129_129949

theorem puppy_price (P : ℕ) (kittens_price : ℕ) (total_earnings : ℕ) :
  (kittens_price = 2 * 6) → (total_earnings = 17) → (kittens_price + P = total_earnings) → P = 5 :=
by
  intros h1 h2 h3
  sorry

end puppy_price_l129_129949


namespace number_of_zeros_of_g_l129_129996

noncomputable def f (x a : ℝ) := Real.exp x * (x + a)

noncomputable def g (x a : ℝ) := f (x - a) a - x^2

theorem number_of_zeros_of_g (a : ℝ) :
  (if a < 1 then ∃! x, g x a = 0
   else if a = 1 then ∃! x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0
   else ∃! x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0) := sorry

end number_of_zeros_of_g_l129_129996


namespace geometric_sequence_property_l129_129043

theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a 1 / a 0) (h₁ : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_property_l129_129043


namespace compare_xyz_l129_129671

open Real

noncomputable def x : ℝ := 6 * log 3 / log 64
noncomputable def y : ℝ := (1 / 3) * log 64 / log 3
noncomputable def z : ℝ := (3 / 2) * log 3 / log 8

theorem compare_xyz : x > y ∧ y > z := 
by {
  sorry
}

end compare_xyz_l129_129671


namespace sum_of_fractions_and_decimal_l129_129087

theorem sum_of_fractions_and_decimal : 
    (3 / 25 : ℝ) + (1 / 5) + 55.21 = 55.53 :=
by 
  sorry

end sum_of_fractions_and_decimal_l129_129087


namespace valid_outfit_combinations_l129_129336

theorem valid_outfit_combinations :
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  total_combinations - invalid_combinations = 205 :=
by
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  have h : total_combinations - invalid_combinations = 205 := sorry
  exact h

end valid_outfit_combinations_l129_129336


namespace find_rate_l129_129235

-- Definitions of conditions
def Principal : ℝ := 2500
def Amount : ℝ := 3875
def Time : ℝ := 12

-- Main statement we are proving
theorem find_rate (P : ℝ) (A : ℝ) (T : ℝ) (R : ℝ) 
    (hP : P = Principal) 
    (hA : A = Amount) 
    (hT : T = Time) 
    (hR : R = (A - P) * 100 / (P * T)) : R = 55 / 12 := 
by 
  sorry

end find_rate_l129_129235


namespace matthew_more_strawberries_than_betty_l129_129999

noncomputable def B : ℕ := 16

theorem matthew_more_strawberries_than_betty (M N : ℕ) 
  (h1 : M > B)
  (h2 : M = 2 * N) 
  (h3 : B + M + N = 70) : M - B = 20 :=
by
  sorry

end matthew_more_strawberries_than_betty_l129_129999


namespace selling_price_per_pound_l129_129837

-- Definitions based on conditions
def cost_per_pound_type1 : ℝ := 2.00
def cost_per_pound_type2 : ℝ := 3.00
def weight_type1 : ℝ := 64
def weight_type2 : ℝ := 16
def total_weight : ℝ := 80

-- The selling price per pound of the mixture
theorem selling_price_per_pound :
  let total_cost := (weight_type1 * cost_per_pound_type1) + (weight_type2 * cost_per_pound_type2)
  (total_cost / total_weight) = 2.20 :=
by
  sorry

end selling_price_per_pound_l129_129837


namespace fraction_relationships_l129_129303

variable (p r s u : ℚ)

theorem fraction_relationships (h1 : p / r = 8) (h2 : s / r = 5) (h3 : s / u = 1 / 3) :
  u / p = 15 / 8 :=
sorry

end fraction_relationships_l129_129303


namespace number_of_BMWs_sold_l129_129534

theorem number_of_BMWs_sold (total_cars_sold : ℕ)
  (percent_Ford percent_Nissan percent_Chevrolet : ℕ)
  (h_total : total_cars_sold = 300)
  (h_percent_Ford : percent_Ford = 18)
  (h_percent_Nissan : percent_Nissan = 25)
  (h_percent_Chevrolet : percent_Chevrolet = 20) :
  (300 * (100 - (percent_Ford + percent_Nissan + percent_Chevrolet)) / 100) = 111 :=
by
  -- We assert that the calculated number of BMWs is 111
  sorry

end number_of_BMWs_sold_l129_129534


namespace smallest_b_l129_129880

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7) (h4 : 2 + a ≤ b) : b = 9 / 2 :=
by
  sorry

end smallest_b_l129_129880


namespace merchant_spent_initially_500_rubles_l129_129678

theorem merchant_spent_initially_500_rubles
  (x : ℕ)
  (h1 : x + 100 > x)
  (h2 : x + 220 > x + 100)
  (h3 : x * (x + 220) = (x + 100) * (x + 100))
  : x = 500 := sorry

end merchant_spent_initially_500_rubles_l129_129678


namespace number_of_children_on_bus_l129_129720

theorem number_of_children_on_bus (initial_children : ℕ) (additional_children : ℕ) (total_children : ℕ) 
  (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = 64 :=
by
  sorry

end number_of_children_on_bus_l129_129720


namespace reception_time_l129_129119

-- Definitions of conditions
def noon : ℕ := 12 * 60 -- define noon in minutes
def rabbit_walk_speed (v : ℕ) : Prop := v > 0
def rabbit_run_speed (v : ℕ) : Prop := 2 * v > 0
def distance (D : ℕ) : Prop := D > 0
def delay (minutes : ℕ) : Prop := minutes = 10

-- Definition of the problem
theorem reception_time (v D : ℕ) (h_v : rabbit_walk_speed v) (h_D : distance D) (h_delay : delay 10) :
  noon + (D / v) * 2 / 3 = 12 * 60 + 40 :=
by sorry

end reception_time_l129_129119


namespace triangle_cos_C_correct_l129_129333

noncomputable def triangle_cos_C (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : ℝ :=
  Real.cos C -- This will be defined correctly in the proof phase.

theorem triangle_cos_C_correct (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : 
  triangle_cos_C A B C hABC hSinA hCosB = 16 / 65 :=
sorry

end triangle_cos_C_correct_l129_129333


namespace alloy_chromium_l129_129423

variable (x : ℝ)

theorem alloy_chromium (h : 0.15 * 15 + 0.08 * x = 0.101 * (15 + x)) : x = 35 := by
  sorry

end alloy_chromium_l129_129423


namespace sandy_correct_sums_l129_129885

variable (c i : ℕ)

theorem sandy_correct_sums (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by {
  -- Proof goes here
  sorry
}

end sandy_correct_sums_l129_129885


namespace emily_page_production_difference_l129_129645

variables (p h : ℕ)

def first_day_pages (p h : ℕ) : ℕ := p * h
def second_day_pages (p h : ℕ) : ℕ := (p - 3) * (h + 3)
def page_difference (p h : ℕ) : ℕ := second_day_pages p h - first_day_pages p h

theorem emily_page_production_difference (h : ℕ) (p_eq_3h : p = 3 * h) :
  page_difference p h = 6 * h - 9 :=
by sorry

end emily_page_production_difference_l129_129645


namespace football_game_initial_population_l129_129240

theorem football_game_initial_population (B G : ℕ) (h1 : G = 240)
  (h2 : (3 / 4 : ℚ) * B + (7 / 8 : ℚ) * G = 480) : B + G = 600 :=
sorry

end football_game_initial_population_l129_129240


namespace solve_trig_eq_l129_129914

noncomputable def arccos (x : ℝ) : ℝ := sorry

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  -3 * (Real.cos x) ^ 2 + 5 * (Real.sin x) + 1 = 0 ↔
  (x = Real.arcsin (1 / 3) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (1 / 3) + 2 * k * Real.pi) :=
sorry

end solve_trig_eq_l129_129914


namespace three_point_sixty_eight_as_fraction_l129_129663

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end three_point_sixty_eight_as_fraction_l129_129663


namespace ferris_wheel_seats_l129_129294

def number_of_people_per_seat := 6
def total_number_of_people := 84

def number_of_seats := total_number_of_people / number_of_people_per_seat

theorem ferris_wheel_seats : number_of_seats = 14 := by
  sorry

end ferris_wheel_seats_l129_129294


namespace teachers_students_relationship_l129_129773

variables (m n k l : ℕ)

theorem teachers_students_relationship
  (teachers_count : m > 0)
  (students_count : n > 0)
  (students_per_teacher : k > 0)
  (teachers_per_student : l > 0)
  (h1 : ∀ p ∈ (Finset.range m), (Finset.card (Finset.range k)) = k)
  (h2 : ∀ s ∈ (Finset.range n), (Finset.card (Finset.range l)) = l) :
  m * k = n * l :=
sorry

end teachers_students_relationship_l129_129773


namespace length_of_platform_l129_129598

theorem length_of_platform (length_of_train speed_of_train time_to_cross : ℕ) 
    (h1 : length_of_train = 450) (h2 : speed_of_train = 126) (h3 : time_to_cross = 20) :
    ∃ length_of_platform : ℕ, length_of_platform = 250 := 
by 
  sorry

end length_of_platform_l129_129598


namespace range_of_m_l129_129332

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x ^ 2 + 24 * x + 5 * m) / 8

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ c : ℝ, G x m = (x + c) ^ 2 ∧ c ^ 2 = 3) → 4 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l129_129332


namespace toy_spending_ratio_l129_129953

theorem toy_spending_ratio :
  ∃ T : ℝ, 204 - T > 0 ∧ 51 = (204 - T) / 2 ∧ (T / 204) = 1 / 2 :=
by
  sorry

end toy_spending_ratio_l129_129953


namespace probability_two_red_or_blue_correct_l129_129167

noncomputable def probability_two_red_or_blue_sequential : ℚ := 1 / 5

theorem probability_two_red_or_blue_correct :
  let total_marbles := 15
  let red_blue_marbles := 7
  let first_draw_prob := (7 : ℚ) / 15
  let second_draw_prob := (6 : ℚ) / 14
  first_draw_prob * second_draw_prob = probability_two_red_or_blue_sequential :=
by
  sorry

end probability_two_red_or_blue_correct_l129_129167


namespace intersection_of_lines_l129_129337

theorem intersection_of_lines :
  ∃ (x y : ℚ), (6 * x - 5 * y = 15) ∧ (8 * x + 3 * y = 1) ∧ x = 25 / 29 ∧ y = -57 / 29 :=
by
  sorry

end intersection_of_lines_l129_129337


namespace no_positive_integer_solutions_l129_129077

theorem no_positive_integer_solutions (A : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) :
  ¬(∃ x : ℕ, x^2 - 2 * A * x + A0 = 0) :=
by sorry

end no_positive_integer_solutions_l129_129077


namespace quadratic_roots_l129_129354

theorem quadratic_roots (p q r : ℝ) (h : p ≠ q) (k : ℝ) :
  (p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) →
  ((p * (q - r)) * k^2 + (q * (r - p)) * k + r * (p - q) = 0) →
  k = - (r * (p - q)) / (p * (q - r)) :=
by
  sorry

end quadratic_roots_l129_129354


namespace custom_mul_2021_1999_l129_129571

axiom custom_mul : ℕ → ℕ → ℕ

axiom custom_mul_id1 : ∀ (A : ℕ), custom_mul A A = 0
axiom custom_mul_id2 : ∀ (A B C : ℕ), custom_mul A (custom_mul B C) = custom_mul A B + C

theorem custom_mul_2021_1999 : custom_mul 2021 1999 = 22 := by
  sorry

end custom_mul_2021_1999_l129_129571


namespace length_of_major_axis_l129_129266

def ellipse_length_major_axis (a b : ℝ) : ℝ := 2 * a

theorem length_of_major_axis : ellipse_length_major_axis 4 1 = 8 :=
by
  unfold ellipse_length_major_axis
  norm_num

end length_of_major_axis_l129_129266


namespace add_fractions_l129_129666

-- Define the two fractions
def frac1 := 7 / 8
def frac2 := 9 / 12

-- The problem: addition of the two fractions and expressing in simplest form
theorem add_fractions : frac1 + frac2 = (13 : ℚ) / 8 := 
by 
  sorry

end add_fractions_l129_129666


namespace chord_ratio_l129_129783

theorem chord_ratio {FQ HQ : ℝ} (h : EQ * FQ = GQ * HQ) (h_eq : EQ = 5) (h_gq : GQ = 12) : 
  FQ / HQ = 12 / 5 :=
by
  rw [h_eq, h_gq] at h
  sorry

end chord_ratio_l129_129783


namespace wire_division_l129_129139

theorem wire_division (initial_length : ℝ) (num_parts : ℕ) (final_length : ℝ) :
  initial_length = 69.76 ∧ num_parts = 8 ∧
  final_length = (initial_length / num_parts) / num_parts →
  final_length = 1.09 :=
by
  sorry

end wire_division_l129_129139


namespace necessary_but_not_sufficient_condition_l129_129878

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b : ℝ × ℝ := (2, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Statement: Prove x > 0 is a necessary but not sufficient condition for the angle between vectors a and b to be acute.
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (dot_product (vector_a x) vector_b > 0) ↔ (x > 0) := 
sorry

end necessary_but_not_sufficient_condition_l129_129878


namespace find_m_l129_129923

variable (a : ℝ × ℝ := (2, 3))
variable (b : ℝ × ℝ := (-1, 2))

def isCollinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 - u.2 * v.1 = 0

theorem find_m (m : ℝ) (h : isCollinear (2 * m - 4, 3 * m + 8) (4, -1)) : m = -2 :=
by {
  sorry
}

end find_m_l129_129923


namespace ratio_A_B_l129_129800

variable (A B C : ℕ)

theorem ratio_A_B 
  (h1: A + B + C = 98) 
  (h2: B = 30) 
  (h3: (B : ℚ) / C = 5 / 8) 
  : (A : ℚ) / B = 2 / 3 :=
sorry

end ratio_A_B_l129_129800


namespace rose_needs_more_money_l129_129247

theorem rose_needs_more_money 
    (paintbrush_cost : ℝ)
    (paints_cost : ℝ)
    (easel_cost : ℝ)
    (money_rose_has : ℝ) :
    paintbrush_cost = 2.40 →
    paints_cost = 9.20 →
    easel_cost = 6.50 →
    money_rose_has = 7.10 →
    (paintbrush_cost + paints_cost + easel_cost - money_rose_has) = 11 :=
by
  intros
  sorry

end rose_needs_more_money_l129_129247


namespace maximum_value_of_f_l129_129342

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - 3 * x else -2 * x + 1

theorem maximum_value_of_f : ∃ (m : ℝ), (∀ x : ℝ, f x ≤ m) ∧ (m = 2) := by
  sorry

end maximum_value_of_f_l129_129342


namespace max_min_f_l129_129756

noncomputable def f (x : ℝ) : ℝ := 
  5 * Real.cos x ^ 2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem max_min_f :
  (∃ x : ℝ, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x : ℝ, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end max_min_f_l129_129756


namespace compute_trig_expression_l129_129375

theorem compute_trig_expression : 
  (1 - 1 / (Real.cos (37 * Real.pi / 180))) *
  (1 + 1 / (Real.sin (53 * Real.pi / 180))) *
  (1 - 1 / (Real.sin (37 * Real.pi / 180))) *
  (1 + 1 / (Real.cos (53 * Real.pi / 180))) = 1 :=
sorry

end compute_trig_expression_l129_129375


namespace daniel_age_is_13_l129_129009

-- Define Aunt Emily's age
def aunt_emily_age : ℕ := 48

-- Define Brianna's age as a third of Aunt Emily's age
def brianna_age : ℕ := aunt_emily_age / 3

-- Define that Daniel's age is 3 years less than Brianna's age
def daniel_age : ℕ := brianna_age - 3

-- Theorem to prove Daniel's age is 13 given the conditions
theorem daniel_age_is_13 :
  brianna_age = aunt_emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age = 13 :=
  sorry

end daniel_age_is_13_l129_129009


namespace smallest_bisecting_segment_l129_129889

-- Define a structure for a triangle in a plane
structure Triangle (α β γ : Type u) :=
(vertex1 : α) 
(vertex2 : β) 
(vertex3 : γ) 
(area : ℝ)

-- Define a predicate for an excellent line
def is_excellent_line {α β γ : Type u} (T : Triangle α β γ) (A : α) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line excellent here, e.g., dividing area in half
sorry

-- Define a function to get the length of a line segment within the triangle
def length_within_triangle {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : ℝ :=
-- compute the length of the segment within the triangle
sorry

-- Define predicates for triangles with specific properties like medians
def is_median {α β γ : Type u} (T : Triangle α β γ) (line : ℝ → ℝ → ℝ) : Prop :=
-- define properties that make a line a median
sorry

theorem smallest_bisecting_segment {α β γ : Type u} (T : Triangle α β γ) (A : α) (median : ℝ → ℝ → ℝ) : 
  (∀ line, is_excellent_line T A line → length_within_triangle T line ≥ length_within_triangle T median) →
  median = line  := 
-- show that the median from the vertex opposite the smallest angle has the smallest segment
sorry

end smallest_bisecting_segment_l129_129889


namespace percentage_increase_l129_129839

variables (P : ℝ) (buy_price : ℝ := 0.60 * P) (sell_price : ℝ := 1.08000000000000007 * P)

theorem percentage_increase (h: (0.60 : ℝ) * P = buy_price) (h1: (1.08000000000000007 : ℝ) * P = sell_price) :
  ((sell_price - buy_price) / buy_price) * 100 = 80.00000000000001 :=
  sorry

end percentage_increase_l129_129839


namespace find_m_l129_129814

def a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3)
def b : ℝ × ℝ := (1, -1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) (h : dot_product (a m) b = 2) : m = 3 :=
by sorry

end find_m_l129_129814


namespace digit_one_not_in_mean_l129_129573

def seq : List ℕ := [5, 55, 555, 5555, 55555, 555555, 5555555, 55555555, 555555555]

noncomputable def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem digit_one_not_in_mean :
  ¬(∃ d, d ∈ (arithmetic_mean seq).digits 10 ∧ d = 1) :=
sorry

end digit_one_not_in_mean_l129_129573


namespace inequality_solution_minimum_value_l129_129289

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem inequality_solution :
  {x : ℝ | f x > 7} = {x | x > 4 ∨ x < -3} :=
by
  sorry

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : ∀ x, f x ≥ m + n) :
  m + n = 3 →
  (m^2 + n^2 ≥ 9 / 2 ∧ (m = 3 / 2 ∧ n = 3 / 2)) :=
by
  sorry

end inequality_solution_minimum_value_l129_129289


namespace linear_inequality_solution_l129_129992

theorem linear_inequality_solution (a b : ℝ)
  (h₁ : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -3 ∨ x > 1)) :
  ∀ x : ℝ, a * x + b < 0 ↔ x < 3 / 2 :=
by
  sorry

end linear_inequality_solution_l129_129992


namespace integer_solutions_m3_eq_n3_plus_n_l129_129817

theorem integer_solutions_m3_eq_n3_plus_n (m n : ℤ) (h : m^3 = n^3 + n) : m = 0 ∧ n = 0 :=
sorry

end integer_solutions_m3_eq_n3_plus_n_l129_129817


namespace find_x_l129_129716

theorem find_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 4) : x = 2 :=
by
  sorry

end find_x_l129_129716


namespace tangent_series_identity_l129_129013

noncomputable def series_tangent (x : ℝ) : ℝ := ∑' n, (1 / (2 ^ n)) * Real.tan (x / (2 ^ n))

theorem tangent_series_identity (x : ℝ) : 
  (1 / x) - (1 / Real.tan x) = series_tangent x := 
sorry

end tangent_series_identity_l129_129013


namespace magnesium_is_limiting_l129_129002

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting_l129_129002


namespace total_spent_target_l129_129911

theorem total_spent_target (face_moisturizer_cost : ℕ) (body_lotion_cost : ℕ) (face_moisturizers_bought : ℕ) (body_lotions_bought : ℕ) (christy_multiplier : ℕ) :
  face_moisturizer_cost = 50 →
  body_lotion_cost = 60 →
  face_moisturizers_bought = 2 →
  body_lotions_bought = 4 →
  christy_multiplier = 2 →
  (face_moisturizers_bought * face_moisturizer_cost + body_lotions_bought * body_lotion_cost) * (1 + christy_multiplier) = 1020 := by
  sorry

end total_spent_target_l129_129911


namespace collinear_vectors_m_n_sum_l129_129564

theorem collinear_vectors_m_n_sum (m n : ℕ)
  (h1 : (2, 3, m) = (2 * n, 6, 8)) :
  m + n = 6 :=
sorry

end collinear_vectors_m_n_sum_l129_129564


namespace other_root_l129_129199

theorem other_root (m : ℤ) (h : (∀ x : ℤ, x^2 - x + m = 0 → (x = 2))) : (¬ ∃ y : ℤ, (y^2 - y + m = 0 ∧ y ≠ 2 ∧ y ≠ -1) ) := 
by {
  sorry
}

end other_root_l129_129199


namespace problem_inequality_l129_129995

def f (x : ℝ) : ℝ := abs (x - 1)

def A := {x : ℝ | -1 < x ∧ x < 1}

theorem problem_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) : 
  f (a * b) > f a - f b := by
  sorry

end problem_inequality_l129_129995


namespace number_of_bottles_l129_129084

-- Define the weights and total weight based on given conditions
def weight_of_two_bags_chips : ℕ := 800
def total_weight_five_bags_and_juices : ℕ := 2200
def weight_difference_chip_Juice : ℕ := 350

-- Considering 1 bag of chips weighs 400 g (derived from the condition)
def weight_of_one_bag_chips : ℕ := 400
def weight_of_one_bottle_juice : ℕ := weight_of_one_bag_chips - weight_difference_chip_Juice

-- Define the proof of the question
theorem number_of_bottles :
  (total_weight_five_bags_and_juices - (5 * weight_of_one_bag_chips)) / weight_of_one_bottle_juice = 4 := by sorry

end number_of_bottles_l129_129084


namespace solve_3x_plus_5_squared_l129_129611

theorem solve_3x_plus_5_squared (x : ℝ) (h : 5 * x - 6 = 15 * x + 21) : 
  3 * (x + 5) ^ 2 = 2523 / 100 :=
by
  sorry

end solve_3x_plus_5_squared_l129_129611


namespace geometric_sequence_sum_a_l129_129812

theorem geometric_sequence_sum_a (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, S n = 2^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 1 then S 1 else S n - S (n - 1)) :
  a = -1 :=
by
  sorry

end geometric_sequence_sum_a_l129_129812


namespace earnings_difference_l129_129221

noncomputable def investment_ratio_a : ℕ := 3
noncomputable def investment_ratio_b : ℕ := 4
noncomputable def investment_ratio_c : ℕ := 5

noncomputable def return_ratio_a : ℕ := 6
noncomputable def return_ratio_b : ℕ := 5
noncomputable def return_ratio_c : ℕ := 4

noncomputable def total_earnings : ℕ := 2900

noncomputable def earnings_a (x y : ℕ) : ℚ := (investment_ratio_a * return_ratio_a * x * y) / 100
noncomputable def earnings_b (x y : ℕ) : ℚ := (investment_ratio_b * return_ratio_b * x * y) / 100

theorem earnings_difference (x y : ℕ) (h : (investment_ratio_a * return_ratio_a * x * y + investment_ratio_b * return_ratio_b * x * y + investment_ratio_c * return_ratio_c * x * y) / 100 = total_earnings) :
  earnings_b x y - earnings_a x y = 100 := by
  sorry

end earnings_difference_l129_129221


namespace framed_painting_ratio_correct_l129_129065

/-- Define the conditions -/
def painting_height : ℕ := 30
def painting_width : ℕ := 20
def width_ratio : ℕ := 3

/-- Calculate the framed dimensions and check the area conditions -/
def framed_smaller_dimension (x : ℕ) : ℕ := painting_width + 2 * x
def framed_larger_dimension (x : ℕ) : ℕ := painting_height + 6 * x

theorem framed_painting_ratio_correct (x : ℕ) (h : (painting_width + 2 * x) * (painting_height + 6 * x) = 2 * (painting_width * painting_height)) :
  framed_smaller_dimension x / framed_larger_dimension x = 4 / 7 :=
by
  sorry

end framed_painting_ratio_correct_l129_129065


namespace carson_total_seed_fertilizer_l129_129270

-- Definitions based on the conditions
variable (F S : ℝ)
variable (h_seed : S = 45)
variable (h_relation : S = 3 * F)

-- Theorem stating the total amount of seed and fertilizer used
theorem carson_total_seed_fertilizer : S + F = 60 := by
  -- Use the given conditions to relate and calculate the total
  sorry

end carson_total_seed_fertilizer_l129_129270


namespace evaluate_expression_l129_129063

def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem evaluate_expression : 
  (binom 2 5 * 3 ^ 5) / binom 10 5 = 0 := by
  -- Given conditions:
  have h1 : binom 2 5 = 0 := by sorry
  have h2 : binom 10 5 = 252 := by sorry
  -- Proof goal:
  sorry

end evaluate_expression_l129_129063


namespace central_number_l129_129877

theorem central_number (C : ℕ) (verts : Finset ℕ) (h : verts = {1, 2, 7, 8, 9, 13, 14}) :
  (∀ T ∈ {t | ∃ a b c, (a + b + c) % 3 = 0 ∧ a ∈ verts ∧ b ∈ verts ∧ c ∈ verts}, (T + C) % 3 = 0) →
  C = 9 :=
by
  sorry

end central_number_l129_129877


namespace solve_equation_l129_129826

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l129_129826


namespace goldfish_in_each_pond_l129_129117

variable (x : ℕ)
variable (l1 h1 l2 h2 : ℕ)

-- Conditions
def cond1 : Prop := l1 + h1 = x ∧ l2 + h2 = x
def cond2 : Prop := 4 * l1 = 3 * h1
def cond3 : Prop := 3 * l2 = 5 * h2
def cond4 : Prop := l2 = l1 + 33

theorem goldfish_in_each_pond : cond1 x l1 h1 l2 h2 ∧ cond2 l1 h1 ∧ cond3 l2 h2 ∧ cond4 l1 l2 → 
  x = 168 := 
by 
  sorry

end goldfish_in_each_pond_l129_129117


namespace betsy_to_cindy_ratio_l129_129164

-- Definitions based on the conditions
def cindy_time : ℕ := 12
def tina_time : ℕ := cindy_time + 6
def betsy_time : ℕ := tina_time / 3

-- Theorem statement to prove
theorem betsy_to_cindy_ratio :
  (betsy_time : ℚ) / cindy_time = 1 / 2 :=
by sorry

end betsy_to_cindy_ratio_l129_129164


namespace f_odd_f_shift_f_in_range_find_f_7_5_l129_129506

def f : ℝ → ℝ := sorry  -- We define the function f (implementation is not needed here)

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem f_shift (x : ℝ) : f (x + 2) = -f x := sorry

theorem f_in_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = x := sorry

theorem find_f_7_5 : f 7.5 = 0.5 :=
by
  sorry

end f_odd_f_shift_f_in_range_find_f_7_5_l129_129506


namespace mult_63_37_l129_129035

theorem mult_63_37 : 63 * 37 = 2331 :=
by {
  sorry
}

end mult_63_37_l129_129035


namespace min_value_fraction_l129_129856

theorem min_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 3) : 
  ∃ v, v = (x + y) / x ∧ v = -2 := 
by 
  sorry

end min_value_fraction_l129_129856


namespace point_in_fourth_quadrant_l129_129919

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l129_129919


namespace first_machine_defect_probability_l129_129415

/-- Probability that a randomly selected defective item was made by the first machine is 0.5 
given certain conditions. -/
theorem first_machine_defect_probability :
  let PFirstMachine := 0.4
  let PSecondMachine := 0.6
  let DefectRateFirstMachine := 0.03
  let DefectRateSecondMachine := 0.02
  let TotalDefectProbability := PFirstMachine * DefectRateFirstMachine + PSecondMachine * DefectRateSecondMachine
  let PDefectGivenFirstMachine := PFirstMachine * DefectRateFirstMachine / TotalDefectProbability
  PDefectGivenFirstMachine = 0.5 :=
by
  sorry

end first_machine_defect_probability_l129_129415


namespace compute_expression_l129_129496

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l129_129496


namespace intersection_M_N_l129_129898

open Set

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_M_N :
  M ∩ N = { x | -2 ≤ x ∧ x ≤ -1 } := by
  sorry

end intersection_M_N_l129_129898


namespace sum_and_product_of_reciprocals_l129_129091

theorem sum_and_product_of_reciprocals (x y : ℝ) (h_sum : x + y = 12) (h_prod : x * y = 32) :
  (1/x + 1/y = 3/8) ∧ (1/x * 1/y = 1/32) :=
by
  sorry

end sum_and_product_of_reciprocals_l129_129091


namespace calculate_expression_l129_129693

theorem calculate_expression :
  (5 / 19) * ((19 / 5) * (16 / 3) + (14 / 3) * (19 / 5)) = 10 :=
by
  sorry

end calculate_expression_l129_129693


namespace evaluate_pow_l129_129555

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l129_129555


namespace function_symmetric_about_point_l129_129411

theorem function_symmetric_about_point :
  ∃ x₀ y₀, (x₀, y₀) = (Real.pi / 3, 0) ∧ ∀ x y, y = Real.sin (2 * x + Real.pi / 3) →
    (Real.sin (2 * (2 * x₀ - x) + Real.pi / 3) = y) :=
sorry

end function_symmetric_about_point_l129_129411


namespace slope_of_perpendicular_line_l129_129398

-- Define what it means to be the slope of a line in a certain form
def slope_of_line (a b c : ℝ) (m : ℝ) : Prop :=
  b ≠ 0 ∧ m = -a / b

-- Define what it means for two slopes to be perpendicular
def are_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Given conditions
def given_line : Prop := slope_of_line 4 5 20 (-4 / 5)

-- The theorem to be proved
theorem slope_of_perpendicular_line : ∃ m : ℝ, given_line ∧ are_perpendicular_slopes (-4 / 5) m ∧ m = 5 / 4 :=
  sorry

end slope_of_perpendicular_line_l129_129398


namespace relationship_among_a_b_c_l129_129404

noncomputable def a : ℝ := 0.99 ^ (1.01 : ℝ)
noncomputable def b : ℝ := 1.01 ^ (0.99 : ℝ)
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l129_129404


namespace infinite_solutions_b_l129_129186

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l129_129186


namespace prime_solution_exists_l129_129371

theorem prime_solution_exists :
  ∃ (p q r : ℕ), p.Prime ∧ q.Prime ∧ r.Prime ∧ (p + q^2 = r^4) ∧ (p = 7) ∧ (q = 3) ∧ (r = 2) := 
by
  sorry

end prime_solution_exists_l129_129371


namespace meals_per_day_l129_129364

-- Definitions based on given conditions
def number_of_people : Nat := 6
def total_plates_used : Nat := 144
def number_of_days : Nat := 4
def plates_per_meal : Nat := 2

-- Theorem to prove
theorem meals_per_day : (total_plates_used / number_of_days) / plates_per_meal / number_of_people = 3 :=
by
  sorry

end meals_per_day_l129_129364


namespace coin_tosses_l129_129525

theorem coin_tosses (n : ℤ) (h : (1/2 : ℝ)^n = 0.125) : n = 3 :=
by
  sorry

end coin_tosses_l129_129525


namespace solve_problem_l129_129126

def problem_statement : Prop :=
  ⌊ (2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011) ⌋ = 8

theorem solve_problem : problem_statement := 
  by sorry

end solve_problem_l129_129126


namespace imag_part_of_complex_l129_129095

open Complex

theorem imag_part_of_complex : (im ((5 + I) / (1 + I))) = -2 :=
by
  sorry

end imag_part_of_complex_l129_129095


namespace exists_x_y_l129_129093

theorem exists_x_y (n : ℕ) (hn : 0 < n) :
  ∃ x y : ℕ, n < x ∧ ¬ x ∣ y ∧ x^x ∣ y^y :=
by sorry

end exists_x_y_l129_129093


namespace num_valid_k_values_l129_129219

theorem num_valid_k_values :
  ∃ (s : Finset ℕ), s = { 1, 2, 3, 6, 9, 18 } ∧ s.card = 6 :=
by
  sorry

end num_valid_k_values_l129_129219


namespace millie_initial_bracelets_l129_129607

theorem millie_initial_bracelets (n : ℕ) (h1 : n - 2 = 7) : n = 9 :=
sorry

end millie_initial_bracelets_l129_129607


namespace inequality_proof_l129_129330

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 :=
by
  sorry

end inequality_proof_l129_129330


namespace total_cost_in_dollars_l129_129202

theorem total_cost_in_dollars :
  (500 * 3 + 300 * 2) / 100 = 21 := 
by
  sorry

end total_cost_in_dollars_l129_129202


namespace exists_x_y_with_specific_difference_l129_129774

theorem exists_x_y_with_specific_difference :
  ∃ x y : ℤ, (2 * x^2 + 8 * y = 26) ∧ (x - y = 26) := 
sorry

end exists_x_y_with_specific_difference_l129_129774


namespace tan_beta_identity_l129_129686

theorem tan_beta_identity (α β : ℝ) (h1 : Real.tan α = 1/3) (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 1/7 :=
sorry

end tan_beta_identity_l129_129686


namespace rectangle_length_width_difference_l129_129823

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : y = 1 / 3 * x)
  (h2 : 2 * x + 2 * y = 32)
  (h3 : Real.sqrt (x^2 + y^2) = 17) :
  abs (x - y) = 8 :=
sorry

end rectangle_length_width_difference_l129_129823


namespace circumference_of_minor_arc_l129_129254

-- Given:
-- 1. Three points (D, E, F) are on a circle with radius 25
-- 2. The angle ∠EFD = 120°

-- We need to prove that the length of the minor arc DE is 50π / 3
theorem circumference_of_minor_arc 
  (D E F : Point) 
  (r : ℝ) (h : r = 25) 
  (angleEFD : ℝ) 
  (hAngle : angleEFD = 120) 
  (circumference : ℝ) 
  (hCircumference : circumference = 2 * Real.pi * r) :
  arc_length_DE = 50 * Real.pi / 3 :=
by
  sorry

end circumference_of_minor_arc_l129_129254


namespace range_of_a_l129_129089

open Real

noncomputable def p (a : ℝ) := ∀ (x : ℝ), x ≥ 1 → (2 * x - 3 * a) ≥ 0
noncomputable def q (a : ℝ) := (0 < 2 * a - 1) ∧ (2 * a - 1 < 1)

theorem range_of_a (a : ℝ) : p a ∧ q a ↔ (1/2 < a ∧ a ≤ 2/3) := by
  sorry

end range_of_a_l129_129089


namespace total_cost_is_15_75_l129_129583

def price_sponge : ℝ := 4.20
def price_shampoo : ℝ := 7.60
def price_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05
def total_cost_before_tax : ℝ := price_sponge + price_shampoo + price_soap
def tax_amount : ℝ := tax_rate * total_cost_before_tax
def total_cost_including_tax : ℝ := total_cost_before_tax + tax_amount

theorem total_cost_is_15_75 : total_cost_including_tax = 15.75 :=
by sorry

end total_cost_is_15_75_l129_129583


namespace rooms_already_painted_l129_129040

-- Define the conditions as variables and hypotheses
variables (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
variables (h1 : total_rooms = 10)
variables (h2 : hours_per_room = 8)
variables (h3 : remaining_hours = 16)

-- Define the theorem stating the number of rooms already painted
theorem rooms_already_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 10) (h2 : hours_per_room = 8) (h3 : remaining_hours = 16) :
  (total_rooms - (remaining_hours / hours_per_room) = 8) :=
sorry

end rooms_already_painted_l129_129040


namespace best_fitting_model_is_model1_l129_129269

noncomputable def model1_R2 : ℝ := 0.98
noncomputable def model2_R2 : ℝ := 0.80
noncomputable def model3_R2 : ℝ := 0.54
noncomputable def model4_R2 : ℝ := 0.35

theorem best_fitting_model_is_model1 :
  model1_R2 > model2_R2 ∧ model1_R2 > model3_R2 ∧ model1_R2 > model4_R2 :=
by
  sorry

end best_fitting_model_is_model1_l129_129269


namespace max_correct_answers_l129_129719

variable (c w b : ℕ) -- Define c, w, b as natural numbers

theorem max_correct_answers (h1 : c + w + b = 30) (h2 : 4 * c - w = 70) : c ≤ 20 := by
  sorry

end max_correct_answers_l129_129719


namespace interval_of_decrease_l129_129805

noncomputable def f (x : ℝ) := x * Real.exp x + 1

theorem interval_of_decrease : {x : ℝ | x < -1} = {x : ℝ | (x + 1) * Real.exp x < 0} :=
by
  sorry

end interval_of_decrease_l129_129805


namespace eq_solutions_of_equation_l129_129540

open Int

theorem eq_solutions_of_equation (x y : ℤ) :
  ((x, y) = (0, -4) ∨ (x, y) = (0, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-4, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-6, 6) ∨
   (x, y) = (0, 0) ∨ (x, y) = (-10, 4)) ↔
  (x - y) * (x - y) = (x - y + 6) * (x + y) :=
sorry

end eq_solutions_of_equation_l129_129540


namespace squares_area_relation_l129_129723

/-- 
Given:
1. $\alpha$ such that $\angle 1 = \angle 2 = \angle 3 = \alpha$
2. The areas of the squares are given by:
   - $S_A = \cos^4 \alpha$
   - $S_D = \sin^4 \alpha$
   - $S_B = \cos^2 \alpha \sin^2 \alpha$
   - $S_C = \cos^2 \alpha \sin^2 \alpha$

Prove that:
$S_A \cdot S_D = S_B \cdot S_C$
--/

theorem squares_area_relation (α : ℝ) :
  (Real.cos α)^4 * (Real.sin α)^4 = (Real.cos α)^2 * (Real.sin α)^2 * (Real.cos α)^2 * (Real.sin α)^2 :=
by sorry

end squares_area_relation_l129_129723


namespace domain_of_f_l129_129698

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ (5 + Real.sqrt 13) / 2 ∧ x ≠ (5 - Real.sqrt 13) / 2 → ∃ y : ℝ, y = f x :=
by
  sorry

end domain_of_f_l129_129698


namespace remainder_3203_4507_9929_mod_75_l129_129453

theorem remainder_3203_4507_9929_mod_75 :
  (3203 * 4507 * 9929) % 75 = 34 :=
by
  have h1 : 3203 % 75 = 53 := sorry
  have h2 : 4507 % 75 = 32 := sorry
  have h3 : 9929 % 75 = 29 := sorry
  -- complete the proof using modular arithmetic rules.
  sorry

end remainder_3203_4507_9929_mod_75_l129_129453


namespace range_of_a_l129_129851

variable (a : ℝ) (f : ℝ → ℝ)
axiom func_def : ∀ x, f x = a^x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom decreasing : ∀ m n : ℝ, m > n → f m < f n

theorem range_of_a : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l129_129851


namespace solve_xy_eq_x_plus_y_l129_129050

theorem solve_xy_eq_x_plus_y (x y : ℤ) (h : x * y = x + y) : (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by {
  sorry
}

end solve_xy_eq_x_plus_y_l129_129050


namespace range_of_m_empty_solution_set_inequality_l129_129448

theorem range_of_m_empty_solution_set_inequality (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 ≥ 0 → false) ↔ -4 < m ∧ m < 0 := 
sorry

end range_of_m_empty_solution_set_inequality_l129_129448


namespace expand_polynomial_correct_l129_129444

open Polynomial

noncomputable def expand_polynomial : Polynomial ℤ :=
  (C 3 * X^3 - C 2 * X^2 + X - C 4) * (C 4 * X^2 - C 2 * X + C 5)

theorem expand_polynomial_correct :
  expand_polynomial = C 12 * X^5 - C 14 * X^4 + C 23 * X^3 - C 28 * X^2 + C 13 * X - C 20 :=
by sorry

end expand_polynomial_correct_l129_129444


namespace length_of_other_parallel_side_l129_129867

theorem length_of_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : 323 = 1/2 * (20 + b) * 17) :
  b = 18 :=
sorry

end length_of_other_parallel_side_l129_129867


namespace positive_integer_conditions_l129_129746

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) :
  (∃ q : ℕ, q > 0 ∧ (5 * p + 36) = q * (2 * p - 9)) ↔ (p = 5 ∨ p = 6 ∨ p = 9 ∨ p = 18) :=
by sorry

end positive_integer_conditions_l129_129746


namespace curves_intersect_four_points_l129_129879

theorem curves_intersect_four_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 4 * a^2 ∧ y = x^2 - 2 * a) → (a > 1/3)) :=
sorry

end curves_intersect_four_points_l129_129879


namespace max_angle_B_l129_129379

-- We define the necessary terms to state our problem
variables {A B C : Real} -- The angles of triangle ABC
variables {cot_A cot_B cot_C : Real} -- The cotangents of angles A, B, and C

-- The main theorem stating that given the conditions the maximum value of angle B is pi/3
theorem max_angle_B (h1 : cot_B = (cot_A + cot_C) / 2) (h2 : A + B + C = Real.pi) :
  B ≤ Real.pi / 3 := by
  sorry

end max_angle_B_l129_129379


namespace problem_statement_l129_129527

def binary_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a^2 - b^2)

theorem problem_statement : binary_op (binary_op 8 6) 2 = 821 / 429 := 
by sorry

end problem_statement_l129_129527


namespace simplify_expression_l129_129499

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := 
by
  sorry

end simplify_expression_l129_129499


namespace largest_integer_n_l129_129981

theorem largest_integer_n (n : ℤ) (h : n^2 - 13 * n + 40 < 0) : n = 7 :=
by
  sorry

end largest_integer_n_l129_129981


namespace no_positive_integer_solutions_l129_129690

theorem no_positive_integer_solutions (x n r : ℕ) (h1 : x > 1) (h2 : x > 0) (h3 : n > 0) (h4 : r > 0) :
  ¬(x^(2*n + 1) = 2^r + 1 ∨ x^(2*n + 1) = 2^r - 1) :=
sorry

end no_positive_integer_solutions_l129_129690


namespace least_x_value_l129_129801

variable (a b : ℕ)
variable (positive_int_a : 0 < a)
variable (positive_int_b : 0 < b)
variable (h : 2 * a^5 = 3 * b^2)

theorem least_x_value (h : 2 * a^5 = 3 * b^2) (positive_int_a : 0 < a) (positive_int_b : 0 < b) : ∃ x, x = 15552 ∧ x = 2 * a^5 ∧ x = 3 * b^2 :=
sorry

end least_x_value_l129_129801


namespace repeating_decimal_sum_l129_129220

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l129_129220


namespace max_number_soap_boxes_l129_129736

-- Definition of dimensions and volumes
def carton_length : ℕ := 25
def carton_width : ℕ := 42
def carton_height : ℕ := 60
def soap_box_length : ℕ := 7
def soap_box_width : ℕ := 12
def soap_box_height : ℕ := 5

def volume (l w h : ℕ) : ℕ := l * w * h

-- Volumes of the carton and soap box
def carton_volume : ℕ := volume carton_length carton_width carton_height
def soap_box_volume : ℕ := volume soap_box_length soap_box_width soap_box_height

-- The maximum number of soap boxes that can be placed in the carton
def max_soap_boxes : ℕ := carton_volume / soap_box_volume

theorem max_number_soap_boxes :
  max_soap_boxes = 150 :=
by
  -- Proof here
  sorry

end max_number_soap_boxes_l129_129736


namespace tan_22_5_expression_l129_129927

theorem tan_22_5_expression :
  let a := 2
  let b := 1
  let c := 0
  let d := 0
  let t := Real.tan (Real.pi / 8)
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
  t = (Real.sqrt a) - (Real.sqrt b) + (Real.sqrt c) - d → 
  a + b + c + d = 3 :=
by
  intros
  exact sorry

end tan_22_5_expression_l129_129927


namespace xy_range_l129_129541

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end xy_range_l129_129541


namespace exist_column_remove_keeps_rows_distinct_l129_129843

theorem exist_column_remove_keeps_rows_distinct 
    (n : ℕ) 
    (table : Fin n → Fin n → Char) 
    (h_diff_rows : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, table i k ≠ table j k) 
    : ∃ col_to_remove : Fin n, ∀ i j : Fin n, i ≠ j → (table i ≠ table j) :=
sorry

end exist_column_remove_keeps_rows_distinct_l129_129843


namespace boxes_count_l129_129345

theorem boxes_count (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) : (total_notebooks / notebooks_per_box) = 3 :=
by
  sorry

end boxes_count_l129_129345


namespace clean_car_time_l129_129835

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end clean_car_time_l129_129835


namespace gcd_2000_7700_l129_129785

theorem gcd_2000_7700 : Nat.gcd 2000 7700 = 100 := by
  -- Prime factorizations of 2000 and 7700
  have fact_2000 : 2000 = 2^4 * 5^3 := sorry
  have fact_7700 : 7700 = 2^2 * 5^2 * 7 * 11 := sorry
  -- Proof of gcd
  sorry

end gcd_2000_7700_l129_129785


namespace janet_total_earnings_l129_129936

def hourly_wage_exterminator := 70
def hourly_work_exterminator := 20
def sculpture_price_per_pound := 20
def sculpture_1_weight := 5
def sculpture_2_weight := 7

theorem janet_total_earnings :
  (hourly_wage_exterminator * hourly_work_exterminator) +
  (sculpture_price_per_pound * sculpture_1_weight) +
  (sculpture_price_per_pound * sculpture_2_weight) = 1640 := by
  sorry

end janet_total_earnings_l129_129936


namespace bricks_needed_for_courtyard_l129_129787

noncomputable def total_bricks_required (courtyard_length courtyard_width : ℝ)
  (brick_length_cm brick_width_cm : ℝ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_length := brick_length_cm / 100
  let brick_width := brick_width_cm / 100
  let brick_area := brick_length * brick_width
  courtyard_area / brick_area

theorem bricks_needed_for_courtyard :
  total_bricks_required 35 24 15 8 = 70000 := by
  sorry

end bricks_needed_for_courtyard_l129_129787


namespace compute_seventy_five_squared_minus_thirty_five_squared_l129_129490

theorem compute_seventy_five_squared_minus_thirty_five_squared :
  75^2 - 35^2 = 4400 := by
  sorry

end compute_seventy_five_squared_minus_thirty_five_squared_l129_129490


namespace min_m_l129_129352

theorem min_m (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
by
  sorry

end min_m_l129_129352


namespace train_speed_l129_129206

/-- A train that crosses a pole in a certain time of 7 seconds and is 210 meters long has a speed of 108 kilometers per hour. -/
theorem train_speed (time_to_cross: ℝ) (length_of_train: ℝ) (speed_kmh : ℝ) 
  (H_time: time_to_cross = 7) (H_length: length_of_train = 210) 
  (conversion_factor: ℝ := 3.6) : speed_kmh = 108 :=
by
  have speed_mps : ℝ := length_of_train / time_to_cross
  have speed_kmh_calc : ℝ := speed_mps * conversion_factor
  sorry

end train_speed_l129_129206


namespace standard_equation_of_ellipse_l129_129058

-- Define the conditions
def isEccentricity (e : ℝ) := e = (Real.sqrt 3) / 3
def segmentLength (L : ℝ) := L = (4 * Real.sqrt 3) / 3

-- Define properties
def is_ellipse (a b c : ℝ) := a > b ∧ b > 0 ∧ (a^2 = b^2 + c^2) ∧ (c = (Real.sqrt 3) / 3 * a)

-- The problem statement
theorem standard_equation_of_ellipse
(a b c : ℝ) (E L : ℝ)
(hE : isEccentricity E)
(hL : segmentLength L)
(h : is_ellipse a b c)
: (a = Real.sqrt 3) ∧ (c = 1) ∧ (b = Real.sqrt 2) ∧ (segmentLength L)
  → ( ∀ x y : ℝ, ((x^2 / 3) + (y^2 / 2) = 1) ) := by
  sorry

end standard_equation_of_ellipse_l129_129058


namespace quadratic_root_value_l129_129256

theorem quadratic_root_value (m : ℝ) :
  ∃ m, (∀ x, x^2 - m * x - 3 = 0 → x = -2) → m = -1/2 :=
by
  sorry

end quadratic_root_value_l129_129256


namespace divisor_between_40_and_50_l129_129318

theorem divisor_between_40_and_50 (n : ℕ) (h1 : 40 ≤ n) (h2 : n ≤ 50) (h3 : n ∣ (2^36 - 1)) : n = 49 :=
sorry

end divisor_between_40_and_50_l129_129318


namespace rectangles_in_grid_squares_in_grid_l129_129988

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 31 → v_lines = 31 → 
  (∃ rect_count : ℕ, rect_count = 216225) :=
by
  intros h_lines_eq v_lines_eq
  sorry

theorem squares_in_grid (n : ℕ) : n = 31 → (∃ square_count : ℕ, square_count = 6975) :=
by
  intros n_eq
  sorry

end rectangles_in_grid_squares_in_grid_l129_129988


namespace arithmetic_sequence_common_difference_l129_129982
-- Lean 4 Proof Statement


theorem arithmetic_sequence_common_difference 
  (a : ℕ) (n : ℕ) (d : ℕ) (S_n : ℕ) (a_n : ℕ) 
  (h1 : a = 2) 
  (h2 : a_n = 29) 
  (h3 : S_n = 155) 
  (h4 : S_n = n * (a + a_n) / 2) 
  (h5 : a_n = a + (n - 1) * d) 
  : d = 3 := 
by 
  sorry

end arithmetic_sequence_common_difference_l129_129982


namespace gcd_m_n_l129_129072

noncomputable def m : ℕ := 5 * 11111111
noncomputable def n : ℕ := 111111111

theorem gcd_m_n : gcd m n = 11111111 := by
  sorry

end gcd_m_n_l129_129072


namespace card_at_42_is_8_spade_l129_129526

-- Conditions Definition
def cards_sequence : List String := 
  ["A♥", "A♠", "2♥", "2♠", "3♥", "3♠", "4♥", "4♠", "5♥", "5♠", "6♥", "6♠", "7♥", "7♠", "8♥", "8♠",
   "9♥", "9♠", "10♥", "10♠", "J♥", "J♠", "Q♥", "Q♠", "K♥", "K♠"]

-- Proposition to be proved
theorem card_at_42_is_8_spade :
  cards_sequence[(41 % 26)] = "8♠" :=
by sorry

end card_at_42_is_8_spade_l129_129526


namespace a_plus_b_is_18_over_5_l129_129426

noncomputable def a_b_sum (a b : ℚ) : Prop :=
  (∃ (x y : ℚ), x = 2 ∧ y = 3 ∧ x = (1 / 3) * y + a ∧ y = (1 / 5) * x + b) → a + b = (18 / 5)

-- No proof provided, just the statement.
theorem a_plus_b_is_18_over_5 (a b : ℚ) : a_b_sum a b :=
sorry

end a_plus_b_is_18_over_5_l129_129426


namespace actual_distance_l129_129597

theorem actual_distance (d_map : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) (H1 : d_map = 20)
    (H2 : scale_inches = 0.5) (H3 : scale_miles = 10) : 
    d_map * (scale_miles / scale_inches) = 400 := 
by
  sorry

end actual_distance_l129_129597


namespace find_huabei_number_l129_129142

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end find_huabei_number_l129_129142


namespace find_number_l129_129325

theorem find_number (k r n : ℤ) (hk : k = 38) (hr : r = 7) (h : n = 23 * k + r) : n = 881 := 
  by
  sorry

end find_number_l129_129325


namespace value_of_a_l129_129396

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_a (a : ℝ) :
  (1 / log_base 2 a) + (1 / log_base 3 a) + (1 / log_base 4 a) + (1 / log_base 5 a) = 7 / 4 ↔
  a = 120 ^ (4 / 7) :=
by
  sorry

end value_of_a_l129_129396


namespace total_spent_l129_129195

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = 0.90 * B
def condition2 : Prop := B = D + 15

-- Question
theorem total_spent : condition1 B D ∧ condition2 B D → B + D = 285 := 
by
  intros h
  sorry

end total_spent_l129_129195


namespace negation_of_p_l129_129562

variable (x y : ℝ)

def proposition_p := ∀ x y : ℝ, x^2 + y^2 - 1 > 0 

theorem negation_of_p : (¬ proposition_p) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end negation_of_p_l129_129562


namespace solve_speeds_ratio_l129_129425

noncomputable def speeds_ratio (v_A v_B : ℝ) : Prop :=
  v_A / v_B = 1 / 3

theorem solve_speeds_ratio (v_A v_B : ℝ) (h1 : ∃ t : ℝ, t = 1 ∧ v_A = 300 - v_B ∧ v_A = v_B ∧ v_B = 300) 
  (h2 : ∃ t : ℝ, t = 7 ∧ 7 * v_A = 300 - 7 * v_B ∧ 7 * v_A = 300 - v_B ∧ 7 * v_B = v_A): 
    speeds_ratio v_A v_B :=
sorry

end solve_speeds_ratio_l129_129425


namespace nancy_kept_chips_correct_l129_129319

/-- Define the initial conditions -/
def total_chips : ℕ := 22
def chips_to_brother : ℕ := 7
def chips_to_sister : ℕ := 5

/-- Define the number of chips Nancy kept -/
def chips_kept : ℕ := total_chips - (chips_to_brother + chips_to_sister)

theorem nancy_kept_chips_correct : chips_kept = 10 := by
  /- This is a placeholder. The proof would go here. -/
  sorry

end nancy_kept_chips_correct_l129_129319


namespace rectangle_area_eq_2a_squared_l129_129067

variable {α : Type} [Semiring α] (a : α)

-- Conditions
def width (a : α) : α := a
def length (a : α) : α := 2 * a

-- Proof statement
theorem rectangle_area_eq_2a_squared (a : α) : (length a) * (width a) = 2 * a^2 := 
sorry

end rectangle_area_eq_2a_squared_l129_129067


namespace min_value_k_l129_129350

variables (x : ℕ → ℚ) (k n c : ℚ)

theorem min_value_k
  (k_gt_one : k > 1) -- condition that k > 1
  (n_gt_2018 : n > 2018) -- condition that n > 2018
  (n_odd : n % 2 = 1) -- condition that n is odd
  (non_zero_rational : ∀ i : ℕ, x i ≠ 0) -- non-zero rational numbers x₁, x₂, ..., xₙ
  (not_all_equal : ∃ i j : ℕ, x i ≠ x j) -- they are not all equal
  (relations : ∀ i : ℕ, x i + k / x (i + 1) = c) -- given relations
  : k = 4 :=
sorry

end min_value_k_l129_129350


namespace ambika_candles_count_l129_129625

-- Definitions
def Aniyah_candles (A : ℕ) : ℕ := 6 * A
def combined_candles (A : ℕ) : ℕ := A + Aniyah_candles A

-- Problem Statement:
theorem ambika_candles_count : ∃ A : ℕ, combined_candles A = 28 ∧ A = 4 :=
by
  sorry

end ambika_candles_count_l129_129625


namespace system_of_equations_has_integer_solutions_l129_129952

theorem system_of_equations_has_integer_solutions (a b : ℤ) :
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end system_of_equations_has_integer_solutions_l129_129952


namespace exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l129_129480

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d > 1 ∧ d < n

def consecutiveComposites (start n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < n → isComposite (start + i)

theorem exists_nine_consecutive_composites :
  ∃ start, start + 8 ≤ 500 ∧ consecutiveComposites start 9 :=
sorry

theorem exists_eleven_consecutive_composites :
  ∃ start, start + 10 ≤ 500 ∧ consecutiveComposites start 11 :=
sorry

end exists_nine_consecutive_composites_exists_eleven_consecutive_composites_l129_129480


namespace cannot_be_sum_of_two_or_more_consecutive_integers_l129_129750

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem cannot_be_sum_of_two_or_more_consecutive_integers (n : ℕ) :
  (¬∃ k m : ℕ, k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ is_power_of_two n :=
by
  sorry

end cannot_be_sum_of_two_or_more_consecutive_integers_l129_129750


namespace solution_set_inequality_l129_129639

theorem solution_set_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) :=
by sorry

end solution_set_inequality_l129_129639


namespace bound_on_k_l129_129790

variables {n k : ℕ}
variables (a : ℕ → ℕ) (h1 : 1 ≤ k) (h2 : ∀ i j, 1 ≤ i → j ≤ k → i < j → a i < a j)
variables (h3 : ∀ i, a i ≤ n) (h4 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → a i ≠ a j))
variables (h5 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → ∀ m p, m ≤ p → m ≤ k → p ≤ k → a i + a j ≠ a m + a p))

theorem bound_on_k : k ≤ Nat.floor (Real.sqrt (2 * n) + 1) :=
sorry

end bound_on_k_l129_129790


namespace hyperbola_eccentricity_l129_129181

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h2 : ∀ c : ℝ, c - a^2 / c = 2 * a) :
  e = 1 + Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l129_129181


namespace correction_amount_l129_129468

variable (x : ℕ)

def half_dollar := 50
def quarter := 25
def nickel := 5
def dime := 10

theorem correction_amount : 
  ∀ x, (x * (half_dollar - quarter)) - (x * (dime - nickel)) = 20 * x := by
  intros x 
  sorry

end correction_amount_l129_129468


namespace abigail_score_l129_129169

theorem abigail_score (sum_20 : ℕ) (sum_21 : ℕ) (h1 : sum_20 = 1700) (h2 : sum_21 = 1806) : (sum_21 - sum_20) = 106 :=
by
  sorry

end abigail_score_l129_129169


namespace minimum_daily_expense_l129_129316

-- Defining the context
variables (x y : ℕ)
def total_capacity (x y : ℕ) : ℕ := 24 * x + 30 * y
def cost (x y : ℕ) : ℕ := 320 * x + 504 * y

theorem minimum_daily_expense :
  (total_capacity x y ≥ 180) →
  (x ≤ 8) →
  (y ≤ 4) →
  cost x y = 2560 := sorry

end minimum_daily_expense_l129_129316


namespace largest_integer_x_l129_129601

theorem largest_integer_x (x : ℤ) : (x / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) → x ≤ 7 ∧ (7 / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) :=
by
  sorry

end largest_integer_x_l129_129601


namespace unique_providers_count_l129_129475

theorem unique_providers_count :
  let num_children := 4
  let num_providers := 25
  (∀ s : Fin num_children, s.val < num_providers)
  → num_providers * (num_providers - 1) * (num_providers - 2) * (num_providers - 3) = 303600
:= sorry

end unique_providers_count_l129_129475


namespace purely_imaginary_number_eq_l129_129428

theorem purely_imaginary_number_eq (z : ℂ) (a : ℝ) (i : ℂ) (h_imag : z.im = 0 ∧ z = 0 ∧ (3 - i) * z = a + i + i) :
  a = 1 / 3 :=
  sorry

end purely_imaginary_number_eq_l129_129428


namespace average_salary_l129_129849

def A_salary : ℝ := 9000
def B_salary : ℝ := 5000
def C_salary : ℝ := 11000
def D_salary : ℝ := 7000
def E_salary : ℝ := 9000
def number_of_people : ℝ := 5
def total_salary : ℝ := A_salary + B_salary + C_salary + D_salary + E_salary

theorem average_salary : (total_salary / number_of_people) = 8200 := by
  sorry

end average_salary_l129_129849


namespace train_length_approx_l129_129137

noncomputable def speed_kmh_to_ms (v: ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def length_of_train (v_kmh: ℝ) (time_s: ℝ) : ℝ :=
  (speed_kmh_to_ms v_kmh) * time_s

theorem train_length_approx (v_kmh: ℝ) (time_s: ℝ) (L: ℝ) 
  (h1: v_kmh = 58) 
  (h2: time_s = 9) 
  (h3: L = length_of_train v_kmh time_s) : 
  |L - 145| < 1 :=
  by sorry

end train_length_approx_l129_129137


namespace total_grains_in_grey_regions_l129_129891

def total_grains_circle1 : ℕ := 87
def total_grains_circle2 : ℕ := 110
def white_grains_circle1 : ℕ := 68
def white_grains_circle2 : ℕ := 68

theorem total_grains_in_grey_regions : total_grains_circle1 - white_grains_circle1 + (total_grains_circle2 - white_grains_circle2) = 61 :=
by
  sorry

end total_grains_in_grey_regions_l129_129891


namespace correct_equation_l129_129860

namespace MathProblem

def is_two_digit_positive_integer (P : ℤ) : Prop :=
  10 ≤ P ∧ P < 100

def equation_A : Prop :=
  ∀ x : ℤ, x^2 + (-98)*x + 2001 = (x - 29) * (x - 69)

def equation_B : Prop :=
  ∀ x : ℤ, x^2 + (-110)*x + 2001 = (x - 23) * (x - 87)

def equation_C : Prop :=
  ∀ x : ℤ, x^2 + 110*x + 2001 = (x + 23) * (x + 87)

def equation_D : Prop :=
  ∀ x : ℤ, x^2 + 98*x + 2001 = (x + 29) * (x + 69)

theorem correct_equation :
  is_two_digit_positive_integer 98 ∧ equation_D :=
  sorry

end MathProblem

end correct_equation_l129_129860


namespace candy_mixture_price_l129_129819

theorem candy_mixture_price
  (price_first_per_kg : ℝ) (price_second_per_kg : ℝ) (weight_ratio : ℝ) (weight_second : ℝ) 
  (h1 : price_first_per_kg = 10) 
  (h2 : price_second_per_kg = 15) 
  (h3 : weight_ratio = 3) 
  : (price_first_per_kg * weight_ratio * weight_second + price_second_per_kg * weight_second) / 
    (weight_ratio * weight_second + weight_second) = 11.25 :=
by
  sorry

end candy_mixture_price_l129_129819


namespace max_log2_x_2log2_y_l129_129740

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem max_log2_x_2log2_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y^2 = 2) :
  log2 x + 2 * log2 y ≤ 0 :=
sorry

end max_log2_x_2log2_y_l129_129740


namespace division_expression_is_7_l129_129556

noncomputable def evaluate_expression : ℝ :=
  1 / 2 / 3 / 4 / 5 / (6 / 7 / 8 / 9 / 10)

theorem division_expression_is_7 : evaluate_expression = 7 :=
by
  sorry

end division_expression_is_7_l129_129556


namespace sin_y_gt_half_x_l129_129590

theorem sin_y_gt_half_x (x y : ℝ) (hx : x ≤ 90) (h : Real.sin y = (3 / 4) * Real.sin x) : y > x / 2 :=
by
  sorry

end sin_y_gt_half_x_l129_129590


namespace proof_problem_l129_129772

-- Given condition
variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b)
variable (h3 : Real.log a + Real.log (b ^ 2) ≥ 2 * a + (b ^ 2) / 2 - 2)

-- Proof statement
theorem proof_problem : a - 2 * b = 1/2 - 2 * Real.sqrt 2 :=
by
  sorry

end proof_problem_l129_129772


namespace rons_height_l129_129208

variable (R : ℝ)

theorem rons_height
  (depth_eq_16_ron_height : 16 * R = 208) :
  R = 13 :=
by {
  sorry
}

end rons_height_l129_129208


namespace necessary_but_not_sufficient_l129_129760

section geometric_progression

variables {a b c : ℝ}

def geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a = b / r ∧ c = b * r

def necessary_condition (a b c : ℝ) : Prop :=
  a * c = b^2

theorem necessary_but_not_sufficient :
  (geometric_progression a b c → necessary_condition a b c) ∧
  (¬ (necessary_condition a b c → geometric_progression a b c)) :=
by sorry

end geometric_progression

end necessary_but_not_sufficient_l129_129760


namespace find_A_l129_129970

variable (x ω φ b A : ℝ)

-- Given conditions
axiom cos_squared_eq : 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b
axiom A_gt_zero : A > 0

-- Lean 4 statement to prove
theorem find_A : A = Real.sqrt 2 :=
by
  sorry

end find_A_l129_129970


namespace find_n_l129_129098

theorem find_n (x y : ℤ) (n : ℕ) (h1 : (x:ℝ)^n + (y:ℝ)^n = 91) (h2 : (x:ℝ) * y = 11.999999999999998) :
  n = 3 := 
sorry

end find_n_l129_129098


namespace smallest_integer_condition_l129_129969

theorem smallest_integer_condition :
  ∃ (x : ℕ) (d : ℕ) (n : ℕ) (p : ℕ), x = 1350 ∧ d = 1 ∧ n = 450 ∧ p = 2 ∧
  x = 10^p * d + n ∧
  n = x / 19 ∧
  (1 ≤ d ∧ d ≤ 9 ∧ 10^p * d % 18 = 0) :=
sorry

end smallest_integer_condition_l129_129969


namespace apples_in_box_ratio_mixed_fruits_to_total_l129_129618

variable (total_fruits : Nat) (oranges : Nat) (peaches : Nat) (apples : Nat) (mixed_fruits : Nat)
variable (one_fourth_of_box_contains_oranges : oranges = total_fruits / 4)
variable (half_as_many_peaches_as_oranges : peaches = oranges / 2)
variable (five_times_as_many_apples_as_peaches : apples = 5 * peaches)
variable (mixed_fruits_double_peaches : mixed_fruits = 2 * peaches)
variable (total_fruits_56 : total_fruits = 56)

theorem apples_in_box : apples = 35 := by
  sorry

theorem ratio_mixed_fruits_to_total : mixed_fruits / total_fruits = 1 / 4 := by
  sorry

end apples_in_box_ratio_mixed_fruits_to_total_l129_129618


namespace goods_train_length_l129_129752

noncomputable def speed_kmh : ℕ := 72  -- Speed of the goods train in km/hr
noncomputable def platform_length : ℕ := 280  -- Length of the platform in meters
noncomputable def time_seconds : ℕ := 26  -- Time taken to cross the platform in seconds
noncomputable def speed_mps : ℤ := speed_kmh * 1000 / 3600 -- Speed of the goods train in meters/second

theorem goods_train_length : 20 * time_seconds = 280 + 240 :=
by
  sorry

end goods_train_length_l129_129752


namespace number_of_liars_l129_129489

/-- Definition of conditions -/
def total_islands : Nat := 17
def population_per_island : Nat := 119

-- Conditions based on the problem description
def islands_yes_first_question : Nat := 7
def islands_no_first_question : Nat := total_islands - islands_yes_first_question

def islands_no_second_question : Nat := 7
def islands_yes_second_question : Nat := total_islands - islands_no_second_question

def minimum_knights_for_no_second_question : Nat := 60  -- At least 60 knights

/-- Main theorem -/
theorem number_of_liars : 
  ∃ x y: Nat, 
    (x + (islands_no_first_question - y) = islands_yes_first_question ∧ 
     y - x = 3 ∧ 
     60 * x + 59 * y + 119 * (islands_no_first_question - y) = 1010 ∧
     (total_islands * population_per_island - 1010 = 1013)) := by
  sorry

end number_of_liars_l129_129489


namespace student_count_estimate_l129_129416

theorem student_count_estimate 
  (n : Nat) 
  (h1 : 80 ≤ n) 
  (h2 : 100 ≤ n) 
  (h3 : 20 * n = 8000) : 
  n = 400 := 
by 
  sorry

end student_count_estimate_l129_129416


namespace complex_magnitude_l129_129130

open Complex

noncomputable def complexZ : ℂ := sorry -- Definition of complex number z

theorem complex_magnitude (z : ℂ) (h : (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I) : abs z = Real.sqrt 5 :=
sorry

end complex_magnitude_l129_129130


namespace graph_of_transformed_function_l129_129298

theorem graph_of_transformed_function
  (f : ℝ → ℝ)
  (hf : f⁻¹ 1 = 0) :
  f (1 - 1) = 1 :=
by
  sorry

end graph_of_transformed_function_l129_129298


namespace mark_owes_linda_l129_129464

-- Define the payment per room and the number of rooms painted
def payment_per_room := (13 : ℚ) / 3
def rooms_painted := (8 : ℚ) / 5

-- State the theorem and the proof
theorem mark_owes_linda : (payment_per_room * rooms_painted) = (104 : ℚ) / 15 := by
  sorry

end mark_owes_linda_l129_129464


namespace find_number_l129_129479

theorem find_number (x : ℕ) (h : 24 * x = 2376) : x = 99 :=
by
  sorry

end find_number_l129_129479


namespace sum_opposite_signs_eq_zero_l129_129857

theorem sum_opposite_signs_eq_zero (x y : ℝ) (h : x * y < 0) : x + y = 0 :=
sorry

end sum_opposite_signs_eq_zero_l129_129857


namespace max_E_l129_129897

def E (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  x₁ + x₂ + x₃ + x₄ -
  x₁ * x₂ - x₁ * x₃ - x₁ * x₄ -
  x₂ * x₃ - x₂ * x₄ - x₃ * x₄ +
  x₁ * x₂ * x₃ + x₁ * x₂ * x₄ +
  x₁ * x₃ * x₄ + x₂ * x₃ * x₄ -
  x₁ * x₂ * x₃ * x₄

theorem max_E (x₁ x₂ x₃ x₄ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 1) (h₅ : 0 ≤ x₃) (h₆ : x₃ ≤ 1) (h₇ : 0 ≤ x₄) (h₈ : x₄ ≤ 1) : 
  E x₁ x₂ x₃ x₄ ≤ 1 :=
sorry

end max_E_l129_129897


namespace two_digit_sum_of_original_and_reverse_l129_129353

theorem two_digit_sum_of_original_and_reverse
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9) -- a is a digit
  (h2 : 0 ≤ b ∧ b ≤ 9) -- b is a digit
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_of_original_and_reverse_l129_129353


namespace motorist_routes_birmingham_to_sheffield_l129_129472

-- Definitions for the conditions
def routes_bristol_to_birmingham : ℕ := 6
def routes_sheffield_to_carlisle : ℕ := 2
def total_routes_bristol_to_carlisle : ℕ := 36

-- The proposition that should be proven
theorem motorist_routes_birmingham_to_sheffield : 
  ∃ x : ℕ, routes_bristol_to_birmingham * x * routes_sheffield_to_carlisle = total_routes_bristol_to_carlisle ∧ x = 3 :=
sorry

end motorist_routes_birmingham_to_sheffield_l129_129472


namespace polyhedron_volume_is_correct_l129_129131

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 12
  let num_squares := 3
  let square_area := side_length * side_length
  let cube_volume := side_length ^ 3
  let polyhedron_volume := cube_volume / 2
  polyhedron_volume

theorem polyhedron_volume_is_correct :
  volume_of_polyhedron = 864 :=
by
  sorry

end polyhedron_volume_is_correct_l129_129131


namespace price_of_5_pound_bag_l129_129138

-- Definitions based on conditions
def price_10_pound_bag : ℝ := 20.42
def price_25_pound_bag : ℝ := 32.25
def min_pounds : ℝ := 65
def max_pounds : ℝ := 80
def total_min_cost : ℝ := 98.77

-- Define the sought price of the 5-pound bag in the hypothesis
variable {price_5_pound_bag : ℝ}

-- The theorem to prove based on the given conditions
theorem price_of_5_pound_bag
  (h₁ : price_10_pound_bag = 20.42)
  (h₂ : price_25_pound_bag = 32.25)
  (h₃ : min_pounds = 65)
  (h₄ : max_pounds = 80)
  (h₅ : total_min_cost = 98.77) :
  price_5_pound_bag = 2.02 :=
sorry

end price_of_5_pound_bag_l129_129138


namespace cans_of_type_B_purchased_l129_129070

variable (T P R : ℕ)

-- Conditions
def cost_per_can_A : ℕ := P / T
def cost_per_can_B : ℕ := 2 * cost_per_can_A T P
def quarters_in_dollar : ℕ := 4

-- Question and proof target
theorem cans_of_type_B_purchased (T P R : ℕ) (hT : T > 0) (hP : P > 0) (hR : R > 0) :
  (4 * R) / (2 * P / T) = 2 * R * T / P :=
by
  sorry

end cans_of_type_B_purchased_l129_129070


namespace lowest_die_exactly_3_prob_l129_129834

noncomputable def fair_die_prob_at_least (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 6 then (6 - n + 1) / 6 else 0

noncomputable def prob_lowest_die_exactly_3 : ℚ :=
  let p_at_least_3 := fair_die_prob_at_least 3
  let p_at_least_4 := fair_die_prob_at_least 4
  (p_at_least_3 ^ 4) - (p_at_least_4 ^ 4)

theorem lowest_die_exactly_3_prob :
  prob_lowest_die_exactly_3 = 175 / 1296 := by
  sorry

end lowest_die_exactly_3_prob_l129_129834


namespace rate_per_kg_mangoes_l129_129302

theorem rate_per_kg_mangoes 
  (weight_grapes : ℕ) 
  (rate_grapes : ℕ) 
  (weight_mangoes : ℕ) 
  (total_paid : ℕ)
  (total_grapes_cost : ℕ)
  (total_mangoes_cost : ℕ)
  (rate_mangoes : ℕ) 
  (h1 : weight_grapes = 14) 
  (h2 : rate_grapes = 54)
  (h3 : weight_mangoes = 10) 
  (h4 : total_paid = 1376) 
  (h5 : total_grapes_cost = weight_grapes * rate_grapes)
  (h6 : total_mangoes_cost = total_paid - total_grapes_cost) 
  (h7 : rate_mangoes = total_mangoes_cost / weight_mangoes):
  rate_mangoes = 62 :=
by
  sorry

end rate_per_kg_mangoes_l129_129302


namespace max_prime_area_of_rectangle_with_perimeter_40_is_19_l129_129685

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Given conditions: perimeter of 40 units; perimeter condition and area as prime number.
def max_prime_area_of_rectangle_with_perimeter_40 : Prop :=
  ∃ (l w : ℕ), l + w = 20 ∧ is_prime (l * (20 - l)) ∧
  ∀ (l' w' : ℕ), l' + w' = 20 → is_prime (l' * (20 - l')) → (l * (20 - l)) ≥ (l' * (20 - l'))

theorem max_prime_area_of_rectangle_with_perimeter_40_is_19 :
  max_prime_area_of_rectangle_with_perimeter_40 :=
sorry

end max_prime_area_of_rectangle_with_perimeter_40_is_19_l129_129685


namespace tomatoes_left_l129_129986

theorem tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) (eaters : ℕ) (final_tomatoes : ℕ)  
  (h_initial : initial_tomatoes = 21)
  (h_fraction : fraction_eaten = 1 / 3)
  (h_eaters : eaters = 2)
  (h_final : final_tomatoes = initial_tomatoes - initial_tomatoes * fraction_eaten) :
  final_tomatoes = 14 := by
  sorry

end tomatoes_left_l129_129986


namespace g_g_x_has_two_distinct_real_roots_iff_l129_129027

noncomputable def g (d x : ℝ) := x^2 - 4 * x + d

def has_two_distinct_real_roots (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem g_g_x_has_two_distinct_real_roots_iff (d : ℝ) :
  has_two_distinct_real_roots (g d ∘ g d) ↔ d = 8 := sorry

end g_g_x_has_two_distinct_real_roots_iff_l129_129027


namespace least_number_remainder_seven_exists_l129_129210

theorem least_number_remainder_seven_exists :
  ∃ x : ℕ, x ≡ 7 [MOD 11] ∧ x ≡ 7 [MOD 17] ∧ x ≡ 7 [MOD 21] ∧ x ≡ 7 [MOD 29] ∧ x ≡ 7 [MOD 35] ∧ 
           x ≡ 1547 [MOD Nat.lcm 11 (Nat.lcm 17 (Nat.lcm 21 (Nat.lcm 29 35)))] :=
  sorry

end least_number_remainder_seven_exists_l129_129210


namespace abs_diff_p_q_l129_129502

theorem abs_diff_p_q (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
by 
  sorry

end abs_diff_p_q_l129_129502


namespace constantin_mother_deposit_return_l129_129510

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l129_129510


namespace divide_into_two_groups_l129_129339

theorem divide_into_two_groups (n : ℕ) (A : Fin n → Type) 
  (acquaintances : (Fin n) → (Finset (Fin n)))
  (c : (Fin n) → ℕ) (d : (Fin n) → ℕ) :
  (∀ i : Fin n, c i = (acquaintances i).card) →
  ∃ G1 G2 : Finset (Fin n), G1 ∩ G2 = ∅ ∧ G1 ∪ G2 = Finset.univ ∧
  (∀ i : Fin n, d i = (acquaintances i ∩ (if i ∈ G1 then G2 else G1)).card ∧ d i ≥ (c i) / 2) :=
by 
  sorry

end divide_into_two_groups_l129_129339


namespace area_not_covered_by_small_squares_l129_129536

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def large_square_area : ℕ := large_square_side_length ^ 2
def small_square_area : ℕ := small_square_side_length ^ 2
def uncovered_area : ℕ := large_square_area - small_square_area

theorem area_not_covered_by_small_squares :
  uncovered_area = 84 := by
  sorry

end area_not_covered_by_small_squares_l129_129536


namespace min_cost_for_boxes_l129_129029

theorem min_cost_for_boxes
  (box_length: ℕ) (box_width: ℕ) (box_height: ℕ)
  (cost_per_box: ℝ) (total_volume: ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : cost_per_box = 1.30)
  (h5 : total_volume = 3060000) :
  ∃ cost: ℝ, cost = 663 :=
by
  sorry

end min_cost_for_boxes_l129_129029


namespace arthur_bought_2_hamburgers_on_second_day_l129_129650

theorem arthur_bought_2_hamburgers_on_second_day
  (H D X: ℕ)
  (h1: 3 * H + 4 * D = 10)
  (h2: D = 1)
  (h3: 2 * X + 3 * D = 7):
  X = 2 :=
by
  sorry

end arthur_bought_2_hamburgers_on_second_day_l129_129650


namespace constant_function_l129_129051

theorem constant_function {f : ℕ → ℕ} (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) : ∃ c : ℕ, ∀ x, f x = c := 
sorry

end constant_function_l129_129051


namespace difference_of_squares_l129_129431

def a : ℕ := 601
def b : ℕ := 597

theorem difference_of_squares : a^2 - b^2 = 4792 :=
by {
  sorry
}

end difference_of_squares_l129_129431


namespace sum_of_x_coordinates_on_parabola_l129_129470

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1

-- Define the points P and Q on the parabola
variables {x1 x2 : ℝ}

-- The Lean theorem statement: 
theorem sum_of_x_coordinates_on_parabola 
  (h1 : parabola x1 = 1) 
  (h2 : parabola x2 = 1) : 
  x1 + x2 = 2 :=
sorry

end sum_of_x_coordinates_on_parabola_l129_129470


namespace smallest_n_for_sqrt_18n_integer_l129_129121

theorem smallest_n_for_sqrt_18n_integer :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (∃ k : ℕ, k^2 = 18 * m) → n <= m) ∧ (∃ k : ℕ, k^2 = 18 * n) :=
sorry

end smallest_n_for_sqrt_18n_integer_l129_129121


namespace measure_of_angle_is_135_l129_129263

noncomputable def degree_measure_of_angle (x : ℝ) : Prop :=
  (x = 3 * (180 - x)) ∧ (2 * x + (180 - x) = 180) -- Combining all conditions

theorem measure_of_angle_is_135 (x : ℝ) (h : degree_measure_of_angle x) : x = 135 :=
by sorry

end measure_of_angle_is_135_l129_129263


namespace geometric_series_common_ratio_l129_129243

theorem geometric_series_common_ratio (a : ℕ → ℚ) (q : ℚ) (h1 : a 1 + a 3 = 10) 
(h2 : a 4 + a 6 = 5 / 4) 
(h_geom : ∀ n : ℕ, a (n + 1) = a n * q) : q = 1 / 2 :=
sorry

end geometric_series_common_ratio_l129_129243


namespace fraction_value_l129_129942

theorem fraction_value (x y : ℝ) (h1 : 2 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 5) (h3 : (∃ m : ℤ, x = m * y)) : x / y = -2 :=
sorry

end fraction_value_l129_129942


namespace geom_series_sum_l129_129446

/-- The sum of the first six terms of the geometric series 
    with first term a = 1 and common ratio r = (1 / 4) is 1365 / 1024. -/
theorem geom_series_sum : 
  let a : ℚ := 1
  let r : ℚ := 1 / 4
  let n : ℕ := 6
  (a * (1 - r^n) / (1 - r)) = 1365 / 1024 :=
by
  sorry

end geom_series_sum_l129_129446


namespace necessary_not_sufficient_condition_not_sufficient_condition_l129_129016

theorem necessary_not_sufficient_condition (x : ℝ) :
  (1 < x ∧ x < 4) → (|x - 2| < 1) := sorry

theorem not_sufficient_condition (x : ℝ) :
  (|x - 2| < 1) → (1 < x ∧ x < 4) := sorry

end necessary_not_sufficient_condition_not_sufficient_condition_l129_129016


namespace product_of_all_possible_N_l129_129191

theorem product_of_all_possible_N (A B N : ℝ) 
  (h1 : A = B + N)
  (h2 : A - 4 = B + N - 4)
  (h3 : B + 5 = B + 5)
  (h4 : |((B + N - 4) - (B + 5))| = 1) :
  ∃ N₁ N₂ : ℝ, (|N₁ - 9| = 1 ∧ |N₂ - 9| = 1) ∧ N₁ * N₂ = 80 :=
by {
  -- We know the absolute value equation leads to two solutions
  -- hence we will consider N₁ and N₂ such that |N - 9| = 1
  -- which eventually yields N = 10 and N = 8, making their product 80.
  sorry
}

end product_of_all_possible_N_l129_129191


namespace point_on_same_side_as_l129_129122

def f (x y : ℝ) : ℝ := 2 * x - y + 1

theorem point_on_same_side_as (x1 y1 : ℝ) (h : f 1 2 > 0) : f 1 0 > 0 := sorry

end point_on_same_side_as_l129_129122


namespace problem_1_problem_2_l129_129213

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l129_129213


namespace find_a_l129_129038

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }
def setB : Set ℝ := { x | Real.log (x^2 - 5 * x + 8) / Real.log 2 = 1 }
def setC (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }

-- Proof statement to find the value of a
theorem find_a (a : ℝ) : setA ∩ setC a = ∅ → setB ∩ setC a ≠ ∅ → a = -2 := by
  sorry

end find_a_l129_129038


namespace proof_problem_l129_129565

theorem proof_problem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^3 + b^3 = 2 * a * b) : a^2 + b^2 ≤ 1 + a * b := 
sorry

end proof_problem_l129_129565


namespace bianca_ate_candies_l129_129873

-- Definitions based on the conditions
def total_candies : ℕ := 32
def pieces_per_pile : ℕ := 5
def number_of_piles : ℕ := 4

-- The statement to prove
theorem bianca_ate_candies : 
  total_candies - (pieces_per_pile * number_of_piles) = 12 := 
by 
  sorry

end bianca_ate_candies_l129_129873


namespace original_price_of_apples_l129_129868

-- Define variables and conditions
variables (P : ℝ)

-- The conditions of the problem
def price_increase_condition := 1.25 * P * 8 = 64

-- The theorem stating the original price per pound of apples
theorem original_price_of_apples (h : price_increase_condition P) : P = 6.40 :=
sorry

end original_price_of_apples_l129_129868


namespace find_second_number_l129_129711

theorem find_second_number (a b c : ℚ) (h1 : a + b + c = 98) (h2 : a = (2 / 3) * b) (h3 : c = (8 / 5) * b) : b = 30 :=
by sorry

end find_second_number_l129_129711


namespace part1_part2_l129_129586

section
variable (x a : ℝ)
def p (x a : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem part1 (h : a = 1) (hq : q x) (hp : p x a) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h : ∀ x, q x → p x a) : 1 ≤ a ∧ a ≤ 2 := by
  sorry
end

end part1_part2_l129_129586


namespace n_power_of_two_if_2_pow_n_plus_one_odd_prime_l129_129308

-- Definition: a positive integer n is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Theorem: if 2^n +1 is an odd prime, then n must be a power of 2
theorem n_power_of_two_if_2_pow_n_plus_one_odd_prime (n : ℕ) (hp : Prime (2^n + 1)) (hn : Odd (2^n + 1)) : is_power_of_two n :=
by
  sorry

end n_power_of_two_if_2_pow_n_plus_one_odd_prime_l129_129308


namespace find_x_l129_129358

-- Define the angles as real numbers representing degrees.
variable (angle_SWR angle_WRU angle_x : ℝ)

-- Conditions given in the problem
def conditions (angle_SWR angle_WRU angle_x : ℝ) : Prop :=
  angle_SWR = 50 ∧ angle_WRU = 30 ∧ angle_SWR = angle_WRU + angle_x

-- Main theorem to prove that x = 20 given the conditions
theorem find_x (angle_SWR angle_WRU angle_x : ℝ) :
  conditions angle_SWR angle_WRU angle_x → angle_x = 20 := by
  sorry

end find_x_l129_129358


namespace smaller_angle_between_east_and_northwest_l129_129317

theorem smaller_angle_between_east_and_northwest
  (rays : ℕ)
  (each_angle : ℕ)
  (direction : ℕ → ℝ)
  (h1 : rays = 10)
  (h2 : each_angle = 36)
  (h3 : direction 0 = 0) -- ray at due North
  (h4 : direction 3 = 90) -- ray at due East
  (h5 : direction 5 = 135) -- ray at due Northwest
  : direction 5 - direction 3 = each_angle :=
by
  -- to be proved
  sorry

end smaller_angle_between_east_and_northwest_l129_129317


namespace sum_of_squares_greater_than_cubics_l129_129476

theorem sum_of_squares_greater_than_cubics (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a)
  : 
  (2 * (a + b + c) * (a^2 + b^2 + c^2)) / 3 > a^3 + b^3 + c^3 + a * b * c := 
by 
  sorry

end sum_of_squares_greater_than_cubics_l129_129476


namespace correct_calculation_l129_129529

theorem correct_calculation (a b m : ℤ) : 
  (¬((a^3)^2 = a^5)) ∧ ((-2 * m^3)^2 = 4 * m^6) ∧ (¬(a^6 / a^2 = a^3)) ∧ (¬((a + b)^2 = a^2 + b^2)) := 
by
  sorry

end correct_calculation_l129_129529


namespace cube_painting_probability_l129_129225

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end cube_painting_probability_l129_129225


namespace spelling_bee_initial_students_l129_129192

theorem spelling_bee_initial_students (x : ℕ) 
    (h1 : (2 / 3) * x = 2 / 3 * x)
    (h2 : (3 / 4) * ((1 / 3) * x) = 3 / 4 * (1 / 3 * x))
    (h3 : (1 / 3) * x * (1 / 4) = 30) : 
  x = 120 :=
sorry

end spelling_bee_initial_students_l129_129192


namespace angle_BAC_eq_69_l129_129099

-- Definitions and conditions
def AM_Squared_EQ_CM_MN (AM CM MN : ℝ) : Prop := AM^2 = CM * MN
def AM_EQ_MK (AM MK : ℝ) : Prop := AM = MK
def angle_AMN_EQ_CMK (angle_AMN angle_CMK : ℝ) : Prop := angle_AMN = angle_CMK
def angle_B : ℝ := 47
def angle_C : ℝ := 64

-- Final proof statement
theorem angle_BAC_eq_69 (AM CM MN MK : ℝ)
  (h1: AM_Squared_EQ_CM_MN AM CM MN)
  (h2: AM_EQ_MK AM MK)
  (h3: angle_AMN_EQ_CMK 70 70) -- Placeholder angle values since angles must be given/defined
  : ∃ angle_BAC : ℝ, angle_BAC = 69 :=
sorry

end angle_BAC_eq_69_l129_129099


namespace quadratic_root_inequality_l129_129348

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l129_129348


namespace min_goals_in_previous_three_matches_l129_129606

theorem min_goals_in_previous_three_matches 
  (score1 score2 score3 score4 : ℕ)
  (total_after_seven_matches : ℕ)
  (previous_three_goal_sum : ℕ) :
  score1 = 18 →
  score2 = 12 →
  score3 = 15 →
  score4 = 14 →
  total_after_seven_matches ≥ 100 →
  previous_three_goal_sum = total_after_seven_matches - (score1 + score2 + score3 + score4) →
  (previous_three_goal_sum / 3 : ℝ) < ((score1 + score2 + score3 + score4) / 4 : ℝ) →
  previous_three_goal_sum ≥ 41 :=
by
  sorry

end min_goals_in_previous_three_matches_l129_129606


namespace chang_apple_problem_l129_129198

theorem chang_apple_problem 
  (A : ℝ)
  (h1 : 0.50 * A * 0.50 + 0.25 * A * 0.10 + 0.15 * A * 0.30 + 0.10 * A * 0.20 = 80)
  : A = 235 := 
sorry

end chang_apple_problem_l129_129198


namespace winner_percentage_l129_129748

theorem winner_percentage (V_winner V_margin V_total : ℕ) (h_winner: V_winner = 806) (h_margin: V_margin = 312) (h_total: V_total = V_winner + (V_winner - V_margin)) :
  ((V_winner: ℚ) / V_total) * 100 = 62 := by
  sorry

end winner_percentage_l129_129748


namespace evaluate_expression_l129_129384

theorem evaluate_expression :
  3000 * (3000 ^ 1500 + 3000 ^ 1500) = 2 * 3000 ^ 1501 :=
by sorry

end evaluate_expression_l129_129384


namespace function_properties_l129_129946

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ ⦃a b : ℝ⦄, 0 < a → a < b → b < 2 → f b < f a) := by
  sorry

end function_properties_l129_129946


namespace arithmetic_expression_l129_129458

theorem arithmetic_expression :
  (30 / (10 + 2 - 5) + 4) * 7 = 58 :=
by
  sorry

end arithmetic_expression_l129_129458


namespace relatively_prime_perfect_squares_l129_129560

theorem relatively_prime_perfect_squares (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_eq : (1:ℚ) / a + (1:ℚ) / b = (1:ℚ) / c) :
    ∃ x y z : ℤ, (a + b = x^2 ∧ a - c = y^2 ∧ b - c = z^2) :=
  sorry

end relatively_prime_perfect_squares_l129_129560


namespace rate_of_mangoes_per_kg_l129_129147

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg_l129_129147


namespace find_probabilities_l129_129279

theorem find_probabilities (p_1 p_3 : ℝ)
  (h1 : p_1 + 0.15 + p_3 + 0.25 + 0.35 = 1)
  (h2 : p_3 = 4 * p_1) :
  p_1 = 0.05 ∧ p_3 = 0.20 :=
by
  sorry

end find_probabilities_l129_129279


namespace union_when_m_equals_4_subset_implies_m_range_l129_129113

-- Define the sets and conditions
def set_A := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: When m = 4, find the union of A and B
theorem union_when_m_equals_4 : ∀ x, x ∈ set_A ∪ set_B 4 ↔ -2 ≤ x ∧ x ≤ 7 :=
by sorry

-- Problem 2: If B ⊆ A, find the range of the real number m
theorem subset_implies_m_range (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≤ 3 :=
by sorry

end union_when_m_equals_4_subset_implies_m_range_l129_129113


namespace smallest_number_divisible_by_conditions_l129_129341

theorem smallest_number_divisible_by_conditions (N : ℕ) (X : ℕ) (H1 : (N - 12) % 8 = 0) (H2 : (N - 12) % 12 = 0)
(H3 : (N - 12) % X = 0) (H4 : (N - 12) % 24 = 0) (H5 : (N - 12) / 24 = 276) : N = 6636 :=
by
  sorry

end smallest_number_divisible_by_conditions_l129_129341


namespace find_m_no_solution_l129_129634

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end find_m_no_solution_l129_129634


namespace candy_distribution_problem_l129_129392

theorem candy_distribution_problem (n : ℕ) :
  (n - 1) * (n - 2) / 2 - 3 * (n/2 - 1) / 6 = n + 1 → n = 18 :=
sorry

end candy_distribution_problem_l129_129392


namespace linear_function_passes_through_point_l129_129237

theorem linear_function_passes_through_point :
  ∀ x y : ℝ, y = -2 * x - 6 → (x = -4 → y = 2) :=
by
  sorry

end linear_function_passes_through_point_l129_129237


namespace percentage_of_money_spent_is_80_l129_129945

-- Define the cost of items
def cheeseburger_cost : ℕ := 3
def milkshake_cost : ℕ := 5
def cheese_fries_cost : ℕ := 8

-- Define the amount of money Jim and his cousin brought
def jim_money : ℕ := 20
def cousin_money : ℕ := 10

-- Define the total cost of the meal
def total_cost : ℕ :=
  2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money they brought
def combined_money : ℕ := jim_money + cousin_money

-- Define the percentage of combined money spent
def percentage_spent : ℕ :=
  (total_cost * 100) / combined_money

theorem percentage_of_money_spent_is_80 :
  percentage_spent = 80 :=
by
  -- proof goes here
  sorry

end percentage_of_money_spent_is_80_l129_129945


namespace geometric_sequence_problem_l129_129284

noncomputable def geometric_sequence_solution (a_1 a_2 a_3 a_4 a_5 q : ℝ) : Prop :=
  (a_5 - a_1 = 15) ∧
  (a_4 - a_2 = 6) ∧
  (a_3 = 4 ∧ q = 2 ∨ a_3 = -4 ∧ q = 1/2)

theorem geometric_sequence_problem :
  ∃ a_1 a_2 a_3 a_4 a_5 q : ℝ, geometric_sequence_solution a_1 a_2 a_3 a_4 a_5 q :=
by
  sorry

end geometric_sequence_problem_l129_129284


namespace line_through_two_points_l129_129081

theorem line_through_two_points (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4)) :
  ∃ (m b : ℝ), (∀ x y : ℝ, (y = m * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ m = -7 ∧ b = 11 := by
  sorry

end line_through_two_points_l129_129081


namespace triangular_25_eq_325_l129_129033

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l129_129033


namespace sum_of_translated_parabolas_l129_129853

noncomputable def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := - (a * x^2 + b * x + c)

noncomputable def translated_right (a b c : ℝ) (x : ℝ) : ℝ := parabola_equation a b c (x - 3)

noncomputable def translated_left (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 3)

theorem sum_of_translated_parabolas (a b c x : ℝ) : 
  (translated_right a b c x) + (translated_left a b c x) = -12 * a * x - 6 * b :=
sorry

end sum_of_translated_parabolas_l129_129853


namespace current_price_of_soda_l129_129482

theorem current_price_of_soda (C S : ℝ) (h1 : 1.25 * C = 15) (h2 : C + S = 16) : 1.5 * S = 6 :=
by
  sorry

end current_price_of_soda_l129_129482


namespace calculation_result_l129_129173

theorem calculation_result :
  (-1) * (-4) + 2^2 / (7 - 5) = 6 :=
by
  sorry

end calculation_result_l129_129173


namespace polynomial_relation_l129_129704

theorem polynomial_relation (x y : ℕ) :
  (x = 1 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 4) ∨ 
  (x = 3 ∧ y = 9) ∨ 
  (x = 4 ∧ y = 16) ∨ 
  (x = 5 ∧ y = 25) → 
  y = x^2 := 
by
  sorry

end polynomial_relation_l129_129704


namespace simplify_fraction_l129_129584

theorem simplify_fraction :
  ( (3 * 5 * 7 : ℚ) / (9 * 11 * 13) ) * ( (7 * 9 * 11 * 15) / (3 * 5 * 14) ) = 15 / 26 :=
by
  sorry

end simplify_fraction_l129_129584


namespace good_carrots_l129_129370

theorem good_carrots (haley_picked : ℕ) (mom_picked : ℕ) (bad_carrots : ℕ) :
  haley_picked = 39 → mom_picked = 38 → bad_carrots = 13 →
  (haley_picked + mom_picked - bad_carrots) = 64 :=
by
  sorry  -- Proof is omitted.

end good_carrots_l129_129370


namespace ratio_of_points_l129_129293

theorem ratio_of_points (B J S : ℕ) 
  (h1 : B = J + 20) 
  (h2 : B + J + S = 160) 
  (h3 : B = 45) : 
  B / S = 1 / 2 :=
  sorry

end ratio_of_points_l129_129293


namespace new_person_weight_l129_129667

theorem new_person_weight (W : ℝ) (N : ℝ) (old_weight : ℝ) (average_increase : ℝ) (num_people : ℕ)
  (h1 : num_people = 8)
  (h2 : old_weight = 45)
  (h3 : average_increase = 6)
  (h4 : (W - old_weight + N) / num_people = W / num_people + average_increase) :
  N = 93 :=
by
  sorry

end new_person_weight_l129_129667


namespace range_of_a_l129_129569

theorem range_of_a (a : ℝ) :
  (1 ∉ {x : ℝ | x^2 - 2 * x + a > 0}) → a ≤ 1 :=
by
  sorry

end range_of_a_l129_129569


namespace find_m_l129_129349

-- Definition of the function as a direct proportion function with respect to x
def isDirectProportion (m : ℝ) : Prop :=
  m^2 - 8 = 1

-- Definition of the graph passing through the second and fourth quadrants
def passesThroughQuadrants (m : ℝ) : Prop :=
  m - 2 < 0

-- The theorem combining the conditions and proving the correct value of m
theorem find_m (m : ℝ) 
  (h1 : isDirectProportion m)
  (h2 : passesThroughQuadrants m) : 
  m = -3 :=
  sorry

end find_m_l129_129349


namespace coordinates_of_point_P_in_third_quadrant_l129_129728

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ := abs P.1
noncomputable def distance_from_x_axis (P : ℝ × ℝ) : ℝ := abs P.2

theorem coordinates_of_point_P_in_third_quadrant : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 < 0 ∧ distance_from_x_axis P = 2 ∧ distance_from_y_axis P = 5 ∧ P = (-5, -2) :=
by
  sorry

end coordinates_of_point_P_in_third_quadrant_l129_129728


namespace inhabitable_fraction_l129_129844

theorem inhabitable_fraction 
  (total_land_fraction : ℚ)
  (inhabitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1 / 3)
  (h2 : inhabitable_land_fraction = 3 / 4):
  total_land_fraction * inhabitable_land_fraction = 1 / 4 := 
by
  sorry

end inhabitable_fraction_l129_129844


namespace abs_inequality_solution_l129_129146

theorem abs_inequality_solution (x : ℝ) : |2 * x + 1| - 2 * |x - 1| > 0 ↔ x > 1 / 4 :=
sorry

end abs_inequality_solution_l129_129146


namespace curve_of_constant_width_l129_129566

structure Curve :=
  (is_convex : Prop)

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def rotate_180 (K : Curve) (O : Point) : Curve := sorry

def sum_curves (K1 K2 : Curve) : Curve := sorry

def is_circle_with_radius (K : Curve) (r : ℝ) : Prop := sorry

def constant_width (K : Curve) (w : ℝ) : Prop := sorry

theorem curve_of_constant_width {K : Curve} {O : Point} {h : ℝ} :
  K.is_convex →
  (K' : Curve) → K' = rotate_180 K O →
  is_circle_with_radius (sum_curves K K') h →
  constant_width K h :=
by 
  sorry

end curve_of_constant_width_l129_129566


namespace intersection_complement_l129_129624

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_complement :
  A ∩ ({x | x < -1 ∨ x > 3} : Set ℝ) = {x | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_complement_l129_129624


namespace Iris_total_spent_l129_129940

theorem Iris_total_spent :
  let jackets := 3
  let cost_per_jacket := 10
  let shorts := 2
  let cost_per_short := 6
  let pants := 4
  let cost_per_pant := 12
  jackets * cost_per_jacket + shorts * cost_per_short + pants * cost_per_pant = 90 := by
  sorry

end Iris_total_spent_l129_129940


namespace trig_identity_cos_add_l129_129175

open Real

theorem trig_identity_cos_add (x : ℝ) (h1 : sin (π / 3 - x) = 3 / 5) (h2 : π / 2 < x ∧ x < π) :
  cos (x + π / 6) = 3 / 5 :=
by
  sorry

end trig_identity_cos_add_l129_129175


namespace athlete_D_is_selected_l129_129096

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected_l129_129096


namespace repeating_decimal_division_l129_129887

def repeating_decimal_142857 : ℚ := 1 / 7
def repeating_decimal_2_857143 : ℚ := 20 / 7

theorem repeating_decimal_division :
  (repeating_decimal_142857 / repeating_decimal_2_857143) = 1 / 20 :=
by
  sorry

end repeating_decimal_division_l129_129887


namespace Martha_should_buy_84oz_of_apples_l129_129547

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end Martha_should_buy_84oz_of_apples_l129_129547


namespace simplified_value_of_expression_l129_129798

theorem simplified_value_of_expression :
  (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := 
by
  sorry

end simplified_value_of_expression_l129_129798


namespace geometric_sequence_common_ratio_l129_129962

theorem geometric_sequence_common_ratio (a1 a2 a3 : ℤ) (r : ℤ)
  (h1 : a1 = 9) (h2 : a2 = -18) (h3 : a3 = 36) (h4 : a2 / a1 = r) (h5 : a3 = a2 * r) :
  r = -2 := 
sorry

end geometric_sequence_common_ratio_l129_129962


namespace savings_fraction_l129_129172

variable (P : ℝ) -- worker's monthly take-home pay, assumed to be a real number
variable (f : ℝ) -- fraction of the take-home pay that she saves each month, assumed to be a real number

-- Condition: 12 times the fraction saved monthly should equal 8 times the amount not saved monthly.
axiom condition : 12 * f * P = 8 * (1 - f) * P

-- Prove: the fraction saved each month is 2/5
theorem savings_fraction : f = 2 / 5 := 
by
  sorry

end savings_fraction_l129_129172


namespace calculation_is_one_l129_129176

noncomputable def calc_expression : ℝ :=
  (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (Real.pi / 3) - Real.sqrt 12

theorem calculation_is_one : calc_expression = 1 :=
by
  -- Each of the steps involved in calculating should match the problem's steps
  -- 1. (1/2)⁻¹ = 2
  -- 2. (2021 + π)^0 = 1
  -- 3. 4 * sin(π / 3) = 2√3 with sin(60°) = √3/2
  -- 4. sqrt(12) = 2√3
  -- Hence 2 - 1 + 2√3 - 2√3 = 1
  sorry

end calculation_is_one_l129_129176


namespace one_third_way_l129_129563

theorem one_third_way (x₁ x₂ : ℚ) (w₁ w₂ : ℕ) (h₁ : x₁ = 1/4) (h₂ : x₂ = 3/4) (h₃ : w₁ = 2) (h₄ : w₂ = 1) : 
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5 / 12 :=
by 
  rw [h₁, h₂, h₃, h₄]
  -- Simplification of the weighted average to get 5/12
  sorry

end one_third_way_l129_129563


namespace find_total_people_find_children_l129_129894

variables (x m : ℕ)

-- Given conditions translated into Lean

def group_b_more_people (x : ℕ) := x + 4
def sum_is_18_times_difference (x : ℕ) := (x + (x + 4)) = 18 * ((x + 4) - x)
def children_b_less_than_three_times (m : ℕ) := (3 * m) - 2
def adult_ticket_price := 100
def children_ticket_price := (100 * 60) / 100
def same_amount_spent (x m : ℕ) := 100 * (x - m) + (100 * 60 / 100) * m = 100 * ((group_b_more_people x) - (children_b_less_than_three_times m)) + (100 * 60 / 100) * (children_b_less_than_three_times m)

-- Proving the two propositions (question == answer given conditions)

theorem find_total_people (x : ℕ) (hx : sum_is_18_times_difference x) : x = 34 ∧ (group_b_more_people x) = 38 :=
by {
  sorry -- proof for x = 34 and group_b_people = 38 given that sum_is_18_times_difference x
}

theorem find_children (m : ℕ) (x : ℕ) (hx : sum_is_18_times_difference x) (hm : same_amount_spent x m) : m = 6 ∧ (children_b_less_than_three_times m) = 16 :=
by {
  sorry -- proof for m = 6 and children_b_people = 16 given sum_is_18_times_difference x and same_amount_spent x m
}

end find_total_people_find_children_l129_129894


namespace china_math_olympiad_34_2023_l129_129388

-- Defining the problem conditions and verifying the minimum and maximum values of S.
theorem china_math_olympiad_34_2023 {a b c d e : ℝ}
  (h1 : a ≥ -1)
  (h2 : b ≥ -1)
  (h3 : c ≥ -1)
  (h4 : d ≥ -1)
  (h5 : e ≥ -1)
  (h6 : a + b + c + d + e = 5) :
  (-512 ≤ (a + b) * (b + c) * (c + d) * (d + e) * (e + a)) ∧
  ((a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288) :=
sorry

end china_math_olympiad_34_2023_l129_129388


namespace rabbit_parent_genotype_l129_129211

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l129_129211


namespace cycling_speed_l129_129654

-- Definitions based on given conditions.
def ratio_L_B : ℕ := 1
def ratio_B_L : ℕ := 2
def area_of_park : ℕ := 20000
def time_in_minutes : ℕ := 6

-- The question translated to Lean 4 statement.
theorem cycling_speed (L B : ℕ) (h1 : ratio_L_B * B = ratio_B_L * L)
  (h2 : L * B = area_of_park)
  (h3 : B = 2 * L) :
  (2 * L + 2 * B) / (time_in_minutes / 60) = 6000 := by
  sorry

end cycling_speed_l129_129654


namespace num_workers_in_factory_l129_129828

theorem num_workers_in_factory 
  (average_salary_total : ℕ → ℕ → ℕ)
  (old_supervisor_salary : ℕ)
  (average_salary_9_new : ℕ)
  (new_supervisor_salary : ℕ) :
  ∃ (W : ℕ), 
  average_salary_total (W + 1) 430 = W * 430 + 870 ∧ 
  average_salary_9_new = 9 * 390 ∧ 
  W + 1 = (9 * 390 - 510 + 870) / 430 := 
by {
  sorry
}

end num_workers_in_factory_l129_129828


namespace general_formula_sequence_l129_129519

variable {a : ℕ → ℝ}

-- Definitions and assumptions
def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n - 2 * a (n + 1) + a (n + 2) = 0

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 4

-- The proof problem
theorem general_formula_sequence (a : ℕ → ℝ)
  (h1 : recurrence_relation a)
  (h2 : initial_conditions a) :
  ∀ n : ℕ, a n = 2 * n :=

sorry

end general_formula_sequence_l129_129519


namespace find_alpha_l129_129695

open Real

def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem find_alpha (α : ℝ) (h1 : alpha_is_acute α) (h2 : sin (α - 10 * (pi / 180)) = sqrt 3 / 2) : α = 70 * (pi / 180) :=
sorry

end find_alpha_l129_129695


namespace number_of_students_with_no_pets_l129_129847

-- Define the number of students in the class
def total_students : ℕ := 25

-- Define the number of students with cats
def students_with_cats : ℕ := (3 * total_students) / 5

-- Define the number of students with dogs
def students_with_dogs : ℕ := (20 * total_students) / 100

-- Define the number of students with elephants
def students_with_elephants : ℕ := 3

-- Calculate the number of students with no pets
def students_with_no_pets : ℕ := total_students - (students_with_cats + students_with_dogs + students_with_elephants)

-- Statement to be proved
theorem number_of_students_with_no_pets : students_with_no_pets = 2 :=
sorry

end number_of_students_with_no_pets_l129_129847


namespace speed_of_train_l129_129775

-- Conditions
def train_length : ℝ := 180
def total_length : ℝ := 195
def time_cross_bridge : ℝ := 30

-- Conversion factor for units (1 m/s = 3.6 km/hr)
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem speed_of_train : 
  (total_length - train_length) / time_cross_bridge * conversion_factor = 23.4 :=
sorry

end speed_of_train_l129_129775


namespace two_roots_iff_a_gt_neg1_l129_129102

theorem two_roots_iff_a_gt_neg1 (a : ℝ) :
  (∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*|x1 + 1| = a) ∧ (x2^2 + 2*x2 + 2*|x2 + 1| = a)) ↔ a > -1 :=
by sorry

end two_roots_iff_a_gt_neg1_l129_129102


namespace classroom_student_count_l129_129672

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l129_129672


namespace cathy_can_win_l129_129036

theorem cathy_can_win (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (f : ℕ → ℕ) (hf : ∀ i, f i < n + 1), (∀ i j, (i < j) → (f i < f j) → (f j = f i + 1)) → n ≤ 2^(k-1)) :=
sorry

end cathy_can_win_l129_129036


namespace exam_rule_l129_129846

variable (P R Q : Prop)

theorem exam_rule (hp : P ∧ R → Q) : ¬ Q → ¬ P ∨ ¬ R :=
by
  sorry

end exam_rule_l129_129846


namespace compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l129_129023

-- Problem 1
theorem compare_sqrt_difference : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := 
  sorry

-- Problem 2
theorem minimize_materials_plan (x y : ℝ) (h : x > y) : 
  4 * x + 6 * y > 3 * x + 7 * y := 
  sorry

-- Problem 3
theorem compare_a_inv (a : ℝ) (h : a > 0) : 
  (0 < a ∧ a < 1) → a < 1 / a ∧ (a = 1 → a = 1 / a) ∧ (a > 1 → a > 1 / a) :=
  sorry

end compare_sqrt_difference_minimize_materials_plan_compare_a_inv_l129_129023


namespace factorization_correct_l129_129082

-- Define the input expression
def expr (x y : ℝ) : ℝ := 2 * x^3 - 18 * x * y^2

-- Define the factorized form
def factorized_expr (x y : ℝ) : ℝ := 2 * x * (x + 3*y) * (x - 3*y)

-- Prove that the original expression is equal to the factorized form
theorem factorization_correct (x y : ℝ) : expr x y = factorized_expr x y := 
by sorry

end factorization_correct_l129_129082


namespace bob_tiller_swath_width_l129_129492

theorem bob_tiller_swath_width
  (plot_width plot_length : ℕ)
  (tilling_rate_seconds_per_foot : ℕ)
  (total_tilling_minutes : ℕ)
  (total_area : ℕ)
  (tilled_length : ℕ)
  (swath_width : ℕ) :
  plot_width = 110 →
  plot_length = 120 →
  tilling_rate_seconds_per_foot = 2 →
  total_tilling_minutes = 220 →
  total_area = plot_width * plot_length →
  tilled_length = (total_tilling_minutes * 60) / tilling_rate_seconds_per_foot →
  swath_width = total_area / tilled_length →
  swath_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end bob_tiller_swath_width_l129_129492


namespace cost_of_each_math_book_l129_129882

-- Define the given conditions
def total_books : ℕ := 90
def math_books : ℕ := 53
def history_books : ℕ := total_books - math_books
def history_book_cost : ℕ := 5
def total_price : ℕ := 397

-- The required theorem
theorem cost_of_each_math_book (M : ℕ) (H : 53 * M + history_books * history_book_cost = total_price) : M = 4 :=
by
  sorry

end cost_of_each_math_book_l129_129882


namespace taco_truck_revenue_l129_129652

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l129_129652


namespace find_polynomial_R_l129_129182

-- Define the polynomials S(x), Q(x), and the remainder R(x)

noncomputable def S (x : ℝ) := 7 * x ^ 31 + 3 * x ^ 13 + 10 * x ^ 11 - 5 * x ^ 9 - 10 * x ^ 7 + 5 * x ^ 5 - 2
noncomputable def Q (x : ℝ) := x ^ 4 + x ^ 3 + x ^ 2 + x + 1
noncomputable def R (x : ℝ) := 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3

-- Statement of the proof
theorem find_polynomial_R :
  ∃ (P : ℝ → ℝ), ∀ x : ℝ, S x = P x * Q x + R x := sorry

end find_polynomial_R_l129_129182


namespace smallest_number_of_rectangles_needed_l129_129973

-- Define the dimensions of the rectangle
def rectangle_area (length width : ℕ) : ℕ := length * width

-- Define the side length of the square
def square_side_length : ℕ := 12

-- Define the number of rectangles needed to cover the square horizontally
def num_rectangles_to_cover_square : ℕ := (square_side_length / 3) * (square_side_length / 4)

-- The theorem must state the total number of rectangles required
theorem smallest_number_of_rectangles_needed : num_rectangles_to_cover_square = 16 := 
by
  -- Proof details are skipped using sorry
  sorry

end smallest_number_of_rectangles_needed_l129_129973


namespace proof_problem_l129_129501

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l129_129501


namespace remainder_349_div_13_l129_129896

theorem remainder_349_div_13 : 349 % 13 = 11 := 
by 
  sorry

end remainder_349_div_13_l129_129896


namespace floor_length_l129_129006

variable (b l : ℝ)

theorem floor_length :
  (l = 3 * b) →
  (3 * b ^ 2 = 128) →
  l = 19.59 :=
by
  intros h1 h2
  sorry

end floor_length_l129_129006


namespace necessary_and_sufficient_condition_l129_129134

theorem necessary_and_sufficient_condition (a b : ℝ) (h : a * b ≠ 0) : 
  a - b = 1 ↔ a^3 - b^3 - a * b - a^2 - b^2 = 0 := by
  sorry

end necessary_and_sufficient_condition_l129_129134


namespace smallest_positive_multiple_l129_129460

theorem smallest_positive_multiple (a : ℕ) :
  (37 * a) % 97 = 7 → 37 * a = 481 :=
sorry

end smallest_positive_multiple_l129_129460


namespace ArletteAge_l129_129357

/-- Define the ages of Omi, Kimiko, and Arlette -/
def OmiAge (K : ℕ) : ℕ := 2 * K
def KimikoAge : ℕ := 28   /- K = 28 -/
def averageAge (O K A : ℕ) : Prop := (O + K + A) / 3 = 35

/-- Prove Arlette's age given the conditions -/
theorem ArletteAge (A : ℕ) (h1 : A + OmiAge KimikoAge + KimikoAge = 3 * 35) : A = 21 := by
  /- Hypothesis h1 unpacks the third condition into equality involving O, K, and A -/
  sorry

end ArletteAge_l129_129357


namespace solve_inequality_l129_129260

-- We will define the conditions and corresponding solution sets
def solution_set (a x : ℝ) : Prop :=
  (a < -1 ∧ (x > -a ∨ x < 1)) ∨
  (a = -1 ∧ x ≠ 1) ∨
  (a > -1 ∧ (x < -a ∨ x > 1))

theorem solve_inequality (a x : ℝ) :
  (x - 1) * (x + a) > 0 ↔ solution_set a x :=
by
  sorry

end solve_inequality_l129_129260


namespace conic_section_pair_of_lines_l129_129055

theorem conic_section_pair_of_lines : 
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = 0 → (2 * x - 3 * y = 0 ∨ 2 * x + 3 * y = 0)) :=
by
  sorry

end conic_section_pair_of_lines_l129_129055


namespace quadratic_equation_statements_l129_129700

theorem quadratic_equation_statements (a b c : ℝ) (h₀ : a ≠ 0) :
  (if -4 * a * c > 0 then (b^2 - 4 * a * c) > 0 else false) ∧
  ¬((b^2 - 4 * a * c > 0) → (b^2 - 4 * c * a > 0)) ∧
  ¬((c^2 * a + c * b + c = 0) → (a * c + b + 1 = 0)) ∧
  ¬(∀ (x₀ : ℝ), (a * x₀^2 + b * x₀ + c = 0) → (b^2 - 4 * a * c = (2 * a * x₀ - b)^2)) :=
by
    sorry

end quadratic_equation_statements_l129_129700


namespace parabola_no_intersect_l129_129922

theorem parabola_no_intersect (m : ℝ) : 
  (¬ ∃ x : ℝ, -x^2 - 6*x + m = 0 ) ↔ m < -9 :=
by
  sorry

end parabola_no_intersect_l129_129922


namespace number_of_valid_pairs_l129_129312

theorem number_of_valid_pairs :
  (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2044 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ↔
  ((∃ (x y : ℕ), 2^2100 < 5^900 ∧ 5^900 < 2^2101)) → 
  (∃ (count : ℕ), count = 900) :=
by sorry

end number_of_valid_pairs_l129_129312


namespace yoongi_number_division_l129_129017

theorem yoongi_number_division (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end yoongi_number_division_l129_129017


namespace spring_expenses_l129_129015

noncomputable def expense_by_end_of_february : ℝ := 0.6
noncomputable def expense_by_end_of_may : ℝ := 1.8
noncomputable def spending_during_spring_months := expense_by_end_of_may - expense_by_end_of_february

-- Lean statement for the proof problem
theorem spring_expenses : spending_during_spring_months = 1.2 := by
  sorry

end spring_expenses_l129_129015


namespace smallest_n_factorial_l129_129205

theorem smallest_n_factorial (a b c m n : ℕ) (h1 : a + b + c = 2020)
(h2 : c > a + 100)
(h3 : m * 10^n = a! * b! * c!)
(h4 : ¬ (10 ∣ m)) : 
  n = 499 :=
sorry

end smallest_n_factorial_l129_129205


namespace perpendicular_graphs_solve_a_l129_129296

theorem perpendicular_graphs_solve_a (a : ℝ) : 
  (∀ x y : ℝ, 2 * y + x + 3 = 0 → 3 * y + a * x + 2 = 0 → 
  ∀ m1 m2 : ℝ, (y = m1 * x + b1 → m1 = -1 / 2) →
  (y = m2 * x + b2 → m2 = -a / 3) →
  m1 * m2 = -1) → a = -6 :=
by
  sorry

end perpendicular_graphs_solve_a_l129_129296


namespace calculate_expression_l129_129120

theorem calculate_expression (a b c d : ℤ) (h1 : 3^0 = 1) (h2 : (-1 / 2 : ℚ)^(-2 : ℤ) = 4) : 
  (202 : ℤ) * 3^0 + (-1 / 2 : ℚ)^(-2 : ℤ) = 206 :=
by
  sorry

end calculate_expression_l129_129120


namespace cos_fourth_minus_sin_fourth_l129_129651

theorem cos_fourth_minus_sin_fourth (α : ℝ) (h : Real.sin α = (Real.sqrt 5) / 5) :
  Real.cos α ^ 4 - Real.sin α ^ 4 = 3 / 5 := 
sorry

end cos_fourth_minus_sin_fourth_l129_129651


namespace remainder_of_prime_when_divided_by_240_l129_129653

theorem remainder_of_prime_when_divided_by_240 (n : ℕ) (hn : n > 0) (hp : Nat.Prime (2^n + 1)) : (2^n + 1) % 240 = 17 := 
sorry

end remainder_of_prime_when_divided_by_240_l129_129653


namespace range_of_a_l129_129447

open Complex Real

theorem range_of_a (a : ℝ) (h : abs (1 + a * Complex.I) ≤ 2) : a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_a_l129_129447


namespace impossible_to_achieve_desired_piles_l129_129793

def initial_piles : List ℕ := [51, 49, 5]

def desired_piles : List ℕ := [52, 48, 5]

def combine_piles (x y : ℕ) : ℕ := x + y

def divide_pile (x : ℕ) (h : x % 2 = 0) : List ℕ := [x / 2, x / 2]

theorem impossible_to_achieve_desired_piles :
  ∀ (piles : List ℕ), 
    (piles = initial_piles) →
    (∀ (p : List ℕ), 
      (p = desired_piles) → 
      False) :=
sorry

end impossible_to_achieve_desired_piles_l129_129793


namespace minimize_f_l129_129351

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l129_129351


namespace joseph_drives_more_l129_129533

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end joseph_drives_more_l129_129533


namespace find_number_l129_129395

theorem find_number : 
  (15^2 * 9^2) / x = 51.193820224719104 → x = 356 :=
by
  sorry

end find_number_l129_129395


namespace number_of_true_propositions_l129_129863

theorem number_of_true_propositions :
  let P1 := false -- Swinging on a swing can be regarded as a translation motion.
  let P2 := false -- Two lines intersected by a third line have equal corresponding angles.
  let P3 := true  -- There is one and only one line passing through a point parallel to a given line.
  let P4 := false -- Angles that are not vertical angles are not equal.
  (if P1 then 1 else 0) + (if P2 then 1 else 0) + (if P3 then 1 else 0) + (if P4 then 1 else 0) = 1 :=
by
  sorry

end number_of_true_propositions_l129_129863


namespace problem_l129_129821

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l129_129821


namespace cos_135_eq_neg_inv_sqrt2_l129_129799

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l129_129799


namespace more_spent_on_keychains_bracelets_than_tshirts_l129_129005

-- Define the conditions as variables
variable (spent_keychains_bracelets spent_total_spent : ℝ)
variable (spent_keychains_bracelets_eq : spent_keychains_bracelets = 347.00)
variable (spent_total_spent_eq : spent_total_spent = 548.00)

-- Using these conditions, define the problem to prove the desired result
theorem more_spent_on_keychains_bracelets_than_tshirts :
  spent_keychains_bracelets - (spent_total_spent - spent_keychains_bracelets) = 146.00 :=
by
  rw [spent_keychains_bracelets_eq, spent_total_spent_eq]
  sorry

end more_spent_on_keychains_bracelets_than_tshirts_l129_129005


namespace sum_of_common_ratios_l129_129481

variable (m x y : ℝ)
variable (h₁ : x ≠ y)
variable (h₂ : a2 = m * x)
variable (h₃ : a3 = m * x^2)
variable (h₄ : b2 = m * y)
variable (h₅ : b3 = m * y^2)
variable (h₆ : a3 - b3 = 3 * (a2 - b2))

theorem sum_of_common_ratios : x + y = 3 :=
by
  sorry

end sum_of_common_ratios_l129_129481


namespace white_roses_count_l129_129977

def total_flowers : ℕ := 6284
def red_roses : ℕ := 1491
def yellow_carnations : ℕ := 3025
def white_roses : ℕ := total_flowers - (red_roses + yellow_carnations)

theorem white_roses_count :
  white_roses = 1768 := by
  sorry

end white_roses_count_l129_129977


namespace prime_square_minus_five_not_div_by_eight_l129_129983

theorem prime_square_minus_five_not_div_by_eight (p : ℕ) (prime_p : Prime p) (p_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) :=
sorry

end prime_square_minus_five_not_div_by_eight_l129_129983


namespace dodecahedron_equilateral_triangles_l129_129373

-- Definitions reflecting the conditions
def vertices_of_dodecahedron := 20
def faces_of_dodecahedron := 12
def vertices_per_face := 5
def equilateral_triangles_per_face := 5

theorem dodecahedron_equilateral_triangles :
  (faces_of_dodecahedron * equilateral_triangles_per_face) = 60 := by
  sorry

end dodecahedron_equilateral_triangles_l129_129373


namespace boxcar_capacity_ratio_l129_129665

theorem boxcar_capacity_ratio :
  ∀ (total_capacity : ℕ)
    (num_red num_blue num_black : ℕ)
    (black_capacity blue_capacity : ℕ)
    (red_capacity : ℕ),
    num_red = 3 →
    num_blue = 4 →
    num_black = 7 →
    black_capacity = 4000 →
    blue_capacity = 2 * black_capacity →
    total_capacity = 132000 →
    total_capacity = num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity →
    (red_capacity / blue_capacity = 3) :=
by
  intros total_capacity num_red num_blue num_black black_capacity blue_capacity red_capacity
         h_num_red h_num_blue h_num_black h_black_capacity h_blue_capacity h_total_capacity h_combined_capacity
  sorry

end boxcar_capacity_ratio_l129_129665


namespace range_of_a_l129_129001

-- Define sets P and M
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def M (a : ℝ) : Set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (1 + a)}

-- Prove the range of a
theorem range_of_a (a : ℝ) : (P ∩ (M a) = P) ↔ (a ≥ 1) :=
by 
  sorry

end range_of_a_l129_129001


namespace quadratic_solution_1_l129_129961

theorem quadratic_solution_1 :
  (∃ x, x^2 - 4 * x + 3 = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

end quadratic_solution_1_l129_129961


namespace ratio_of_hexagon_areas_l129_129637

open Real

-- Define the given conditions about the hexagon and the midpoints
structure Hexagon :=
  (s : ℝ)
  (regular : True)
  (midpoints : True)

theorem ratio_of_hexagon_areas (h : Hexagon) : 
  let s := 2
  ∃ (area_ratio : ℝ), area_ratio = 4 / 7 :=
by
  sorry

end ratio_of_hexagon_areas_l129_129637


namespace petya_vacation_days_l129_129617

-- Defining the conditions
def total_days : ℕ := 90

def swims (d : ℕ) : Prop := d % 2 = 0
def shops (d : ℕ) : Prop := d % 3 = 0
def solves_math (d : ℕ) : Prop := d % 5 = 0

def does_all (d : ℕ) : Prop := swims d ∧ shops d ∧ solves_math d

def does_any_task (d : ℕ) : Prop := swims d ∨ shops d ∨ solves_math d

-- "Pleasant" days definition: swims, not shops, not solves math
def is_pleasant_day (d : ℕ) : Prop := swims d ∧ ¬shops d ∧ ¬solves_math d
-- "Boring" days definition: does nothing
def is_boring_day (d : ℕ) : Prop := ¬does_any_task d

-- Theorem stating the number of pleasant and boring days
theorem petya_vacation_days :
  (∃ pleasant_days : Finset ℕ, pleasant_days.card = 24 ∧ ∀ d ∈ pleasant_days, is_pleasant_day d)
  ∧ (∃ boring_days : Finset ℕ, boring_days.card = 24 ∧ ∀ d ∈ boring_days, is_boring_day d) :=
by
  sorry

end petya_vacation_days_l129_129617


namespace sufficient_but_not_necessary_l129_129088

variable (x : ℚ)

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

theorem sufficient_but_not_necessary :
  (is_integer x → is_integer (2 * x + 1)) ∧
  (¬ (is_integer (2 * x + 1) → is_integer x)) :=
by
  sorry

end sufficient_but_not_necessary_l129_129088


namespace exponent_product_l129_129960

variables {a m n : ℝ}

theorem exponent_product (h1 : a^m = 2) (h2 : a^n = 8) : a^m * a^n = 16 :=
by
  sorry

end exponent_product_l129_129960


namespace prudence_nap_is_4_hours_l129_129991

def prudence_nap_length (total_sleep : ℕ) (weekdays_sleep : ℕ) (weekend_sleep : ℕ) (weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  (total_sleep - (weekdays_sleep + weekend_sleep) * total_weeks) / (2 * total_weeks)

theorem prudence_nap_is_4_hours
  (total_sleep weekdays_sleep weekend_sleep total_weeks : ℕ) :
  total_sleep = 200 ∧ weekdays_sleep = 5 * 6 ∧ weekend_sleep = 2 * 9 ∧ total_weeks = 4 →
  prudence_nap_length total_sleep weekdays_sleep weekend_sleep total_weeks total_weeks = 4 :=
by
  intros
  sorry

end prudence_nap_is_4_hours_l129_129991


namespace distance_between_A_and_B_is_40_l129_129905

theorem distance_between_A_and_B_is_40
  (v1 v2 : ℝ)
  (h1 : ∃ t: ℝ, t = (40 / 2) / v1 ∧ t = (40 - 24) / v2)
  (h2 : ∃ t: ℝ, t = (40 - 15) / v1 ∧ t = 40 / (2 * v2)) :
  40 = 40 := by
  sorry

end distance_between_A_and_B_is_40_l129_129905


namespace complex_pure_imaginary_l129_129578

theorem complex_pure_imaginary (a : ℂ) : (∃ (b : ℂ), (2 - I) * (a + 2 * I) = b * I) → a = -1 :=
by
  sorry

end complex_pure_imaginary_l129_129578


namespace point_in_third_quadrant_l129_129163

theorem point_in_third_quadrant (m n : ℝ) (h1 : m > 0) (h2 : n > 0) : (-m < 0) ∧ (-n < 0) :=
by
  sorry

end point_in_third_quadrant_l129_129163


namespace minimize_expression_l129_129418

theorem minimize_expression : 
  let a := -1
  let b := -0.5
  (a + b) ≤ (a - b) ∧ (a + b) ≤ (a * b) ∧ (a + b) ≤ (a / b) := by
  let a := -1
  let b := -0.5
  sorry

end minimize_expression_l129_129418


namespace eval_neg64_pow_two_thirds_l129_129545

theorem eval_neg64_pow_two_thirds : (-64 : Real)^(2/3) = 16 := 
by 
  sorry

end eval_neg64_pow_two_thirds_l129_129545


namespace correct_statements_l129_129007

-- Define the function and the given conditions
def f : ℝ → ℝ := sorry

lemma not_constant (h: ∃ x y: ℝ, x ≠ y ∧ f x ≠ f y) : true := sorry
lemma periodic (x : ℝ) : f (x - 1) = f (x + 1) := sorry
lemma symmetric (x : ℝ) : f (2 - x) = f x := sorry

-- The statements we want to prove
theorem correct_statements : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (1 - x) = f (1 + x)) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x)
:= by
  sorry

end correct_statements_l129_129007


namespace triangle_larger_segment_cutoff_l129_129200

open Real

theorem triangle_larger_segment_cutoff (a b c h s₁ s₂ : ℝ) (habc : a = 35) (hbc : b = 85) (hca : c = 90)
  (hh : h = 90)
  (eq₁ : a^2 = s₁^2 + h^2)
  (eq₂ : b^2 = s₂^2 + h^2)
  (h_sum : s₁ + s₂ = c) :
  max s₁ s₂ = 78.33 :=
by
  sorry

end triangle_larger_segment_cutoff_l129_129200


namespace arithmetic_seq_term_ratio_l129_129112

-- Assume two arithmetic sequences a and b
def arithmetic_seq_a (n : ℕ) : ℕ := sorry
def arithmetic_seq_b (n : ℕ) : ℕ := sorry

-- Sum of first n terms of the sequences
def sum_a (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_a |>.sum
def sum_b (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_b |>.sum

-- The given condition: Sn / Tn = (7n + 2) / (n + 3)
axiom sum_condition (n : ℕ) : (sum_a n) / (sum_b n) = (7 * n + 2) / (n + 3)

-- The goal: a4 / b4 = 51 / 10
theorem arithmetic_seq_term_ratio : (arithmetic_seq_a 4 : ℚ) / (arithmetic_seq_b 4 : ℚ) = 51 / 10 :=
by
  sorry

end arithmetic_seq_term_ratio_l129_129112


namespace interval_proof_l129_129262

noncomputable def valid_interval (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y

theorem interval_proof : ∀ x : ℝ, valid_interval x ↔ (0 ≤ x ∧ x < 4) :=
by
  sorry

end interval_proof_l129_129262


namespace class_B_more_uniform_than_class_A_l129_129616

-- Definitions based on the given problem
def class_height_variance (class_name : String) : ℝ :=
  if class_name = "A" then 3.24 else if class_name = "B" then 1.63 else 0

-- The theorem statement proving that Class B has more uniform heights (smaller variance)
theorem class_B_more_uniform_than_class_A :
  class_height_variance "B" < class_height_variance "A" :=
by
  sorry

end class_B_more_uniform_than_class_A_l129_129616


namespace problem_l129_129285

-- Condition that defines s and t
def s : ℤ := 4
def t : ℤ := 3

theorem problem (s t : ℤ) (h_s : s = 4) (h_t : t = 3) : s - 2 * t = -2 := by
  sorry

end problem_l129_129285


namespace factorization_of_polynomial_l129_129049

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end factorization_of_polynomial_l129_129049


namespace largest_power_of_2_divides_n_l129_129132

def n : ℤ := 17^4 - 13^4

theorem largest_power_of_2_divides_n : ∃ (k : ℕ), 2^4 = k ∧ 2^k ∣ n ∧ ¬ (2^(k + 1) ∣ n) := by
  sorry

end largest_power_of_2_divides_n_l129_129132


namespace find_original_number_l129_129763

theorem find_original_number (x : ℚ) (h : 1 + 1 / x = 8 / 3) : x = 3 / 5 := by
  sorry

end find_original_number_l129_129763


namespace soccer_games_total_l129_129928

variable (wins losses ties total_games : ℕ)

theorem soccer_games_total
    (h1 : losses = 9)
    (h2 : 4 * wins + 3 * losses + ties = 8 * total_games) :
    total_games = 24 :=
by
  sorry

end soccer_games_total_l129_129928


namespace alpha_beta_value_l129_129930

theorem alpha_beta_value :
  ∃ α β : ℝ, (α^2 - 2 * α - 4 = 0) ∧ (β^2 - 2 * β - 4 = 0) ∧ (α + β = 2) ∧ (α^3 + 8 * β + 6 = 30) :=
by
  sorry

end alpha_beta_value_l129_129930


namespace factorization_correct_l129_129367

theorem factorization_correct : 
  ¬(∃ x : ℝ, -x^2 + 4 * x = -x * (x + 4)) ∧
  ¬(∃ x y: ℝ, x^2 + x * y + x = x * (x + y)) ∧
  (∀ x y: ℝ, x * (x - y) + y * (y - x) = (x - y)^2) ∧
  ¬(∃ x : ℝ, x^2 - 4 * x + 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correct_l129_129367


namespace determinant_roots_cubic_eq_l129_129714

noncomputable def determinant_of_matrix (a b c : ℝ) : ℝ :=
  a * (b * c - 1) - (c - 1) + (1 - b)

theorem determinant_roots_cubic_eq {a b c p q r : ℝ}
  (h1 : a + b + c = p)
  (h2 : a * b + b * c + c * a = q)
  (h3 : a * b * c = r) :
  determinant_of_matrix a b c = r - p + 2 :=
by {
  sorry
}

end determinant_roots_cubic_eq_l129_129714


namespace solve_system_l129_129636

theorem solve_system (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (eq1 : a * y + b * x = c)
  (eq2 : c * x + a * z = b)
  (eq3 : b * z + c * y = a) :
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧ 
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧ 
  z = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end solve_system_l129_129636


namespace distance_between_points_l129_129114

theorem distance_between_points :
  let p1 := (-4, 17)
  let p2 := (12, -1)
  let distance := Real.sqrt ((12 - (-4))^2 + (-1 - 17)^2)
  distance = 2 * Real.sqrt 145 := sorry

end distance_between_points_l129_129114


namespace arithmetic_sequence_a3_l129_129365

theorem arithmetic_sequence_a3 (a1 d : ℤ) (h : a1 + (a1 + d) + (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d) = 20) : 
  a1 + 2 * d = 4 := by
  sorry

end arithmetic_sequence_a3_l129_129365


namespace hexagon_monochromatic_triangle_probability_l129_129822

open Classical

-- Define the total number of edges in the hexagon
def total_edges : ℕ := 15

-- Define the number of triangles from 6 vertices
def total_triangles : ℕ := Nat.choose 6 3

-- Define the probability that a given triangle is not monochromatic
def prob_not_monochromatic_triangle : ℚ := 3 / 4

-- Calculate the probability of having at least one monochromatic triangle
def prob_at_least_one_monochromatic_triangle : ℚ := 
  1 - (prob_not_monochromatic_triangle ^ total_triangles)

theorem hexagon_monochromatic_triangle_probability :
  abs ((prob_at_least_one_monochromatic_triangle : ℝ) - 0.9968) < 0.0001 :=
by
  sorry

end hexagon_monochromatic_triangle_probability_l129_129822


namespace trees_still_left_l129_129288

theorem trees_still_left 
  (initial_trees : ℕ) 
  (trees_died : ℕ) 
  (trees_cut : ℕ) 
  (initial_trees_eq : initial_trees = 86) 
  (trees_died_eq : trees_died = 15) 
  (trees_cut_eq : trees_cut = 23) 
  : initial_trees - (trees_died + trees_cut) = 48 :=
by
  sorry

end trees_still_left_l129_129288


namespace find_perpendicular_line_l129_129449

theorem find_perpendicular_line (x y : ℝ) (h₁ : y = (1/2) * x + 1)
    (h₂ : (x, y) = (2, 0)) : y = -2 * x + 4 :=
sorry

end find_perpendicular_line_l129_129449


namespace complement_M_in_U_l129_129776

def M (x : ℝ) : Prop := 0 < x ∧ x < 2

def complement_M (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

theorem complement_M_in_U (x : ℝ) : ¬ M x ↔ complement_M x :=
by sorry

end complement_M_in_U_l129_129776


namespace auction_theorem_l129_129676

def auctionProblem : Prop :=
  let starting_value := 300
  let harry_bid_round1 := starting_value + 200
  let alice_bid_round1 := harry_bid_round1 * 2
  let bob_bid_round1 := harry_bid_round1 * 3
  let highest_bid_round1 := bob_bid_round1
  let carol_bid_round2 := highest_bid_round1 * 1.5
  let sum_previous_increases := (harry_bid_round1 - starting_value) + 
                                 (alice_bid_round1 - harry_bid_round1) + 
                                 (bob_bid_round1 - harry_bid_round1)
  let dave_bid_round2 := carol_bid_round2 + sum_previous_increases
  let highest_other_bid_round3 := dave_bid_round2
  let harry_final_bid_round3 := 6000
  let difference := harry_final_bid_round3 - highest_other_bid_round3
  difference = 2050

theorem auction_theorem : auctionProblem :=
by
  sorry

end auction_theorem_l129_129676


namespace calc_expression_l129_129159

theorem calc_expression :
  15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 :=
by
  sorry

end calc_expression_l129_129159


namespace minimize_expr_l129_129168

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b * c = 8)

-- Define the target expression and the proof goal
def expr := (3 * a + b) * (a + 3 * c) * (2 * b * c + 4)

-- Prove the main statement
theorem minimize_expr : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c = 8) ∧ expr a b c = 384 :=
sorry

end minimize_expr_l129_129168


namespace sequence_two_cases_l129_129958

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ≥ a (n-1)) ∧  -- nondecreasing
  (∃ n m, n ≠ m ∧ a n ≠ a m) ∧  -- nonconstant
  (∀ n, a n ∣ n^2)  -- a_n | n^2

theorem sequence_two_cases (a : ℕ → ℕ) :
  sequence_property a →
  (∃ n1, ∀ n ≥ n1, a n = n) ∨ (∃ n2, ∀ n ≥ n2, a n = n^2) :=
by {
  sorry
}

end sequence_two_cases_l129_129958


namespace fraction_value_l129_129575

theorem fraction_value : (1998 - 998) / 1000 = 1 :=
by
  sorry

end fraction_value_l129_129575


namespace problem_1_problem_2_l129_129935

noncomputable def f (x a : ℝ) : ℝ := abs x + 2 * abs (x - a)

theorem problem_1 (x : ℝ) : (f x 1 ≤ 4) ↔ (- 2 / 3 ≤ x ∧ x ≤ 2) := 
sorry

theorem problem_2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ (4 ≤ a) := 
sorry

end problem_1_problem_2_l129_129935


namespace range_of_a_l129_129171

-- Define the input conditions and requirements, and then state the theorem.
def is_acute_angle_cos_inequality (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2

theorem range_of_a (a : ℝ) :
  is_acute_angle_cos_inequality a 1 3 ∧ is_acute_angle_cos_inequality 1 3 a ∧
  is_acute_angle_cos_inequality 3 a 1 ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end range_of_a_l129_129171


namespace original_cost_before_changes_l129_129768

variable (C : ℝ)

theorem original_cost_before_changes (h : 2 * C * 1.20 = 480) : C = 200 :=
by
  -- proof goes here
  sorry

end original_cost_before_changes_l129_129768


namespace range_of_a_l129_129963

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (a ≤ 0 ∨ a ≥ 4) :=
by {
  sorry
}

end range_of_a_l129_129963


namespace thief_speed_l129_129129

theorem thief_speed
  (distance_initial : ℝ := 100 / 1000) -- distance (100 meters converted to kilometers)
  (policeman_speed : ℝ := 10) -- speed of the policeman in km/hr
  (thief_distance : ℝ := 400 / 1000) -- distance thief runs in kilometers (400 meters converted)
  : ∃ V_t : ℝ, V_t = 8 :=
by
  sorry

end thief_speed_l129_129129


namespace coordinates_of_P_l129_129779

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end coordinates_of_P_l129_129779


namespace three_digit_cubes_divisible_by_8_l129_129866

theorem three_digit_cubes_divisible_by_8 : ∃ (S : Finset ℕ), S.card = 2 ∧ ∀ x ∈ S, x ^ 3 ≥ 100 ∧ x ^ 3 ≤ 999 ∧ x ^ 3 % 8 = 0 :=
by
  sorry

end three_digit_cubes_divisible_by_8_l129_129866


namespace find_radius_of_circle_l129_129687

theorem find_radius_of_circle
  (a b R : ℝ)
  (h1 : R^2 = a * b) :
  R = Real.sqrt (a * b) :=
by
  sorry

end find_radius_of_circle_l129_129687


namespace sandy_initial_carrots_l129_129045

-- Defining the conditions
def sam_took : ℕ := 3
def sandy_left : ℕ := 3

-- The statement to be proven
theorem sandy_initial_carrots :
  (sandy_left + sam_took = 6) :=
by
  sorry

end sandy_initial_carrots_l129_129045


namespace bruce_total_amount_paid_l129_129567

-- Definitions for quantities and rates
def quantity_of_grapes : Nat := 8
def rate_per_kg_grapes : Nat := 70
def quantity_of_mangoes : Nat := 11
def rate_per_kg_mangoes : Nat := 55

-- Calculate individual costs
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes

-- Calculate total amount paid
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Statement to prove
theorem bruce_total_amount_paid : total_amount_paid = 1165 := by
  -- Proof is intentionally left as a placeholder
  sorry

end bruce_total_amount_paid_l129_129567


namespace tables_left_l129_129222

theorem tables_left (original_tables number_of_customers_per_table current_customers : ℝ) 
(h1 : original_tables = 44.0)
(h2 : number_of_customers_per_table = 8.0)
(h3 : current_customers = 256) : 
(original_tables - current_customers / number_of_customers_per_table) = 12.0 :=
by
  sorry

end tables_left_l129_129222


namespace total_time_spent_l129_129259

def chess_game_duration_hours : ℕ := 20
def chess_game_duration_minutes : ℕ := 15
def additional_analysis_time : ℕ := 22
def total_expected_time : ℕ := 1237

theorem total_time_spent : 
  (chess_game_duration_hours * 60 + chess_game_duration_minutes + additional_analysis_time) = total_expected_time :=
  by
    sorry

end total_time_spent_l129_129259


namespace quadrilateral_area_l129_129466

def vertex1 : ℝ × ℝ := (2, 1)
def vertex2 : ℝ × ℝ := (4, 3)
def vertex3 : ℝ × ℝ := (7, 1)
def vertex4 : ℝ × ℝ := (4, 6)

noncomputable def shoelace_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
       (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area :
  shoelace_area vertex1 vertex2 vertex3 vertex4 = 7.5 :=
by
  sorry

end quadrilateral_area_l129_129466


namespace fraction_simplification_l129_129376

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (2 * x - 5) / (x ^ 2 - 1) + 3 / (1 - x) = - (x + 8) / (x ^ 2 - 1) :=
  sorry

end fraction_simplification_l129_129376


namespace subtracting_five_equals_thirtyfive_l129_129436

variable (x : ℕ)

theorem subtracting_five_equals_thirtyfive (h : x - 5 = 35) : x / 5 = 8 :=
sorry

end subtracting_five_equals_thirtyfive_l129_129436


namespace solve_for_x_l129_129709

theorem solve_for_x (x y : ℚ) (h1 : 2 * x - 3 * y = 15) (h2 : x + 2 * y = 8) : x = 54 / 7 :=
sorry

end solve_for_x_l129_129709


namespace find_central_angle_l129_129498

-- We define the given conditions.
def radius : ℝ := 2
def area : ℝ := 8

-- We state the theorem that we need to prove.
theorem find_central_angle (R : ℝ) (A : ℝ) (hR : R = radius) (hA : A = area) :
  ∃ α : ℝ, α = 4 :=
by
  sorry

end find_central_angle_l129_129498


namespace value_set_for_a_non_empty_proper_subsets_l129_129494

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

theorem value_set_for_a (M : Set ℝ) : 
  (∀ (a : ℝ), B a ⊆ A → a ∈ M) :=
sorry

theorem non_empty_proper_subsets (M : Set ℝ) :
  M = {0, 3, -3} →
  (∃ S : Set (Set ℝ), S = {{0}, {3}, {-3}, {0, 3}, {0, -3}, {3, -3}}) :=
sorry

end value_set_for_a_non_empty_proper_subsets_l129_129494


namespace find_lawn_length_l129_129283

theorem find_lawn_length
  (width_lawn : ℕ)
  (road_width : ℕ)
  (cost_total : ℕ)
  (cost_per_sqm : ℕ)
  (total_area_roads : ℕ)
  (area_roads_length : ℕ)
  (area_roads_breadth : ℕ)
  (length_lawn : ℕ) :
  width_lawn = 60 →
  road_width = 10 →
  cost_total = 3600 →
  cost_per_sqm = 3 →
  total_area_roads = cost_total / cost_per_sqm →
  area_roads_length = road_width * length_lawn →
  area_roads_breadth = road_width * (width_lawn - road_width) →
  total_area_roads = area_roads_length + area_roads_breadth →
  length_lawn = 70 :=
by
  intros h_width_lawn h_road_width h_cost_total h_cost_per_sqm h_total_area_roads h_area_roads_length h_area_roads_breadth h_total_area_roads_eq
  sorry

end find_lawn_length_l129_129283


namespace mike_gave_4_marbles_l129_129196

noncomputable def marbles_given (original_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  original_marbles - remaining_marbles

theorem mike_gave_4_marbles (original_marbles remaining_marbles given_marbles : ℕ) 
  (h1 : original_marbles = 8) (h2 : remaining_marbles = 4) (h3 : given_marbles = marbles_given original_marbles remaining_marbles) : given_marbles = 4 :=
by
  sorry

end mike_gave_4_marbles_l129_129196


namespace trishul_invested_percentage_less_than_raghu_l129_129738

variable {T V R : ℝ}

def vishal_invested_more (T V : ℝ) : Prop :=
  V = 1.10 * T

def total_sum_of_investments (T V : ℝ) : Prop :=
  T + V + 2300 = 6647

def raghu_investment : ℝ := 2300

theorem trishul_invested_percentage_less_than_raghu
  (h1 : vishal_invested_more T V)
  (h2 : total_sum_of_investments T V) :
  ((raghu_investment - T) / raghu_investment) * 100 = 10 :=
  sorry

end trishul_invested_percentage_less_than_raghu_l129_129738


namespace sum_of_three_integers_l129_129467

theorem sum_of_three_integers (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : a * b * c = 5^3) : a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l129_129467


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l129_129241

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l129_129241


namespace perpendicular_vectors_l129_129014

/-- Given vectors a and b, prove that m = 6 if a is perpendicular to b -/
theorem perpendicular_vectors {m : ℝ} (h₁ : (1, 5, -2) = (1, 5, -2)) (h₂ : ∃ m : ℝ, (m, 2, m+2) = (m, 2, m+2)) (h₃ : (1 * m + 5 * 2 + (-2) * (m + 2) = 0)) :
  m = 6 :=
sorry

end perpendicular_vectors_l129_129014


namespace factory_output_exceeds_by_20_percent_l129_129054

theorem factory_output_exceeds_by_20_percent 
  (planned_output : ℝ) (actual_output : ℝ)
  (h_planned : planned_output = 20)
  (h_actual : actual_output = 24) :
  ((actual_output - planned_output) / planned_output) * 100 = 20 := 
by
  sorry

end factory_output_exceeds_by_20_percent_l129_129054


namespace certain_number_is_five_hundred_l129_129143

theorem certain_number_is_five_hundred (x : ℝ) (h : 0.60 * x = 0.50 * 600) : x = 500 := 
by sorry

end certain_number_is_five_hundred_l129_129143


namespace min_value_reciprocal_sum_l129_129552

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end min_value_reciprocal_sum_l129_129552


namespace product_b6_b8_is_16_l129_129696

-- Given conditions
variable (a : ℕ → ℝ) -- Sequence a_n
variable (b : ℕ → ℝ) -- Sequence b_n

-- Condition 1: Arithmetic sequence a_n and non-zero
axiom a_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_non_zero : ∃ n, a n ≠ 0

-- Condition 2: Equation 2a_3 - a_7^2 + 2a_n = 0
axiom a_satisfies_eq : ∀ n : ℕ, 2 * a 3 - (a 7) ^ 2 + 2 * a n = 0

-- Condition 3: Geometric sequence b_n with b_7 = a_7
axiom b_is_geometric : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n
axiom b7_equals_a7 : b 7 = a 7

-- Prove statement
theorem product_b6_b8_is_16 : b 6 * b 8 = 16 := sorry

end product_b6_b8_is_16_l129_129696


namespace cyclists_meet_time_l129_129861

theorem cyclists_meet_time 
  (v1 v2 : ℕ) (C : ℕ) (h1 : v1 = 7) (h2 : v2 = 8) (hC : C = 675) : 
  C / (v1 + v2) = 45 :=
by
  sorry

end cyclists_meet_time_l129_129861


namespace product_has_no_linear_term_l129_129643

theorem product_has_no_linear_term (m : ℝ) (h : ((x : ℝ) → (x - m) * (x - 3) = x^2 + 3 * m)) : m = -3 := 
by
  sorry

end product_has_no_linear_term_l129_129643


namespace intersection_points_l129_129280

theorem intersection_points (g : ℝ → ℝ) (hg_inv : Function.Injective g) : 
  ∃ n, n = 3 ∧ ∀ x, g (x^3) = g (x^5) ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
by {
  sorry
}

end intersection_points_l129_129280


namespace value_of_a3_minus_a2_l129_129179

theorem value_of_a3_minus_a2 : 
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S n = n^2) ∧ (S 3 - S 2 - (S 2 - S 1)) = 2) :=
sorry

end value_of_a3_minus_a2_l129_129179


namespace find_triples_of_positive_integers_l129_129057

theorem find_triples_of_positive_integers :
  ∀ (x y z : ℕ), 2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023 ↔ 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 3 ∧ y = 3 ∧ z = 2) := 
by 
  sorry

end find_triples_of_positive_integers_l129_129057


namespace find_n_from_lcm_gcf_l129_129933

open scoped Classical

noncomputable def LCM (a b : ℕ) : ℕ := sorry
noncomputable def GCF (a b : ℕ) : ℕ := sorry

theorem find_n_from_lcm_gcf (n m : ℕ) (h1 : LCM n m = 48) (h2 : GCF n m = 18) (h3 : m = 16) : n = 54 :=
by sorry

end find_n_from_lcm_gcf_l129_129933


namespace subset_A_imp_range_a_disjoint_A_imp_range_a_l129_129841

-- Definition of sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

-- Proof problem for Question 1
theorem subset_A_imp_range_a (a : ℝ) (h : A ⊆ B a) : 
  (4 / 3) ≤ a ∧ a ≤ 2 ∧ a ≠ 0 :=
sorry

-- Proof problem for Question 2
theorem disjoint_A_imp_range_a (a : ℝ) (h : A ∩ B a = ∅) : 
  a ≤ (2 / 3) ∨ a ≥ 4 :=
sorry

end subset_A_imp_range_a_disjoint_A_imp_range_a_l129_129841


namespace problem_solution_l129_129931

variable (a b c d m : ℝ)

-- Conditions
def opposite_numbers (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def absolute_value_eq (m : ℝ) : Prop := |m| = 3

theorem problem_solution
  (h1 : opposite_numbers a b)
  (h2 : reciprocals c d)
  (h3 : absolute_value_eq m) :
  (a + b) / 2023 - 4 * (c * d) + m^2 = 5 :=
by
  sorry

end problem_solution_l129_129931


namespace geometric_sequence_second_term_l129_129921

theorem geometric_sequence_second_term (a r : ℝ) 
  (h_fifth_term : a * r^4 = 48) 
  (h_sixth_term : a * r^5 = 72) : 
  a * r = 1152 / 81 := by
  sorry

end geometric_sequence_second_term_l129_129921


namespace birdseed_needed_weekly_birdseed_needed_l129_129862

def parakeet_daily_consumption := 2
def parrot_daily_consumption := 14
def finch_daily_consumption := parakeet_daily_consumption / 2
def num_parakeets := 3
def num_parrots := 2
def num_finches := 4
def days_in_week := 7

theorem birdseed_needed :
  num_parakeets * parakeet_daily_consumption +
  num_parrots * parrot_daily_consumption +
  num_finches * finch_daily_consumption = 38 :=
by
  sorry

theorem weekly_birdseed_needed :
  38 * days_in_week = 266 :=
by
  sorry

end birdseed_needed_weekly_birdseed_needed_l129_129862


namespace students_distribution_l129_129021

theorem students_distribution (students villages : ℕ) (h_students : students = 4) (h_villages : villages = 3) :
  ∃ schemes : ℕ, schemes = 36 := 
sorry

end students_distribution_l129_129021


namespace percentage_of_ll_watchers_l129_129248

theorem percentage_of_ll_watchers 
  (T : ℕ) 
  (IS : ℕ) 
  (ME : ℕ) 
  (E2 : ℕ) 
  (A3 : ℕ) 
  (total_residents : T = 600)
  (is_watchers : IS = 210)
  (me_watchers : ME = 300)
  (e2_watchers : E2 = 108)
  (a3_watchers : A3 = 21)
  (at_least_one_show : IS + (by sorry) + ME - E2 + A3 = T) :
  ∃ x : ℕ, (x * 100 / T) = 115 :=
by sorry

end percentage_of_ll_watchers_l129_129248


namespace find_natural_solution_l129_129505

theorem find_natural_solution (x y : ℕ) (h : y^6 + 2 * y^3 - y^2 + 1 = x^3) : x = 1 ∧ y = 0 :=
by
  sorry

end find_natural_solution_l129_129505


namespace MitchWorks25Hours_l129_129600

noncomputable def MitchWorksHours : Prop :=
  let weekday_earnings_rate := 3
  let weekend_earnings_rate := 6
  let weekly_earnings := 111
  let weekend_hours := 6
  let weekday_hours (x : ℕ) := 5 * x
  let weekend_earnings := weekend_hours * weekend_earnings_rate
  let weekday_earnings (x : ℕ) := x * weekday_earnings_rate
  let total_weekday_earnings (x : ℕ) := weekly_earnings - weekend_earnings
  ∀ (x : ℕ), weekday_earnings x = total_weekday_earnings x → x = 25

theorem MitchWorks25Hours : MitchWorksHours := by
  sorry

end MitchWorks25Hours_l129_129600


namespace quadratic_minimization_l129_129004

theorem quadratic_minimization : 
  ∃ x : ℝ, ∀ y : ℝ, (x^2 - 12 * x + 36 ≤ y^2 - 12 * y + 36) ∧ x^2 - 12 * x + 36 = 0 :=
by
  sorry

end quadratic_minimization_l129_129004


namespace smallest_solution_of_quadratic_eq_l129_129184

theorem smallest_solution_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ < x₂) ∧ (x₁^2 + 10 * x₁ - 40 = 0) ∧ (x₂^2 + 10 * x₂ - 40 = 0) ∧ x₁ = -8 :=
by {
  sorry
}

end smallest_solution_of_quadratic_eq_l129_129184


namespace triangle_angle_sine_identity_l129_129915

theorem triangle_angle_sine_identity (A B C : ℝ) (n : ℤ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n + 1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) :=
by
  sorry

end triangle_angle_sine_identity_l129_129915


namespace journey_time_ratio_l129_129520

theorem journey_time_ratio (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 48
  let T2 := D / 32
  (T2 / T1) = 3 / 2 :=
by
  sorry

end journey_time_ratio_l129_129520


namespace election_problem_l129_129292

theorem election_problem :
  ∃ (n : ℕ), n = (10 * 9) * Nat.choose 8 3 :=
  by
  use 5040
  sorry

end election_problem_l129_129292


namespace circle_area_l129_129890

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = (2 * r) ^ 2) : π * r ^ 2 = π ^ (1 / 3) :=
by
  sorry

end circle_area_l129_129890


namespace total_books_l129_129272

theorem total_books (x : ℕ) (h1 : 3 * x + 2 * x + (3 / 2) * x > 3000) : 
  ∃ (T : ℕ), T = 3 * x + 2 * x + (3 / 2) * x ∧ T > 3000 ∧ T = 3003 := 
by 
  -- Our theorem states there exists an integer T such that the total number of books is 3003.
  sorry

end total_books_l129_129272


namespace min_small_containers_needed_l129_129158

def medium_container_capacity : ℕ := 450
def small_container_capacity : ℕ := 28

theorem min_small_containers_needed : ⌈(medium_container_capacity : ℝ) / small_container_capacity⌉ = 17 :=
by
  sorry

end min_small_containers_needed_l129_129158


namespace abscissa_range_of_point_P_l129_129548

-- Definitions based on the conditions from the problem
def y_function (x : ℝ) : ℝ := 4 - 3 * x
def point_P (x y : ℝ) : Prop := y = y_function x
def ordinate_greater_than_negative_five (y : ℝ) : Prop := y > -5

-- Theorem statement combining the above definitions
theorem abscissa_range_of_point_P (x y : ℝ) :
  point_P x y →
  ordinate_greater_than_negative_five y →
  x < 3 :=
sorry

end abscissa_range_of_point_P_l129_129548


namespace ratio_playground_landscape_l129_129641

-- Defining the conditions
def breadth := 420
def length := breadth / 6
def playground_area := 4200
def landscape_area := length * breadth

-- Stating the theorem to prove the ratio is 1:7
theorem ratio_playground_landscape :
  (playground_area.toFloat / landscape_area.toFloat) = (1.0 / 7.0) :=
by
  sorry

end ratio_playground_landscape_l129_129641


namespace find_remainder_l129_129047

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l129_129047


namespace joe_list_possibilities_l129_129987

theorem joe_list_possibilities :
  let balls := 15
  let draws := 4
  (balls ^ draws = 50625) := 
by
  let balls := 15
  let draws := 4
  sorry

end joe_list_possibilities_l129_129987


namespace sunflower_cans_l129_129884

theorem sunflower_cans (total_seeds seeds_per_can : ℕ) (h_total_seeds : total_seeds = 54) (h_seeds_per_can : seeds_per_can = 6) :
  total_seeds / seeds_per_can = 9 :=
by sorry

end sunflower_cans_l129_129884


namespace Nicole_fewer_questions_l129_129535

-- Definitions based on the given conditions
def Nicole_correct : ℕ := 22
def Cherry_correct : ℕ := 17
def Kim_correct : ℕ := Cherry_correct + 8

-- Theorem to prove the number of fewer questions Nicole answered compared to Kim
theorem Nicole_fewer_questions : Kim_correct - Nicole_correct = 3 :=
by
  -- We set up the definitions
  let Nicole_correct := 22
  let Cherry_correct := 17
  let Kim_correct := Cherry_correct + 8
  -- The proof will be filled in here. 
  -- The goal theorem statement is filled with 'sorry' to bypass the actual proof.
  have : Kim_correct - Nicole_correct = 3 := sorry
  exact this

end Nicole_fewer_questions_l129_129535


namespace vertex_on_x_axis_segment_cut_on_x_axis_l129_129275

-- Define the quadratic function
def quadratic_func (k x : ℝ) : ℝ :=
  (k + 2) * x^2 - 2 * k * x + 3 * k

-- The conditions to prove
theorem vertex_on_x_axis (k : ℝ) :
  (4 * k^2 - 4 * 3 * k * (k + 2) = 0) ↔ (k = 0 ∨ k = -3) :=
sorry

theorem segment_cut_on_x_axis (k : ℝ) :
  ((2 * k / (k + 2))^2 - 12 * k / (k + 2) = 16) ↔ (k = -8/3 ∨ k = -1) :=
sorry

end vertex_on_x_axis_segment_cut_on_x_axis_l129_129275


namespace how_many_people_in_group_l129_129217

-- Definition of the conditions
def ratio_likes_football : ℚ := 24 / 60
def ratio_plays_football_given_likes : ℚ := 1 / 2
def expected_to_play_football : ℕ := 50

-- Combining the ratios to get the fraction of total people playing football
def ratio_plays_football : ℚ := ratio_likes_football * ratio_plays_football_given_likes

-- Total number of people in the group
def total_people_in_group : ℕ := 250

-- Proof statement
theorem how_many_people_in_group (expected_to_play_football : ℕ) : 
  ratio_plays_football * total_people_in_group = expected_to_play_football :=
by {
  -- Directly using our definitions
  sorry
}

end how_many_people_in_group_l129_129217


namespace call_cost_per_minute_l129_129903

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l129_129903


namespace problem_l129_129387

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem problem (a b : ℝ) (h : f a b (-2) = 2) : f a b 2 = -10 :=
by
  sorry

end problem_l129_129387


namespace number_equals_14_l129_129073

theorem number_equals_14 (n : ℕ) (h1 : 2^n - 2^(n-2) = 3 * 2^12) (h2 : n = 14) : n = 14 := 
by 
  sorry

end number_equals_14_l129_129073


namespace mr_bird_exact_speed_l129_129183

-- Define the properties and calculating the exact speed
theorem mr_bird_exact_speed (d t : ℝ) (h1 : d = 50 * (t + 1 / 12)) (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 :=
by 
  -- skipping the proof
  sorry

end mr_bird_exact_speed_l129_129183


namespace kaleb_first_load_pieces_l129_129662

-- Definitions of given conditions
def total_pieces : ℕ := 39
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- Definition for calculation of pieces in equal loads
def pieces_in_equal_loads : ℕ := num_equal_loads * pieces_per_load

-- Definition for pieces in the first load
def pieces_in_first_load : ℕ := total_pieces - pieces_in_equal_loads

-- Statement to prove that the pieces in the first load is 19
theorem kaleb_first_load_pieces : pieces_in_first_load = 19 := 
by
  -- The proof is skipped
  sorry

end kaleb_first_load_pieces_l129_129662


namespace savings_after_20_days_l129_129133

-- Definitions based on conditions
def daily_earnings : ℕ := 80
def days_worked : ℕ := 20
def total_spent : ℕ := 1360

-- Prove the savings after 20 days
theorem savings_after_20_days : daily_earnings * days_worked - total_spent = 240 :=
by
  sorry

end savings_after_20_days_l129_129133


namespace parametric_to_ellipse_parametric_to_line_l129_129620

-- Define the conditions and the corresponding parametric equations
variable (φ t : ℝ) (x y : ℝ)

-- The first parametric equation converted to the ordinary form
theorem parametric_to_ellipse (h1 : x = 5 * Real.cos φ) (h2 : y = 4 * Real.sin φ) :
  (x ^ 2 / 25) + (y ^ 2 / 16) = 1 := sorry

-- The second parametric equation converted to the ordinary form
theorem parametric_to_line (h3 : x = 1 - 3 * t) (h4 : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := sorry

end parametric_to_ellipse_parametric_to_line_l129_129620


namespace at_least_one_negative_l129_129071

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) : a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l129_129071


namespace polynomial_is_first_degree_l129_129465

theorem polynomial_is_first_degree (k m : ℝ) (h : (k - 1) = 0) : k = 1 :=
by
  sorry

end polynomial_is_first_degree_l129_129465


namespace quadratic_sums_l129_129209

variables {α : Type} [CommRing α] {a b c : α}

theorem quadratic_sums 
  (h₁ : ∀ (a b c : α), a + b ≠ 0 ∧ b + c ≠ 0 ∧ c + a ≠ 0)
  (h₂ : ∀ (r₁ r₂ : α), 
    (r₁^2 + a * r₁ + b = 0 ∧ r₂^2 + b * r₂ + c = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₃ : ∀ (r₁ r₂ : α), 
    (r₁^2 + b * r₁ + c = 0 ∧ r₂^2 + c * r₂ + a = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₄ : ∀ (r₁ r₂ : α), 
    (r₁^2 + c * r₁ + a = 0 ∧ r₂^2 + a * r₂ + b = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0) :
  a^2 + b^2 + c^2 = 18 ∧
  a^2 * b + b^2 * c + c^2 * a = 27 ∧
  a^3 * b^2 + b^3 * c^2 + c^3 * a^2 = -162 :=
sorry

end quadratic_sums_l129_129209


namespace number_is_twenty_l129_129034

-- We state that if \( \frac{30}{100}x = \frac{15}{100} \times 40 \), then \( x = 20 \)
theorem number_is_twenty (x : ℝ) (h : (30 / 100) * x = (15 / 100) * 40) : x = 20 :=
by
  sorry

end number_is_twenty_l129_129034


namespace find_other_number_l129_129155

def HCF (a b : ℕ) : ℕ := sorry
def LCM (a b : ℕ) : ℕ := sorry

theorem find_other_number (B : ℕ) 
 (h1 : HCF 24 B = 15) 
 (h2 : LCM 24 B = 312) 
 : B = 195 := 
by
  sorry

end find_other_number_l129_129155


namespace solution_set_even_function_l129_129657

/-- Let f be an even function, and for x in [0, ∞), f(x) = x - 1. Determine the solution set for the inequality f(x) > 1.
We prove that the solution set is {x | x < -2 or x > 2}. -/
theorem solution_set_even_function (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, 0 ≤ x → f x = x - 1) :
  {x : ℝ | f x > 1} = {x | x < -2 ∨ x > 2} :=
by
  sorry  -- Proof steps go here.

end solution_set_even_function_l129_129657


namespace ratio_eq_thirteen_fifths_l129_129733

theorem ratio_eq_thirteen_fifths
  (a b c : ℝ)
  (h₁ : b / a = 4)
  (h₂ : c / b = 2) :
  (a + b + c) / (a + b) = 13 / 5 :=
sorry

end ratio_eq_thirteen_fifths_l129_129733


namespace total_annual_gain_l129_129204

theorem total_annual_gain (x : ℝ) 
    (Lakshmi_share : ℝ) 
    (Lakshmi_share_eq: Lakshmi_share = 12000) : 
    (3 * Lakshmi_share = 36000) :=
by
  sorry

end total_annual_gain_l129_129204


namespace smallest_X_divisible_by_60_l129_129178

/-
  Let \( T \) be a positive integer consisting solely of 0s and 1s.
  If \( X = \frac{T}{60} \) and \( X \) is an integer, prove that the smallest possible value of \( X \) is 185.
-/
theorem smallest_X_divisible_by_60 (T X : ℕ) 
  (hT_digit : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) 
  (h1 : X = T / 60) 
  (h2 : T % 60 = 0) : 
  X = 185 :=
sorry

end smallest_X_divisible_by_60_l129_129178


namespace percentage_markup_l129_129993

theorem percentage_markup (selling_price cost_price : ℚ)
  (h_selling_price : selling_price = 8325)
  (h_cost_price : cost_price = 7239.13) :
  ((selling_price - cost_price) / cost_price) * 100 = 15 := 
sorry

end percentage_markup_l129_129993


namespace cylinder_radius_range_l129_129902

theorem cylinder_radius_range :
  (V : ℝ) → (h : ℝ) → (r : ℝ) →
  V = 20 * Real.pi →
  h = 2 →
  (V = Real.pi * r^2 * h) →
  3 < r ∧ r < 4 :=
by
  -- Placeholder for the proof
  intro V h r hV hh hV_eq
  sorry

end cylinder_radius_range_l129_129902


namespace parabola_ellipse_focus_l129_129669

theorem parabola_ellipse_focus (p : ℝ) :
  (∃ (x y : ℝ), x^2 = 2 * p * y ∧ y = -1 ∧ x = 0) →
  p = -2 :=
by
  sorry

end parabola_ellipse_focus_l129_129669


namespace solution_set_range_ineq_l129_129315

theorem solution_set_range_ineq (m : ℝ) :
  ∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0 ↔ (-5: ℝ)⁻¹ < m ∧ m ≤ 3 :=
by
  sorry

end solution_set_range_ineq_l129_129315


namespace chewing_gum_company_revenue_l129_129022

theorem chewing_gum_company_revenue (R : ℝ) :
  let projected_revenue := 1.25 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 60 := 
by
  sorry

end chewing_gum_company_revenue_l129_129022


namespace pet_snake_cost_l129_129668

theorem pet_snake_cost (original_amount left_amount snake_cost : ℕ) 
  (h1 : original_amount = 73) 
  (h2 : left_amount = 18)
  (h3 : snake_cost = original_amount - left_amount) : 
  snake_cost = 55 := 
by 
  sorry

end pet_snake_cost_l129_129668


namespace right_triangle_sum_of_squares_l129_129429

   theorem right_triangle_sum_of_squares {AB AC BC : ℝ} (h_right: AB^2 + AC^2 = BC^2) (h_hypotenuse: BC = 1) :
     AB^2 + AC^2 + BC^2 = 2 :=
   by
     sorry
   
end right_triangle_sum_of_squares_l129_129429


namespace expand_expression_l129_129383

theorem expand_expression (y : ℝ) : (7 * y + 12) * 3 * y = 21 * y ^ 2 + 36 * y := by
  sorry

end expand_expression_l129_129383


namespace part1_part2_part3_l129_129149

-- Part 1
theorem part1 (a b : ℝ) : 3*(a-b)^2 - 6*(a-b)^2 + 2*(a-b)^2 = -(a-b)^2 :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x^2 - 2*y = 4) : 3*x^2 - 6*y - 21 = -9 :=
sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) : 
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 :=
sorry

end part1_part2_part3_l129_129149


namespace taylor_family_reunion_adults_l129_129895

def number_of_kids : ℕ := 45
def number_of_tables : ℕ := 14
def people_per_table : ℕ := 12
def total_people := number_of_tables * people_per_table

theorem taylor_family_reunion_adults : total_people - number_of_kids = 123 := by
  sorry

end taylor_family_reunion_adults_l129_129895


namespace xy_identity_l129_129998

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l129_129998


namespace area_of_square_ABCD_l129_129405

theorem area_of_square_ABCD :
  (∃ (x y : ℝ), 2 * x + 2 * y = 40) →
  ∃ (s : ℝ), s = 20 ∧ s * s = 400 :=
by
  sorry

end area_of_square_ABCD_l129_129405


namespace graph_intersect_x_axis_exactly_once_l129_129104

theorem graph_intersect_x_axis_exactly_once (a : ℝ) :
    (∀ x : ℝ, (a-1) * x^2 - 4 * x + 2 * a = 0 → x = -(1/2)) ∨ -- Quadratic condition with one real root giving unique intersection
    ((a-1) = 0 ∧ ∃ x : ℝ, -4 * x + 2 * a = 0) -- Linear condition giving unique intersection
    ↔ a = -1 ∨ a = 2 ∨ a = 1 :=
by
    sorry

end graph_intersect_x_axis_exactly_once_l129_129104


namespace solve_m_problem_l129_129813

theorem solve_m_problem :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0) →
  m ∈ Set.Ico (-1/4 : ℝ) 2 :=
sorry

end solve_m_problem_l129_129813


namespace number_of_white_balls_l129_129228

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end number_of_white_balls_l129_129228


namespace value_of_x_plus_y_l129_129848

theorem value_of_x_plus_y (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end value_of_x_plus_y_l129_129848


namespace eight_digit_not_perfect_square_l129_129917

theorem eight_digit_not_perfect_square : ∀ x : ℕ, 0 ≤ x ∧ x ≤ 9999 → ¬ ∃ y : ℤ, (99990000 + x) = y * y := 
by
  intros x hx
  intro h
  obtain ⟨y, hy⟩ := h
  sorry

end eight_digit_not_perfect_square_l129_129917


namespace rebecca_haircuts_l129_129189

-- Definitions based on the conditions
def charge_per_haircut : ℕ := 30
def charge_per_perm : ℕ := 40
def charge_per_dye_job : ℕ := 60
def dye_cost_per_job : ℕ := 10
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def tips : ℕ := 50
def total_amount : ℕ := 310

-- Define the unknown number of haircuts scheduled
variable (H : ℕ)

-- Statement of the proof problem
theorem rebecca_haircuts :
  charge_per_haircut * H + charge_per_perm * num_perms + charge_per_dye_job * num_dye_jobs
  - dye_cost_per_job * num_dye_jobs + tips = total_amount → H = 4 :=
by
  sorry

end rebecca_haircuts_l129_129189


namespace squared_expression_is_matching_string_l129_129329

theorem squared_expression_is_matching_string (n : ℕ) (h : n > 0) :
  let a := (10^n - 1) / 9
  let term1 := 4 * a * (9 * a + 2)
  let term2 := 10 * a + 1
  let term3 := 6 * a
  let exp := term1 + term2 - term3
  Nat.sqrt exp = 6 * a + 1 := by
  sorry

end squared_expression_is_matching_string_l129_129329


namespace find_ratio_AF_FB_l129_129990

-- Define the vector space over reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions of points A, B, C, D, F, P
variables (a b c d f p : V)

-- Given conditions as hypotheses
variables (h1 : (p = 2 / 5 • a + 3 / 5 • d))
variables (h2 : (p = 5 / 7 • f + 2 / 7 • c))
variables (hd : (d = 1 / 3 • b + 2 / 3 • c))
variables (hf : (f = 1 / 4 • a + 3 / 4 • b))

-- Theorem statement
theorem find_ratio_AF_FB : (41 : ℝ) / 15 = (41 : ℝ) / 15 := 
by sorry

end find_ratio_AF_FB_l129_129990


namespace functional_eq_solution_l129_129111

theorem functional_eq_solution (f : ℝ → ℝ) 
  (H : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
by 
  sorry

end functional_eq_solution_l129_129111


namespace smallest_n_l129_129932

theorem smallest_n (n : ℕ) (h1 : n % 6 = 5) (h2 : n % 7 = 4) (h3 : n > 20) : n = 53 :=
sorry

end smallest_n_l129_129932


namespace area_of_square_l129_129943

theorem area_of_square (a : ℝ) (h : a = 12) : a * a = 144 := by
  rw [h]
  norm_num

end area_of_square_l129_129943


namespace nth_term_150_l129_129390

-- Conditions
def a : ℕ := 2
def d : ℕ := 5
def arithmetic_sequence (n : ℕ) : ℕ := a + (n - 1) * d

-- Question and corresponding answer proof
theorem nth_term_150 : arithmetic_sequence 150 = 747 := by
  sorry

end nth_term_150_l129_129390


namespace greatest_common_divisor_l129_129808

theorem greatest_common_divisor (n : ℕ) (h1 : ∃ d : ℕ, d = gcd 180 n ∧ (∃ (l : List ℕ), l.length = 5 ∧ ∀ x : ℕ, x ∈ l → x ∣ d)) :
  ∃ x : ℕ, x = 27 :=
by
  sorry

end greatest_common_divisor_l129_129808


namespace census_survey_is_suitable_l129_129559

def suitable_for_census (s: String) : Prop :=
  s = "Understand the vision condition of students in a class"

theorem census_survey_is_suitable :
  suitable_for_census "Understand the vision condition of students in a class" :=
by
  sorry

end census_survey_is_suitable_l129_129559


namespace locus_of_midpoint_l129_129508

open Real

noncomputable def circumcircle_eq (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := 1
  let b := 3
  let r2 := 5
  (a, b, r2)

theorem locus_of_midpoint (A B C N : ℝ × ℝ) :
  N = (6, 2) ∧ A = (0, 1) ∧ B = (2, 1) ∧ C = (3, 4) → 
  let P := (7 / 2, 5 / 2)
  let r2 := 5 / 4
  ∃ x y : ℝ, 
  (x, y) = P ∧ (x - 7 / 2)^2 + (y - 5 / 2)^2 = r2 :=
by sorry

end locus_of_midpoint_l129_129508


namespace prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l129_129900

def prob_has_bio_test : ℚ := 5 / 8
def prob_not_has_chem_test : ℚ := 1 / 2

theorem prob_not_has_bio_test : 1 - 5 / 8 = 3 / 8 := by
  sorry

theorem combined_prob_neither_bio_nor_chem :
  (1 - 5 / 8) * (1 / 2) = 3 / 16 := by
  sorry

end prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l129_129900


namespace negation_of_proposition_true_l129_129190

theorem negation_of_proposition_true (a b : ℝ) : 
  (¬ ((a > b) → (∀ c : ℝ, c ^ 2 ≠ 0 → a * c ^ 2 > b * c ^ 2)) = true) :=
by
  sorry

end negation_of_proposition_true_l129_129190


namespace find_k_from_direction_vector_l129_129437

/-- Given points p1 and p2, the direction vector's k component
    is -3 when the x component is 3. -/
theorem find_k_from_direction_vector
  (p1 : ℤ × ℤ) (p2 : ℤ × ℤ)
  (h1 : p1 = (2, -1))
  (h2 : p2 = (-4, 5))
  (dv_x : ℤ) (dv_k : ℤ)
  (h3 : (dv_x, dv_k) = (3, -3)) :
  True :=
by
  sorry

end find_k_from_direction_vector_l129_129437


namespace guiding_normal_vector_l129_129769

noncomputable def ellipsoid (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 - 6

def point_M0 : ℝ × ℝ × ℝ := (1, -1, 1)

def normal_vector (x y z : ℝ) : ℝ × ℝ × ℝ := (
  2 * x,
  4 * y,
  6 * z
)

theorem guiding_normal_vector : normal_vector 1 (-1) 1 = (2, -4, 6) :=
by
  sorry

end guiding_normal_vector_l129_129769


namespace total_nephews_l129_129445

noncomputable def Alden_past_nephews : ℕ := 50
noncomputable def Alden_current_nephews : ℕ := 2 * Alden_past_nephews
noncomputable def Vihaan_current_nephews : ℕ := Alden_current_nephews + 60

theorem total_nephews :
  Alden_current_nephews + Vihaan_current_nephews = 260 := 
by
  sorry

end total_nephews_l129_129445


namespace intersection_is_2_l129_129282

noncomputable def M : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def N : Set ℝ := {x | x^2 ≥ 2 * x}
noncomputable def intersection : Set ℝ := M ∩ N

theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l129_129282


namespace maximum_abc_value_l129_129461

theorem maximum_abc_value:
  (∀ (a b c : ℝ), (0 < a ∧ a < 3) ∧ (0 < b ∧ b < 3) ∧ (0 < c ∧ c < 3) ∧ (∀ x : ℝ, (x^4 + a * x^3 + b * x^2 + c * x + 1) ≠ 0) → (abc ≤ 18.75)) :=
sorry

end maximum_abc_value_l129_129461


namespace flower_shop_ratio_l129_129488

theorem flower_shop_ratio (V C T R : ℕ) 
(total_flowers : V + C + T + R > 0)
(tulips_ratio : T = V / 4)
(roses_tulips_equal : R = T)
(carnations_fraction : C = 2 / 3 * (V + T + R + C)) 
: V / C = 1 / 3 := 
by
  -- Proof omitted
  sorry

end flower_shop_ratio_l129_129488


namespace line_intersects_ellipse_l129_129459

theorem line_intersects_ellipse
  (m : ℝ) :
  ∃ P : ℝ × ℝ, P = (3, 2) ∧ ((m + 2) * P.1 - (m + 4) * P.2 + 2 - m = 0) ∧ 
  (P.1^2 / 25 + P.2^2 / 9 < 1) :=
by 
  sorry

end line_intersects_ellipse_l129_129459


namespace remainder_of_large_number_l129_129094

theorem remainder_of_large_number :
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  last_four_digits % 16 = 9 := 
by
  let number := 65985241545898754582556898522454889
  let last_four_digits := 4889
  show last_four_digits % 16 = 9
  sorry

end remainder_of_large_number_l129_129094


namespace age_difference_constant_l129_129824

theorem age_difference_constant (seokjin_age_mother_age_diff : ∀ (t : ℕ), 33 - 7 = 26) : 
  ∀ (n : ℕ), 33 + n - (7 + n) = 26 := 
by
  sorry

end age_difference_constant_l129_129824


namespace modulo_calculation_l129_129809

theorem modulo_calculation : (68 * 97 * 113) % 25 = 23 := by
  sorry

end modulo_calculation_l129_129809


namespace cubes_painted_on_one_side_l129_129753

def is_cube_painted_on_one_side (l w h : ℕ) (cube_size : ℕ) : ℕ :=
  let top_bottom := (l - 2) * (w - 2) * 2
  let front_back := (l - 2) * (h - 2) * 2
  let left_right := (w - 2) * (h - 2) * 2
  top_bottom + front_back + left_right

theorem cubes_painted_on_one_side (l w h cube_size : ℕ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) (h_cube_size : cube_size = 1) :
  is_cube_painted_on_one_side l w h cube_size = 22 :=
by
  sorry

end cubes_painted_on_one_side_l129_129753


namespace circle_equation_AB_diameter_l129_129537

theorem circle_equation_AB_diameter (A B : ℝ × ℝ) :
  A = (1, -4) → B = (-5, 4) →
  ∃ C : ℝ × ℝ, C = (-2, 0) ∧ ∃ r : ℝ, r = 5 ∧ (∀ x y : ℝ, (x + 2)^2 + y^2 = 25) :=
by intros h1 h2; sorry

end circle_equation_AB_diameter_l129_129537


namespace range_of_a_l129_129761

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l129_129761


namespace joe_has_more_shirts_l129_129287

theorem joe_has_more_shirts (alex_shirts : ℕ) (ben_shirts : ℕ) (ben_joe_diff : ℕ)
  (h_a : alex_shirts = 4)
  (h_b : ben_shirts = 15)
  (h_bj : ben_shirts = joe_shirts + ben_joe_diff)
  (h_bj_diff : ben_joe_diff = 8) :
  joe_shirts - alex_shirts = 3 :=
by {
  sorry
}

end joe_has_more_shirts_l129_129287


namespace Peggy_needs_to_add_stamps_l129_129764

theorem Peggy_needs_to_add_stamps :
  ∀ (Peggy_stamps Bert_stamps Ernie_stamps : ℕ),
  Peggy_stamps = 75 →
  Ernie_stamps = 3 * Peggy_stamps →
  Bert_stamps = 4 * Ernie_stamps →
  Bert_stamps - Peggy_stamps = 825 :=
by
  intros Peggy_stamps Bert_stamps Ernie_stamps hPeggy hErnie hBert
  sorry

end Peggy_needs_to_add_stamps_l129_129764


namespace max_distinct_dance_counts_l129_129771

theorem max_distinct_dance_counts (B G : ℕ) (hB : B = 29) (hG : G = 15) 
  (dance_with : ℕ → ℕ → Prop)
  (h_dance_limit : ∀ b g, dance_with b g → b ≤ B ∧ g ≤ G) :
  ∃ max_counts : ℕ, max_counts = 29 :=
by
  -- The statement of the theorem. Proof is omitted.
  sorry

end max_distinct_dance_counts_l129_129771


namespace asymptote_problem_l129_129151

-- Definitions for the problem
def r (x : ℝ) : ℝ := -3 * (x + 2) * (x - 1)
def s (x : ℝ) : ℝ := (x + 2) * (x - 4)

-- Assertion to prove
theorem asymptote_problem : r (-1) / s (-1) = 6 / 5 :=
by {
  -- This is where the proof would be carried out
  sorry
}

end asymptote_problem_l129_129151


namespace sum_of_squares_of_roots_l129_129745

theorem sum_of_squares_of_roots (α β : ℝ)
  (h_root1 : 10 * α^2 - 14 * α - 24 = 0)
  (h_root2 : 10 * β^2 - 14 * β - 24 = 0)
  (h_distinct : α ≠ β) :
  α^2 + β^2 = 169 / 25 :=
sorry

end sum_of_squares_of_roots_l129_129745


namespace sequence_expression_l129_129507

theorem sequence_expression {a : ℕ → ℝ} (h1 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4)
  (h2 : a 1 = 1) (h3 : ∀ n, a n > 0) : ∀ n, a n = Real.sqrt (4 * n - 3) := by
  sorry

end sequence_expression_l129_129507


namespace increasing_on_interval_l129_129000

theorem increasing_on_interval (a : ℝ) : (∀ x : ℝ, x > 1/2 → (2 * x + a + 1 / x^2) ≥ 0) → a ≥ -3 :=
by
  intros h
  -- Rest of the proof would go here
  sorry

end increasing_on_interval_l129_129000


namespace fraction_equal_decimal_l129_129574

theorem fraction_equal_decimal : (1 / 4) = 0.25 :=
sorry

end fraction_equal_decimal_l129_129574


namespace largest_n_l129_129816

def a_n (n : ℕ) (d_a : ℤ) : ℤ := 1 + (n-1) * d_a
def b_n (n : ℕ) (d_b : ℤ) : ℤ := 3 + (n-1) * d_b

theorem largest_n (d_a d_b : ℤ) (n : ℕ) :
  (a_n n d_a * b_n n d_b = 2304 ∧ a_n 1 d_a = 1 ∧ b_n 1 d_b = 3) 
  → n ≤ 20 := 
sorry

end largest_n_l129_129816


namespace angle_between_v1_v2_l129_129356

-- Define vectors
def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (4, 6)

-- Define the dot product function
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Define the magnitude function
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the cosine of the angle between two vectors
noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ := (dot_product a b) / (magnitude a * magnitude b)

-- Define the angle in degrees between two vectors
noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := Real.arccos (cos_theta a b) * (180 / Real.pi)

-- The statement to prove
theorem angle_between_v1_v2 : angle_between_vectors v1 v2 = Real.arccos (-6 * Real.sqrt 13 / 65) * (180 / Real.pi) :=
sorry

end angle_between_v1_v2_l129_129356


namespace gain_amount_l129_129074

theorem gain_amount (gain_percent : ℝ) (gain : ℝ) (amount : ℝ) 
  (h_gain_percent : gain_percent = 1) 
  (h_gain : gain = 0.70) 
  : amount = 70 :=
by
  sorry

end gain_amount_l129_129074


namespace addition_in_sets_l129_129781

theorem addition_in_sets (a b : ℤ) (hA : ∃ k : ℤ, a = 2 * k) (hB : ∃ k : ℤ, b = 2 * k + 1) : ∃ k : ℤ, a + b = 2 * k + 1 :=
by
  sorry

end addition_in_sets_l129_129781


namespace total_points_scored_l129_129681

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l129_129681


namespace determine_x_l129_129153

theorem determine_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 18 * y + x - 2 = 0) : x = 9 / 5 :=
sorry

end determine_x_l129_129153


namespace union_eq_l129_129207

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l129_129207


namespace swim_distance_downstream_l129_129514

theorem swim_distance_downstream 
  (V_m V_s : ℕ) 
  (t d : ℕ) 
  (h1 : V_m = 9) 
  (h2 : t = 3) 
  (h3 : 3 * (V_m - V_s) = 18) : 
  t * (V_m + V_s) = 36 := 
by 
  sorry

end swim_distance_downstream_l129_129514


namespace election_winner_margin_l129_129422

theorem election_winner_margin (V : ℝ) 
    (hV: V = 3744 / 0.52) 
    (w_votes: ℝ := 3744) 
    (l_votes: ℝ := 0.48 * V) :
    w_votes - l_votes = 288 := by
  sorry

end election_winner_margin_l129_129422


namespace no_real_a_values_l129_129443

noncomputable def polynomial_with_no_real_root (a : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 ≠ 0
  
theorem no_real_a_values :
  ∀ a : ℝ, (∃ x : ℝ, x^4 + a^2 * x^3 - 2 * x^2 + a * x + 4 = 0) → false :=
by sorry

end no_real_a_values_l129_129443


namespace quadratic_function_expr_value_of_b_minimum_value_of_m_l129_129984

-- Problem 1: Proving the quadratic function expression
theorem quadratic_function_expr (x : ℝ) (b c : ℝ)
  (h1 : (0:ℝ) = x^2 + b * 0 + c)
  (h2 : -b / 2 = (1:ℝ)) :
  x^2 - 2 * x + 4 = x^2 + b * x + c := sorry

-- Problem 2: Proving specific values of b
theorem value_of_b (b c : ℝ)
  (h1 : b^2 - c = 0)
  (h2 : ∀ x : ℝ, (b - 3 ≤ x ∧ x ≤ b → (x^2 + b * x + c ≥ 21))) :
  b = -Real.sqrt 7 ∨ b = 4 := sorry

-- Problem 3: Proving the minimum value of m
theorem minimum_value_of_m (x : ℝ) (m : ℝ)
  (h1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x^2 + x + m ≥ x^2 - 2 * x + 4) :
  m = 4 := sorry

end quadratic_function_expr_value_of_b_minimum_value_of_m_l129_129984


namespace net_rate_of_pay_l129_129320

theorem net_rate_of_pay :
  ∀ (duration_travel : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (earnings_rate : ℝ) (gas_cost : ℝ),
  duration_travel = 3 → speed = 50 → fuel_efficiency = 30 → earnings_rate = 0.75 → gas_cost = 2.50 →
  (earnings_rate * speed * duration_travel - (speed * duration_travel / fuel_efficiency) * gas_cost) / duration_travel = 33.33 :=
by
  intros duration_travel speed fuel_efficiency earnings_rate gas_cost
  intros h1 h2 h3 h4 h5
  sorry

end net_rate_of_pay_l129_129320


namespace length_of_first_platform_l129_129069

theorem length_of_first_platform 
  (t1 t2 : ℝ) 
  (length_train : ℝ) 
  (length_second_platform : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (speed_eq : (t1 + length_train) / time1 = (length_second_platform + length_train) / time2) 
  (time1_eq : time1 = 15) 
  (time2_eq : time2 = 20) 
  (length_train_eq : length_train = 100) 
  (length_second_platform_eq: length_second_platform = 500) :
  t1 = 350 := 
  by 
  sorry

end length_of_first_platform_l129_129069


namespace sequence_sum_l129_129127

def arithmetic_seq (a₀ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₀ + n * d

def geometric_seq (b₀ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₀ * r^(n)

theorem sequence_sum :
  let a : ℕ → ℕ := arithmetic_seq 3 1
  let b : ℕ → ℕ := geometric_seq 1 2
  b (a 0) + b (a 1) + b (a 2) + b (a 3) = 60 :=
  by
    let a : ℕ → ℕ := arithmetic_seq 3 1
    let b : ℕ → ℕ := geometric_seq 1 2
    have h₀ : a 0 = 3 := by rfl
    have h₁ : a 1 = 4 := by rfl
    have h₂ : a 2 = 5 := by rfl
    have h₃ : a 3 = 6 := by rfl
    have hsum : b 3 + b 4 + b 5 + b 6 = 60 := by sorry
    exact hsum

end sequence_sum_l129_129127


namespace percentage_calculation_l129_129934

variable (x : Real)
variable (hx : x > 0)

theorem percentage_calculation : 
  ∃ p : Real, p = (0.18 * x) / (x + 20) * 100 :=
sorry

end percentage_calculation_l129_129934


namespace boys_without_calculators_l129_129959

theorem boys_without_calculators (total_boys total_students students_with_calculators girls_with_calculators : ℕ) 
    (h1 : total_boys = 20) 
    (h2 : total_students = 40) 
    (h3 : students_with_calculators = 30) 
    (h4 : girls_with_calculators = 18) : 
    (total_boys - (students_with_calculators - girls_with_calculators)) = 8 :=
by
  sorry

end boys_without_calculators_l129_129959


namespace problem1_problem2_problem3_problem4_l129_129080

-- Problem 1
theorem problem1 (x : ℝ) (h : x * (5 * x + 4) = 5 * x + 4) : x = -4 / 5 ∨ x = 1 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : -3 * x^2 + 22 * x - 24 = 0) : x = 6 ∨ x = 4 / 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x + 8) * (x + 1) = -12) : x = -4 ∨ x = -5 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (h : (3 * x + 2) * (x + 3) = x + 14) : x = -4 ∨ x = 2 / 3 := 
sorry

end problem1_problem2_problem3_problem4_l129_129080


namespace sets_satisfying_union_l129_129749

open Set

theorem sets_satisfying_union :
  {B : Set ℕ | {1, 2} ∪ B = {1, 2, 3}} = { {3}, {1, 3}, {2, 3}, {1, 2, 3} } :=
by
  sorry

end sets_satisfying_union_l129_129749


namespace range_k_domain_f_l129_129944

theorem range_k_domain_f :
  (∀ x : ℝ, x^2 - 6*k*x + k + 8 ≥ 0) ↔ (-8/9 ≤ k ∧ k ≤ 1) :=
sorry

end range_k_domain_f_l129_129944


namespace find_b_l129_129845

variables {a b : ℝ}

theorem find_b (h1 : (x - 3) * (x - a) = x^2 - b * x - 10) : b = -1/3 :=
  sorry

end find_b_l129_129845


namespace sum_of_integers_between_60_and_460_ending_in_2_is_10280_l129_129658

-- We define the sequence.
def endsIn2Seq : List Int := List.range' 62 (452 + 1 - 62) 10  -- Generates [62, 72, ..., 452]

-- The sum of the sequence.
def sumEndsIn2Seq : Int := endsIn2Seq.sum

-- The theorem to prove the desired sum.
theorem sum_of_integers_between_60_and_460_ending_in_2_is_10280 :
  sumEndsIn2Seq = 10280 := by
  -- Proof is omitted
  sorry

end sum_of_integers_between_60_and_460_ending_in_2_is_10280_l129_129658


namespace John_lost_socks_l129_129261

theorem John_lost_socks (initial_socks remaining_socks : ℕ) (H1 : initial_socks = 20) (H2 : remaining_socks = 14) : initial_socks - remaining_socks = 6 :=
by
-- Proof steps can be skipped
sorry

end John_lost_socks_l129_129261


namespace triangle_inequality_l129_129512

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
by
  sorry

end triangle_inequality_l129_129512


namespace power_mod_eq_remainder_l129_129157

theorem power_mod_eq_remainder (b m e : ℕ) (hb : b = 17) (hm : m = 23) (he : e = 2090) : 
  b^e % m = 12 := 
  by sorry

end power_mod_eq_remainder_l129_129157


namespace movie_theater_total_revenue_l129_129605

noncomputable def revenue_from_matinee_tickets : ℕ := 20 * 5 * 1 / 2 + 180 * 5
noncomputable def revenue_from_evening_tickets : ℕ := 150 * 12 * 9 / 10 + 75 * 12 * 75 / 100 + 75 * 12
noncomputable def revenue_from_3d_tickets : ℕ := 60 * 23 + 25 * 20 * 85 / 100 + 15 * 20
noncomputable def revenue_from_late_night_tickets : ℕ := 30 * 10 * 12 / 10 + 20 * 10

noncomputable def total_revenue : ℕ :=
  revenue_from_matinee_tickets + revenue_from_evening_tickets +
  revenue_from_3d_tickets + revenue_from_late_night_tickets

theorem movie_theater_total_revenue : total_revenue = 6810 := by
  sorry

end movie_theater_total_revenue_l129_129605


namespace value_of_B_minus_3_plus_A_l129_129765

theorem value_of_B_minus_3_plus_A (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 :=
by 
  sorry

end value_of_B_minus_3_plus_A_l129_129765


namespace find_positive_integer_l129_129455

theorem find_positive_integer (n : ℕ) (h1 : n % 14 = 0) (h2 : 676 ≤ n ∧ n ≤ 702) : n = 700 :=
sorry

end find_positive_integer_l129_129455


namespace area_of_square_l129_129346

theorem area_of_square (ABCD MN : ℝ) (h1 : 4 * (ABCD / 4) = ABCD) (h2 : MN = 3) : ABCD = 64 :=
by
  sorry

end area_of_square_l129_129346


namespace amoeba_count_after_week_l129_129394

-- Definition of the initial conditions
def amoeba_splits_daily (n : ℕ) : ℕ := 2^n

-- Theorem statement translating the problem to Lean
theorem amoeba_count_after_week : amoeba_splits_daily 7 = 128 :=
by
  sorry

end amoeba_count_after_week_l129_129394


namespace min_value_frac_l129_129966

open Real

theorem min_value_frac (a b : ℝ) (h1 : a + b = 1/2) (h2 : a > 0) (h3 : b > 0) :
    (4 / a + 1 / b) = 18 :=
sorry

end min_value_frac_l129_129966


namespace maximize_area_l129_129594

-- Define the variables and constants
variables {x y p : ℝ}

-- Define the conditions
def perimeter (x y p : ℝ) := (2 * x + 2 * y = p)
def area (x y : ℝ) := x * y

-- The theorem statement with conditions
theorem maximize_area (h : perimeter x y p) : x = y → x = p / 4 :=
by
  sorry

end maximize_area_l129_129594


namespace ratio_problem_l129_129090

theorem ratio_problem 
  (x y z w : ℚ) 
  (h1 : x / y = 12) 
  (h2 : z / y = 4) 
  (h3 : z / w = 3 / 4) : 
  w / x = 4 / 9 := 
  sorry

end ratio_problem_l129_129090


namespace symmetry_about_origin_l129_129796

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define the function v based on f and g
def v (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x * |g x|

-- The theorem statement
theorem symmetry_about_origin (f g : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even g) : 
  ∀ x : ℝ, v f g (-x) = -v f g x := 
by
  sorry

end symmetry_about_origin_l129_129796


namespace tan_beta_is_neg3_l129_129409

theorem tan_beta_is_neg3 (α β : ℝ) (h1 : Real.tan α = -2) (h2 : Real.tan (α + β) = 1) : Real.tan β = -3 := 
sorry

end tan_beta_is_neg3_l129_129409


namespace gabby_fruit_total_l129_129078

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l129_129078


namespace frog_eyes_count_l129_129044

def total_frog_eyes (a b c : ℕ) (eyesA eyesB eyesC : ℕ) : ℕ :=
  a * eyesA + b * eyesB + c * eyesC

theorem frog_eyes_count :
  let a := 2
  let b := 1
  let c := 3
  let eyesA := 2
  let eyesB := 3
  let eyesC := 4
  total_frog_eyes a b c eyesA eyesB eyesC = 19 := by
  sorry

end frog_eyes_count_l129_129044


namespace yellow_marbles_in_C_l129_129664

theorem yellow_marbles_in_C 
  (Y : ℕ)
  (conditionA : 4 - 2 ≠ 6)
  (conditionB : 6 - 1 ≠ 6)
  (conditionC1 : 3 > Y → 3 - Y = 6)
  (conditionC2 : Y > 3 → Y - 3 = 6) :
  Y = 9 :=
by
  sorry

end yellow_marbles_in_C_l129_129664


namespace local_minimum_of_reflected_function_l129_129359

noncomputable def f : ℝ → ℝ := sorry

theorem local_minimum_of_reflected_function (f : ℝ → ℝ) (x_0 : ℝ) (h1 : x_0 ≠ 0) (h2 : ∃ ε > 0, ∀ x, abs (x - x_0) < ε → f x ≤ f x_0) :
  ∃ δ > 0, ∀ x, abs (x - (-x_0)) < δ → -f (-x) ≥ -f (-x_0) :=
sorry

end local_minimum_of_reflected_function_l129_129359


namespace solution_set_l129_129267

/-- Definition: integer solutions (a, b, c) with c ≤ 94 that satisfy the equation -/
def int_solutions (a b c : ℤ) : Prop :=
  c ≤ 94 ∧ (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c

/-- Proposition: The integer solutions (a, b, c) that satisfy the equation are exactly these -/
theorem solution_set :
  { (a, b, c) : ℤ × ℤ × ℤ  | int_solutions a b c } =
  { (3, 7, 41), (4, 6, 44), (5, 5, 45), (6, 4, 44), (7, 3, 41) } :=
by
  sorry

end solution_set_l129_129267


namespace total_handshakes_tournament_l129_129742

/-- 
In a women's doubles tennis tournament, four teams of two women competed. After the tournament, 
each woman shook hands only once with each of the other players, except with her own partner.
Prove that the total number of unique handshakes is 24.
-/
theorem total_handshakes_tournament : 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  total_handshakes = 24 :=
by 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  have : total_handshakes = 24 := sorry
  exact this

end total_handshakes_tournament_l129_129742


namespace problem_l129_129242

theorem problem (x y : ℝ) 
  (h1 : |x + y - 9| = -(2 * x - y + 3) ^ 2) :
  x = 2 ∧ y = 7 :=
sorry

end problem_l129_129242


namespace pow_two_grows_faster_than_square_l129_129673

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square_l129_129673


namespace intersection_of_A_and_B_l129_129755

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l129_129755


namespace total_playtime_l129_129589

noncomputable def lena_playtime_minutes : ℕ := 210
noncomputable def brother_playtime_minutes (lena_playtime: ℕ) : ℕ := lena_playtime + 17
noncomputable def sister_playtime_minutes (brother_playtime: ℕ) : ℕ := 2 * brother_playtime

theorem total_playtime
  (lena_playtime : ℕ)
  (brother_playtime : ℕ)
  (sister_playtime : ℕ)
  (h_lena : lena_playtime = lena_playtime_minutes)
  (h_brother : brother_playtime = brother_playtime_minutes lena_playtime)
  (h_sister : sister_playtime = sister_playtime_minutes brother_playtime) :
  lena_playtime + brother_playtime + sister_playtime = 891 := 
  by sorry

end total_playtime_l129_129589


namespace convert_quadratic_to_general_form_l129_129521

theorem convert_quadratic_to_general_form
  (x : ℝ)
  (h : 3 * x * (x - 3) = 4) :
  3 * x ^ 2 - 9 * x - 4 = 0 :=
by
  sorry

end convert_quadratic_to_general_form_l129_129521


namespace D_time_to_complete_job_l129_129301

-- Let A_rate be the rate at which A works (jobs per hour)
-- Let D_rate be the rate at which D works (jobs per hour)
def A_rate : ℚ := 1 / 3
def combined_rate : ℚ := 1 / 2

-- We need to prove that D_rate, the rate at which D works alone, is 1/6 jobs per hour
def D_rate := 1 / 6

-- And thus, that D can complete the job in 6 hours
theorem D_time_to_complete_job :
  (A_rate + D_rate = combined_rate) → (1 / D_rate) = 6 :=
by
  sorry

end D_time_to_complete_job_l129_129301


namespace range_of_m_value_of_m_l129_129989

-- Defining the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- The condition for having real roots
def has_real_roots (a b c : ℝ) := b^2 - 4 * a * c ≥ 0

-- First part: Range of values for m
theorem range_of_m (m : ℝ) : has_real_roots 1 (-2) (m - 1) ↔ m ≤ 2 := sorry

-- Second part: Finding m when x₁² + x₂² = 6x₁x₂
theorem value_of_m 
  (x₁ x₂ m : ℝ) (h₁ : quadratic_eq 1 (-2) (m - 1) x₁) (h₂ : quadratic_eq 1 (-2) (m - 1) x₂) 
  (h_sum : x₁ + x₂ = 2) (h_prod : x₁ * x₂ = m - 1) (h_condition : x₁^2 + x₂^2 = 6 * (x₁ * x₂)) : 
  m = 3 / 2 := sorry

end range_of_m_value_of_m_l129_129989


namespace increasing_function_range_l129_129097

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x - 1 else x + 1

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 / 2 < a ∧ a ≤ 2) :=
sorry

end increasing_function_range_l129_129097


namespace bob_deli_total_cost_l129_129517

-- Definitions based on the problem's conditions
def sandwich_cost : ℕ := 5
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount_threshold : ℕ := 50
def discount_amount : ℕ := 10

-- The total initial cost without discount
def initial_total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The final cost after applying discount if applicable
def final_cost : ℕ :=
  if initial_total_cost > discount_threshold then
    initial_total_cost - discount_amount
  else
    initial_total_cost

-- Statement to prove
theorem bob_deli_total_cost : final_cost = 55 := by
  sorry

end bob_deli_total_cost_l129_129517


namespace find_y_l129_129380

theorem find_y 
  (x y : ℝ) 
  (h1 : (6 : ℝ) = (1/2 : ℝ) * x) 
  (h2 : y = (1/2 : ℝ) * 10) 
  (h3 : x * y = 60) 
: y = 5 := 
by 
  sorry

end find_y_l129_129380


namespace fraction_b_plus_c_over_a_l129_129313

variable (a b c d : ℝ)

theorem fraction_b_plus_c_over_a :
  (a ≠ 0) →
  (a * 4^3 + b * 4^2 + c * 4 + d = 0) →
  (a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) →
  (b + c) / a = -13 :=
by
  intros h₁ h₂ h₃ 
  sorry

end fraction_b_plus_c_over_a_l129_129313


namespace line_intersects_circle_l129_129116

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*y = 0) ∧ (y - 1 = k * (x - 1)) :=
sorry

end line_intersects_circle_l129_129116


namespace number_of_pages_in_book_l129_129281

-- Define the conditions using variables and hypotheses
variables (P : ℝ) (h1 : 0.30 * P = 150)

-- State the theorem to be proved
theorem number_of_pages_in_book : P = 500 :=
by
  -- Proof would go here, but we use sorry to skip it
  sorry

end number_of_pages_in_book_l129_129281


namespace sum_coordinates_is_60_l129_129497

theorem sum_coordinates_is_60 :
  let points := [(5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13), (5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7)]
  let x_coords_sum := (5 + Real.sqrt 91) + (5 - Real.sqrt 91) + (5 + Real.sqrt 91) + (5 - Real.sqrt 91)
  let y_coords_sum := 13 + 13 + 7 + 7
  x_coords_sum + y_coords_sum = 60 :=
by
  sorry

end sum_coordinates_is_60_l129_129497


namespace arithmetic_sequence_sum_properties_l129_129627

theorem arithmetic_sequence_sum_properties {S : ℕ → ℝ} {a : ℕ → ℝ} (d : ℝ)
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  let a6 := (S 6 - S 5)
  let a7 := (S 7 - S 6)
  (d = a7 - a6) →
  d < 0 ∧ S 12 > 0 ∧ ¬(∀ n, S n = S 11) ∧ abs a6 > abs a7 :=
by
  sorry

end arithmetic_sequence_sum_properties_l129_129627


namespace tan_two_x_is_odd_l129_129381

noncomputable def tan_two_x (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_two_x_is_odd :
  ∀ x : ℝ,
  (∀ k : ℤ, x ≠ (k * Real.pi / 2) + (Real.pi / 4)) →
  tan_two_x (-x) = -tan_two_x x :=
by
  sorry

end tan_two_x_is_odd_l129_129381


namespace correct_divisor_l129_129703

theorem correct_divisor (X : ℕ) (D : ℕ) (H1 : X = 24 * 87) (H2 : X / D = 58) : D = 36 :=
by
  sorry

end correct_divisor_l129_129703


namespace smallest_integer_solution_m_l129_129092

theorem smallest_integer_solution_m :
  (∃ x y m : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) →
  ∃ m : ℤ, (∀ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) ↔ m = -1 :=
by
  sorry

end smallest_integer_solution_m_l129_129092


namespace units_digit_27_mul_46_l129_129951

-- Define the function to calculate the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement: The units digit of 27 * 46 is 2
theorem units_digit_27_mul_46 : units_digit (27 * 46) = 2 :=
  sorry

end units_digit_27_mul_46_l129_129951


namespace dave_ice_cubes_total_l129_129201

theorem dave_ice_cubes_total : 
  let trayA_initial := 2
  let trayA_final := trayA_initial + 7
  let trayB := (1 / 3) * trayA_final
  let trayC := 2 * trayA_final
  trayA_final + trayB + trayC = 30 := by
  sorry

end dave_ice_cubes_total_l129_129201


namespace sum_of_vertices_l129_129832

theorem sum_of_vertices (vertices_rectangle : ℕ) (vertices_pentagon : ℕ) 
  (h_rect : vertices_rectangle = 4) (h_pent : vertices_pentagon = 5) : 
  vertices_rectangle + vertices_pentagon = 9 :=
by
  sorry

end sum_of_vertices_l129_129832


namespace f_g_of_3_l129_129062

def f (x : ℤ) : ℤ := 2 * x + 3
def g (x : ℤ) : ℤ := x^3 - 6

theorem f_g_of_3 : f (g 3) = 45 := by
  sorry

end f_g_of_3_l129_129062


namespace twice_a_minus_4_nonnegative_l129_129585

theorem twice_a_minus_4_nonnegative (a : ℝ) : 2 * a - 4 ≥ 0 ↔ 2 * a - 4 = 0 ∨ 2 * a - 4 > 0 := 
by
  sorry

end twice_a_minus_4_nonnegative_l129_129585


namespace area_enclosed_by_graph_l129_129725

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end area_enclosed_by_graph_l129_129725


namespace action_figures_more_than_books_proof_l129_129997

-- Definitions for the conditions
def books := 3
def action_figures_initial := 4
def action_figures_added := 2

-- Definition for the total action figures
def action_figures_total := action_figures_initial + action_figures_added

-- Definition for the number difference
def action_figures_more_than_books := action_figures_total - books

-- Proof statement
theorem action_figures_more_than_books_proof : action_figures_more_than_books = 3 :=
by
  sorry

end action_figures_more_than_books_proof_l129_129997


namespace house_selling_price_l129_129777

theorem house_selling_price
  (original_price : ℝ := 80000)
  (profit_rate : ℝ := 0.20)
  (commission_rate : ℝ := 0.05):
  original_price + (original_price * profit_rate) + (original_price * commission_rate) = 100000 := by
  sorry

end house_selling_price_l129_129777


namespace value_of_m_minus_n_l129_129432

variables {a b : ℕ}
variables {m n : ℤ}

def are_like_terms (m n : ℤ) : Prop :=
  (m - 2 = 4) ∧ (n + 7 = 4)

theorem value_of_m_minus_n (h : are_like_terms m n) : m - n = 9 :=
by
  sorry

end value_of_m_minus_n_l129_129432


namespace circle_radius_l129_129967

theorem circle_radius :
  ∃ r : ℝ, ∀ x y : ℝ, (x^2 - 8 * x + y^2 + 4 * y + 16 = 0) → r = 2 :=
sorry

end circle_radius_l129_129967


namespace find_a5_l129_129400

variables {a : ℕ → ℝ}  -- represent the arithmetic sequence

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom a3_a8_sum : a 3 + a 8 = 22
axiom a6_value : a 6 = 8
axiom arithmetic : is_arithmetic_sequence a

-- Target proof statement
theorem find_a5 (a : ℕ → ℝ) (arithmetic : is_arithmetic_sequence a) (a3_a8_sum : a 3 + a 8 = 22) (a6_value : a 6 = 8) : a 5 = 14 :=
by {
  sorry
}

end find_a5_l129_129400


namespace solve_quadratic_1_solve_quadratic_2_l129_129393

theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 - 7 * x - 1 = 0 ↔ 
  (x = (7 + Real.sqrt 57) / 4 ∨ x = (7 - Real.sqrt 57) / 4) := 
by 
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 3)^2 = 10 * x - 15 ↔ 
  (x = 3 / 2 ∨ x = 4) := 
by 
  sorry

end solve_quadratic_1_solve_quadratic_2_l129_129393


namespace convex_m_gons_two_acute_angles_l129_129916

noncomputable def count_convex_m_gons_with_two_acute_angles (m n : ℕ) (P : Finset ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem convex_m_gons_two_acute_angles {m n : ℕ} {P : Finset ℕ}
  (hP : P.card = 2 * n + 1)
  (hmn : 4 < m ∧ m < n) :
  count_convex_m_gons_with_two_acute_angles m n P = 
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
sorry

end convex_m_gons_two_acute_angles_l129_129916


namespace concert_duration_is_805_l129_129295

def hours_to_minutes (hours : ℕ) : ℕ :=
  hours * 60

def total_duration (hours : ℕ) (extra_minutes : ℕ) : ℕ :=
  hours_to_minutes hours + extra_minutes

theorem concert_duration_is_805 : total_duration 13 25 = 805 :=
by
  -- Proof skipped
  sorry

end concert_duration_is_805_l129_129295


namespace area_of_L_shaped_figure_l129_129516

theorem area_of_L_shaped_figure :
  let large_rect_area := 10 * 7
  let small_rect_area := 4 * 3
  large_rect_area - small_rect_area = 58 := by
  sorry

end area_of_L_shaped_figure_l129_129516


namespace truncated_cone_sphere_radius_l129_129408

noncomputable def radius_of_sphere (r1 r2 h : ℝ) : ℝ := 
  (Real.sqrt (h^2 + (r1 - r2)^2)) / 2

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 h : ℝ), r1 = 20 → r2 = 6 → h = 15 → radius_of_sphere r1 r2 h = Real.sqrt 421 / 2 :=
by
  intros r1 r2 h h1 h2 h3
  simp [radius_of_sphere]
  rw [h1, h2, h3]
  sorry

end truncated_cone_sphere_radius_l129_129408


namespace geometric_sequence_sum_div_l129_129052

theorem geometric_sequence_sum_div :
  ∀ {a : ℕ → ℝ} {q : ℝ},
  (∀ n, a (n + 1) = a n * q) →
  q = -1 / 3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros a q geometric_seq common_ratio
  sorry

end geometric_sequence_sum_div_l129_129052


namespace specific_certain_event_l129_129046

theorem specific_certain_event :
  ∀ (A B C D : Prop), 
    (¬ A) →
    (¬ B) →
    (¬ C) →
    D →
    D :=
by
  intros A B C D hA hB hC hD
  exact hD

end specific_certain_event_l129_129046


namespace gray_eyed_black_haired_students_l129_129018

theorem gray_eyed_black_haired_students (total_students : ℕ) 
  (green_eyed_red_haired : ℕ) (black_haired : ℕ) (gray_eyed : ℕ) 
  (h_total : total_students = 50)
  (h_green_eyed_red_haired : green_eyed_red_haired = 17)
  (h_black_haired : black_haired = 27)
  (h_gray_eyed : gray_eyed = 23) :
  ∃ (gray_eyed_black_haired : ℕ), gray_eyed_black_haired = 17 :=
by sorry

end gray_eyed_black_haired_students_l129_129018


namespace colorful_triangle_in_complete_graph_l129_129925

open SimpleGraph

theorem colorful_triangle_in_complete_graph (n : ℕ) (h : n ≥ 3) (colors : Fin n → Fin n → Fin (n - 1)) :
  ∃ (u v w : Fin n), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ colors u v ≠ colors v w ∧ colors v w ≠ colors w u ∧ colors w u ≠ colors u v :=
  sorry

end colorful_triangle_in_complete_graph_l129_129925


namespace sequence_general_term_l129_129596

theorem sequence_general_term (a : ℕ → ℤ) : 
  (∀ n, a n = (-1)^(n + 1) * (3 * n - 2)) ↔ 
  (a 1 = 1 ∧ a 2 = -4 ∧ a 3 = 7 ∧ a 4 = -10 ∧ a 5 = 13) :=
by
  sorry

end sequence_general_term_l129_129596


namespace no_distinct_ordered_pairs_l129_129277

theorem no_distinct_ordered_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) :
  (x^2 * y^2)^2 - 14 * x^2 * y^2 + 49 ≠ 0 :=
by
  sorry

end no_distinct_ordered_pairs_l129_129277


namespace order_of_means_l129_129744

variables (a b : ℝ)
-- a and b are positive and unequal
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : a ≠ b

-- Definitions of the means
noncomputable def AM : ℝ := (a + b) / 2
noncomputable def GM : ℝ := Real.sqrt (a * b)
noncomputable def HM : ℝ := (2 * a * b) / (a + b)
noncomputable def QM : ℝ := Real.sqrt ((a^2 + b^2) / 2)

-- The theorem to prove the order of the means
theorem order_of_means (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  QM a b > AM a b ∧ AM a b > GM a b ∧ GM a b > HM a b :=
sorry

end order_of_means_l129_129744


namespace quadratic_rewrite_l129_129326

theorem quadratic_rewrite  (a b c x : ℤ) (h : 25 * x^2 + 30 * x - 35 = 0) (hp : 25 * x^2 + 30 * x + 9 = (5 * x + 3) ^ 2)
(hc : c = 44) : a = 5 → b = 3 → a + b + c = 52 := 
by
  intro ha hb
  sorry

end quadratic_rewrite_l129_129326


namespace inscribed_circle_area_ratio_l129_129328

theorem inscribed_circle_area_ratio
  (R : ℝ) -- Radius of the original circle
  (r : ℝ) -- Radius of the inscribed circle
  (h : R = 3 * r) -- Relationship between the radii based on geometry problem
  :
  (π * R^2) / (π * r^2) = 9 :=
by sorry

end inscribed_circle_area_ratio_l129_129328


namespace least_positive_integer_divisible_by_three_primes_l129_129926

-- Define the next three distinct primes larger than 5
def prime1 := 7
def prime2 := 11
def prime3 := 13

-- Define the product of these primes
def prod := prime1 * prime2 * prime3

-- Statement of the theorem
theorem least_positive_integer_divisible_by_three_primes : prod = 1001 :=
by
  sorry

end least_positive_integer_divisible_by_three_primes_l129_129926


namespace smallest_prime_factor_in_C_l129_129551

def smallest_prime_factor_def (n : Nat) : Nat :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  sorry /- Define a function to find the smallest prime factor of a number n -/

def is_prime (p : Nat) : Prop :=
  2 ≤ p ∧ ∀ d : Nat, 2 ≤ d → d ∣ p → d = p

def in_set (x : Nat) : Prop :=
  x = 64 ∨ x = 66 ∨ x = 67 ∨ x = 68 ∨ x = 71

theorem smallest_prime_factor_in_C : ∀ x, in_set x → 
  (smallest_prime_factor_def x = 2 ∨ smallest_prime_factor_def x = 67 ∨ smallest_prime_factor_def x = 71) :=
by
  intro x hx
  cases hx with
  | inl hx  => sorry
  | inr hx  => sorry

end smallest_prime_factor_in_C_l129_129551


namespace smallest_b_l129_129786

theorem smallest_b (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : a * b * c = 360) : b = 3 :=
sorry

end smallest_b_l129_129786


namespace cricket_run_target_l129_129717

/-- Assuming the run rate in the first 15 overs and the required run rate for the next 35 overs to
reach a target, prove that the target number of runs is 275. -/
theorem cricket_run_target
  (run_rate_first_15 : ℝ := 3.2)
  (overs_first_15 : ℝ := 15)
  (run_rate_remaining_35 : ℝ := 6.485714285714286)
  (overs_remaining_35 : ℝ := 35)
  (runs_first_15 := run_rate_first_15 * overs_first_15)
  (runs_remaining_35 := run_rate_remaining_35 * overs_remaining_35)
  (target_runs := runs_first_15 + runs_remaining_35) :
  target_runs = 275 := by
  sorry

end cricket_run_target_l129_129717


namespace four_consecutive_integers_divisible_by_12_l129_129324

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l129_129324


namespace tenth_term_geom_seq_l129_129118

theorem tenth_term_geom_seq :
  let a := (5 : ℚ)
  let r := (4 / 3 : ℚ)
  let n := 10
  (a * r^(n - 1)) = (1310720 / 19683 : ℚ) :=
by
  sorry

end tenth_term_geom_seq_l129_129118


namespace number_of_boys_is_50_l129_129156

-- Definitions for conditions:
def total_students : Nat := 100
def boys (x : Nat) : Nat := x
def girls (x : Nat) : Nat := x

-- Theorem statement:
theorem number_of_boys_is_50 (x : Nat) (g : Nat) (h1 : x + g = total_students) (h2 : g = boys x) : boys x = 50 :=
by
  sorry

end number_of_boys_is_50_l129_129156


namespace age_of_other_man_l129_129770

-- Definitions of the given conditions
def average_age_increase (avg_men : ℕ → ℝ) (men_removed women_avg : ℕ) : Prop :=
  avg_men 8 + 2 = avg_men 6 + women_avg / 2

def one_man_age : ℕ := 24
def women_avg : ℕ := 30

-- Statement of the problem to prove
theorem age_of_other_man (avg_men : ℕ → ℝ) (other_man : ℕ) :
  average_age_increase avg_men 24 women_avg →
  other_man = 20 :=
sorry

end age_of_other_man_l129_129770


namespace trig_identity_l129_129450

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := 
by
  sorry

end trig_identity_l129_129450


namespace distinct_arrays_for_48_chairs_with_conditions_l129_129726

theorem distinct_arrays_for_48_chairs_with_conditions : 
  ∃ n : ℕ, n = 7 ∧ 
    ∀ (m r c : ℕ), 
      m = 48 ∧ 
      2 ≤ r ∧ 
      2 ≤ c ∧ 
      r * c = m ↔ 
      (∃ (k : ℕ), 
         ((k = (m / r) ∧ r * (m / r) = m) ∨ (k = (m / c) ∧ c * (m / c) = m)) ∧ 
         r * c = m) → 
    n = 7 :=
by
  sorry

end distinct_arrays_for_48_chairs_with_conditions_l129_129726


namespace largest_n_divisible_by_n_plus_10_l129_129343

theorem largest_n_divisible_by_n_plus_10 :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧ ∀ m : ℕ, ((m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 := 
sorry

end largest_n_divisible_by_n_plus_10_l129_129343


namespace alcohol_percentage_in_new_mixture_l129_129679

theorem alcohol_percentage_in_new_mixture :
  let afterShaveLotionVolume := 200
  let afterShaveLotionConcentration := 0.35
  let solutionVolume := 75
  let solutionConcentration := 0.15
  let waterVolume := 50
  let totalVolume := afterShaveLotionVolume + solutionVolume + waterVolume
  let alcoholVolume := (afterShaveLotionVolume * afterShaveLotionConcentration) + (solutionVolume * solutionConcentration)
  let alcoholPercentage := (alcoholVolume / totalVolume) * 100
  alcoholPercentage = 25 := 
  sorry

end alcohol_percentage_in_new_mixture_l129_129679


namespace solve_for_x_l129_129025

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l129_129025


namespace total_students_l129_129232

-- Given definitions
def basketball_count : ℕ := 7
def cricket_count : ℕ := 5
def both_count : ℕ := 3

-- The goal to prove
theorem total_students : basketball_count + cricket_count - both_count = 9 :=
by
  sorry

end total_students_l129_129232


namespace negation_proposition_l129_129174

theorem negation_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l129_129174


namespace problem1_problem2_problem3_l129_129550

/-- Problem 1: Calculate 25 * 26 * 8 and show it equals 5200 --/
theorem problem1 : 25 * 26 * 8 = 5200 := 
sorry

/-- Problem 2: Calculate 340 * 40 / 17 and show it equals 800 --/
theorem problem2 : 340 * 40 / 17 = 800 := 
sorry

/-- Problem 3: Calculate 440 * 15 + 480 * 15 + 79 * 15 + 15 and show it equals 15000 --/
theorem problem3 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := 
sorry

end problem1_problem2_problem3_l129_129550


namespace find_c_l129_129870

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem find_c (a b m c : ℝ) (h1 : ∀ x, f x a b ≥ 0)
  (h2 : ∀ x, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
by
  sorry

end find_c_l129_129870


namespace last_digit_of_sum_of_powers_l129_129677

theorem last_digit_of_sum_of_powers {a b c d : ℕ} 
  (h1 : a = 2311) (h2 : b = 5731) (h3 : c = 3467) (h4 : d = 6563) 
  : (a^b + c^d) % 10 = 4 := by
  sorry

end last_digit_of_sum_of_powers_l129_129677


namespace platform_length_l129_129644

noncomputable def length_of_platform (L : ℝ) : Prop :=
  ∃ (a : ℝ), 
    -- Train starts from rest
    (0 : ℝ) * 24 + (1/2) * a * 24^2 = 300 ∧
    -- Train crosses a platform in 39 seconds
    (0 : ℝ) * 39 + (1/2) * a * 39^2 = 300 + L ∧
    -- Constant acceleration found
    a = (25 : ℝ) / 24

-- Claim that length of platform should be 492.19 meters
theorem platform_length : length_of_platform 492.19 :=
sorry

end platform_length_l129_129644


namespace elisa_target_amount_l129_129806

def elisa_current_amount : ℕ := 37
def elisa_additional_amount : ℕ := 16

theorem elisa_target_amount : elisa_current_amount + elisa_additional_amount = 53 :=
by
  sorry

end elisa_target_amount_l129_129806


namespace evaluate_expression_at_x_eq_2_l129_129278

theorem evaluate_expression_at_x_eq_2 : (3 * 2 + 4)^2 - 10 * 2 = 80 := by
  sorry

end evaluate_expression_at_x_eq_2_l129_129278


namespace angle_B_value_l129_129561

noncomputable def degree_a (A : ℝ) : Prop := A = 30 ∨ A = 60

noncomputable def degree_b (A B : ℝ) : Prop := B = 3 * A - 60

theorem angle_B_value (A B : ℝ) 
  (h1 : B = 3 * A - 60)
  (h2 : A = 30 ∨ A = 60) :
  B = 30 ∨ B = 120 :=
by
  sorry

end angle_B_value_l129_129561


namespace tigers_in_zoo_l129_129125

-- Given definitions
def ratio_lions_tigers := 3 / 4
def number_of_lions := 21
def number_of_tigers := 28

-- Problem statement
theorem tigers_in_zoo : (number_of_lions : ℚ) / 3 * 4 = number_of_tigers := by
  sorry

end tigers_in_zoo_l129_129125


namespace simplify_expression_eq_l129_129539

noncomputable def simplified_expression (b : ℝ) : ℝ :=
  (Real.rpow (Real.rpow (b ^ 16) (1 / 8)) (1 / 4)) ^ 3 *
  (Real.rpow (Real.rpow (b ^ 16) (1 / 4)) (1 / 8)) ^ 3

theorem simplify_expression_eq (b : ℝ) (hb : 0 < b) :
  simplified_expression b = b ^ 3 :=
by sorry

end simplify_expression_eq_l129_129539


namespace ratio_of_final_to_initial_l129_129203

theorem ratio_of_final_to_initial (P : ℝ) (R : ℝ) (T : ℝ) (hR : R = 0.02) (hT : T = 50) :
  let SI := P * R * T
  let A := P + SI
  A / P = 2 :=
by
  sorry

end ratio_of_final_to_initial_l129_129203


namespace simplify_and_find_ratio_l129_129511

theorem simplify_and_find_ratio (m : ℤ) (c d : ℤ) (h : (5 * m + 15) / 5 = c * m + d) : d / c = 3 := by
  sorry

end simplify_and_find_ratio_l129_129511


namespace regular_polygon_is_octagon_l129_129732

theorem regular_polygon_is_octagon (n : ℕ) (interior_angle exterior_angle : ℝ) :
  interior_angle = 3 * exterior_angle ∧ interior_angle + exterior_angle = 180 → n = 8 :=
by
  intros h
  sorry

end regular_polygon_is_octagon_l129_129732


namespace sam_distance_when_meeting_l129_129145

theorem sam_distance_when_meeting :
  ∃ t : ℝ, (35 = 2 * t + 5 * t) ∧ (5 * t = 25) :=
by
  sorry

end sam_distance_when_meeting_l129_129145


namespace smallest_positive_integer_form_l129_129100

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d = 1205 * m + 27090 * n ∧ (∀ e, e > 0 → (∃ x y : ℤ, d = 1205 * x + 27090 * y) → d ≤ e) :=
sorry

end smallest_positive_integer_form_l129_129100


namespace no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l129_129485

theorem no_triangle_sum_of_any_two_angles_lt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) :=
by
  sorry

theorem no_triangle_sum_of_any_two_angles_gt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120) :=
by
  sorry

end no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l129_129485


namespace sandbox_area_l129_129706

def length : ℕ := 312
def width : ℕ := 146
def area : ℕ := 45552

theorem sandbox_area : length * width = area := by
  sorry

end sandbox_area_l129_129706


namespace rank_from_left_l129_129441

theorem rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h_total : total_students = 31) (h_right : rank_from_right = 21) : 
  rank_from_left = 11 := by
  sorry

end rank_from_left_l129_129441


namespace john_drinks_42_quarts_per_week_l129_129577

def gallons_per_day : ℝ := 1.5
def quarts_per_gallon : ℝ := 4
def days_per_week : ℕ := 7

theorem john_drinks_42_quarts_per_week :
  gallons_per_day * quarts_per_gallon * days_per_week = 42 := sorry

end john_drinks_42_quarts_per_week_l129_129577


namespace smallest_area_right_triangle_l129_129974

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l129_129974


namespace cube_with_holes_l129_129892

-- Definitions and conditions
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def depth_hole : ℝ := 1
def number_of_holes : ℕ := 6

-- Prove that the total surface area including inside surfaces is 144 square meters
def total_surface_area_including_inside_surfaces : ℝ :=
  let original_surface_area := 6 * (edge_length_cube ^ 2)
  let area_removed_per_hole := side_length_hole ^ 2
  let area_exposed_inside_per_hole := 2 * (side_length_hole * depth_hole) + area_removed_per_hole
  original_surface_area - number_of_holes * area_removed_per_hole + number_of_holes * area_exposed_inside_per_hole

-- Prove that the total volume of material removed is 24 cubic meters
def total_volume_removed : ℝ :=
  number_of_holes * (side_length_hole ^ 2 * depth_hole)

theorem cube_with_holes :
  total_surface_area_including_inside_surfaces = 144 ∧ total_volume_removed = 24 :=
by
  sorry

end cube_with_holes_l129_129892


namespace div_sub_eq_l129_129031

theorem div_sub_eq : 0.24 / 0.004 - 0.1 = 59.9 := by
  sorry

end div_sub_eq_l129_129031


namespace compare_solutions_l129_129212

theorem compare_solutions 
  (c d p q : ℝ) 
  (hc : c ≠ 0) 
  (hp : p ≠ 0) :
  (-d / c) < (-q / p) ↔ (q / p) < (d / c) :=
by
  sorry

end compare_solutions_l129_129212


namespace cos_omega_x_3_zeros_interval_l129_129245

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end cos_omega_x_3_zeros_interval_l129_129245


namespace cos_double_angle_l129_129718

theorem cos_double_angle (α : ℝ) (h : Real.sin ((Real.pi / 6) + α) = 1 / 3) :
  Real.cos ((2 * Real.pi / 3) - 2 * α) = -7 / 9 := by
  sorry

end cos_double_angle_l129_129718


namespace max_acute_angles_l129_129253

theorem max_acute_angles (n : ℕ) : 
  ∃ k : ℕ, k ≤ (2 * n / 3) + 1 :=
sorry

end max_acute_angles_l129_129253


namespace triangle_property_l129_129377

theorem triangle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_perimeter : a + b + c = 12) (h_inradius : 2 * (a + b + c) = 24) :
    ¬((a^2 + b^2 = c^2) ∨ (a^2 + b^2 > c^2) ∨ (c^2 > a^2 + b^2)) := 
sorry

end triangle_property_l129_129377


namespace find_a11_l129_129401

variable (a : ℕ → ℝ)

axiom geometric_seq (a : ℕ → ℝ) (r : ℝ) : ∀ n, a (n + 1) = a n * r

variable (r : ℝ)
variable (h3 : a 3 = 4)
variable (h7 : a 7 = 12)

theorem find_a11 : a 11 = 36 := by
  sorry

end find_a11_l129_129401


namespace infections_first_wave_l129_129487

theorem infections_first_wave (x : ℕ)
  (h1 : 4 * x * 14 = 21000) : x = 375 :=
  sorry

end infections_first_wave_l129_129487


namespace total_hours_played_l129_129994

-- Definitions based on conditions
def Nathan_hours_per_day : ℕ := 3
def Nathan_weeks : ℕ := 2
def days_per_week : ℕ := 7

def Tobias_hours_per_day : ℕ := 5
def Tobias_weeks : ℕ := 1

-- Calculating total hours
def Nathan_total_hours := Nathan_hours_per_day * days_per_week * Nathan_weeks
def Tobias_total_hours := Tobias_hours_per_day * days_per_week * Tobias_weeks

-- Theorem statement
theorem total_hours_played : Nathan_total_hours + Tobias_total_hours = 77 := by
  -- Proof would go here
  sorry

end total_hours_played_l129_129994


namespace sum_of_zeros_of_even_function_is_zero_l129_129689

open Function

theorem sum_of_zeros_of_even_function_is_zero (f : ℝ → ℝ) (hf: Even f) (hx: ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) :
  x1 + x2 + x3 + x4 = 0 := by
  sorry

end sum_of_zeros_of_even_function_is_zero_l129_129689


namespace ellen_bought_chairs_l129_129804

-- Define the conditions
def cost_per_chair : ℕ := 15
def total_amount_spent : ℕ := 180

-- State the theorem to be proven
theorem ellen_bought_chairs :
  (total_amount_spent / cost_per_chair = 12) := 
sorry

end ellen_bought_chairs_l129_129804


namespace find_larger_number_l129_129230

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1000) 
  (h2 : L = 10 * S + 10) : 
  L = 1110 :=
sorry

end find_larger_number_l129_129230


namespace boat_travel_difference_l129_129875

-- Define the speeds
variables (a b : ℝ) (ha : a > b)

-- Define the travel times
def downstream_time := 3
def upstream_time := 2

-- Define the distances
def downstream_distance := downstream_time * (a + b)
def upstream_distance := upstream_time * (a - b)

-- Prove the mathematical statement
theorem boat_travel_difference : downstream_distance a b - upstream_distance a b = a + 5 * b := by
  -- sorry can be used to skip the proof
  sorry

end boat_travel_difference_l129_129875


namespace no_real_values_of_p_for_equal_roots_l129_129239

theorem no_real_values_of_p_for_equal_roots (p : ℝ) : ¬ ∃ (p : ℝ), (p^2 - 2*p + 5 = 0) :=
by sorry

end no_real_values_of_p_for_equal_roots_l129_129239


namespace arithmetic_sequence_root_sum_l129_129670

theorem arithmetic_sequence_root_sum (a : ℕ → ℝ) (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) 
    (h_roots : (a 3) * (a 8) + 3 * (a 3) + 3 * (a 8) - 18 = 0) : a 5 + a 6 = 3 := by
  sorry

end arithmetic_sequence_root_sum_l129_129670


namespace least_possible_perimeter_l129_129478

theorem least_possible_perimeter (x : ℕ) (h1 : 27 < x) (h2 : x < 75) :
  24 + 51 + x = 103 :=
by
  sorry

end least_possible_perimeter_l129_129478


namespace jenny_correct_number_l129_129727

theorem jenny_correct_number (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 :=
by
  sorry

end jenny_correct_number_l129_129727


namespace problem_one_problem_two_l129_129161

noncomputable def f (x m : ℝ) : ℝ := x^2 - (m-1) * x + 2 * m

theorem problem_one (m : ℝ) : (∀ x : ℝ, 0 < x → f x m > 0) ↔ (-2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5) :=
by
  sorry

theorem problem_two (m : ℝ) : (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x m = 0) ↔ (m ∈ Set.Ioo (-2 : ℝ) 0) :=
by
  sorry

end problem_one_problem_two_l129_129161


namespace find_C_and_D_l129_129871

noncomputable def C : ℚ := 15 / 8
noncomputable def D : ℚ := 17 / 8

theorem find_C_and_D (x : ℚ) (h₁ : x ≠ 9) (h₂ : x ≠ -7) :
  (4 * x - 6) / ((x - 9) * (x + 7)) = C / (x - 9) + D / (x + 7) :=
by sorry

end find_C_and_D_l129_129871


namespace original_price_of_sarees_l129_129144

theorem original_price_of_sarees (P : ℝ):
  (0.80 * P) * 0.95 = 152 → P = 200 :=
by
  intro h1
  -- You can omit the proof here because the task requires only the statement.
  sorry

end original_price_of_sarees_l129_129144


namespace cost_of_5_spoons_l129_129893

theorem cost_of_5_spoons (cost_per_set : ℕ) (num_spoons_per_set : ℕ) (num_spoons_needed : ℕ)
  (h1 : cost_per_set = 21) (h2 : num_spoons_per_set = 7) (h3 : num_spoons_needed = 5) :
  (cost_per_set / num_spoons_per_set) * num_spoons_needed = 15 :=
by
  sorry

end cost_of_5_spoons_l129_129893


namespace project_completion_days_l129_129110

theorem project_completion_days 
  (total_mandays : ℕ)
  (initial_workers : ℕ)
  (leaving_workers : ℕ)
  (remaining_workers : ℕ)
  (days_total : ℕ) :
  total_mandays = 200 →
  initial_workers = 10 →
  leaving_workers = 4 →
  remaining_workers = 6 →
  days_total = 40 :=
by
  intros h0 h1 h2 h3
  sorry

end project_completion_days_l129_129110


namespace red_flowers_count_l129_129103

-- Let's define the given conditions
def total_flowers : ℕ := 10
def white_flowers : ℕ := 2
def blue_percentage : ℕ := 40

-- Calculate the number of blue flowers
def blue_flowers : ℕ := (blue_percentage * total_flowers) / 100

-- The property we want to prove is the number of red flowers
theorem red_flowers_count :
  total_flowers - (blue_flowers + white_flowers) = 4 :=
by
  sorry

end red_flowers_count_l129_129103


namespace functional_equation_solution_l129_129483

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)) :
  (f = fun x => 0) ∨ (f = id) ∨ (f = fun x => -x) :=
sorry

end functional_equation_solution_l129_129483


namespace number_of_even_divisors_of_factorial_eight_l129_129855

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end number_of_even_divisors_of_factorial_eight_l129_129855


namespace polygon_interior_angle_l129_129310

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end polygon_interior_angle_l129_129310


namespace alpha_epsilon_time_difference_l129_129621

def B := 100
def M := 120
def A := B - 10

theorem alpha_epsilon_time_difference : M - A = 30 := by
  sorry

end alpha_epsilon_time_difference_l129_129621


namespace cubic_sum_identity_l129_129615

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 40) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 637 :=
by
  sorry

end cubic_sum_identity_l129_129615


namespace sodium_thiosulfate_properties_l129_129572

def thiosulfate_structure : Type := sorry
-- Define the structure of S2O3^{2-} with S-S bond
def has_s_s_bond (ion : thiosulfate_structure) : Prop := sorry
-- Define the formation reaction
def formed_by_sulfite_reaction (ion : thiosulfate_structure) : Prop := sorry

theorem sodium_thiosulfate_properties :
  ∃ (ion : thiosulfate_structure),
    has_s_s_bond ion ∧ formed_by_sulfite_reaction ion :=
by
  sorry

end sodium_thiosulfate_properties_l129_129572


namespace polar_conversion_equiv_l129_129576

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_equiv : polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  sorry

end polar_conversion_equiv_l129_129576


namespace candy_eaten_l129_129579

theorem candy_eaten (x : ℕ) (initial_candy eaten_more remaining : ℕ) (h₁ : initial_candy = 22) (h₂ : eaten_more = 5) (h₃ : remaining = 8) (h₄ : initial_candy - x - eaten_more = remaining) : x = 9 :=
by
  -- proof
  sorry

end candy_eaten_l129_129579


namespace min_value_fraction_l129_129032

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : a < (2 / 3) * b) (h3 : c ≥ b^2 / (3 * a)) : 
  ∃ x : ℝ, (∀ y : ℝ, y ≥ x → y ≥ 1) ∧ (x = 1) :=
by
  sorry

end min_value_fraction_l129_129032


namespace algebraic_expression_defined_iff_l129_129053

theorem algebraic_expression_defined_iff (x : ℝ) : (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end algebraic_expression_defined_iff_l129_129053


namespace sum_of_sequences_l129_129580

def sequence1 := [2, 14, 26, 38, 50]
def sequence2 := [12, 24, 36, 48, 60]
def sequence3 := [5, 15, 25, 35, 45]

theorem sum_of_sequences :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := 
by 
  sorry

end sum_of_sequences_l129_129580


namespace proof_problem_l129_129929

open Real

noncomputable def p : Prop := ∃ x : ℝ, x - 2 > log x / log 10
noncomputable def q : Prop := ∀ x : ℝ, x^2 > 0

theorem proof_problem :
  (p ∧ ¬q) := by
  sorry

end proof_problem_l129_129929


namespace starWars_earnings_correct_l129_129613

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end starWars_earnings_correct_l129_129613


namespace number_division_l129_129570

theorem number_division (n : ℕ) (h1 : n / 25 = 5) (h2 : n % 25 = 2) : n = 127 :=
by
  sorry

end number_division_l129_129570


namespace rectangle_perimeter_l129_129635

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter_l129_129635


namespace product_and_quotient_l129_129066

theorem product_and_quotient : (16 * 0.0625 / 4 * 0.5 * 2) = (1 / 4) :=
by
  -- The proof steps would go here
  sorry

end product_and_quotient_l129_129066


namespace range_of_x_l129_129430

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_x (x : ℝ) (h₀ : -1 < x ∧ x < 1) (h₁ : f 0 = 0) (h₂ : f (1 - x) + f (1 - x^2) < 0) :
  1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end range_of_x_l129_129430


namespace sum_of_arithmetic_sequence_15_terms_l129_129920

/-- An arithmetic sequence starts at 3 and has a common difference of 4.
    Prove that the sum of the first 15 terms of this sequence is 465. --/
theorem sum_of_arithmetic_sequence_15_terms :
  let a := 3
  let d := 4
  let n := 15
  let aₙ := a + (n - 1) * d
  (n / 2) * (a + aₙ) = 465 :=
by
  sorry

end sum_of_arithmetic_sequence_15_terms_l129_129920


namespace rectangle_area_increase_l129_129543

-- Definitions to match the conditions
variables {l w : ℝ}

-- The statement 
theorem rectangle_area_increase (h1 : l > 0) (h2 : w > 0) :
  (((1.15 * l) * (1.2 * w) - (l * w)) / (l * w)) * 100 = 38 :=
by
  sorry

end rectangle_area_increase_l129_129543


namespace not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l129_129224

noncomputable def f (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem not_bounded_on_neg_infty_zero (a x : ℝ) (h : a = 1) : 
  ¬ ∃ M > 0, ∀ x < 0, |f a x| ≤ M :=
by sorry

-- Problem (2)
theorem range_of_a_bounded_on_zero_infty (a : ℝ) : 
  (∀ x ≥ 0, |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l129_129224


namespace reduce_consumption_percentage_l129_129462

theorem reduce_consumption_percentage :
  ∀ (current_rate old_rate : ℝ), 
  current_rate = 20 → 
  old_rate = 16 → 
  ((current_rate - old_rate) / old_rate * 100) = 25 :=
by
  intros current_rate old_rate h_current h_old
  sorry

end reduce_consumption_percentage_l129_129462


namespace number_of_correct_answers_l129_129859

-- We define variables C (number of correct answers) and W (number of wrong answers).
variables (C W : ℕ)

-- Define the conditions given in the problem.
def conditions :=
  C + W = 75 ∧ 4 * C - W = 125

-- Define the theorem which states that the number of correct answers is 40.
theorem number_of_correct_answers
  (h : conditions C W) :
  C = 40 :=
sorry

end number_of_correct_answers_l129_129859


namespace problem1_problem2_l129_129754

-- Problem 1: Prove that (-11) + 8 + (-4) = -7
theorem problem1 : (-11) + 8 + (-4) = -7 := by
  sorry

-- Problem 2: Prove that -1^2023 - |1 - 1/3| * (-3/2)^2 = -(5/2)
theorem problem2 : (-1 : ℚ)^2023 - abs (1 - 1/3) * (-3/2)^2 = -(5/2) := by
  sorry

end problem1_problem2_l129_129754


namespace range_of_a_l129_129980

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ a > 3 ∨ a < -3 :=
by
  sorry

end range_of_a_l129_129980


namespace dagger_evaluation_l129_129546

def dagger (a b : ℚ) : ℚ :=
match a, b with
| ⟨m, n, _, _⟩, ⟨p, q, _, _⟩ => (m * p : ℚ) * (q / n : ℚ)

theorem dagger_evaluation : dagger (3/7) (11/4) = 132/7 := by
  sorry

end dagger_evaluation_l129_129546


namespace area_of_trapezoid_l129_129327

theorem area_of_trapezoid
  (r : ℝ)
  (AD BC : ℝ)
  (center_on_base : Bool)
  (height : ℝ)
  (area : ℝ)
  (inscribed_circle : r = 6)
  (base_AD : AD = 8)
  (base_BC : BC = 4)
  (K_height : height = 4 * Real.sqrt 2)
  (calc_area : area = (1 / 2) * (AD + BC) * height)
  : area = 32 * Real.sqrt 2 := by
  sorry

end area_of_trapezoid_l129_129327


namespace diana_principal_charge_l129_129238

theorem diana_principal_charge :
  ∃ P : ℝ, P > 0 ∧ (P + P * 0.06 = 63.6) ∧ P = 60 :=
by
  use 60
  sorry

end diana_principal_charge_l129_129238


namespace merchant_markup_percentage_l129_129869

theorem merchant_markup_percentage (CP MP SP : ℝ) (x : ℝ) (H_CP : CP = 100)
  (H_MP : MP = CP + (x / 100 * CP)) 
  (H_SP_discount : SP = MP * 0.80) 
  (H_SP_profit : SP = CP * 1.12) : 
  x = 40 := 
by
  sorry

end merchant_markup_percentage_l129_129869


namespace kenya_peanuts_eq_133_l129_129438

def num_peanuts_jose : Nat := 85
def additional_peanuts_kenya : Nat := 48

def peanuts_kenya (jose_peanuts : Nat) (additional_peanuts : Nat) : Nat :=
  jose_peanuts + additional_peanuts

theorem kenya_peanuts_eq_133 : peanuts_kenya num_peanuts_jose additional_peanuts_kenya = 133 := by
  sorry

end kenya_peanuts_eq_133_l129_129438


namespace circle_equations_l129_129780

-- Given conditions: the circle passes through points O(0,0), A(1,1), B(4,2)
-- Prove the general equation of the circle and the standard equation 

theorem circle_equations : 
  ∃ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
                      (x, y) = (0, 0) ∨ (x, y) = (1, 1) ∨ (x, y) = (4, 2)) ∧
  (D = -8) ∧ (E = 6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y = 0 ↔ (x - 4)^2 + (y + 3)^2 = 25) :=
sorry

end circle_equations_l129_129780


namespace parallelogram_probability_l129_129061

theorem parallelogram_probability (P Q R S : ℝ × ℝ) 
  (hP : P = (4, 2)) 
  (hQ : Q = (-2, -2)) 
  (hR : R = (-6, -6)) 
  (hS : S = (0, -2)) :
  let parallelogram_area := 24 -- given the computed area based on provided geometry
  let divided_area := parallelogram_area / 2
  let not_above_x_axis_area := divided_area
  (not_above_x_axis_area / parallelogram_area) = (1 / 2) :=
by
  sorry

end parallelogram_probability_l129_129061


namespace tank_dimension_l129_129713

theorem tank_dimension (cost_per_sf : ℝ) (total_cost : ℝ) (length1 length3 : ℝ) (surface_area : ℝ) (dimension : ℝ) :
  cost_per_sf = 20 ∧ total_cost = 1520 ∧ 
  length1 = 4 ∧ length3 = 2 ∧ 
  surface_area = total_cost / cost_per_sf ∧
  12 * dimension + 16 = surface_area → dimension = 5 :=
by
  intro h
  obtain ⟨hcps, htac, hl1, hl3, hsa, heq⟩ := h
  sorry

end tank_dimension_l129_129713


namespace James_vegetable_intake_in_third_week_l129_129471

noncomputable def third_week_vegetable_intake : ℝ :=
  let asparagus_per_day_first_week : ℝ := 0.25
  let broccoli_per_day_first_week : ℝ := 0.25
  let cauliflower_per_day_first_week : ℝ := 0.5

  let asparagus_per_day_second_week := 2 * asparagus_per_day_first_week
  let broccoli_per_day_second_week := 3 * broccoli_per_day_first_week
  let cauliflower_per_day_second_week := cauliflower_per_day_first_week * 1.75
  let spinach_per_day_second_week : ℝ := 0.5
  
  let daily_intake_second_week := asparagus_per_day_second_week +
                                  broccoli_per_day_second_week +
                                  cauliflower_per_day_second_week +
                                  spinach_per_day_second_week
  
  let kale_per_day_third_week : ℝ := 0.5
  let zucchini_per_day_third_week : ℝ := 0.15
  
  let daily_intake_third_week := asparagus_per_day_second_week +
                                 broccoli_per_day_second_week +
                                 cauliflower_per_day_second_week +
                                 spinach_per_day_second_week +
                                 kale_per_day_third_week +
                                 zucchini_per_day_third_week
  
  daily_intake_third_week * 7

theorem James_vegetable_intake_in_third_week : 
  third_week_vegetable_intake = 22.925 :=
  by
    sorry

end James_vegetable_intake_in_third_week_l129_129471


namespace problem_statement_l129_129477

-- Definitions and conditions
def f (x : ℝ) : ℝ := x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

-- Given the specific condition
def f_symmetric_about_1 : Prop := is_symmetric_about f 1

-- We need to prove that this implies g(x) = 3x - 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f_symmetric_about_1 → ∀ x, g x = 3 * x - 2 := 
by
  intro h
  sorry -- Detailed proof is omitted

end problem_statement_l129_129477


namespace percentage_y_more_than_z_l129_129741

theorem percentage_y_more_than_z :
  ∀ (P y x k : ℕ),
    P = 200 →
    740 = x + y + P →
    x = (5 / 4) * y →
    y = P * (1 + k / 100) →
    k = 20 :=
by
  sorry

end percentage_y_more_than_z_l129_129741


namespace total_distance_travelled_l129_129439

theorem total_distance_travelled (D : ℝ) (h1 : (D / 2) / 30 + (D / 2) / 25 = 11) : D = 150 :=
sorry

end total_distance_travelled_l129_129439


namespace probability_of_same_number_l129_129588

theorem probability_of_same_number (m n : ℕ) 
  (hb : m < 250 ∧ m % 20 = 0) 
  (bb : n < 250 ∧ n % 30 = 0) : 
  (∀ (b : ℕ), b < 250 ∧ b % 60 = 0 → ∃ (m n : ℕ), ((m < 250 ∧ m % 20 = 0) ∧ (n < 250 ∧ n % 30 = 0)) → (m = n)) :=
sorry

end probability_of_same_number_l129_129588


namespace min_value_112_l129_129682

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d +
                                c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c)

theorem min_value_112 (a b c d : ℝ) (h : a + b + c + d = 8) : min_value_expr a b c d = 112 :=
  sorry

end min_value_112_l129_129682


namespace min_value_abc2_l129_129107

variables (a b c d : ℝ)

def condition_1 : Prop := a + b = 9 / (c - d)
def condition_2 : Prop := c + d = 25 / (a - b)

theorem min_value_abc2 :
  condition_1 a b c d → condition_2 a b c d → (a^2 + b^2 + c^2 + d^2) = 34 :=
by
  intros h1 h2
  sorry

end min_value_abc2_l129_129107


namespace find_d_l129_129840

-- Given conditions
def line_eq (x y : ℚ) : Prop := y = (3 * x - 4) / 4

def parametrized_eq (v d : ℚ × ℚ) (t x y : ℚ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2)

def distance_eq (x y : ℚ) (t : ℚ) : Prop :=
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = t * t

-- The proof problem statement
theorem find_d (d : ℚ × ℚ) 
  (h_d : d = (7/2, 5/2)) :
  ∀ (x y t : ℚ) (v : ℚ × ℚ) (h_v : v = (3, 1)),
    (x ≥ 3) → 
    line_eq x y → 
    parametrized_eq v d t x y → 
    distance_eq x y t → 
    d = (7/2, 5/2) := 
by 
  intros;
  sorry


end find_d_l129_129840


namespace forty_percent_of_jacquelines_candy_bars_is_120_l129_129948

-- Define the number of candy bars Fred has
def fred_candy_bars : ℕ := 12

-- Define the number of candy bars Uncle Bob has
def uncle_bob_candy_bars : ℕ := fred_candy_bars + 6

-- Define the total number of candy bars Fred and Uncle Bob have together
def total_candy_bars : ℕ := fred_candy_bars + uncle_bob_candy_bars

-- Define the number of candy bars Jacqueline has
def jacqueline_candy_bars : ℕ := 10 * total_candy_bars

-- Define the number of candy bars that is 40% of Jacqueline's total
def forty_percent_jacqueline_candy_bars : ℕ := (40 * jacqueline_candy_bars) / 100

-- The statement to prove
theorem forty_percent_of_jacquelines_candy_bars_is_120 :
  forty_percent_jacqueline_candy_bars = 120 :=
sorry

end forty_percent_of_jacquelines_candy_bars_is_120_l129_129948


namespace remainder_when_divided_by_15_l129_129257

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end remainder_when_divided_by_15_l129_129257


namespace groupD_can_form_triangle_l129_129194

def groupA := (5, 7, 12)
def groupB := (7, 7, 15)
def groupC := (6, 9, 16)
def groupD := (6, 8, 12)

def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem groupD_can_form_triangle : canFormTriangle 6 8 12 :=
by
  -- Proof of the above theorem will follow the example from the solution.
  sorry

end groupD_can_form_triangle_l129_129194


namespace geometric_common_ratio_eq_three_l129_129729

theorem geometric_common_ratio_eq_three 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h_arithmetic_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0) 
  (h_geom_seq : (a 2 + 2 * d) ^ 2 = (a 2 + d) * (a 2 + 5 * d)) : 
  (a 3) / (a 2) = 3 :=
by 
  sorry

end geometric_common_ratio_eq_three_l129_129729


namespace wheres_waldo_books_published_l129_129692

theorem wheres_waldo_books_published (total_minutes : ℕ) (minutes_per_puzzle : ℕ) (puzzles_per_book : ℕ)
  (h1 : total_minutes = 1350) (h2 : minutes_per_puzzle = 3) (h3 : puzzles_per_book = 30) :
  total_minutes / minutes_per_puzzle / puzzles_per_book = 15 :=
by
  sorry

end wheres_waldo_books_published_l129_129692


namespace product_of_digits_l129_129385

theorem product_of_digits (n A B : ℕ) (h1 : n % 6 = 0) (h2 : A + B = 12) (h3 : n = 10 * A + B) : 
  (A * B = 32 ∨ A * B = 36) :=
by 
  sorry

end product_of_digits_l129_129385


namespace arc_PQ_circumference_l129_129918

-- Definitions based on the identified conditions
def radius : ℝ := 24
def angle_PRQ : ℝ := 90

-- The theorem to prove based on the question and correct answer
theorem arc_PQ_circumference : 
  angle_PRQ = 90 → 
  ∃ arc_length : ℝ, arc_length = (2 * Real.pi * radius) / 4 ∧ arc_length = 12 * Real.pi :=
by
  sorry

end arc_PQ_circumference_l129_129918


namespace sum_of_A_B_C_l129_129030

theorem sum_of_A_B_C (A B C : ℕ) (h_pos_A : 0 < A) (h_pos_B : 0 < B) (h_pos_C : 0 < C) (h_rel_prime : Nat.gcd A (Nat.gcd B C) = 1) (h_eq : A * Real.log 3 / Real.log 180 + B * Real.log 5 / Real.log 180 = C) : A + B + C = 4 :=
sorry

end sum_of_A_B_C_l129_129030


namespace determine_a_l129_129904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2 ^ x + 1)

theorem determine_a (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = 1 / 2 :=
by
  sorry

end determine_a_l129_129904


namespace smallest_n_Sn_gt_2023_l129_129309

open Nat

theorem smallest_n_Sn_gt_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 4) →
  (∀ n : ℕ, n > 0 → a n + a (n + 1) = 4 * n + 2) →
  (∀ m : ℕ, S m = if m % 2 = 0 then m ^ 2 + m else m ^ 2 + m + 2) →
  ∃ n : ℕ, S n > 2023 ∧ ∀ k : ℕ, k < n → S k ≤ 2023 :=
sorry

end smallest_n_Sn_gt_2023_l129_129309


namespace additional_charge_is_correct_l129_129881

noncomputable def additional_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) (segment_length : ℝ) : ℝ :=
  let segments := total_distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  charge_for_distance / segments

theorem additional_charge_is_correct :
  additional_charge_per_segment 2.0 3.6 5.15 (2/5) = 0.35 :=
by
  sorry

end additional_charge_is_correct_l129_129881


namespace smallest_d_l129_129941

theorem smallest_d (d : ℝ) : 
  (∃ d, 2 * d = Real.sqrt ((4 * Real.sqrt 3) ^ 2 + (d + 4) ^ 2)) →
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
by
  sorry

end smallest_d_l129_129941


namespace problem_statement_l129_129831

theorem problem_statement (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1)
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) :
  a ^ 2011 * b ^ 2011 + c ^ 2011 = 1 / 2011^2011 :=
by
  sorry

end problem_statement_l129_129831


namespace smallest_hiding_number_l129_129626

/-- Define the concept of "hides" -/
def hides (A B : ℕ) : Prop :=
  ∃ (remove : ℕ → ℕ), remove A = B

/-- The smallest natural number that hides all numbers from 2000 to 2021 is 20012013456789 -/
theorem smallest_hiding_number : hides 20012013456789 2000 ∧ hides 20012013456789 2001 ∧ hides 20012013456789 2002 ∧
    hides 20012013456789 2003 ∧ hides 20012013456789 2004 ∧ hides 20012013456789 2005 ∧ hides 20012013456789 2006 ∧
    hides 20012013456789 2007 ∧ hides 20012013456789 2008 ∧ hides 20012013456789 2009 ∧ hides 20012013456789 2010 ∧
    hides 20012013456789 2011 ∧ hides 20012013456789 2012 ∧ hides 20012013456789 2013 ∧ hides 20012013456789 2014 ∧
    hides 20012013456789 2015 ∧ hides 20012013456789 2016 ∧ hides 20012013456789 2017 ∧ hides 20012013456789 2018 ∧
    hides 20012013456789 2019 ∧ hides 20012013456789 2020 ∧ hides 20012013456789 2021 :=
by
  sorry

end smallest_hiding_number_l129_129626


namespace f_analytical_expression_l129_129252

noncomputable def f (x : ℝ) : ℝ := (2^(x + 1) - 2^(-x)) / 3

theorem f_analytical_expression :
  ∀ x : ℝ, f (-x) + 2 * f x = 2^x :=
by
  sorry

end f_analytical_expression_l129_129252


namespace carla_marble_purchase_l129_129623

variable (started_with : ℕ) (now_has : ℕ) (bought : ℕ)

theorem carla_marble_purchase (h1 : started_with = 53) (h2 : now_has = 187) : bought = 134 := by
  sorry

end carla_marble_purchase_l129_129623


namespace min_value_xy_l129_129244

theorem min_value_xy (x y : ℝ) (h : 1 / x + 2 / y = Real.sqrt (x * y)) : x * y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_xy_l129_129244


namespace Amy_finish_time_l129_129825

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end Amy_finish_time_l129_129825


namespace factorization_identity_l129_129649

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end factorization_identity_l129_129649


namespace part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l129_129141

section part1
variable (x : ℝ)

theorem part1 (a : ℝ) (h : a = 2) : (x + a) * (x - 2 * a + 1) < 0 ↔ -2 < x ∧ x < 3 :=
by
  sorry
end part1

section part2
variable (x a : ℝ)

-- Case: a = 1
theorem part2_a_eq_1 (h : a = 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ False :=
by
  sorry

-- Case: a > 1
theorem part2_a_gt_1 (h : a > 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 1 < x ∧ x < 2 * a - 1 :=
by
  sorry

-- Case: a < 1
theorem part2_a_lt_1 (h : a < 1) : (x - 1) * (x - 2 * a + 1) < 0 ↔ 2 * a - 1 < x ∧ x < 1 :=
by
  sorry
end part2

end part1_part2_a_eq_1_part2_a_gt_1_part2_a_lt_1_l129_129141


namespace cone_volume_increase_l129_129985

theorem cone_volume_increase (r h : ℝ) (k : ℝ) :
  let V := (1/3) * π * r^2 * h
  let h' := 2.60 * h
  let r' := r * (1 + k / 100)
  let V' := (1/3) * π * (r')^2 * h'
  let percentage_increase := ((V' / V) - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by
  sorry

end cone_volume_increase_l129_129985


namespace evaluate_expression_l129_129762

theorem evaluate_expression : -20 + 8 * (10 / 2) - 4 = 16 :=
by
  sorry -- Proof to be completed

end evaluate_expression_l129_129762


namespace petya_friends_count_l129_129739

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l129_129739


namespace factory_produces_6500_toys_per_week_l129_129414

theorem factory_produces_6500_toys_per_week
    (days_per_week : ℕ)
    (toys_per_day : ℕ)
    (h1 : days_per_week = 5)
    (h2 : toys_per_day = 1300) :
    days_per_week * toys_per_day = 6500 := 
by 
  sorry

end factory_produces_6500_toys_per_week_l129_129414


namespace factorize_mn_minus_mn_cubed_l129_129064

theorem factorize_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n ^ 3 = m * n * (1 + n) * (1 - n) :=
by {
  sorry
}

end factorize_mn_minus_mn_cubed_l129_129064


namespace boat_travel_times_l129_129829

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l129_129829


namespace additionalPeopleNeededToMowLawn_l129_129899

def numberOfPeopleNeeded (people : ℕ) (hours : ℕ) : ℕ :=
  (people * 8) / hours

theorem additionalPeopleNeededToMowLawn : numberOfPeopleNeeded 4 3 - 4 = 7 :=
by
  sorry

end additionalPeopleNeededToMowLawn_l129_129899


namespace temperature_at_80_degrees_l129_129675

theorem temperature_at_80_degrees (t : ℝ) :
  (-t^2 + 10 * t + 60 = 80) ↔ (t = 5 + 3 * Real.sqrt 5 ∨ t = 5 - 3 * Real.sqrt 5) := by
  sorry

end temperature_at_80_degrees_l129_129675


namespace g_diff_eq_neg8_l129_129979

noncomputable def g : ℝ → ℝ := sorry

axiom linear_g : ∀ x y : ℝ, g (x + y) = g x + g y

axiom condition_g : ∀ x : ℝ, g (x + 2) - g x = 4

theorem g_diff_eq_neg8 : g 2 - g 6 = -8 :=
by
  sorry

end g_diff_eq_neg8_l129_129979


namespace hyperbola_asymptote_slope_l129_129759

theorem hyperbola_asymptote_slope :
  (∃ m : ℚ, m > 0 ∧ ∀ x : ℚ, ∀ y : ℚ, ((x*x/16 - y*y/25 = 1) → (y = m * x ∨ y = -m * x))) → m = 5/4 :=
sorry

end hyperbola_asymptote_slope_l129_129759


namespace max_yellow_apples_max_total_apples_l129_129782

-- Definitions for the conditions
def num_green_apples : Nat := 10
def num_yellow_apples : Nat := 13
def num_red_apples : Nat := 18

-- Predicate for the stopping condition
def stop_condition (green yellow red : Nat) : Prop :=
  green < yellow ∧ yellow < red

-- Proof problem for maximum number of yellow apples
theorem max_yellow_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → y ≤ 13) →
  yellow ≤ 13 :=
sorry

-- Proof problem for maximum total number of apples
theorem max_total_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → g + y + r ≤ 39) →
  green + yellow + red ≤ 39 :=
sorry

end max_yellow_apples_max_total_apples_l129_129782


namespace successive_product_l129_129421

theorem successive_product (n : ℤ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_l129_129421


namespace no_intersections_root_of_quadratic_l129_129947

theorem no_intersections_root_of_quadratic (x : ℝ) :
  ¬(∃ x, (y = x) ∧ (y = x - 3)) ↔ (x^2 - 3 * x = 0) := by
  sorry

end no_intersections_root_of_quadratic_l129_129947
