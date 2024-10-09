import Mathlib

namespace find_middle_number_l1588_158874

theorem find_middle_number
  (S1 S2 M : ℤ)
  (h1 : S1 = 6 * 5)
  (h2 : S2 = 6 * 7)
  (h3 : 13 * 9 = S1 + M + S2) :
  M = 45 :=
by
  -- proof steps would go here
  sorry

end find_middle_number_l1588_158874


namespace find_a_l1588_158871

theorem find_a (a : ℝ) (h : ∃ x, x = 3 ∧ x^2 + a * x + a - 1 = 0) : a = -2 :=
sorry

end find_a_l1588_158871


namespace train_length_correct_l1588_158838

noncomputable def train_speed_kmph : ℝ := 60
noncomputable def train_time_seconds : ℝ := 15

noncomputable def length_of_train : ℝ :=
  let speed_mps := train_speed_kmph * 1000 / 3600
  speed_mps * train_time_seconds

theorem train_length_correct :
  length_of_train = 250.05 :=
by
  -- Proof goes here
  sorry

end train_length_correct_l1588_158838


namespace least_5_digit_divisible_l1588_158806

theorem least_5_digit_divisible (n : ℕ) (h1 : n ≥ 10000) (h2 : n < 100000)
  (h3 : 15 ∣ n) (h4 : 12 ∣ n) (h5 : 18 ∣ n) : n = 10080 :=
by
  sorry

end least_5_digit_divisible_l1588_158806


namespace speed_in_still_water_l1588_158808

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 35

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 :=
by
  sorry

end speed_in_still_water_l1588_158808


namespace sum_of_roots_l1588_158821

theorem sum_of_roots (b : ℝ) (x : ℝ) (y : ℝ) :
  (x^2 - b * x + 20 = 0) ∧ (y^2 - b * y + 20 = 0) ∧ (x * y = 20) -> (x + y = b) := 
by
  sorry

end sum_of_roots_l1588_158821


namespace solve_system_of_equations_l1588_158847

theorem solve_system_of_equations : ∃ (x y : ℝ), 4 * x + y = 6 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 :=
by
  existsi (1 : ℝ)
  existsi (2 : ℝ)
  sorry

end solve_system_of_equations_l1588_158847


namespace parabola_focus_l1588_158893

theorem parabola_focus (p : ℝ) (hp : p > 0) :
    ∀ (x y : ℝ), (x = 2 * p * y^2) ↔ (x, y) = (1 / (8 * p), 0) :=
by 
  sorry

end parabola_focus_l1588_158893


namespace camilla_blueberry_jelly_beans_l1588_158872

theorem camilla_blueberry_jelly_beans (b c : ℕ) (h1 : b = 2 * c) (h2 : b - 10 = 3 * (c - 10)) : b = 40 := 
sorry

end camilla_blueberry_jelly_beans_l1588_158872


namespace days_provisions_initially_meant_l1588_158864

theorem days_provisions_initially_meant (x : ℕ) (h1 : 250 * x = 200 * 50) : x = 40 :=
by sorry

end days_provisions_initially_meant_l1588_158864


namespace ordered_pairs_1944_l1588_158813

theorem ordered_pairs_1944 :
  ∃ n : ℕ, (∀ x y : ℕ, (x * y = 1944 ↔ x > 0 ∧ y > 0)) → n = 24 :=
by
  sorry

end ordered_pairs_1944_l1588_158813


namespace second_carpenter_days_l1588_158842

theorem second_carpenter_days (x : ℚ) (h1 : 1 / 5 + 1 / x = 1 / 2) : x = 10 / 3 :=
by
  sorry

end second_carpenter_days_l1588_158842


namespace sum_of_D_coordinates_l1588_158855

-- Definition of the midpoint condition
def is_midpoint (N C D : ℝ × ℝ) : Prop :=
  N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2

-- Given points
def N : ℝ × ℝ := (5, -1)
def C : ℝ × ℝ := (11, 10)

-- Statement of the problem
theorem sum_of_D_coordinates :
  ∃ D : ℝ × ℝ, is_midpoint N C D ∧ (D.1 + D.2 = -13) :=
  sorry

end sum_of_D_coordinates_l1588_158855


namespace problem_inequality_I_problem_inequality_II_l1588_158866

theorem problem_inequality_I (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  1 / a + 1 / b ≥ 4 := sorry

theorem problem_inequality_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := sorry

end problem_inequality_I_problem_inequality_II_l1588_158866


namespace range_of_m_inequality_a_b_l1588_158849

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 2|

theorem range_of_m (m : ℝ) : (∀ x, f x ≥ |m - 1|) → -2 ≤ m ∧ m ≤ 4 :=
sorry

theorem inequality_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a^2 + b^2 = 2) : 
  a + b ≥ 2 * a * b :=
sorry

end range_of_m_inequality_a_b_l1588_158849


namespace smallest_n_property_l1588_158861

noncomputable def smallest_n : ℕ := 13

theorem smallest_n_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (x * y * z ∣ (x + y + z) ^ smallest_n) :=
by
  intros x y z hx hy hz hxy hyz hzx
  use smallest_n
  sorry

end smallest_n_property_l1588_158861


namespace questionnaires_drawn_from_D_l1588_158814

theorem questionnaires_drawn_from_D (a b c d : ℕ) (A_s B_s C_s D_s: ℕ) (common_diff: ℕ)
  (h1 : a + b + c + d = 1000)
  (h2 : b = a + common_diff)
  (h3 : c = a + 2 * common_diff)
  (h4 : d = a + 3 * common_diff)
  (h5 : A_s = 30 - common_diff)
  (h6 : B_s = 30)
  (h7 : C_s = 30 + common_diff)
  (h8 : D_s = 30 + 2 * common_diff)
  (h9 : A_s + B_s + C_s + D_s = 150)
  : D_s = 60 := sorry

end questionnaires_drawn_from_D_l1588_158814


namespace ratio_of_squares_l1588_158833

def square_inscribed_triangle_1 (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  x = 24 / 7

def square_inscribed_triangle_2 (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  y = 10 / 3

theorem ratio_of_squares (x y : ℝ) 
  (hx : square_inscribed_triangle_1 x) 
  (hy : square_inscribed_triangle_2 y) : 
  x / y = 36 / 35 := 
by sorry

end ratio_of_squares_l1588_158833


namespace maximum_t_l1588_158801

theorem maximum_t {a b t : ℝ} (ha : 0 < a) (hb : a < b) (ht : b < t)
  (h_condition : b * Real.log a < a * Real.log b) : t ≤ Real.exp 1 :=
sorry

end maximum_t_l1588_158801


namespace max_x_satisfying_ineq_l1588_158844

theorem max_x_satisfying_ineq : ∃ (x : ℤ), (x ≤ 1 ∧ ∀ (y : ℤ), (y > x → y > 1) ∧ (y ≤ 1 → (y : ℚ) / 3 + 7 / 4 < 9 / 4)) := 
by
  sorry

end max_x_satisfying_ineq_l1588_158844


namespace linearly_dependent_k_l1588_158857

theorem linearly_dependent_k (k : ℝ) : 
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨1, k⟩ : ℝ × ℝ) = (0, 0)) ↔ k = 3 / 2 :=
by
  sorry

end linearly_dependent_k_l1588_158857


namespace libraryRoomNumber_l1588_158858

-- Define the conditions
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def isPrime (n : ℕ) : Prop := Nat.Prime n
def isEven (n : ℕ) : Prop := n % 2 = 0
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def hasDigit7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Main theorem
theorem libraryRoomNumber (n : ℕ) (h1 : isTwoDigit n)
  (h2 : (isPrime n ∧ isEven n ∧ isDivisibleBy5 n ∧ hasDigit7 n) ↔ false)
  : n % 10 = 0 := 
sorry

end libraryRoomNumber_l1588_158858


namespace find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l1588_158834

theorem find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square :
  ∃ n : ℕ, (4^n + 5^n) = k^2 ↔ n = 1 :=
by
  sorry

end find_n_such_that_4_pow_n_plus_5_pow_n_is_perfect_square_l1588_158834


namespace arithmetic_sequence_sum_equals_product_l1588_158851

theorem arithmetic_sequence_sum_equals_product :
  ∃ (a_1 a_2 a_3 : ℤ), (a_2 = a_1 + d) ∧ (a_3 = a_1 + 2 * d) ∧ 
    a_1 ≠ 0 ∧ (a_1 + a_2 + a_3 = a_1 * a_2 * a_3) ∧ 
    (∃ d x : ℤ, x ≠ 0 ∧ d ≠ 0 ∧ 
    ((x = 1 ∧ d = 1) ∨ (x = -3 ∧ d = 1) ∨ (x = 3 ∧ d = -1) ∨ (x = -1 ∧ d = -1))) :=
sorry

end arithmetic_sequence_sum_equals_product_l1588_158851


namespace train_length_l1588_158840

-- Define the given conditions
def train_cross_time : ℕ := 40 -- time in seconds
def train_speed_kmh : ℕ := 144 -- speed in km/h

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 5) / 18 

def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh

-- Theorem statement
theorem train_length :
  train_speed_ms * train_cross_time = 1600 :=
by
  sorry

end train_length_l1588_158840


namespace find_f_2_l1588_158867

def f (x : ℕ) : ℤ := sorry

axiom func_def : ∀ x : ℕ, f (x + 1) = x^2 - x

theorem find_f_2 : f 2 = 0 :=
by
  sorry

end find_f_2_l1588_158867


namespace evening_water_usage_is_6_l1588_158869

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end evening_water_usage_is_6_l1588_158869


namespace minimum_number_of_girls_l1588_158897

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l1588_158897


namespace senate_arrangement_l1588_158817

def countArrangements : ℕ :=
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- The calculation for arrangements considering fixed elements, and permutations adjusted for rotation
  12 * (Nat.factorial 10 / 2)

theorem senate_arrangement :
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- Total ways to arrange the members around the table under the given conditions
  countArrangements = 21772800 :=
by
  sorry

end senate_arrangement_l1588_158817


namespace martha_to_doris_ratio_l1588_158865

-- Define the amounts involved
def initial_amount : ℕ := 21
def doris_spent : ℕ := 6
def remaining_after_doris : ℕ := initial_amount - doris_spent
def final_amount : ℕ := 12
def martha_spent : ℕ := remaining_after_doris - final_amount

-- State the theorem about the ratio
theorem martha_to_doris_ratio : martha_spent * 2 = doris_spent :=
by
  -- Detailed proof is skipped
  sorry

end martha_to_doris_ratio_l1588_158865


namespace student_l1588_158843

-- Definition of the conditions
def mistaken_calculation (x : ℤ) : ℤ :=
  x + 10

def correct_calculation (x : ℤ) : ℤ :=
  x + 5

-- Theorem statement: Prove that the student's result is 10 more than the correct result
theorem student's_error {x : ℤ} : mistaken_calculation x = correct_calculation x + 5 :=
by
  sorry

end student_l1588_158843


namespace talent_show_l1588_158839

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l1588_158839


namespace sufficient_but_not_necessary_l1588_158836

theorem sufficient_but_not_necessary (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  (a > b ∧ b > 0 ∧ c > 0) → (a / (a + c) > b / (b + c)) :=
by
  intros
  sorry

end sufficient_but_not_necessary_l1588_158836


namespace candy_problem_l1588_158816

theorem candy_problem (N a S : ℕ) 
  (h1 : ∀ i : ℕ, i < N → a = S - 7 - a)
  (h2 : ∀ i : ℕ, i < N → a > 1)
  (h3 : S = N * a) : 
  S = 21 :=
by
  sorry

end candy_problem_l1588_158816


namespace rotameter_gas_phase_measurement_l1588_158805

theorem rotameter_gas_phase_measurement
  (liquid_inch_per_lpm : ℝ) (liquid_liter_per_minute : ℝ) (gas_inch_movement_ratio : ℝ) (gas_liter_passed : ℝ) :
  liquid_inch_per_lpm = 2.5 → liquid_liter_per_minute = 60 → gas_inch_movement_ratio = 0.5 → gas_liter_passed = 192 →
  (gas_inch_movement_ratio * liquid_inch_per_lpm * gas_liter_passed / liquid_liter_per_minute) = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_gas_phase_measurement_l1588_158805


namespace remainder_of_4_pow_a_div_10_l1588_158881

theorem remainder_of_4_pow_a_div_10 (a : ℕ) (h1 : a > 0) (h2 : a % 2 = 0) :
  (4 ^ a) % 10 = 6 :=
by sorry

end remainder_of_4_pow_a_div_10_l1588_158881


namespace det_new_matrix_l1588_158889

variables {a b c d : ℝ}

theorem det_new_matrix (h : a * d - b * c = 5) : (a - c) * d - (b - d) * c = 5 :=
by sorry

end det_new_matrix_l1588_158889


namespace non_neg_int_solutions_l1588_158824

theorem non_neg_int_solutions : 
  ∀ (x y : ℕ), 2 * x ^ 2 + 2 * x * y - x + y = 2020 → 
               (x = 0 ∧ y = 2020) ∨ (x = 1 ∧ y = 673) :=
by
  sorry

end non_neg_int_solutions_l1588_158824


namespace grasshopper_jump_is_31_l1588_158862

def frog_jump : ℕ := 35
def total_jump : ℕ := 66
def grasshopper_jump := total_jump - frog_jump

theorem grasshopper_jump_is_31 : grasshopper_jump = 31 := 
by
  unfold grasshopper_jump
  sorry

end grasshopper_jump_is_31_l1588_158862


namespace two_bags_remainder_l1588_158825

-- Given conditions
variables (n : ℕ)

-- Assume n ≡ 8 (mod 11)
def satisfied_mod_condition : Prop := n % 11 = 8

-- Prove that 2n ≡ 5 (mod 11)
theorem two_bags_remainder (h : satisfied_mod_condition n) : (2 * n) % 11 = 5 :=
by 
  unfold satisfied_mod_condition at h
  sorry

end two_bags_remainder_l1588_158825


namespace A_inter_B_empty_iff_A_union_B_eq_B_iff_l1588_158827

open Set

variable (a x : ℝ)

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem A_inter_B_empty_iff {a : ℝ} :
  (A a ∩ B = ∅) ↔ 0 ≤ a ∧ a ≤ 4 :=
by 
  sorry

theorem A_union_B_eq_B_iff {a : ℝ} :
  (A a ∪ B = B) ↔ a < -4 :=
by
  sorry

end A_inter_B_empty_iff_A_union_B_eq_B_iff_l1588_158827


namespace republicans_in_house_l1588_158800

theorem republicans_in_house (D R : ℕ) (h1 : D + R = 434) (h2 : R = D + 30) : R = 232 :=
by sorry

end republicans_in_house_l1588_158800


namespace total_potatoes_l1588_158826

theorem total_potatoes (cooked_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) (H1 : cooked_potatoes = 7) (H2 : time_per_potato = 5) (H3 : remaining_time = 45) : (cooked_potatoes + (remaining_time / time_per_potato) = 16) :=
by
  sorry

end total_potatoes_l1588_158826


namespace carrot_servings_l1588_158894

theorem carrot_servings (C : ℕ) 
  (H1 : ∀ (corn_servings : ℕ), corn_servings = 5 * C)
  (H2 : ∀ (green_bean_servings : ℕ) (corn_servings : ℕ), green_bean_servings = corn_servings / 2)
  (H3 : ∀ (plot_plants : ℕ), plot_plants = 9)
  (H4 : ∀ (total_servings : ℕ) 
         (carrot_servings : ℕ)
         (corn_servings : ℕ)
         (green_bean_servings : ℕ), 
         total_servings = carrot_servings + corn_servings + green_bean_servings ∧
         total_servings = 306) : 
  C = 4 := 
    sorry

end carrot_servings_l1588_158894


namespace anna_grams_l1588_158829

-- Definitions based on conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℝ := 15
def anna_cost_per_gram : ℝ := 20
def combined_cost : ℝ := 1450

-- Statement to prove
theorem anna_grams : (combined_cost - (gary_grams * gary_cost_per_gram)) / anna_cost_per_gram = 50 :=
by 
  sorry

end anna_grams_l1588_158829


namespace range_of_a_l1588_158812

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 2 < 0) ↔ (a^2 ≤ 8) :=
by
  sorry

end range_of_a_l1588_158812


namespace find_f_of_neg_2_l1588_158899

theorem find_f_of_neg_2
  (f : ℚ → ℚ)
  (h : ∀ (x : ℚ), x ≠ 0 → 3 * f (1/x) + 2 * f x / x = x^2)
  : f (-2) = 13/5 :=
sorry

end find_f_of_neg_2_l1588_158899


namespace wrench_force_inversely_proportional_l1588_158841

theorem wrench_force_inversely_proportional (F L : ℝ) (F1 F2 L1 L2 : ℝ) 
    (h1 : F1 = 375) 
    (h2 : L1 = 9) 
    (h3 : L2 = 15) 
    (h4 : ∀ L : ℝ, F * L = F1 * L1) : F2 = 225 :=
by
  sorry

end wrench_force_inversely_proportional_l1588_158841


namespace central_vs_northern_chess_match_l1588_158877

noncomputable def schedule_chess_match : Nat :=
  let players_team1 := ["A", "B", "C"];
  let players_team2 := ["X", "Y", "Z"];
  let total_games := 3 * 3 * 3;
  let games_per_round := 4;
  let total_rounds := 7;
  Nat.factorial total_rounds

theorem central_vs_northern_chess_match :
    schedule_chess_match = 5040 :=
by
  sorry

end central_vs_northern_chess_match_l1588_158877


namespace complement_is_empty_l1588_158818

def U : Set ℕ := {1, 3}
def A : Set ℕ := {1, 3}

theorem complement_is_empty : (U \ A) = ∅ := 
by 
  sorry

end complement_is_empty_l1588_158818


namespace jessica_speed_last_40_l1588_158875

theorem jessica_speed_last_40 
  (total_distance : ℕ)
  (total_time_min : ℕ)
  (first_segment_avg_speed : ℕ)
  (second_segment_avg_speed : ℕ)
  (last_segment_avg_speed : ℕ) :
  total_distance = 120 →
  total_time_min = 120 →
  first_segment_avg_speed = 50 →
  second_segment_avg_speed = 60 →
  last_segment_avg_speed = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end jessica_speed_last_40_l1588_158875


namespace at_least_one_true_l1588_158837

theorem at_least_one_true (p q : Prop) (h : ¬(p ∨ q) = false) : p ∨ q :=
by
  sorry

end at_least_one_true_l1588_158837


namespace find_x_value_l1588_158815

-- Define the condition as a hypothesis
def condition (x : ℝ) : Prop := (x / 4) - x - (3 / 6) = 1

-- State the theorem
theorem find_x_value (x : ℝ) (h : condition x) : x = -2 := 
by sorry

end find_x_value_l1588_158815


namespace Sophie_donuts_problem_l1588_158876

noncomputable def total_cost_before_discount (cost_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  cost_per_box * num_boxes

noncomputable def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost - discount

noncomputable def total_donuts (donuts_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  donuts_per_box * num_boxes

noncomputable def donuts_left (total_donuts : ℕ) (donuts_given_away : ℕ) : ℕ :=
  total_donuts - donuts_given_away

theorem Sophie_donuts_problem
  (budget : ℝ)
  (cost_per_box : ℝ)
  (discount_rate : ℝ)
  (num_boxes : ℕ)
  (donuts_per_box : ℕ)
  (donuts_given_to_mom : ℕ)
  (donuts_given_to_sister : ℕ)
  (half_dozen : ℕ) :
  budget = 50 →
  cost_per_box = 12 →
  discount_rate = 0.10 →
  num_boxes = 4 →
  donuts_per_box = 12 →
  donuts_given_to_mom = 12 →
  donuts_given_to_sister = 6 →
  half_dozen = 6 →
  total_cost_after_discount (total_cost_before_discount cost_per_box num_boxes) (discount_amount (total_cost_before_discount cost_per_box num_boxes) discount_rate) = 43.2 ∧
  donuts_left (total_donuts donuts_per_box num_boxes) (donuts_given_to_mom + donuts_given_to_sister) = 30 :=
by
  sorry

end Sophie_donuts_problem_l1588_158876


namespace relationship_among_a_b_c_l1588_158831

-- Defining the properties and conditions of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Defining the function f based on the condition
noncomputable def f (x m : ℝ) : ℝ := 2 ^ |x - m| - 1

-- Defining the constants a, b, c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5) 0
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2) 0
noncomputable def c : ℝ := f 0 0

-- The theorem stating the relationship among a, b, and c
theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l1588_158831


namespace inequality_trig_l1588_158820

theorem inequality_trig 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (hz : 0 < z ∧ z < (π / 2)) :
  (π / 2) + 2 * (Real.sin x) * (Real.cos y) + 2 * (Real.sin y) * (Real.cos z) > 
  (Real.sin (2 * x)) + (Real.sin (2 * y)) + (Real.sin (2 * z)) :=
by
  sorry  -- The proof is omitted

end inequality_trig_l1588_158820


namespace max_checkers_on_chessboard_l1588_158810

theorem max_checkers_on_chessboard (n : ℕ) : 
  ∃ k : ℕ, k = 2 * n * (n / 2) := sorry

end max_checkers_on_chessboard_l1588_158810


namespace quadratic_floor_eq_more_than_100_roots_l1588_158870

open Int

theorem quadratic_floor_eq_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ (S : Finset ℤ), S.card > 100 ∧ ∀ x ∈ S, ⌊(x : ℝ) ^ 2⌋ + p * x + q = 0 :=
by
  sorry

end quadratic_floor_eq_more_than_100_roots_l1588_158870


namespace no_real_intersection_l1588_158809

theorem no_real_intersection (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x * f y = y * f x) 
  (h2 : f 1 = -1) : ¬∃ x : ℝ, f x = x^2 + 1 :=
by
  sorry

end no_real_intersection_l1588_158809


namespace smallest_c_in_progressions_l1588_158859

def is_arithmetic_progression (a b c : ℤ) : Prop := b - a = c - b

def is_geometric_progression (b c a : ℤ) : Prop := c^2 = a*b

theorem smallest_c_in_progressions :
  ∃ (a b c : ℤ), is_arithmetic_progression a b c ∧ is_geometric_progression b c a ∧ 
  (∀ (a' b' c' : ℤ), is_arithmetic_progression a' b' c' ∧ is_geometric_progression b' c' a' → c ≤ c') ∧ c = 2 :=
by
  sorry

end smallest_c_in_progressions_l1588_158859


namespace vanya_meets_mother_opposite_dir_every_4_minutes_l1588_158890

-- Define the parameters
def lake_perimeter : ℝ := sorry  -- Length of the lake's perimeter, denoted as l
def mother_time_lap : ℝ := 12    -- Time taken by the mother to complete one lap (in minutes)
def vanya_time_overtake : ℝ := 12 -- Time taken by Vanya to overtake the mother (in minutes)

-- Define speeds
noncomputable def mother_speed : ℝ := lake_perimeter / mother_time_lap
noncomputable def vanya_speed : ℝ := 2 * lake_perimeter / vanya_time_overtake

-- Define their relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := mother_speed + vanya_speed

-- Prove that the meeting interval is 4 minutes
theorem vanya_meets_mother_opposite_dir_every_4_minutes :
  (lake_perimeter / relative_speed) = 4 := by
  sorry

end vanya_meets_mother_opposite_dir_every_4_minutes_l1588_158890


namespace range_of_a_l1588_158828

noncomputable def f (x a : ℝ) := (x^2 + a * x + 11) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 3) ↔ (a ≥ -8 / 3) :=
by sorry

end range_of_a_l1588_158828


namespace x_intercept_of_perpendicular_line_is_16_over_3_l1588_158873

theorem x_intercept_of_perpendicular_line_is_16_over_3 :
  (∃ x : ℚ, (∃ y : ℚ, (4 * x - 3 * y = 12))
    ∧ (∃ x y : ℚ, (y = - (3 / 4) * x + 4 ∧ y = 0) ∧ x = 16 / 3)) :=
by {
  sorry
}

end x_intercept_of_perpendicular_line_is_16_over_3_l1588_158873


namespace minimum_value_m_l1588_158856

noncomputable def f (x : ℝ) (phi : ℝ) : ℝ :=
  Real.sin (2 * x + phi)

theorem minimum_value_m (phi : ℝ) (m : ℝ) (h1 : |phi| < Real.pi / 2)
  (h2 : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x (Real.pi / 6) ≤ m) :
  m = -1 / 2 :=
by
  sorry

end minimum_value_m_l1588_158856


namespace perpendicular_lines_sum_is_minus_four_l1588_158878

theorem perpendicular_lines_sum_is_minus_four 
  (a b c : ℝ) 
  (h1 : (a * 2) / (4 * 5) = 1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * (-2) + b = 0) : 
  a + b + c = -4 := 
sorry

end perpendicular_lines_sum_is_minus_four_l1588_158878


namespace noncongruent_triangles_count_l1588_158846

/-- Prove that the number of noncongruent integer-sided triangles 
with positive area and perimeter less than 20, 
which are neither equilateral, isosceles, nor right triangles, is 15. -/
theorem noncongruent_triangles_count : 
  ∃ n : ℕ, 
  (∀ (a b c : ℕ) (h : a ≤ b ∧ b ≤ c),
    a + b + c < 20 ∧ a + b > c ∧ a^2 + b^2 ≠ c^2 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → n ≥ 15) :=
sorry

end noncongruent_triangles_count_l1588_158846


namespace current_short_trees_l1588_158822

-- Definitions of conditions in a)
def tall_trees : ℕ := 44
def short_trees_planted : ℕ := 57
def total_short_trees_after_planting : ℕ := 98

-- Statement to prove the question == answer given conditions
theorem current_short_trees (S : ℕ) (h : S + short_trees_planted = total_short_trees_after_planting) : S = 41 :=
by
  -- Proof would go here
  sorry

end current_short_trees_l1588_158822


namespace value_of_f2_l1588_158888

noncomputable def f : ℕ → ℕ :=
  sorry

axiom f_condition : ∀ x : ℕ, f (x + 1) = 2 * x + 3

theorem value_of_f2 : f 2 = 5 :=
by sorry

end value_of_f2_l1588_158888


namespace log_equation_l1588_158850

theorem log_equation :
  (3 / (Real.log 1000^4 / Real.log 8)) + (4 / (Real.log 1000^4 / Real.log 9)) = 3 :=
by
  sorry

end log_equation_l1588_158850


namespace multiple_of_4_and_6_sum_even_l1588_158892

theorem multiple_of_4_and_6_sum_even (a b : ℤ) (h₁ : ∃ m : ℤ, a = 4 * m) (h₂ : ∃ n : ℤ, b = 6 * n) : ∃ k : ℤ, (a + b) = 2 * k :=
by
  sorry

end multiple_of_4_and_6_sum_even_l1588_158892


namespace solver_inequality_l1588_158835

theorem solver_inequality (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → (x ≥ 3) :=
by
  intro h
  sorry

end solver_inequality_l1588_158835


namespace base_length_of_isosceles_triangle_l1588_158803

noncomputable def isosceles_triangle_base_length (height : ℝ) (radius : ℝ) : ℝ :=
  if height = 25 ∧ radius = 8 then 80 / 3 else 0

theorem base_length_of_isosceles_triangle :
  isosceles_triangle_base_length 25 8 = 80 / 3 :=
by
  -- skipping the proof
  sorry

end base_length_of_isosceles_triangle_l1588_158803


namespace fill_tank_time_l1588_158895

-- Define the rates at which the pipes fill the tank
noncomputable def rate_A := (1:ℝ)/50
noncomputable def rate_B := (1:ℝ)/75

-- Define the combined rate of both pipes
noncomputable def combined_rate := rate_A + rate_B

-- Define the time to fill the tank at the combined rate
noncomputable def time_to_fill := 1 / combined_rate

-- The theorem that states the time taken to fill the tank is 30 hours
theorem fill_tank_time : time_to_fill = 30 := sorry

end fill_tank_time_l1588_158895


namespace smallest_w_l1588_158845

def fact_936 : ℕ := 2^3 * 3^1 * 13^1

theorem smallest_w (w : ℕ) (h_w_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (12^2 ∣ 936 * w) → w = 36 :=
by
  sorry

end smallest_w_l1588_158845


namespace sale_in_second_month_l1588_158848

-- Define the constants for known sales and average requirement
def sale_first_month : Int := 8435
def sale_third_month : Int := 8855
def sale_fourth_month : Int := 9230
def sale_fifth_month : Int := 8562
def sale_sixth_month : Int := 6991
def average_sale_per_month : Int := 8500
def number_of_months : Int := 6

-- Define the total sales required for six months
def total_sales_required : Int := average_sale_per_month * number_of_months

-- Define the total known sales excluding the second month
def total_known_sales : Int := sale_first_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- The statement to prove: the sale in the second month is 8927
theorem sale_in_second_month : 
  total_sales_required - total_known_sales = 8927 := 
by
  sorry

end sale_in_second_month_l1588_158848


namespace petya_wins_if_and_only_if_m_ne_n_l1588_158898

theorem petya_wins_if_and_only_if_m_ne_n 
  (m n : ℕ) 
  (game : ∀ m n : ℕ, Prop)
  (win_condition : (game m n ↔ m ≠ n)) : 
  Prop := 
by 
  sorry

end petya_wins_if_and_only_if_m_ne_n_l1588_158898


namespace sum_infinite_series_l1588_158882

theorem sum_infinite_series :
  (∑' n : ℕ, (4 * (n + 1) + 1) / (3^(n + 1))) = 7 / 2 :=
sorry

end sum_infinite_series_l1588_158882


namespace statement_B_statement_C_statement_D_l1588_158860

variables (a b : ℝ)

-- Condition: a > 0
axiom a_pos : a > 0

-- Condition: e^a + ln b = 1
axiom eq1 : Real.exp a + Real.log b = 1

-- Statement B: a + ln b < 0
theorem statement_B : a + Real.log b < 0 :=
  sorry

-- Statement C: e^a + b > 2
theorem statement_C : Real.exp a + b > 2 :=
  sorry

-- Statement D: a + b > 1
theorem statement_D : a + b > 1 :=
  sorry

end statement_B_statement_C_statement_D_l1588_158860


namespace Frank_can_buy_7_candies_l1588_158886

def tickets_whack_a_mole := 33
def tickets_skee_ball := 9
def cost_per_candy := 6

def total_tickets := tickets_whack_a_mole + tickets_skee_ball

theorem Frank_can_buy_7_candies : total_tickets / cost_per_candy = 7 := by
  sorry

end Frank_can_buy_7_candies_l1588_158886


namespace julia_total_food_cost_l1588_158823

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end julia_total_food_cost_l1588_158823


namespace boys_in_class_l1588_158807

-- Define the conditions given in the problem
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 4 * (boys + girls) / 7 ∧ girls = 3 * (boys + girls) / 7
def total_students (boys girls : ℕ) : Prop := boys + girls = 49

-- Define the statement to be proved
theorem boys_in_class (boys girls : ℕ) (h1 : ratio_boys_to_girls boys girls) (h2 : total_students boys girls) : boys = 28 :=
by
  sorry

end boys_in_class_l1588_158807


namespace second_player_wins_12_petals_second_player_wins_11_petals_l1588_158891

def daisy_game (n : Nat) : Prop :=
  ∀ (p1_move p2_move : Nat → Nat → Prop), n % 2 = 0 → (∃ k, p1_move n k = false) ∧ (∃ ℓ, p2_move n ℓ = true)

theorem second_player_wins_12_petals : daisy_game 12 := sorry
theorem second_player_wins_11_petals : daisy_game 11 := sorry

end second_player_wins_12_petals_second_player_wins_11_petals_l1588_158891


namespace hexagon_coloring_l1588_158883

def hex_colorings : ℕ := 2

theorem hexagon_coloring :
  ∃ c : ℕ, c = hex_colorings := by
  sorry

end hexagon_coloring_l1588_158883


namespace solution_to_trig_equation_l1588_158887

theorem solution_to_trig_equation (x : ℝ) (k : ℤ) :
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
  (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) →
  (Real.sin (x / 2) = Real.cos (x / 2)) →
  (∃ k : ℤ, x = (π / 2) + 2 * π * ↑k) :=
by sorry

end solution_to_trig_equation_l1588_158887


namespace inequality_solution_l1588_158830

theorem inequality_solution 
  (a b c d e f : ℕ) 
  (h1 : a * d * f > b * c * f)
  (h2 : c * f * b > d * e * b) 
  (h3 : a * f - b * e = 1) 
  : d ≥ b + f := by
  -- Proof goes here
  sorry

end inequality_solution_l1588_158830


namespace value_of_a_l1588_158802

-- Declare and define the given conditions.
def line1 (y : ℝ) := y = 13
def line2 (x t y : ℝ) := y = 3 * x + t

-- Define the proof statement.
theorem value_of_a (a b t : ℝ) (h1 : line1 b) (h2 : line2 a t b) (ht : t = 1) : a = 4 :=
by
  sorry

end value_of_a_l1588_158802


namespace largest_divisor_of_n_squared_l1588_158832

theorem largest_divisor_of_n_squared (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d, d ∣ n^2 → d = 900) : 900 ∣ n^2 :=
by sorry

end largest_divisor_of_n_squared_l1588_158832


namespace find_value_of_a_l1588_158804

-- Define the setting for triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : b^2 - c^2 + 2 * a = 0)
variables (h2 : Real.tan C / Real.tan B = 3)

-- Given conditions and conclusion for the proof problem
theorem find_value_of_a 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) : 
  a = 4 := 
sorry

end find_value_of_a_l1588_158804


namespace polynomial_integer_roots_l1588_158885

theorem polynomial_integer_roots (b1 b2 : ℤ) (x : ℤ) (h : x^3 + b2 * x^2 + b1 * x + 18 = 0) :
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
sorry

end polynomial_integer_roots_l1588_158885


namespace standard_equation_of_ellipse_l1588_158884

-- Define the conditions of the ellipse
def ellipse_condition_A (m n : ℝ) : Prop := n * (5 / 3) ^ 2 = 1
def ellipse_condition_B (m n : ℝ) : Prop := m + n = 1

-- The theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (m n : ℝ) (hA : ellipse_condition_A m n) (hB : ellipse_condition_B m n) :
  m = 16 / 25 ∧ n = 9 / 25 :=
sorry

end standard_equation_of_ellipse_l1588_158884


namespace maximum_expr_value_l1588_158879

theorem maximum_expr_value :
  ∃ (x y e f : ℕ), (e = 4 ∧ x = 3 ∧ y = 2 ∧ f = 0) ∧
  (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4) ∧
  (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ∧
  (y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4) ∧
  (f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4) ∧
  (e ≠ x ∧ e ≠ y ∧ e ≠ f ∧ x ≠ y ∧ x ≠ f ∧ y ≠ f) ∧
  (e * x^y - f = 36) :=
by
  sorry

end maximum_expr_value_l1588_158879


namespace smallest_n_l1588_158852

theorem smallest_n 
    (h1 : ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * r = b ∧ b * r = c ∧ 7 * n + 1 = a + b + c)
    (h2 : ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * s = y ∧ y * s = z ∧ 8 * n + 1 = x + y + z) :
    n = 22 :=
sorry

end smallest_n_l1588_158852


namespace line_perpendicular_to_two_planes_parallel_l1588_158896

-- Declare lines and planes
variables {Line Plane : Type}

-- Define the perpendicular and parallel relationships
variables (perpendicular : Line → Plane → Prop)
variables (parallel : Plane → Plane → Prop)

-- Given conditions
variables (m n : Line) (α β : Plane)
-- The known conditions are:
-- 1. m is perpendicular to α
-- 2. m is perpendicular to β
-- We want to prove:
-- 3. α is parallel to β

theorem line_perpendicular_to_two_planes_parallel (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β :=
sorry

end line_perpendicular_to_two_planes_parallel_l1588_158896


namespace clock_chime_time_l1588_158811

/-- The proven time it takes for a wall clock to strike 12 times at 12 o'clock -/
theorem clock_chime_time :
  (∃ (interval_time : ℝ), (interval_time = 3) ∧ (∃ (time_12_times : ℝ), (time_12_times = interval_time * (12 - 1)) ∧ (time_12_times = 33))) :=
by
  sorry

end clock_chime_time_l1588_158811


namespace min_value_expression_l1588_158863

theorem min_value_expression (a d b c : ℝ) (habd : a ≥ 0 ∧ d ≥ 0) (hbc : b > 0 ∧ c > 0) (h_cond : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expression_l1588_158863


namespace johns_hats_cost_l1588_158853

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l1588_158853


namespace triangle_obtuse_l1588_158819

theorem triangle_obtuse (α β γ : ℝ) 
  (h1 : α ≤ β) (h2 : β < γ) 
  (h3 : α + β + γ = 180) 
  (h4 : α + β < γ) : 
  γ > 90 :=
  sorry

end triangle_obtuse_l1588_158819


namespace solution_set_l1588_158880

theorem solution_set (x : ℝ) : 
  (-2 * x ≤ 6) ∧ (x + 1 < 0) ↔ (-3 ≤ x) ∧ (x < -1) := by
  sorry

end solution_set_l1588_158880


namespace at_least_half_team_B_can_serve_l1588_158868

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end at_least_half_team_B_can_serve_l1588_158868


namespace remainder_of_eggs_is_2_l1588_158854

-- Define the number of eggs each person has
def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25

-- Define total eggs and remainder function
def total_eggs : ℕ := david_eggs + emma_eggs + fiona_eggs
def remainder (a b : ℕ) : ℕ := a % b

-- Prove that the remainder of total eggs divided by 10 is 2
theorem remainder_of_eggs_is_2 : remainder total_eggs 10 = 2 := by
  sorry

end remainder_of_eggs_is_2_l1588_158854
