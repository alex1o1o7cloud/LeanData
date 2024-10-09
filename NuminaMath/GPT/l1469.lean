import Mathlib

namespace simplify_expression_l1469_146999

variable (a b : ℝ)

theorem simplify_expression :
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (- (1 / 2) * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := 
by 
  sorry

end simplify_expression_l1469_146999


namespace total_seashells_after_six_weeks_l1469_146910

theorem total_seashells_after_six_weeks :
  ∀ (a b : ℕ) 
  (initial_a : a = 50) 
  (initial_b : b = 30) 
  (next_a : ∀ k : ℕ, k > 0 → a + 20 = (a + 20) * k) 
  (next_b : ∀ k : ℕ, k > 0 → b * 2 = (b * 2) * k), 
  (a + 20 * 5) + (b * 2 ^ 5) = 1110 :=
by
  intros a b initial_a initial_b next_a next_b
  sorry

end total_seashells_after_six_weeks_l1469_146910


namespace rainy_days_last_week_l1469_146941

theorem rainy_days_last_week (n : ℤ) (R NR : ℕ) (h1 : n * R + 3 * NR = 20)
  (h2 : 3 * NR = n * R + 10) (h3 : R + NR = 7) : R = 2 :=
sorry

end rainy_days_last_week_l1469_146941


namespace intersection_of_A_and_B_l1469_146955

open Set

variable {α : Type}

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 2, 3, 5}
def B : Set ℤ := {x | -1 < x ∧ x < 3}

-- Define the proof problem as a theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 2} :=
by
  sorry

end intersection_of_A_and_B_l1469_146955


namespace factorize_expression_l1469_146922

theorem factorize_expression (a x : ℝ) :
  a * x^2 - 2 * a * x + a = a * (x - 1) ^ 2 := 
sorry

end factorize_expression_l1469_146922


namespace proof_f_g_f3_l1469_146975

def f (x: ℤ) : ℤ := 2*x + 5
def g (x: ℤ) : ℤ := 5*x + 2

theorem proof_f_g_f3 :
  f (g (f 3)) = 119 := by
  sorry

end proof_f_g_f3_l1469_146975


namespace total_spending_l1469_146930

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l1469_146930


namespace intersection_of_M_and_N_l1469_146968

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | x < 1}

theorem intersection_of_M_and_N : (M ∩ N = {x : ℝ | -1 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_M_and_N_l1469_146968


namespace autumn_sales_l1469_146971

theorem autumn_sales (T : ℝ) (spring summer winter autumn : ℝ) 
    (h1 : spring = 3)
    (h2 : summer = 6)
    (h3 : winter = 5)
    (h4 : T = (3 / 0.2)) :
    autumn = 1 :=
by 
  -- Proof goes here
  sorry

end autumn_sales_l1469_146971


namespace percentage_increase_on_sale_l1469_146911

theorem percentage_increase_on_sale (P S : ℝ) (hP : P ≠ 0) (hS : S ≠ 0)
  (h_price_reduction : (0.8 : ℝ) * P * S * (1 + (X / 100)) = 1.44 * P * S) :
  X = 80 := by
  sorry

end percentage_increase_on_sale_l1469_146911


namespace solve_for_q_l1469_146927

theorem solve_for_q (k l q : ℝ) 
  (h1 : 3 / 4 = k / 48)
  (h2 : 3 / 4 = (k + l) / 56)
  (h3 : 3 / 4 = (q - l) / 160) :
  q = 126 :=
  sorry

end solve_for_q_l1469_146927


namespace find_m_l1469_146946

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let d : ℝ := |2| / Real.sqrt (m^2 + 1)
  d = 1

theorem find_m (m : ℝ) : tangent_condition m ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end find_m_l1469_146946


namespace remainder_n_pow_5_minus_n_mod_30_l1469_146900

theorem remainder_n_pow_5_minus_n_mod_30 (n : ℤ) : (n^5 - n) % 30 = 0 := 
by sorry

end remainder_n_pow_5_minus_n_mod_30_l1469_146900


namespace peach_ratios_and_percentages_l1469_146978

def red_peaches : ℕ := 8
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6
def orange_peaches : ℕ := 4
def total_peaches : ℕ := red_peaches + yellow_peaches + green_peaches + orange_peaches

theorem peach_ratios_and_percentages :
  ((green_peaches : ℚ) / total_peaches = 3 / 16) ∧
  ((green_peaches : ℚ) / total_peaches * 100 = 18.75) ∧
  ((yellow_peaches : ℚ) / total_peaches = 7 / 16) ∧
  ((yellow_peaches : ℚ) / total_peaches * 100 = 43.75) :=
by {
  sorry
}

end peach_ratios_and_percentages_l1469_146978


namespace find_larger_number_l1469_146926

theorem find_larger_number (L S : ℤ) (h₁ : L - S = 1000) (h₂ : L = 10 * S + 10) : L = 1110 :=
sorry

end find_larger_number_l1469_146926


namespace not_possible_to_construct_l1469_146985

/-- The frame consists of 54 unit segments. -/
def frame_consists_of_54_units : Prop := sorry

/-- Each part of the construction set consists of three unit segments. -/
def part_is_three_units : Prop := sorry

/-- Each vertex of a cube is shared by three edges. -/
def vertex_shares_three_edges : Prop := sorry

/-- Six segments emerge from the center of the cube. -/
def center_has_six_segments : Prop := sorry

/-- It is not possible to construct the frame with exactly 18 parts. -/
theorem not_possible_to_construct
  (h1 : frame_consists_of_54_units)
  (h2 : part_is_three_units)
  (h3 : vertex_shares_three_edges)
  (h4 : center_has_six_segments) : 
  ¬ ∃ (parts : ℕ), parts = 18 :=
sorry

end not_possible_to_construct_l1469_146985


namespace difference_of_numbers_l1469_146950

variable (x y d : ℝ)

theorem difference_of_numbers
  (h1 : x + y = 5)
  (h2 : x - y = d)
  (h3 : x^2 - y^2 = 50) :
  d = 10 :=
by
  sorry

end difference_of_numbers_l1469_146950


namespace bus_initial_passengers_l1469_146903

theorem bus_initial_passengers (M W : ℕ) 
  (h1 : W = M / 2) 
  (h2 : M - 16 = W + 8) : 
  M + W = 72 :=
sorry

end bus_initial_passengers_l1469_146903


namespace doughnut_completion_time_l1469_146902

noncomputable def time_completion : Prop :=
  let start_time : ℕ := 7 * 60 -- 7:00 AM in minutes
  let quarter_complete_time : ℕ := 10 * 60 + 20 -- 10:20 AM in minutes
  let efficiency_decrease_time : ℕ := 12 * 60 -- 12:00 PM in minutes
  let one_quarter_duration : ℕ := quarter_complete_time - start_time
  let total_time_before_efficiency_decrease : ℕ := 5 * 60 -- from 7:00 AM to 12:00 PM is 5 hours
  let remaining_time_without_efficiency : ℕ := 4 * one_quarter_duration - total_time_before_efficiency_decrease
  let adjusted_remaining_time : ℕ := remaining_time_without_efficiency * 10 / 9 -- decrease by 10% efficiency
  let total_job_duration : ℕ := total_time_before_efficiency_decrease + adjusted_remaining_time
  let completion_time := efficiency_decrease_time + adjusted_remaining_time
  completion_time = 21 * 60 + 15 -- 9:15 PM in minutes

theorem doughnut_completion_time : time_completion :=
  by 
    sorry

end doughnut_completion_time_l1469_146902


namespace subset_M_N_l1469_146958

def is_element_of_M (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)

def is_element_of_N (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)

theorem subset_M_N : ∀ x, is_element_of_M x → is_element_of_N x :=
by
  sorry

end subset_M_N_l1469_146958


namespace find_factor_l1469_146954

-- Define the conditions
def number : ℕ := 9
def expr1 (f : ℝ) : ℝ := (number + 2) * f
def expr2 : ℝ := 24 + number

-- The proof problem statement
theorem find_factor (f : ℝ) : expr1 f = expr2 → f = 3 := by
  sorry

end find_factor_l1469_146954


namespace intersection_of_sets_l1469_146908

def setA : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def setB : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_sets :
  setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end intersection_of_sets_l1469_146908


namespace stability_of_scores_requires_variance_l1469_146986

-- Define the conditions
variable (scores : List ℝ)

-- Define the main theorem
theorem stability_of_scores_requires_variance : True :=
  sorry

end stability_of_scores_requires_variance_l1469_146986


namespace machine_working_time_l1469_146939

theorem machine_working_time (y : ℝ) :
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) → y = 2 :=
by
  sorry

end machine_working_time_l1469_146939


namespace circle_tangent_line_l1469_146928

theorem circle_tangent_line (a : ℝ) : 
  ∃ (a : ℝ), a = 2 ∨ a = -8 := 
by 
  sorry

end circle_tangent_line_l1469_146928


namespace find_cos_C_l1469_146989

noncomputable def cos_C_eq (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : Prop :=
  Real.cos C = 7 / 25

theorem find_cos_C (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) :
  cos_C_eq A B C a b c h1 h2 :=
sorry

end find_cos_C_l1469_146989


namespace find_a_l1469_146972

theorem find_a 
  (x y z a : ℤ)
  (h1 : z + a = -2)
  (h2 : y + z = 1)
  (h3 : x + y = 0) : 
  a = -2 := 
  by 
    sorry

end find_a_l1469_146972


namespace months_for_three_times_collection_l1469_146981

def Kymbrea_collection (n : ℕ) : ℕ := 40 + 3 * n
def LaShawn_collection (n : ℕ) : ℕ := 20 + 5 * n

theorem months_for_three_times_collection : ∃ n : ℕ, LaShawn_collection n = 3 * Kymbrea_collection n ∧ n = 25 := 
by
  sorry

end months_for_three_times_collection_l1469_146981


namespace remainder_division_l1469_146937

theorem remainder_division {N : ℤ} (k : ℤ) (h : N = 125 * k + 40) : N % 15 = 10 :=
sorry

end remainder_division_l1469_146937


namespace find_two_irreducible_fractions_l1469_146931

theorem find_two_irreducible_fractions :
  ∃ (a b d1 d2 : ℕ), 
    (1 ≤ a) ∧ 
    (1 ≤ b) ∧ 
    (gcd a d1 = 1) ∧ 
    (gcd b d2 = 1) ∧ 
    (1 ≤ d1) ∧ 
    (d1 ≤ 100) ∧ 
    (1 ≤ d2) ∧ 
    (d2 ≤ 100) ∧ 
    (a / (d1 : ℚ) + b / (d2 : ℚ) = 86 / 111) := 
by {
  sorry
}

end find_two_irreducible_fractions_l1469_146931


namespace largest_operation_result_is_div_l1469_146977

noncomputable def max_operation_result : ℚ :=
  max (max (-1 + (-1 / 2)) (-1 - (-1 / 2)))
      (max (-1 * (-1 / 2)) (-1 / (-1 / 2)))

theorem largest_operation_result_is_div :
  max_operation_result = 2 := by
  sorry

end largest_operation_result_is_div_l1469_146977


namespace greatest_temp_diff_on_tuesday_l1469_146923

def highest_temp_mon : ℝ := 5
def lowest_temp_mon : ℝ := 2
def highest_temp_tue : ℝ := 4
def lowest_temp_tue : ℝ := -1
def highest_temp_wed : ℝ := 0
def lowest_temp_wed : ℝ := -4

def temp_diff (highest lowest : ℝ) : ℝ :=
  highest - lowest

theorem greatest_temp_diff_on_tuesday : temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_mon lowest_temp_mon 
  ∧ temp_diff highest_temp_tue lowest_temp_tue 
  > temp_diff highest_temp_wed lowest_temp_wed := 
by
  sorry

end greatest_temp_diff_on_tuesday_l1469_146923


namespace rice_in_each_container_l1469_146957

-- Given conditions from the problem
def total_weight_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- A theorem that each container has 25 ounces of rice given the conditions
theorem rice_in_each_container (h : total_weight_pounds * pounds_to_ounces / num_containers = 25) : True :=
  sorry

end rice_in_each_container_l1469_146957


namespace at_least_one_root_l1469_146994

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end at_least_one_root_l1469_146994


namespace shifted_roots_polynomial_l1469_146914

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ :=
  x^3 - 5 * x + 7

-- Define the shifted polynomial
def shifted_polynomial (x : ℝ) : ℝ :=
  x^3 + 9 * x^2 + 22 * x + 19

-- Define the roots condition
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop :=
  p r = 0

-- State the theorem
theorem shifted_roots_polynomial :
  ∀ a b c : ℝ,
    is_root original_polynomial a →
    is_root original_polynomial b →
    is_root original_polynomial c →
    is_root shifted_polynomial (a - 3) ∧
    is_root shifted_polynomial (b - 3) ∧
    is_root shifted_polynomial (c - 3) :=
by
  intros a b c ha hb hc
  sorry

end shifted_roots_polynomial_l1469_146914


namespace garden_square_char_l1469_146919

theorem garden_square_char (s q p x : ℕ) (h1 : p = 28) (h2 : q = p + x) (h3 : q = s^2) (h4 : p = 4 * s) : x = 21 :=
by
  sorry

end garden_square_char_l1469_146919


namespace three_children_meet_l1469_146998

theorem three_children_meet 
  (children : Finset ℕ)
  (visited_times : ℕ → ℕ)
  (meet_at_stand : ℕ → ℕ → Prop)
  (h_children_count : children.card = 7)
  (h_visited_times : ∀ c ∈ children, visited_times c = 3)
  (h_meet_pairwise : ∀ (c1 c2 : ℕ), c1 ∈ children → c2 ∈ children → c1 ≠ c2 → meet_at_stand c1 c2) :
  ∃ (t : ℕ), ∃ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
  c1 ∈ children ∧ c2 ∈ children ∧ c3 ∈ children ∧ 
  meet_at_stand c1 t ∧ meet_at_stand c2 t ∧ meet_at_stand c3 t := 
sorry

end three_children_meet_l1469_146998


namespace intersection_A_B_l1469_146942

def interval_A : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def interval_B : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem intersection_A_B :
  interval_A ∩ interval_B = { x | (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
sorry

end intersection_A_B_l1469_146942


namespace fraction_equality_l1469_146936

-- Defining the hypotheses and the goal
theorem fraction_equality (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 10 :=
by
  sorry

end fraction_equality_l1469_146936


namespace expected_value_correct_l1469_146997

-- Define the probabilities
def prob_8 : ℚ := 3 / 8
def prob_other : ℚ := 5 / 56 -- Derived from the solution steps but using only given conditions explicitly.

-- Define the expected value calculation
def expected_value_die : ℚ :=
  (1 * prob_other) + (2 * prob_other) + (3 * prob_other) + (4 * prob_other) +
  (5 * prob_other) + (6 * prob_other) + (7 * prob_other) + (8 * prob_8)

-- The theorem to prove
theorem expected_value_correct : expected_value_die = 77 / 14 := by
  sorry

end expected_value_correct_l1469_146997


namespace probability_of_B_not_losing_is_70_l1469_146943

-- Define the probabilities as given in the conditions
def prob_A_winning : ℝ := 0.30
def prob_draw : ℝ := 0.50

-- Define the probability of B not losing
def prob_B_not_losing : ℝ := 0.50 + (1 - prob_A_winning - prob_draw)

-- State the theorem
theorem probability_of_B_not_losing_is_70 :
  prob_B_not_losing = 0.70 := by
  sorry -- Proof to be filled in

end probability_of_B_not_losing_is_70_l1469_146943


namespace find_days_jane_indisposed_l1469_146934

-- Define the problem conditions
def John_rate := 1 / 20
def Jane_rate := 1 / 10
def together_rate := John_rate + Jane_rate
def total_task := 1
def total_days := 10

-- The time Jane was indisposed
def days_jane_indisposed (x : ℝ) : Prop :=
  (total_days - x) * together_rate + x * John_rate = total_task

-- Statement we want to prove
theorem find_days_jane_indisposed : ∃ x : ℝ, days_jane_indisposed x ∧ x = 5 :=
by 
  sorry

end find_days_jane_indisposed_l1469_146934


namespace cos_alpha_value_l1469_146993

open Real

theorem cos_alpha_value (α : ℝ) : 
  (sin (α - (π / 3)) = 1 / 5) ∧ (0 < α) ∧ (α < π / 2) → 
  (cos α = (2 * sqrt 6 - sqrt 3) / 10) := 
by
  intros h
  sorry

end cos_alpha_value_l1469_146993


namespace values_of_n_l1469_146980

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l1469_146980


namespace integer_value_of_K_l1469_146935

theorem integer_value_of_K (K : ℤ) : 
  (1000 < K^4 ∧ K^4 < 5000) ∧ K > 1 → K = 6 ∨ K = 7 ∨ K = 8 :=
by sorry

end integer_value_of_K_l1469_146935


namespace parabola_min_value_l1469_146951

theorem parabola_min_value (x : ℝ) : (∃ x, x^2 + 10 * x + 21 = -4) := sorry

end parabola_min_value_l1469_146951


namespace find_value_l1469_146962

variables {p q s u : ℚ}

theorem find_value
  (h1 : p / q = 5 / 6)
  (h2 : s / u = 7 / 15) :
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 :=
sorry

end find_value_l1469_146962


namespace true_propositions_among_converse_inverse_contrapositive_l1469_146945

theorem true_propositions_among_converse_inverse_contrapositive
  (x : ℝ)
  (h1 : x^2 ≥ 1 → x ≥ 1) :
  (if x ≥ 1 then x^2 ≥ 1 else true) ∧ 
  (if x^2 < 1 then x < 1 else true) ∧ 
  (if x < 1 then x^2 < 1 else true) → 
  ∃ n, n = 2 :=
by sorry

end true_propositions_among_converse_inverse_contrapositive_l1469_146945


namespace sequence_a_n_a5_eq_21_l1469_146990

theorem sequence_a_n_a5_eq_21 
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  a 5 = 21 :=
by
  sorry

end sequence_a_n_a5_eq_21_l1469_146990


namespace problem1_problem2_l1469_146907

-- Problem 1

def a : ℚ := -1 / 2
def b : ℚ := -1

theorem problem1 :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3 / 4 :=
by
  sorry

-- Problem 2

def x : ℚ := 1 / 2
def y : ℚ := -2 / 3
axiom condition2 : abs (2 * x - 1) + (3 * y + 2)^2 = 0

theorem problem2 :
  5 * x^2 - (2 * x * y - 3 * (x * y / 3 + 2) + 5 * x^2) = 19 / 3 :=
by
  have h : abs (2 * x - 1) + (3 * y + 2)^2 = 0 := condition2
  sorry

end problem1_problem2_l1469_146907


namespace paper_cups_count_l1469_146920

variables (P C : ℝ) (x : ℕ)

theorem paper_cups_count :
  100 * P + x * C = 7.50 ∧ 20 * P + 40 * C = 1.50 → x = 200 :=
sorry

end paper_cups_count_l1469_146920


namespace effective_speed_against_current_l1469_146970

theorem effective_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (headwind_speed : ℝ)
  (obstacle_reduction_pct : ℝ)
  (h_speed_with_current : speed_with_current = 25)
  (h_speed_of_current : speed_of_current = 4)
  (h_headwind_speed : headwind_speed = 2)
  (h_obstacle_reduction_pct : obstacle_reduction_pct = 0.15) :
  let speed_in_still_water := speed_with_current - speed_of_current
  let speed_against_current_headwind := speed_in_still_water - speed_of_current - headwind_speed
  let reduction_due_to_obstacles := obstacle_reduction_pct * speed_against_current_headwind
  let effective_speed := speed_against_current_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
{
  sorry
}

end effective_speed_against_current_l1469_146970


namespace tank_inflow_rate_l1469_146988

/-- 
  Tanks A and B have the same capacity of 20 liters. Tank A has
  an inflow rate of 2 liters per hour and takes 5 hours longer to
  fill than tank B. Show that the inflow rate in tank B is 4 liters 
  per hour.
-/
theorem tank_inflow_rate (capacity : ℕ) (rate_A : ℕ) (extra_time : ℕ) (rate_B : ℕ) 
  (h1 : capacity = 20) (h2 : rate_A = 2) (h3 : extra_time = 5) (h4 : capacity / rate_A = (capacity / rate_B) + extra_time) :
  rate_B = 4 :=
sorry

end tank_inflow_rate_l1469_146988


namespace gcd_lcm_sum_ge_sum_l1469_146987

theorem gcd_lcm_sum_ge_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  Nat.gcd a b + Nat.lcm a b ≥ a + b := 
sorry

end gcd_lcm_sum_ge_sum_l1469_146987


namespace area_of_centroid_path_l1469_146929

theorem area_of_centroid_path (A B C O G : ℝ) (r : ℝ) (h1 : A ≠ B) 
  (h2 : 2 * r = 30) (h3 : ∀ C, C ≠ A ∧ C ≠ B ∧ dist O C = r) 
  (h4 : dist O G = r / 3) : 
  (π * (r / 3)^2 = 25 * π) :=
by 
  -- def AB := 2 * r -- given AB is a diameter of the circle
  -- def O := (A + B) / 2 -- center of the circle
  -- def G := (A + B + C) / 3 -- centroid of triangle ABC
  sorry

end area_of_centroid_path_l1469_146929


namespace yang_hui_rect_eq_l1469_146918

theorem yang_hui_rect_eq (L W x : ℝ) 
  (h1 : L * W = 864)
  (h2 : L + W = 60)
  (h3 : L = W + x) : 
  (60 - x) / 2 * (60 + x) / 2 = 864 :=
by
  sorry

end yang_hui_rect_eq_l1469_146918


namespace power_equality_l1469_146924

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l1469_146924


namespace sequence_general_term_l1469_146933

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n > 1, a n = 2 * a (n-1) + 1) : a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l1469_146933


namespace cone_base_circumference_l1469_146983

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) (total_angle : ℝ) (C : ℝ) (h1 : r = 6) (h2 : sector_angle = 180) (h3 : total_angle = 360) (h4 : C = 2 * r * Real.pi) :
  (sector_angle / total_angle) * C = 6 * Real.pi :=
by
  -- Skipping proof
  sorry

end cone_base_circumference_l1469_146983


namespace sara_added_onions_l1469_146960

theorem sara_added_onions
  (initial_onions X : ℤ) 
  (h : initial_onions + X - 5 + 9 = initial_onions + 8) :
  X = 4 :=
by
  sorry

end sara_added_onions_l1469_146960


namespace edward_rides_eq_8_l1469_146944

-- Define the initial conditions
def initial_tickets : ℕ := 79
def spent_tickets : ℕ := 23
def cost_per_ride : ℕ := 7

-- Define the remaining tickets after spending at the booth
def remaining_tickets : ℕ := initial_tickets - spent_tickets

-- Define the number of rides Edward could go on
def number_of_rides : ℕ := remaining_tickets / cost_per_ride

-- The goal is to prove that the number of rides is equal to 8.
theorem edward_rides_eq_8 : number_of_rides = 8 := by sorry

end edward_rides_eq_8_l1469_146944


namespace bead_arrangement_probability_l1469_146912

def total_beads := 6
def red_beads := 2
def white_beads := 2
def blue_beads := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 6  -- Based on valid patterns RWBRWB, RWBWRB, and all other permutations for each starting color

def probability_valid := valid_arrangements / total_arrangements

theorem bead_arrangement_probability : probability_valid = 1 / 15 :=
  by
  -- The context and details of the solution steps are omitted as they are not included in the Lean theorem statement.
  -- This statement will skip the proof
  sorry

end bead_arrangement_probability_l1469_146912


namespace Cara_skate_distance_l1469_146932

-- Definitions corresponding to the conditions
def distance_CD : ℝ := 150
def speed_Cara : ℝ := 10
def speed_Dan : ℝ := 6
def angle_Cara_CD : ℝ := 45

-- main theorem based on the problem and given conditions
theorem Cara_skate_distance : ∃ t : ℝ, distance_CD = 150 ∧ speed_Cara = 10 ∧ speed_Dan = 6
                            ∧ angle_Cara_CD = 45 
                            ∧ 10 * t = 253.5 :=
by
  sorry

end Cara_skate_distance_l1469_146932


namespace debby_total_photos_l1469_146956

theorem debby_total_photos (friends_photos family_photos : ℕ) (h1 : friends_photos = 63) (h2 : family_photos = 23) : friends_photos + family_photos = 86 :=
by sorry

end debby_total_photos_l1469_146956


namespace min_red_beads_l1469_146953

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end min_red_beads_l1469_146953


namespace multiply_transformed_l1469_146952

theorem multiply_transformed : (268 * 74 = 19832) → (2.68 * 0.74 = 1.9832) :=
by
  intro h
  sorry

end multiply_transformed_l1469_146952


namespace expected_difference_l1469_146901

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_composite (n : ℕ) : Prop := n = 4 ∨ n = 6 ∨ n = 8

def roll_die : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def probability_eat_sweetened : ℚ := 4 / 7
def probability_eat_unsweetened : ℚ := 3 / 7
def days_in_leap_year : ℕ := 366

def expected_days_unsweetened : ℚ := probability_eat_unsweetened * days_in_leap_year
def expected_days_sweetened : ℚ := probability_eat_sweetened * days_in_leap_year

theorem expected_difference :
  expected_days_sweetened - expected_days_unsweetened = 52.28 := by
  sorry

end expected_difference_l1469_146901


namespace jennifer_money_left_l1469_146984

theorem jennifer_money_left (initial_amount : ℕ) (sandwich_fraction museum_ticket_fraction book_fraction : ℚ) 
  (h_initial : initial_amount = 90) 
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum_ticket : museum_ticket_fraction = 1/6)
  (h_book : book_fraction = 1/2) : 
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_ticket_fraction + initial_amount * book_fraction) = 12 :=
by
  sorry

end jennifer_money_left_l1469_146984


namespace tangent_line_equation_l1469_146921

-- Definitions used as conditions in the problem
def curve (x : ℝ) : ℝ := 2 * x - x^3
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Lean 4 statement representing the proof problem
theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := 1
  let m := deriv curve x₀
  m = -1 ∧ curve x₀ = y₀ →
  ∀ x y : ℝ, x + y - 2 = 0 → curve x₀ + m * (x - x₀) = y :=
by
  -- Proof would go here
  sorry

end tangent_line_equation_l1469_146921


namespace sam_cleaner_meetings_two_times_l1469_146973

open Nat

noncomputable def sam_and_cleaner_meetings (sam_rate cleaner_rate cleaner_stop_time bench_distance : ℕ) : ℕ :=
  let cycle_time := (bench_distance / cleaner_rate) + cleaner_stop_time
  let distance_covered_in_cycle_sam := sam_rate * cycle_time
  let distance_covered_in_cycle_cleaner := bench_distance
  let effective_distance_reduction := distance_covered_in_cycle_cleaner - distance_covered_in_cycle_sam
  let number_of_cycles_until_meeting := bench_distance / effective_distance_reduction
  number_of_cycles_until_meeting + 1

theorem sam_cleaner_meetings_two_times :
  sam_and_cleaner_meetings 3 9 40 300 = 2 :=
by sorry

end sam_cleaner_meetings_two_times_l1469_146973


namespace range_of_a_l1469_146917

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l1469_146917


namespace valentines_count_l1469_146979

theorem valentines_count (x y : ℕ) (h1 : (x = 2 ∧ y = 48) ∨ (x = 48 ∧ y = 2)) : 
  x * y - (x + y) = 46 := by
  sorry

end valentines_count_l1469_146979


namespace initial_bees_l1469_146963

variable (B : ℕ)

theorem initial_bees (h : B + 10 = 26) : B = 16 :=
by sorry

end initial_bees_l1469_146963


namespace property_value_at_beginning_l1469_146965

theorem property_value_at_beginning 
  (r : ℝ) (v3 : ℝ) (V : ℝ) (rate : ℝ) (years : ℕ) 
  (h_rate : rate = 6.25 / 100) 
  (h_years : years = 3) 
  (h_v3 : v3 = 21093) 
  (h_r : r = 1 - rate) 
  (h_V : V * r ^ years = v3) 
  : V = 25656.25 :=
by
  sorry

end property_value_at_beginning_l1469_146965


namespace sum_series_eq_two_l1469_146974

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l1469_146974


namespace numberOfRealSolutions_l1469_146905

theorem numberOfRealSolutions :
  ∀ (x : ℝ), (-4*x + 12)^2 + 1 = (x - 1)^2 → (∃ a b : ℝ, (a ≠ b) ∧ (-4*a + 12)^2 + 1 = (a - 1)^2 ∧ (-4*b + 12)^2 + 1 = (b - 1)^2) := by
  sorry

end numberOfRealSolutions_l1469_146905


namespace diagonal_of_larger_screen_l1469_146909

theorem diagonal_of_larger_screen (d : ℝ) 
  (h1 : ∃ s : ℝ, s^2 = 20^2 + 42) 
  (h2 : ∀ s, d = s * Real.sqrt 2) : 
  d = Real.sqrt 884 :=
by
  sorry

end diagonal_of_larger_screen_l1469_146909


namespace gcd_of_expression_l1469_146967

noncomputable def gcd_expression (a b c d : ℤ) : ℤ :=
  (a - b) * (b - c) * (c - d) * (d - a) * (b - d) * (a - c)

theorem gcd_of_expression : 
  ∀ (a b c d : ℤ), ∃ (k : ℤ), gcd_expression a b c d = 12 * k :=
sorry

end gcd_of_expression_l1469_146967


namespace minimum_value_inequality_l1469_146949

theorem minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → (2 / x + 1 / y) ≥ 9)) :=
by
  -- skipping the proof
  sorry

end minimum_value_inequality_l1469_146949


namespace intersection_complement_l1469_146904

def U : Set ℝ := Set.univ
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | x > 3}

theorem intersection_complement :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l1469_146904


namespace playground_perimeter_km_l1469_146940

def playground_length : ℕ := 360
def playground_width : ℕ := 480

def perimeter_in_meters (length width : ℕ) : ℕ := 2 * (length + width)

def perimeter_in_kilometers (perimeter_m : ℕ) : ℕ := perimeter_m / 1000

theorem playground_perimeter_km :
  perimeter_in_kilometers (perimeter_in_meters playground_length playground_width) = 168 :=
by
  sorry

end playground_perimeter_km_l1469_146940


namespace find_b_l1469_146991

theorem find_b (b : ℚ) : (∃ x y : ℚ, x = 3 ∧ y = -5 ∧ (b * x - (b + 2) * y = b - 3)) → b = -13 / 7 :=
sorry

end find_b_l1469_146991


namespace f_value_l1469_146964

def B := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) : ℝ := sorry

axiom f_property : ∀ x ∈ B, f x + f (2 - (1 / x)) = Real.log (abs (x ^ 2))

theorem f_value : f 2023 = Real.log 2023 :=
by
  sorry

end f_value_l1469_146964


namespace farmer_ploughing_problem_l1469_146976

theorem farmer_ploughing_problem (A D : ℕ) (h1 : A = 120 * D) (h2 : A - 40 = 85 * (D + 2)) : 
  A = 720 ∧ D = 6 :=
by
  sorry

end farmer_ploughing_problem_l1469_146976


namespace original_purchase_price_l1469_146982

-- Define the conditions and question
theorem original_purchase_price (P S : ℝ) (h1 : S = P + 0.25 * S) (h2 : 16 = 0.80 * S - P) : P = 240 :=
by
  -- Proof steps would go here
  sorry

end original_purchase_price_l1469_146982


namespace total_weight_of_rings_l1469_146966

-- Conditions
def weight_orange : ℝ := 0.08333333333333333
def weight_purple : ℝ := 0.3333333333333333
def weight_white : ℝ := 0.4166666666666667

-- Goal
theorem total_weight_of_rings : weight_orange + weight_purple + weight_white = 0.8333333333333333 := by
  sorry

end total_weight_of_rings_l1469_146966


namespace value_of_5_S_3_l1469_146915

def operation_S (a b : ℝ) : ℝ := 4 * a + 6 * b - 2 * a * b

theorem value_of_5_S_3 : operation_S 5 3 = 8 :=
by
  sorry

end value_of_5_S_3_l1469_146915


namespace lower_seat_tickets_l1469_146913

theorem lower_seat_tickets (L U : ℕ) (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end lower_seat_tickets_l1469_146913


namespace area_square_II_is_6a_squared_l1469_146996

-- Problem statement:
-- Given the diagonal of square I is 2a and the area of square II is three times the area of square I,
-- prove that the area of square II is 6a^2

noncomputable def area_square_II (a : ℝ) : ℝ :=
  let side_I := (2 * a) / Real.sqrt 2
  let area_I := side_I ^ 2
  3 * area_I

theorem area_square_II_is_6a_squared (a : ℝ) : area_square_II a = 6 * a ^ 2 :=
by
  sorry

end area_square_II_is_6a_squared_l1469_146996


namespace overlap_length_l1469_146947

noncomputable def length_of_all_red_segments := 98 -- in cm
noncomputable def total_length := 83 -- in cm
noncomputable def number_of_overlaps := 6 -- count

theorem overlap_length :
  ∃ (x : ℝ), length_of_all_red_segments - total_length = number_of_overlaps * x ∧ x = 2.5 := by
  sorry

end overlap_length_l1469_146947


namespace convert_3652_from_base7_to_base10_l1469_146916

def base7ToBase10(n : ℕ) := 
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  d0 * (7^0) + d1 * (7^1) + d2 * (7^2) + d3 * (7^3)

theorem convert_3652_from_base7_to_base10 : base7ToBase10 3652 = 1360 :=
by
  sorry

end convert_3652_from_base7_to_base10_l1469_146916


namespace xiao_hong_mistake_l1469_146995

theorem xiao_hong_mistake (a : ℕ) (h : 31 - a = 12) : 31 + a = 50 :=
by
  sorry

end xiao_hong_mistake_l1469_146995


namespace autograph_value_after_changes_l1469_146969

def initial_value : ℝ := 100
def drop_percent : ℝ := 0.30
def increase_percent : ℝ := 0.40

theorem autograph_value_after_changes :
  let value_after_drop := initial_value * (1 - drop_percent)
  let value_after_increase := value_after_drop * (1 + increase_percent)
  value_after_increase = 98 :=
by
  sorry

end autograph_value_after_changes_l1469_146969


namespace avg_salary_supervisors_l1469_146961

-- Definitions based on the conditions of the problem
def total_workers : Nat := 48
def supervisors : Nat := 6
def laborers : Nat := 42
def avg_salary_total : Real := 1250
def avg_salary_laborers : Real := 950

-- Given the above conditions, we need to prove the average salary of the supervisors.
theorem avg_salary_supervisors :
  (supervisors * (supervisors * total_workers * avg_salary_total - laborers * avg_salary_laborers) / supervisors) = 3350 :=
by
  sorry

end avg_salary_supervisors_l1469_146961


namespace value_of_a_l1469_146906

noncomputable def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ :=
  a * b^3 + c

theorem value_of_a :
  F a 2 3 = F a 3 4 → a = -1 / 19 :=
by
  sorry

end value_of_a_l1469_146906


namespace symmetric_line_eq_l1469_146925

theorem symmetric_line_eq (x y : ℝ) :
  (y = 2 * x + 3) → (y - 1 = x + 1) → (x - 2 * y = 0) :=
by
  intros h1 h2
  sorry

end symmetric_line_eq_l1469_146925


namespace team_leader_prize_l1469_146938

theorem team_leader_prize 
    (number_of_students : ℕ := 10)
    (number_of_team_members : ℕ := 9)
    (team_member_prize : ℕ := 200)
    (additional_leader_prize : ℕ := 90)
    (total_prize : ℕ)
    (leader_prize : ℕ := total_prize - (number_of_team_members * team_member_prize))
    (average_prize : ℕ := (total_prize + additional_leader_prize) / number_of_students)
: leader_prize = 300 := 
by {
  sorry  -- Proof omitted
}

end team_leader_prize_l1469_146938


namespace number_eq_1925_l1469_146992

theorem number_eq_1925 (x : ℝ) (h : x / 7 - x / 11 = 100) : x = 1925 :=
sorry

end number_eq_1925_l1469_146992


namespace area_of_parallelogram_l1469_146959

theorem area_of_parallelogram (base : ℝ) (height : ℝ)
  (h1 : base = 3.6)
  (h2 : height = 2.5 * base) :
  base * height = 32.4 :=
by
  sorry

end area_of_parallelogram_l1469_146959


namespace bounds_on_xyz_l1469_146948

theorem bounds_on_xyz (a x y z : ℝ) (h1 : x + y + z = a)
                      (h2 : x^2 + y^2 + z^2 = (a^2) / 2)
                      (h3 : a > 0) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z) :
                      (0 < x ∧ x ≤ (2 / 3) * a) ∧ 
                      (0 < y ∧ y ≤ (2 / 3) * a) ∧ 
                      (0 < z ∧ z ≤ (2 / 3) * a) :=
sorry

end bounds_on_xyz_l1469_146948
