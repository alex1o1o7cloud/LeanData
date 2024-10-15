import Mathlib

namespace NUMINAMATH_GPT_average_payment_52_installments_l2075_207520

theorem average_payment_52_installments :
  let first_payment : ℕ := 500
  let remaining_payment : ℕ := first_payment + 100
  let num_first_payments : ℕ := 25
  let num_remaining_payments : ℕ := 27
  let total_payments : ℕ := num_first_payments + num_remaining_payments
  let total_paid_first : ℕ := num_first_payments * first_payment
  let total_paid_remaining : ℕ := num_remaining_payments * remaining_payment
  let total_paid : ℕ := total_paid_first + total_paid_remaining
  let average_payment : ℚ := total_paid / total_payments
  average_payment = 551.92 :=
by
  sorry

end NUMINAMATH_GPT_average_payment_52_installments_l2075_207520


namespace NUMINAMATH_GPT_trailing_zeros_1_to_100_l2075_207575

def count_multiples (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def trailing_zeros_in_range (n : ℕ) : ℕ :=
  let multiples_of_5 := count_multiples n 5
  let multiples_of_25 := count_multiples n 25
  multiples_of_5 + multiples_of_25

theorem trailing_zeros_1_to_100 : trailing_zeros_in_range 100 = 24 := by
  sorry

end NUMINAMATH_GPT_trailing_zeros_1_to_100_l2075_207575


namespace NUMINAMATH_GPT_alpha_beta_range_l2075_207589

theorem alpha_beta_range (α β : ℝ) (h1 : - (π / 2) < α) (h2 : α < β) (h3 : β < π) : 
- 3 * (π / 2) < α - β ∧ α - β < 0 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_range_l2075_207589


namespace NUMINAMATH_GPT_total_time_spent_l2075_207596

variable (B I E M EE ST ME : ℝ)

def learn_basic_rules : ℝ := B
def learn_intermediate_level : ℝ := I
def learn_expert_level : ℝ := E
def learn_master_level : ℝ := M
def endgame_exercises : ℝ := EE
def middle_game_strategy_tactics : ℝ := ST
def mentoring : ℝ := ME

theorem total_time_spent :
  B = 2 →
  I = 75 * B →
  E = 50 * (B + I) →
  M = 30 * E →
  EE = 0.25 * I →
  ST = 2 * EE →
  ME = 0.5 * E →
  B + I + E + M + EE + ST + ME = 235664.5 :=
by
  intros hB hI hE hM hEE hST hME
  rw [hB, hI, hE, hM, hEE, hST, hME]
  sorry

end NUMINAMATH_GPT_total_time_spent_l2075_207596


namespace NUMINAMATH_GPT_problem_statement_l2075_207539

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * cos (ω * x + π / 6)

theorem problem_statement (ω : ℝ) (hx : ω = 2 ∨ ω = -2) :
  f ω (π / 3) = -3 ∨ f ω (π / 3) = 0 := by
  unfold f
  cases hx with
  | inl w_eq => sorry
  | inr w_eq => sorry

end NUMINAMATH_GPT_problem_statement_l2075_207539


namespace NUMINAMATH_GPT_doughnut_machine_completion_time_l2075_207581

noncomputable def start_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
noncomputable def one_third_time : ℕ := 11 * 60 + 10  -- 11:10 AM in minutes
noncomputable def total_time_minutes : ℕ := 8 * 60  -- 8 hours in minutes
noncomputable def expected_completion_time : ℕ := 16 * 60 + 30  -- 4:30 PM in minutes

theorem doughnut_machine_completion_time :
  one_third_time - start_time = total_time_minutes / 3 →
  start_time + total_time_minutes = expected_completion_time :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_doughnut_machine_completion_time_l2075_207581


namespace NUMINAMATH_GPT_sequence_properties_l2075_207591

def f (x : ℝ) : ℝ := x^3 + 3 * x

variables {a_5 a_8 : ℝ}
variables {S_12 : ℝ}

axiom a5_condition : (a_5 - 1)^3 + 3 * a_5 = 4
axiom a8_condition : (a_8 - 1)^3 + 3 * a_8 = 2

theorem sequence_properties : (a_5 > a_8) ∧ (S_12 = 12) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_properties_l2075_207591


namespace NUMINAMATH_GPT_simplify_fraction_l2075_207543

theorem simplify_fraction :
  (1 / (1 + Real.sqrt 3) * 1 / (1 - Real.sqrt 5)) = 
  (1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2075_207543


namespace NUMINAMATH_GPT_find_positive_integer_n_l2075_207519

theorem find_positive_integer_n (S : ℕ → ℚ) (hS : ∀ n, S n = n / (n + 1))
  (h : ∃ n : ℕ, S n * S (n + 1) = 3 / 4) : 
  ∃ n : ℕ, n = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_positive_integer_n_l2075_207519


namespace NUMINAMATH_GPT_john_total_spent_is_correct_l2075_207588

noncomputable def john_spent_total (original_cost : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_cost := original_cost - (discount_rate / 100 * original_cost)
  let cost_with_tax := discounted_cost + (sales_tax_rate / 100 * discounted_cost)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_cost_with_tax := lightsaber_cost + (sales_tax_rate / 100 * lightsaber_cost)
  cost_with_tax + lightsaber_cost_with_tax

theorem john_total_spent_is_correct :
  john_spent_total 1200 20 8 = 3628.80 :=
by
  sorry

end NUMINAMATH_GPT_john_total_spent_is_correct_l2075_207588


namespace NUMINAMATH_GPT_red_paint_four_times_blue_paint_total_painted_faces_is_1625_l2075_207565

/-- Given a structure of twenty-five layers of cubes -/
def structure_layers := 25

/-- The number of painted faces from each vertical view -/
def vertical_faces_per_view : ℕ :=
  (structure_layers * (structure_layers + 1)) / 2

/-- The total number of red-painted faces (4 vertical views) -/
def total_red_faces : ℕ :=
  4 * vertical_faces_per_view

/-- The total number of blue-painted faces (1 top view) -/
def total_blue_faces : ℕ :=
  vertical_faces_per_view

theorem red_paint_four_times_blue_paint :
  total_red_faces = 4 * total_blue_faces :=
by sorry

theorem total_painted_faces_is_1625 :
  (4 * vertical_faces_per_view + vertical_faces_per_view) = 1625 :=
by sorry

end NUMINAMATH_GPT_red_paint_four_times_blue_paint_total_painted_faces_is_1625_l2075_207565


namespace NUMINAMATH_GPT_cos_150_degree_l2075_207540

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_150_degree_l2075_207540


namespace NUMINAMATH_GPT_young_people_in_sample_l2075_207515

-- Define the conditions
def total_population (elderly middle_aged young : ℕ) : ℕ :=
  elderly + middle_aged + young

def sample_proportion (sample_size total_pop : ℚ) : ℚ :=
  sample_size / total_pop

def stratified_sample (group_size proportion : ℚ) : ℚ :=
  group_size * proportion

-- Main statement to prove
theorem young_people_in_sample (elderly middle_aged young : ℕ) (sample_size : ℚ) :
  total_population elderly middle_aged young = 108 →
  sample_size = 36 →
  stratified_sample (young : ℚ) (sample_proportion sample_size 108) = 17 :=
by
  intros h_total h_sample_size
  sorry -- proof omitted

end NUMINAMATH_GPT_young_people_in_sample_l2075_207515


namespace NUMINAMATH_GPT_find_negative_number_l2075_207545

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end NUMINAMATH_GPT_find_negative_number_l2075_207545


namespace NUMINAMATH_GPT_barbara_removed_114_sheets_l2075_207527

/-- Given conditions: -/
def bundles (n : ℕ) := 2 * n
def bunches (n : ℕ) := 4 * n
def heaps (n : ℕ) := 20 * n

/-- Barbara removed certain amounts of paper from the chest of drawers. -/
def total_sheets_removed := bundles 3 + bunches 2 + heaps 5

theorem barbara_removed_114_sheets : total_sheets_removed = 114 := by
  -- proof will be inserted here
  sorry

end NUMINAMATH_GPT_barbara_removed_114_sheets_l2075_207527


namespace NUMINAMATH_GPT_solve_fractional_equation_l2075_207535

theorem solve_fractional_equation (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ -6) :
    (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2075_207535


namespace NUMINAMATH_GPT_probability_of_ram_l2075_207505

theorem probability_of_ram 
  (P_ravi : ℝ) (P_both : ℝ) 
  (h_ravi : P_ravi = 1 / 5) 
  (h_both : P_both = 0.11428571428571428) : 
  ∃ P_ram : ℝ, P_ram = 0.5714285714285714 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_ram_l2075_207505


namespace NUMINAMATH_GPT_solve_for_x_l2075_207526

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h_eq : x^4 = 6561) : x = 9 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2075_207526


namespace NUMINAMATH_GPT_class_average_l2075_207533

theorem class_average (x : ℝ) :
  (0.25 * 80 + 0.5 * x + 0.25 * 90 = 75) → x = 65 := by
  sorry

end NUMINAMATH_GPT_class_average_l2075_207533


namespace NUMINAMATH_GPT_candies_shared_l2075_207531

theorem candies_shared (y b d x : ℕ) (h1 : x = 2 * y + 10) (h2 : x = 3 * b + 18) (h3 : x = 5 * d - 55) (h4 : x + y + b + d = 2013) : x = 990 :=
by
  sorry

end NUMINAMATH_GPT_candies_shared_l2075_207531


namespace NUMINAMATH_GPT_bisecting_line_eq_l2075_207521

theorem bisecting_line_eq : ∃ (a : ℝ), (∀ x y : ℝ, (y = a * x) ↔ y = -1 / 6 * x) ∧ 
  (∀ p : ℝ × ℝ, (3 * p.1 - 5 * p.2  = 6 → p.2 = a * p.1) ∧ 
                  (4 * p.1 + p.2 + 6 = 0 → p.2 = a * p.1)) :=
by
  use -1 / 6
  sorry

end NUMINAMATH_GPT_bisecting_line_eq_l2075_207521


namespace NUMINAMATH_GPT_percent_decrease_in_cost_l2075_207528

theorem percent_decrease_in_cost (cost_1990 cost_2010 : ℕ) (h1 : cost_1990 = 35) (h2 : cost_2010 = 5) : 
  ((cost_1990 - cost_2010) * 100 / cost_1990 : ℚ) = 86 := 
by
  sorry

end NUMINAMATH_GPT_percent_decrease_in_cost_l2075_207528


namespace NUMINAMATH_GPT_oxygen_atoms_l2075_207507

theorem oxygen_atoms (x : ℤ) (h : 27 + 16 * x + 3 = 78) : x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_oxygen_atoms_l2075_207507


namespace NUMINAMATH_GPT_largest_four_digit_number_l2075_207510

def is_four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N ≤ 9999

def sum_of_digits (N : ℕ) : ℕ :=
  let a := N / 1000
  let b := (N % 1000) / 100
  let c := (N % 100) / 10
  let d := N % 10
  a + b + c + d

def is_divisible (N S : ℕ) : Prop := N % S = 0

theorem largest_four_digit_number :
  ∃ N : ℕ, is_four_digit_number N ∧ is_divisible N (sum_of_digits N) ∧
  (∀ M : ℕ, is_four_digit_number M ∧ is_divisible M (sum_of_digits M) → N ≥ M) ∧ N = 9990 :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_l2075_207510


namespace NUMINAMATH_GPT_sugar_concentration_after_adding_water_l2075_207509

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end NUMINAMATH_GPT_sugar_concentration_after_adding_water_l2075_207509


namespace NUMINAMATH_GPT_find_a_2016_l2075_207574

theorem find_a_2016 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 1, a (n + 1) = 3 * S n)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)):
  a 2016 = 3 * 4 ^ 2014 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_2016_l2075_207574


namespace NUMINAMATH_GPT_ratio_a_c_l2075_207598

theorem ratio_a_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 1) 
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 32 := 
by sorry

end NUMINAMATH_GPT_ratio_a_c_l2075_207598


namespace NUMINAMATH_GPT_find_added_number_l2075_207590

theorem find_added_number (R D Q X : ℕ) (hR : R = 5) (hD : D = 3 * Q) (hDiv : 113 = D * Q + R) (hD_def : D = 3 * R + X) : 
  X = 3 :=
by
  -- Provide the conditions as assumptions
  sorry

end NUMINAMATH_GPT_find_added_number_l2075_207590


namespace NUMINAMATH_GPT_angle_bisector_length_l2075_207587

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end NUMINAMATH_GPT_angle_bisector_length_l2075_207587


namespace NUMINAMATH_GPT_coin_same_side_probability_l2075_207567

noncomputable def probability_same_side_5_tosses (p : ℚ) := (p ^ 5) + (p ^ 5)

theorem coin_same_side_probability : probability_same_side_5_tosses (1/2) = 1/16 := by
  sorry

end NUMINAMATH_GPT_coin_same_side_probability_l2075_207567


namespace NUMINAMATH_GPT_find_correct_value_l2075_207544

theorem find_correct_value (incorrect_value : ℝ) (subtracted_value : ℝ) (added_value : ℝ) (h_sub : subtracted_value = -added_value)
(h_incorrect : incorrect_value = 8.8) (h_subtracted : subtracted_value = -4.3) (h_added : added_value = 4.3) : incorrect_value + added_value + added_value = 17.4 :=
by
  sorry

end NUMINAMATH_GPT_find_correct_value_l2075_207544


namespace NUMINAMATH_GPT_m_not_in_P_l2075_207558

noncomputable def m : ℝ := Real.sqrt 3
def P : Set ℝ := { x | x^2 - Real.sqrt 2 * x ≤ 0 }

theorem m_not_in_P : m ∉ P := by
  sorry

end NUMINAMATH_GPT_m_not_in_P_l2075_207558


namespace NUMINAMATH_GPT_distinct_values_in_expression_rearrangement_l2075_207586

theorem distinct_values_in_expression_rearrangement : 
  ∀ (exp : ℕ), exp = 3 → 
  (∃ n : ℕ, n = 3 ∧ 
    let a := exp ^ (exp ^ exp)
    let b := exp ^ ((exp ^ exp) ^ exp)
    let c := ((exp ^ exp) ^ exp) ^ exp
    let d := (exp ^ (exp ^ exp)) ^ exp
    let e := (exp ^ exp) ^ (exp ^ exp)
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :=
by
  sorry

end NUMINAMATH_GPT_distinct_values_in_expression_rearrangement_l2075_207586


namespace NUMINAMATH_GPT_gcd_lcm_product_l2075_207522

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 75) (h2 : b = 90) : Nat.gcd a b * Nat.lcm a b = 6750 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l2075_207522


namespace NUMINAMATH_GPT_value_of_expression_l2075_207532

noncomputable def x := (2 : ℚ) / 3
noncomputable def y := (5 : ℚ) / 2

theorem value_of_expression : (1 / 3) * x^8 * y^9 = (5^9 / (2 * 3^9)) := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2075_207532


namespace NUMINAMATH_GPT_coin_exchange_proof_l2075_207556

/-- Prove the coin combination that Petya initially had -/
theorem coin_exchange_proof (x y z : ℕ) (hx : 20 * x + 15 * y + 10 * z = 125) : x = 0 ∧ y = 1 ∧ z = 11 :=
by
  sorry

end NUMINAMATH_GPT_coin_exchange_proof_l2075_207556


namespace NUMINAMATH_GPT_triangle_sides_from_rhombus_l2075_207571

variable (m p q : ℝ)

def is_triangle_side_lengths (BC AC AB : ℝ) :=
  (BC = p + q) ∧
  (AC = m * (p + q) / p) ∧
  (AB = m * (p + q) / q)

theorem triangle_sides_from_rhombus :
  ∃ BC AC AB : ℝ, is_triangle_side_lengths m p q BC AC AB :=
by
  use p + q
  use m * (p + q) / p
  use m * (p + q) / q
  sorry

end NUMINAMATH_GPT_triangle_sides_from_rhombus_l2075_207571


namespace NUMINAMATH_GPT_least_positive_integer_solution_l2075_207577

theorem least_positive_integer_solution : 
  ∃ x : ℕ, x + 3567 ≡ 1543 [MOD 14] ∧ x = 6 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_least_positive_integer_solution_l2075_207577


namespace NUMINAMATH_GPT_recommended_cups_l2075_207512

theorem recommended_cups (current_cups : ℕ) (R : ℕ) : 
  current_cups = 20 →
  R = current_cups + (6 / 10) * current_cups →
  R = 32 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_recommended_cups_l2075_207512


namespace NUMINAMATH_GPT_boatman_current_speed_and_upstream_time_l2075_207594

variables (v : ℝ) (v_T : ℝ) (t_up : ℝ) (t_total : ℝ) (dist : ℝ) (d1 : ℝ) (d2 : ℝ)

theorem boatman_current_speed_and_upstream_time
  (h1 : dist = 12.5)
  (h2 : d1 = 3)
  (h3 : d2 = 5)
  (h4 : t_total = 8)
  (h5 : ∀ t, t = d1 / (v - v_T))
  (h6 : ∀ t, t = d2 / (v + v_T))
  (h7 : dist / (v - v_T) + dist / (v + v_T) = t_total) :
  v_T = 5 / 6 ∧ t_up = 5 := by
  sorry

end NUMINAMATH_GPT_boatman_current_speed_and_upstream_time_l2075_207594


namespace NUMINAMATH_GPT_rectangle_perimeter_l2075_207525

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem rectangle_perimeter
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 l w : ℕ)
  (h1 : a1 + a2 + a3 = a9)
  (h2 : a1 + a2 = a3)
  (h3 : a1 + a3 = a4)
  (h4 : a3 + a4 = a5)
  (h5 : a4 + a5 = a6)
  (h6 : a2 + a3 + a5 = a7)
  (h7 : a2 + a7 = a8)
  (h8 : a1 + a4 + a6 = a9)
  (h9 : a6 + a9 = a7 + a8)
  (h_rel_prime : relatively_prime l w)
  (h_dimensions : l = 61)
  (h_dimensions_w : w = 69) :
  2 * l + 2 * w = 260 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2075_207525


namespace NUMINAMATH_GPT_probability_ephraim_keiko_l2075_207536

-- Define the probability that Ephraim gets a certain number of heads tossing two pennies
def prob_heads_ephraim (n : Nat) : ℚ :=
  if n = 2 then 1 / 4
  else if n = 1 then 1 / 2
  else if n = 0 then 1 / 4
  else 0

-- Define the probability that Keiko gets a certain number of heads tossing one penny
def prob_heads_keiko (n : Nat) : ℚ :=
  if n = 1 then 1 / 2
  else if n = 0 then 1 / 2
  else 0

-- Define the probability that Ephraim and Keiko get the same number of heads
def prob_same_heads : ℚ :=
  (prob_heads_ephraim 0 * prob_heads_keiko 0) + (prob_heads_ephraim 1 * prob_heads_keiko 1) + (prob_heads_ephraim 2 * prob_heads_keiko 2)

-- The statement that requires proof
theorem probability_ephraim_keiko : prob_same_heads = 3 / 8 := 
  sorry

end NUMINAMATH_GPT_probability_ephraim_keiko_l2075_207536


namespace NUMINAMATH_GPT_tan_five_pi_over_four_l2075_207552

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_l2075_207552


namespace NUMINAMATH_GPT_time_spent_on_seals_l2075_207516

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end NUMINAMATH_GPT_time_spent_on_seals_l2075_207516


namespace NUMINAMATH_GPT_digit_2023_in_fractional_expansion_l2075_207502

theorem digit_2023_in_fractional_expansion :
  ∃ d : ℕ, (d = 4) ∧ (∃ n_block : ℕ, n_block = 6 ∧ (∃ p : Nat, p = 2023 ∧ ∃ r : ℕ, r = p % n_block ∧ r = 1)) :=
sorry

end NUMINAMATH_GPT_digit_2023_in_fractional_expansion_l2075_207502


namespace NUMINAMATH_GPT_sqrt_of_4_l2075_207566

theorem sqrt_of_4 (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_GPT_sqrt_of_4_l2075_207566


namespace NUMINAMATH_GPT_local_minimum_point_l2075_207537

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point (a : ℝ) (h : ∃ δ > 0, ∀ x, abs (x - a) < δ → f x ≥ f a) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_local_minimum_point_l2075_207537


namespace NUMINAMATH_GPT_train_length_is_correct_l2075_207529

noncomputable def speed_kmph : ℝ := 72
noncomputable def time_seconds : ℝ := 74.994
noncomputable def tunnel_length_m : ℝ := 1400
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance : ℝ := speed_mps * time_seconds
noncomputable def train_length : ℝ := total_distance - tunnel_length_m

theorem train_length_is_correct :
  train_length = 99.88 := by
  -- the proof will follow here
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l2075_207529


namespace NUMINAMATH_GPT_tree_height_l2075_207514

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end NUMINAMATH_GPT_tree_height_l2075_207514


namespace NUMINAMATH_GPT_directly_proportional_l2075_207513

-- Defining conditions
def A (x y : ℝ) : Prop := y = x + 8
def B (x y : ℝ) : Prop := (2 / (5 * y)) = x
def C (x y : ℝ) : Prop := (2 / 3) * x = y

-- Theorem stating that in the given equations, equation C shows direct proportionality
theorem directly_proportional (x y : ℝ) : C x y ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by
  sorry

end NUMINAMATH_GPT_directly_proportional_l2075_207513


namespace NUMINAMATH_GPT_rate_of_interest_l2075_207564

theorem rate_of_interest (R : ℝ) (h : 5000 * 2 * R / 100 + 3000 * 4 * R / 100 = 2200) : R = 10 := by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l2075_207564


namespace NUMINAMATH_GPT_new_average_commission_is_250_l2075_207582

-- Definitions based on the problem conditions
def C : ℝ := 1000
def n : ℝ := 6
def increase_in_average_commission : ℝ := 150

-- Theorem stating the new average commission is $250
theorem new_average_commission_is_250 (x : ℝ) (h1 : x + increase_in_average_commission = (5 * x + C) / n) :
  x + increase_in_average_commission = 250 := by
  sorry

end NUMINAMATH_GPT_new_average_commission_is_250_l2075_207582


namespace NUMINAMATH_GPT_sequence_bounds_l2075_207538

theorem sequence_bounds (c : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = ↑n + c / ↑n) 
  (h2 : ∀ n : ℕ+, a n ≥ a 3) : 6 ≤ c ∧ c ≤ 12 :=
by 
  -- We will prove that 6 ≤ c and c ≤ 12 given the conditions stated
  sorry

end NUMINAMATH_GPT_sequence_bounds_l2075_207538


namespace NUMINAMATH_GPT_remaining_money_l2075_207547

def initial_amount : ℕ := 10
def spent_on_toy_truck : ℕ := 3
def spent_on_pencil_case : ℕ := 2

theorem remaining_money (initial_amount spent_on_toy_truck spent_on_pencil_case : ℕ) : 
  initial_amount - (spent_on_toy_truck + spent_on_pencil_case) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remaining_money_l2075_207547


namespace NUMINAMATH_GPT_smallest_logarithmic_term_l2075_207573

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_logarithmic_term (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_interval : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) := 
by
  sorry

end NUMINAMATH_GPT_smallest_logarithmic_term_l2075_207573


namespace NUMINAMATH_GPT_pair_not_product_48_l2075_207503

theorem pair_not_product_48:
  (∀(a b : ℤ), (a, b) = (-6, -8)                    → a * b = 48) ∧
  (∀(a b : ℤ), (a, b) = (-4, -12)                   → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (3/4, -64)                  → a * b ≠ 48) ∧
  (∀(a b : ℤ), (a, b) = (3, 16)                     → a * b = 48) ∧
  (∀(a b : ℚ), (a, b) = (4/3, 36)                   → a * b = 48)
  :=
by
  sorry

end NUMINAMATH_GPT_pair_not_product_48_l2075_207503


namespace NUMINAMATH_GPT_tangency_lines_intersect_at_diagonal_intersection_point_l2075_207548

noncomputable def point := Type
noncomputable def line := Type

noncomputable def tangency (C : point) (l : line) : Prop := sorry
noncomputable def circumscribed (Q : point × point × point × point) (C : point) : Prop := sorry
noncomputable def intersects (l1 l2 : line) (P : point) : Prop := sorry
noncomputable def connects_opposite_tangency (Q : point × point × point × point) (l1 l2 : line) : Prop := sorry
noncomputable def diagonals_intersect_at (Q : point × point × point × point) (P : point) : Prop := sorry

theorem tangency_lines_intersect_at_diagonal_intersection_point :
  ∀ (Q : point × point × point × point) (C P : point), 
  circumscribed Q C →
  diagonals_intersect_at Q P →
  ∀ (l1 l2 : line), connects_opposite_tangency Q l1 l2 →
  intersects l1 l2 P :=
sorry

end NUMINAMATH_GPT_tangency_lines_intersect_at_diagonal_intersection_point_l2075_207548


namespace NUMINAMATH_GPT_milkshake_cost_proof_l2075_207569

-- Define the problem
def milkshake_cost (total_money : ℕ) (hamburger_cost : ℕ) (n_hamburgers : ℕ)
                   (n_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_hamburgers_cost := n_hamburgers * hamburger_cost
  let money_after_hamburgers := total_money - total_hamburgers_cost
  let milkshake_cost := (money_after_hamburgers - remaining_money) / n_milkshakes
  milkshake_cost

-- Statement to prove
theorem milkshake_cost_proof : milkshake_cost 120 4 8 6 70 = 3 :=
by
  -- we skip the proof steps as the problem statement does not require it
  sorry

end NUMINAMATH_GPT_milkshake_cost_proof_l2075_207569


namespace NUMINAMATH_GPT_relationship_between_x_x_squared_and_x_cubed_l2075_207570

theorem relationship_between_x_x_squared_and_x_cubed (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : x < x^3 ∧ x^3 < x^2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_x_x_squared_and_x_cubed_l2075_207570


namespace NUMINAMATH_GPT_find_number_l2075_207530

theorem find_number (x : ℕ) (h : x * 9999 = 724817410) : x = 72492 :=
sorry

end NUMINAMATH_GPT_find_number_l2075_207530


namespace NUMINAMATH_GPT_find_initial_number_l2075_207550

theorem find_initial_number (x : ℝ) (h : x + 12.808 - 47.80600000000004 = 3854.002) : x = 3889 := by
  sorry

end NUMINAMATH_GPT_find_initial_number_l2075_207550


namespace NUMINAMATH_GPT_apprentice_daily_output_l2075_207560

namespace Production

variables (x y : ℝ)

theorem apprentice_daily_output
  (h1 : 4 * x + 7 * y = 765)
  (h2 : 6 * x + 2 * y = 765) :
  y = 45 :=
sorry

end Production

end NUMINAMATH_GPT_apprentice_daily_output_l2075_207560


namespace NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l2075_207563

theorem solve_first_equation (x : ℤ) : 4 * x + 3 = 5 * x - 1 → x = 4 :=
by
  intros h
  sorry

theorem solve_second_equation (x : ℤ) : 4 * (x - 1) = 1 - x → x = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l2075_207563


namespace NUMINAMATH_GPT_total_weekly_pay_l2075_207511

theorem total_weekly_pay (Y_pay: ℝ) (X_pay: ℝ) (Y_weekly: Y_pay = 150) (X_weekly: X_pay = 1.2 * Y_pay) : 
  X_pay + Y_pay = 330 :=
by sorry

end NUMINAMATH_GPT_total_weekly_pay_l2075_207511


namespace NUMINAMATH_GPT_remainder_div_l2075_207541

theorem remainder_div (N : ℕ) (n : ℕ) : 
  (N % 2^n) = (N % 10^n % 2^n) ∧ (N % 5^n) = (N % 10^n % 5^n) := by
  sorry

end NUMINAMATH_GPT_remainder_div_l2075_207541


namespace NUMINAMATH_GPT_number_times_half_squared_is_eight_l2075_207559

noncomputable def num : ℝ := 32

theorem number_times_half_squared_is_eight :
  (num * (1 / 2) ^ 2 = 2 ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_number_times_half_squared_is_eight_l2075_207559


namespace NUMINAMATH_GPT_paint_cost_per_quart_l2075_207551

theorem paint_cost_per_quart
  (total_cost : ℝ)
  (coverage_per_quart : ℝ)
  (side_length : ℝ)
  (cost_per_quart : ℝ) 
  (h1 : total_cost = 192)
  (h2 : coverage_per_quart = 10)
  (h3 : side_length = 10) 
  (h4 : cost_per_quart = total_cost / ((6 * side_length ^ 2) / coverage_per_quart))
  : cost_per_quart = 3.20 := 
by 
  sorry

end NUMINAMATH_GPT_paint_cost_per_quart_l2075_207551


namespace NUMINAMATH_GPT_percentage_of_boys_answered_neither_l2075_207518

theorem percentage_of_boys_answered_neither (P_A P_B P_A_and_B : ℝ) (hP_A : P_A = 0.75) (hP_B : P_B = 0.55) (hP_A_and_B : P_A_and_B = 0.50) :
  1 - (P_A + P_B - P_A_and_B) = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_answered_neither_l2075_207518


namespace NUMINAMATH_GPT_problem_inequality_l2075_207579

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x - y + z) * (y - z + x) * (z - x + y) ≤ x * y * z := sorry

end NUMINAMATH_GPT_problem_inequality_l2075_207579


namespace NUMINAMATH_GPT_value_of_b_l2075_207557

noncomputable def k := 675

theorem value_of_b (a b : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) (h4 : a = -12) :
  b = -56.25 := by
  sorry

end NUMINAMATH_GPT_value_of_b_l2075_207557


namespace NUMINAMATH_GPT_f_one_f_a_f_f_a_l2075_207553

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem f_one : f 1 = 5 := by
  sorry

theorem f_a (a : ℝ) : f a = 2 * a + 3 := by
  sorry

theorem f_f_a (a : ℝ) : f (f a) = 4 * a + 9 := by
  sorry

end NUMINAMATH_GPT_f_one_f_a_f_f_a_l2075_207553


namespace NUMINAMATH_GPT_decrypt_message_base7_l2075_207506

noncomputable def base7_to_base10 : Nat := 
  2 * 343 + 5 * 49 + 3 * 7 + 4 * 1

theorem decrypt_message_base7 : base7_to_base10 = 956 := 
by 
  sorry

end NUMINAMATH_GPT_decrypt_message_base7_l2075_207506


namespace NUMINAMATH_GPT_algebra_expression_value_l2075_207555

theorem algebra_expression_value (m : ℝ) (h : m^2 - 3 * m - 1 = 0) : 2 * m^2 - 6 * m + 5 = 7 := by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l2075_207555


namespace NUMINAMATH_GPT_area_of_red_flowers_is_54_l2075_207583

noncomputable def total_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def red_yellow_area (total : ℝ) : ℝ :=
  total / 2

noncomputable def red_area (red_yellow : ℝ) : ℝ :=
  red_yellow / 2

theorem area_of_red_flowers_is_54 :
  total_area 18 12 / 2 / 2 = 54 := 
  by
    sorry

end NUMINAMATH_GPT_area_of_red_flowers_is_54_l2075_207583


namespace NUMINAMATH_GPT_geometric_seq_value_l2075_207561

theorem geometric_seq_value (a : ℕ → ℝ) (h : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end NUMINAMATH_GPT_geometric_seq_value_l2075_207561


namespace NUMINAMATH_GPT_number_of_unanswered_questions_l2075_207580

theorem number_of_unanswered_questions (n p q : ℕ) (h1 : p = 8) (h2 : q = 5) (h3 : n = 20)
(h4: ∃ s, s % 13 = 0) (hy : y = 0 ∨ y = 13) : 
  ∃ k, k = 20 ∨ k = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_unanswered_questions_l2075_207580


namespace NUMINAMATH_GPT_required_vases_l2075_207534

def vase_capacity_roses : Nat := 6
def vase_capacity_tulips : Nat := 8
def vase_capacity_lilies : Nat := 4

def remaining_roses : Nat := 20
def remaining_tulips : Nat := 15
def remaining_lilies : Nat := 5

def vases_for_roses : Nat := (remaining_roses + vase_capacity_roses - 1) / vase_capacity_roses
def vases_for_tulips : Nat := (remaining_tulips + vase_capacity_tulips - 1) / vase_capacity_tulips
def vases_for_lilies : Nat := (remaining_lilies + vase_capacity_lilies - 1) / vase_capacity_lilies

def total_vases_needed : Nat := vases_for_roses + vases_for_tulips + vases_for_lilies

theorem required_vases : total_vases_needed = 8 := by
  sorry

end NUMINAMATH_GPT_required_vases_l2075_207534


namespace NUMINAMATH_GPT_pascal_triangle_row_20_sum_l2075_207504

theorem pascal_triangle_row_20_sum :
  (Nat.choose 20 2) + (Nat.choose 20 3) + (Nat.choose 20 4) = 6175 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_row_20_sum_l2075_207504


namespace NUMINAMATH_GPT_cylinder_radius_eq_3_l2075_207501

theorem cylinder_radius_eq_3 (r : ℝ) : 
  (π * (r + 4)^2 * 3 = π * r^2 * 11) ∧ (r >= 0) → r = 3 :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_radius_eq_3_l2075_207501


namespace NUMINAMATH_GPT_wide_flags_made_l2075_207524

theorem wide_flags_made
  (initial_fabric : ℕ) (square_flag_side : ℕ) (wide_flag_width : ℕ) (wide_flag_height : ℕ)
  (tall_flag_width : ℕ) (tall_flag_height : ℕ) (made_square_flags : ℕ) (made_tall_flags : ℕ)
  (remaining_fabric : ℕ) (used_fabric_for_small_flags : ℕ) (used_fabric_for_tall_flags : ℕ)
  (used_fabric_for_wide_flags : ℕ) (wide_flag_area : ℕ) :
    initial_fabric = 1000 →
    square_flag_side = 4 →
    wide_flag_width = 5 →
    wide_flag_height = 3 →
    tall_flag_width = 3 →
    tall_flag_height = 5 →
    made_square_flags = 16 →
    made_tall_flags = 10 →
    remaining_fabric = 294 →
    used_fabric_for_small_flags = 256 →
    used_fabric_for_tall_flags = 150 →
    used_fabric_for_wide_flags = initial_fabric - remaining_fabric - (used_fabric_for_small_flags + used_fabric_for_tall_flags) →
    wide_flag_area = wide_flag_width * wide_flag_height →
    (used_fabric_for_wide_flags / wide_flag_area) = 20 :=
by
  intros; 
  sorry

end NUMINAMATH_GPT_wide_flags_made_l2075_207524


namespace NUMINAMATH_GPT_problem_AD_l2075_207572

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

open Real

theorem problem_AD :
  (∀ x, 0 < x ∧ x < π / 4 → f x < f (x + 0.01) ∧ g x < g (x + 0.01)) ∧
  (∃ x, x = π / 4 ∧ f x + g x = 1 / 2 + sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_AD_l2075_207572


namespace NUMINAMATH_GPT_hyperbola_focal_product_l2075_207508

-- Define the hyperbola with given equation and point P conditions
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1 }

-- Define properties of vectors related to foci
def perpendicular (v1 v2 : ℝ × ℝ) := (v1.1 * v2.1 + v1.2 * v2.2 = 0)

-- Define the point-focus distance product condition
noncomputable def focalProduct (P F1 F2 : ℝ × ℝ) := (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

theorem hyperbola_focal_product :
  ∀ (a b : ℝ) (F1 F2 P : ℝ × ℝ),
  Hyperbola a b P ∧ perpendicular (P - F1) (P - F2) ∧
  -- Assuming a parabola property ties F1 with a specific value
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 4 * (Real.sqrt  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))) →
  focalProduct P F1 F2 = 14 := by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_product_l2075_207508


namespace NUMINAMATH_GPT_matches_C_won_l2075_207554

variable (A_wins B_wins D_wins total_matches wins_C : ℕ)

theorem matches_C_won 
  (hA : A_wins = 3)
  (hB : B_wins = 1)
  (hD : D_wins = 0)
  (htot : total_matches = 6)
  (h_sum_wins: A_wins + B_wins + D_wins + wins_C = total_matches)
  : wins_C = 2 :=
by
  sorry

end NUMINAMATH_GPT_matches_C_won_l2075_207554


namespace NUMINAMATH_GPT_green_peaches_per_basket_l2075_207592

/-- Define the conditions given in the problem. -/
def n_baskets : ℕ := 7
def n_red_each : ℕ := 10
def n_green_total : ℕ := 14

/-- Prove that there are 2 green peaches in each basket. -/
theorem green_peaches_per_basket : n_green_total / n_baskets = 2 := by
  sorry

end NUMINAMATH_GPT_green_peaches_per_basket_l2075_207592


namespace NUMINAMATH_GPT_tom_trip_cost_l2075_207568

-- Definitions of hourly rates
def rate_6AM_to_10AM := 10
def rate_10AM_to_2PM := 12
def rate_2PM_to_6PM := 15
def rate_6PM_to_10PM := 20

-- Definitions of trip start times and durations
def first_trip_start := 8
def second_trip_start := 14
def third_trip_start := 20

-- Function to calculate the cost for each trip segment
def cost (start_hour : Nat) (duration : Nat) : Nat :=
  if start_hour >= 6 ∧ start_hour < 10 then duration * rate_6AM_to_10AM
  else if start_hour >= 10 ∧ start_hour < 14 then duration * rate_10AM_to_2PM
  else if start_hour >= 14 ∧ start_hour < 18 then duration * rate_2PM_to_6PM
  else if start_hour >= 18 ∧ start_hour < 22 then duration * rate_6PM_to_10PM
  else 0

-- Function to calculate the total trip cost
def total_cost : Nat :=
  cost first_trip_start 2 + cost (first_trip_start + 2) 2 +
  cost second_trip_start 4 +
  cost third_trip_start 4

-- Proof statement
theorem tom_trip_cost : total_cost = 184 := by
  -- The detailed steps of the proof would go here. Replaced with 'sorry' presently to indicate incomplete proof.
  sorry

end NUMINAMATH_GPT_tom_trip_cost_l2075_207568


namespace NUMINAMATH_GPT_find_number_l2075_207562

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2075_207562


namespace NUMINAMATH_GPT_value_of_a_l2075_207542

theorem value_of_a (a : ℝ) (h : (a - 3) * x ^ |a - 2| + 4 = 0) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2075_207542


namespace NUMINAMATH_GPT_female_employees_l2075_207517

theorem female_employees (E M F : ℕ) (h1 : 300 = 300) (h2 : (2/5 : ℚ) * E = (2/5 : ℚ) * M + 300) (h3 : E = M + F) : F = 750 := 
by
  sorry

end NUMINAMATH_GPT_female_employees_l2075_207517


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2075_207597

theorem geometric_sequence_problem
  (q : ℝ) (h_q : |q| ≠ 1) (m : ℕ)
  (a : ℕ → ℝ)
  (h_a1 : a 1 = -1)
  (h_am : a m = a 1 * a 2 * a 3 * a 4 * a 5) 
  (h_gseq : ∀ n, a (n + 1) = a n * q) :
  m = 11 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2075_207597


namespace NUMINAMATH_GPT_possible_values_of_C_l2075_207593

variable {α : Type} [LinearOrderedField α]

-- Definitions of points A, B and C
def pointA (a : α) := a
def pointB (b : α) := b
def pointC (c : α) := c

-- Given condition
def given_condition (a b : α) : Prop := (a + 3) ^ 2 + |b - 1| = 0

-- Function to determine if the folding condition is met
def folding_number_line (A B C : α) : Prop :=
  (C = 2 * A - B ∨ C = 2 * B - A ∨ (A + B) / 2 = C)

-- Theorem to prove the possible values of C
theorem possible_values_of_C (a b : α) (h : given_condition a b) :
  ∃ C : α, folding_number_line (pointA a) (pointB b) (pointC C) ∧ (C = -7 ∨ C = 5 ∨ C = -1) :=
sorry

end NUMINAMATH_GPT_possible_values_of_C_l2075_207593


namespace NUMINAMATH_GPT_factory_earnings_l2075_207549

-- Definition of constants and functions based on the conditions:
def material_A_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def material_B_production (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def convert_B_to_C (material_B : ℕ) : ℕ := material_B / 2
def earnings (amount : ℕ) (price_per_unit : ℕ) : ℕ := amount * price_per_unit

-- Given conditions for the problem:
def hours_machine_1_and_2 : ℕ := 23
def hours_machine_3 : ℕ := 23
def hours_machine_4 : ℕ := 12
def rate_A_machine_1_and_2 : ℕ := 2
def rate_B_machine_1_and_2 : ℕ := 1
def rate_A_machine_3_and_4 : ℕ := 3
def rate_B_machine_3_and_4 : ℕ := 2
def price_A : ℕ := 50
def price_C : ℕ := 100

-- Calculations based on problem conditions:
noncomputable def total_A : ℕ := 
  2 * material_A_production hours_machine_1_and_2 rate_A_machine_1_and_2 + 
  material_A_production hours_machine_3 rate_A_machine_3_and_4 + 
  material_A_production hours_machine_4 rate_A_machine_3_and_4

noncomputable def total_B : ℕ := 
  2 * material_B_production hours_machine_1_and_2 rate_B_machine_1_and_2 + 
  material_B_production hours_machine_3 rate_B_machine_3_and_4 + 
  material_B_production hours_machine_4 rate_B_machine_3_and_4

noncomputable def total_C : ℕ := convert_B_to_C total_B

noncomputable def total_earnings : ℕ :=
  earnings total_A price_A + earnings total_C price_C

-- The theorem to prove the total earnings:
theorem factory_earnings : total_earnings = 15650 :=
by
  sorry

end NUMINAMATH_GPT_factory_earnings_l2075_207549


namespace NUMINAMATH_GPT_chickens_count_l2075_207585

def total_animals := 13
def total_legs := 44
def legs_per_chicken := 2
def legs_per_buffalo := 4

theorem chickens_count : 
  (∃ c b : ℕ, c + b = total_animals ∧ legs_per_chicken * c + legs_per_buffalo * b = total_legs ∧ c = 4) :=
by
  sorry

end NUMINAMATH_GPT_chickens_count_l2075_207585


namespace NUMINAMATH_GPT_minimum_degree_g_l2075_207599

open Polynomial

theorem minimum_degree_g (f g h : Polynomial ℝ) 
  (h_eq : 5 • f + 2 • g = h)
  (deg_f : f.degree = 11)
  (deg_h : h.degree = 12) : 
  ∃ d : ℕ, g.degree = d ∧ d >= 12 := 
sorry

end NUMINAMATH_GPT_minimum_degree_g_l2075_207599


namespace NUMINAMATH_GPT_problem_statement_l2075_207595

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 65 / 2) + 5 / 2)

theorem problem_statement :
  ∃ a b c : ℕ, (x ^ 100 = 2 * x ^ 98 + 16 * x ^ 96 + 13 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 42) ∧ (a + b + c = 337) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2075_207595


namespace NUMINAMATH_GPT_distribute_items_in_identical_bags_l2075_207500

noncomputable def count_ways_to_distribute_items (num_items : ℕ) (num_bags : ℕ) : ℕ :=
  if h : num_items = 5 ∧ num_bags = 3 then 36 else 0

theorem distribute_items_in_identical_bags :
  count_ways_to_distribute_items 5 3 = 36 :=
by
  -- Proof is skipped as per instructions
  sorry

end NUMINAMATH_GPT_distribute_items_in_identical_bags_l2075_207500


namespace NUMINAMATH_GPT_absolute_sum_of_coefficients_l2075_207523

theorem absolute_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (2 - x)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 2^6 →
  a_0 > 0 ∧ a_2 > 0 ∧ a_4 > 0 ∧ a_6 > 0 ∧
  a_1 < 0 ∧ a_3 < 0 ∧ a_5 < 0 → 
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 665 :=
by sorry

end NUMINAMATH_GPT_absolute_sum_of_coefficients_l2075_207523


namespace NUMINAMATH_GPT_kitchen_upgrade_total_cost_l2075_207578

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end NUMINAMATH_GPT_kitchen_upgrade_total_cost_l2075_207578


namespace NUMINAMATH_GPT_equation_of_line_through_point_l2075_207546

theorem equation_of_line_through_point (a T : ℝ) (h : a ≠ 0 ∧ T ≠ 0) :
  ∃ k : ℝ, (k = T / (a^2)) ∧ (k * x + (2 * T / a)) = (k * x + (2 * T / a)) → 
  (T * x - a^2 * y + 2 * T * a = 0) :=
by
  use T / (a^2)
  sorry

end NUMINAMATH_GPT_equation_of_line_through_point_l2075_207546


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l2075_207576

variable (a_n : ℕ → ℝ) -- Sequence term
variable (S_n : ℕ → ℝ) -- Sum of first n terms

-- Conditions given in the problem
axiom sum_first_term : a_n 1 = 2
axiom sum_first_two_terms : a_n 1 + a_n 2 = 7
axiom sum_first_three_terms : a_n 1 + a_n 2 + a_n 3 = 18

-- Expected result to prove
theorem sum_of_first_n_terms 
  (h1 : S_n 1 = 2)
  (h2 : S_n 2 = 7)
  (h3 : S_n 3 = 18) :
  S_n n = (3/2) * ((n * (n + 1) * (2 * n + 1) / 6) - (n * (n + 1) / 2) + 2 * n) :=
sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_l2075_207576


namespace NUMINAMATH_GPT_div_condition_l2075_207584

theorem div_condition (N : ℤ) : (∃ k : ℤ, N^2 - 71 = k * (7 * N + 55)) ↔ (N = 57 ∨ N = -8) := 
by
  sorry

end NUMINAMATH_GPT_div_condition_l2075_207584
