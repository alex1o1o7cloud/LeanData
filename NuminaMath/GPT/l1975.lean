import Mathlib

namespace train_cross_pole_time_l1975_197598

noncomputable def train_time_to_cross_pole (length : ℕ) (speed_km_per_hr : ℕ) : ℕ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  length / speed_m_per_s

theorem train_cross_pole_time :
  train_time_to_cross_pole 100 72 = 5 :=
by
  unfold train_time_to_cross_pole
  sorry

end train_cross_pole_time_l1975_197598


namespace is_not_innovative_54_l1975_197561

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ n = a^2 - b^2

theorem is_not_innovative_54 : ¬ is_innovative 54 :=
sorry

end is_not_innovative_54_l1975_197561


namespace point_a_coordinates_l1975_197585

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end point_a_coordinates_l1975_197585


namespace find_number_l1975_197596

-- We define n, x, y as real numbers
variables (n x y : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := n * (x - y) = 4
def condition2 : Prop := 6 * x - 3 * y = 12

-- Define the theorem we need to prove: If the conditions hold, then n = 2
theorem find_number (h1 : condition1 n x y) (h2 : condition2 x y) : n = 2 := 
sorry

end find_number_l1975_197596


namespace largest_x_l1975_197512

-- Definitions from the conditions
def eleven_times_less_than_150 (x : ℕ) : Prop := 11 * x < 150

-- Statement of the proof problem
theorem largest_x : ∃ x : ℕ, eleven_times_less_than_150 x ∧ ∀ y : ℕ, eleven_times_less_than_150 y → y ≤ x := 
sorry

end largest_x_l1975_197512


namespace find_cos_sin_sum_l1975_197556

-- Define the given condition: tan θ = 5/12 and 180° ≤ θ ≤ 270°.
variable (θ : ℝ)
variable (h₁ : Real.tan θ = 5 / 12)
variable (h₂ : π ≤ θ ∧ θ ≤ 3 * π / 2)

-- Define the main statement to prove.
theorem find_cos_sin_sum : Real.cos θ + Real.sin θ = -17 / 13 := by
  sorry

end find_cos_sin_sum_l1975_197556


namespace det_scaled_matrix_l1975_197592

variable (a b c d : ℝ)
variable (h : Matrix.det ![![a, b], ![c, d]] = 5)

theorem det_scaled_matrix : Matrix.det ![![3 * a, 3 * b], ![4 * c, 4 * d]] = 60 := by
  sorry

end det_scaled_matrix_l1975_197592


namespace polygon_expected_value_l1975_197580

def polygon_expected_sides (area_square : ℝ) (flower_prob : ℝ) (area_flower : ℝ) (hex_sides : ℝ) (pent_sides : ℝ) : ℝ :=
  hex_sides * flower_prob + pent_sides * (area_square - flower_prob)

theorem polygon_expected_value :
  polygon_expected_sides 1 (π - 1) (π - 1) 6 5 = π + 4 :=
by
  -- Proof is skipped
  sorry

end polygon_expected_value_l1975_197580


namespace area_of_region_l1975_197584

noncomputable def circle_radius : ℝ := 3

noncomputable def segment_length : ℝ := 4

theorem area_of_region : ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_l1975_197584


namespace sequence_period_9_l1975_197554

def sequence_periodic (x : ℕ → ℤ) : Prop :=
  ∀ n > 1, x (n + 1) = |x n| - x (n - 1)

theorem sequence_period_9 (x : ℕ → ℤ) :
  sequence_periodic x → ∃ p, p = 9 ∧ ∀ n, x (n + p) = x n :=
by
  sorry

end sequence_period_9_l1975_197554


namespace beta_minus_alpha_l1975_197506

open Real

noncomputable def vector_a (α : ℝ) := (cos α, sin α)
noncomputable def vector_b (β : ℝ) := (cos β, sin β)

theorem beta_minus_alpha (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < π)
  (h₄ : |2 * vector_a α + vector_b β| = |vector_a α - 2 * vector_b β|) :
  β - α = π / 2 :=
sorry

end beta_minus_alpha_l1975_197506


namespace problem_statement_l1975_197508

def line : Type := sorry
def plane : Type := sorry

def perpendicular (l : line) (p : plane) : Prop := sorry
def parallel (l1 l2 : line) : Prop := sorry

variable (m n : line)
variable (α β : plane)

theorem problem_statement (h1 : perpendicular m α) 
                          (h2 : parallel m n) 
                          (h3 : parallel n β) : 
                          perpendicular α β := 
sorry

end problem_statement_l1975_197508


namespace arithmetic_sequence_range_of_m_l1975_197599

-- Conditions
variable {a : ℕ+ → ℝ} -- Sequence of positive terms
variable {S : ℕ+ → ℝ} -- Sum of the first n terms
variable (h : ∀ n, 2 * Real.sqrt (S n) = a n + 1) -- Relationship condition

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ+)
    (h1 : ∀ n, 2 * Real.sqrt (S n) = a n + 1)
    (h2 : S 1 = 1 / 4 * (a 1 + 1)^2) :
    ∃ d : ℝ, ∀ n, a (n + 1) = a n + d :=
sorry

-- Part 2: Find range of m
theorem range_of_m (T : ℕ+ → ℝ)
    (hT : ∀ n, T n = 1 / 4 * n + 1 / 8 * (1 - 1 / (2 * n + 1))) :
    ∃ m : ℝ, (6 / 7 : ℝ) < m ∧ m ≤ 10 / 9 ∧
    (∃ n₁ n₂ n₃ : ℕ+, (n₁ < n₂ ∧ n₂ < n₃) ∧ (∀ n, T n < m ↔ n₁ ≤ n ∧ n ≤ n₃)) :=
sorry

end arithmetic_sequence_range_of_m_l1975_197599


namespace total_scoops_l1975_197544

-- Define the conditions as variables
def flourCups := 3
def sugarCups := 2
def scoopSize := 1/3

-- Define what needs to be proved, i.e., the total amount of scoops needed
theorem total_scoops (flourCups sugarCups : ℚ) (scoopSize : ℚ) : 
  (flourCups / scoopSize) + (sugarCups / scoopSize) = 15 := 
by
  sorry

end total_scoops_l1975_197544


namespace find_pos_int_l1975_197571

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int_l1975_197571


namespace square_area_proof_l1975_197534

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end square_area_proof_l1975_197534


namespace probability_one_white_ball_initial_find_n_if_one_red_ball_l1975_197539

-- Define the initial conditions: 5 red balls and 3 white balls
def initial_red_balls := 5
def initial_white_balls := 3
def total_initial_balls := initial_red_balls + initial_white_balls

-- Define the probability of drawing exactly one white ball initially
def prob_draw_one_white := initial_white_balls / total_initial_balls

-- Define the number of white balls added
variable (n : ℕ)

-- Define the total number of balls after adding n white balls
def total_balls_after_adding := total_initial_balls + n

-- Define the probability of drawing exactly one red ball after adding n white balls
def prob_draw_one_red := initial_red_balls / total_balls_after_adding

-- Prove that the probability of drawing one white ball initially is 3/8
theorem probability_one_white_ball_initial : prob_draw_one_white = 3 / 8 := by
  sorry

-- Prove that, if the probability of drawing one red ball after adding n white balls is 1/2, then n = 2
theorem find_n_if_one_red_ball : prob_draw_one_red = 1 / 2 -> n = 2 := by
  sorry

end probability_one_white_ball_initial_find_n_if_one_red_ball_l1975_197539


namespace find_c_plus_one_over_b_l1975_197518

variable (a b c : ℝ)
variable (habc : a * b * c = 1)
variable (ha : a + (1 / c) = 7)
variable (hb : b + (1 / a) = 35)

theorem find_c_plus_one_over_b : (c + (1 / b) = 11 / 61) :=
by
  have h1 : a * b * c = 1 := habc
  have h2 : a + (1 / c) = 7 := ha
  have h3 : b + (1 / a) = 35 := hb
  sorry

end find_c_plus_one_over_b_l1975_197518


namespace find_number_l1975_197586

theorem find_number (x : ℤ) (h : ((x * 2) - 37 + 25) / 8 = 5) : x = 26 :=
sorry  -- Proof placeholder

end find_number_l1975_197586


namespace func_equiv_l1975_197555

noncomputable def f (x : ℝ) : ℝ := if x = 0 then 0 else x + 1 / x

theorem func_equiv {a b : ℝ} (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) :
  (∀ x, f (2 * x) = a * f x + b * x) ∧ (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y)) :=
sorry

end func_equiv_l1975_197555


namespace complex_magnitude_squared_l1975_197535

open Complex Real

theorem complex_magnitude_squared :
  ∃ (z : ℂ), z + abs z = 3 + 7 * i ∧ abs z ^ 2 = 841 / 9 :=
by
  sorry

end complex_magnitude_squared_l1975_197535


namespace right_triangle_least_side_l1975_197578

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l1975_197578


namespace player_current_average_l1975_197531

theorem player_current_average
  (A : ℕ) -- Assume A is a natural number (non-negative)
  (cond1 : 10 * A + 78 = 11 * (A + 4)) :
  A = 34 :=
by
  sorry

end player_current_average_l1975_197531


namespace arcsin_one_half_eq_pi_six_l1975_197579

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end arcsin_one_half_eq_pi_six_l1975_197579


namespace last_digit_largest_prime_l1975_197543

-- Definition and conditions
def largest_known_prime : ℕ := 2^216091 - 1

-- The statement of the problem we want to prove
theorem last_digit_largest_prime : (largest_known_prime % 10) = 7 := by
  sorry

end last_digit_largest_prime_l1975_197543


namespace intersection_of_log_functions_l1975_197529

theorem intersection_of_log_functions : 
  ∃ x : ℝ, (3 * Real.log x = Real.log (3 * x)) ∧ x = Real.sqrt 3 := 
by 
  sorry

end intersection_of_log_functions_l1975_197529


namespace new_average_weight_l1975_197563

def num_people := 6
def avg_weight1 := 154
def weight_seventh := 133

theorem new_average_weight :
  (num_people * avg_weight1 + weight_seventh) / (num_people + 1) = 151 := by
  sorry

end new_average_weight_l1975_197563


namespace solve_problem_l1975_197542

def f (x : ℝ) : ℝ := x^2 - 4*x + 7
def g (x : ℝ) : ℝ := 2*x + 1

theorem solve_problem : f (g 3) - g (f 3) = 19 := by
  sorry

end solve_problem_l1975_197542


namespace find_value_of_a_plus_b_l1975_197565

variables (a b : ℝ)

theorem find_value_of_a_plus_b
  (h1 : a^3 - 3 * a^2 + 5 * a = 1)
  (h2 : b^3 - 3 * b^2 + 5 * b = 5) :
  a + b = 2 := 
sorry

end find_value_of_a_plus_b_l1975_197565


namespace rationalize_denominator_div_l1975_197594

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l1975_197594


namespace arithmetic_series_first_term_l1975_197546

theorem arithmetic_series_first_term 
  (a d : ℚ)
  (h1 : 15 * (2 * a +  29 * d) = 450)
  (h2 : 15 * (2 * a + 89 * d) = 1650) :
  a = -13 / 3 :=
by
  sorry

end arithmetic_series_first_term_l1975_197546


namespace remainder_sum_of_squares_mod_13_l1975_197590

-- Define the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Prove that the remainder when the sum of squares of the first 20 natural numbers
-- is divided by 13 is 10
theorem remainder_sum_of_squares_mod_13 : sum_of_squares 20 % 13 = 10 := 
by
  -- Here you can imagine the relevant steps or intermediate computations might go, if needed.
  sorry -- Placeholder for the proof.

end remainder_sum_of_squares_mod_13_l1975_197590


namespace framed_painting_ratio_l1975_197503

theorem framed_painting_ratio (x : ℝ) (h : (15 + 2 * x) * (30 + 4 * x) = 900) : (15 + 2 * x) / (30 + 4 * x) = 1 / 2 :=
by
  sorry

end framed_painting_ratio_l1975_197503


namespace product_of_reds_is_red_sum_of_reds_is_red_l1975_197515

noncomputable def color := ℕ → Prop

variables (white red : color)
variable (r : ℕ)

axiom coloring : ∀ n, white n ∨ red n
axiom exists_white : ∃ n, white n
axiom exists_red : ∃ n, red n
axiom sum_of_white_red_is_white : ∀ m n, white m → red n → white (m + n)
axiom prod_of_white_red_is_red : ∀ m n, white m → red n → red (m * n)

theorem product_of_reds_is_red (m n : ℕ) : red m → red n → red (m * n) :=
sorry

theorem sum_of_reds_is_red (m n : ℕ) : red m → red n → red (m + n) :=
sorry

end product_of_reds_is_red_sum_of_reds_is_red_l1975_197515


namespace Monet_paintings_consecutively_l1975_197504

noncomputable def probability_Monet_paintings_consecutively (total_art_pieces Monet_paintings : ℕ) : ℚ :=
  let numerator := 9 * Nat.factorial (total_art_pieces - Monet_paintings) * Nat.factorial Monet_paintings
  let denominator := Nat.factorial total_art_pieces
  numerator / denominator

theorem Monet_paintings_consecutively :
  probability_Monet_paintings_consecutively 12 4 = 18 / 95 := by
  sorry

end Monet_paintings_consecutively_l1975_197504


namespace percentage_difference_l1975_197537

theorem percentage_difference (y : ℝ) (h : y ≠ 0) (x z : ℝ) (hx : x = 5 * y) (hz : z = 1.20 * y) :
  ((z - y) / x * 100) = 4 :=
by
  rw [hz, hx]
  simp
  sorry

end percentage_difference_l1975_197537


namespace dollars_tina_l1975_197511

open Real

theorem dollars_tina (P Q R S T : ℤ)
  (h1 : abs (P - Q) = 21)
  (h2 : abs (Q - R) = 9)
  (h3 : abs (R - S) = 7)
  (h4 : abs (S - T) = 6)
  (h5 : abs (T - P) = 13)
  (h6 : P + Q + R + S + T = 86) :
  T = 16 :=
sorry

end dollars_tina_l1975_197511


namespace evaluate_expression_l1975_197597

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem evaluate_expression : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end evaluate_expression_l1975_197597


namespace simplify_fraction_l1975_197525

variable (d : ℤ)

theorem simplify_fraction (d : ℤ) : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := 
by 
  sorry

end simplify_fraction_l1975_197525


namespace ratio_HP_HA_l1975_197551

-- Given Definitions
variables (A B C P Q H : Type)
variables (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) (h3 : P ≠ Q)
variables (h4 : FootOfAltitudeFrom A H B C) (h5 : OnExtendedLine P A B) (h6 : OnExtendedLine Q A C)
variables (h7 : HP = HQ) (h8 : CyclicQuadrilateral B C P Q)

-- Required Ratio
theorem ratio_HP_HA : HP = HA := sorry

end ratio_HP_HA_l1975_197551


namespace sufficient_but_not_necessary_l1975_197538

theorem sufficient_but_not_necessary (x : ℝ) (h : 2 < x ∧ x < 3) :
  x * (x - 5) < 0 ∧ ∃ y, y * (y - 5) < 0 ∧ (2 ≤ y ∧ y ≤ 3) → False :=
by
  sorry

end sufficient_but_not_necessary_l1975_197538


namespace inequalities_count_three_l1975_197574

theorem inequalities_count_three
  (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  (x^2 + y^2 < a^2 + b^2) ∧ ¬(x^2 - y^2 < a^2 - b^2) ∧ (x^2 * y^3 < a^2 * b^3) ∧ (x^2 / y^3 < a^2 / b^3) := 
sorry

end inequalities_count_three_l1975_197574


namespace correct_mms_packs_used_l1975_197567

variable (num_sundaes_monday : ℕ) (mms_per_sundae_monday : ℕ)
variable (num_sundaes_tuesday : ℕ) (mms_per_sundae_tuesday : ℕ)
variable (mms_per_pack : ℕ)

-- Conditions
def conditions : Prop := 
  num_sundaes_monday = 40 ∧ 
  mms_per_sundae_monday = 6 ∧ 
  num_sundaes_tuesday = 20 ∧
  mms_per_sundae_tuesday = 10 ∧ 
  mms_per_pack = 40

-- Question: How many m&m packs does Kekai use?
def number_of_mms_packs (num_sundaes_monday mms_per_sundae_monday 
                         num_sundaes_tuesday mms_per_sundae_tuesday 
                         mms_per_pack : ℕ) : ℕ := 
  (num_sundaes_monday * mms_per_sundae_monday + num_sundaes_tuesday * mms_per_sundae_tuesday) / mms_per_pack

-- Theorem to prove the correct number of m&m packs used
theorem correct_mms_packs_used (h : conditions num_sundaes_monday mms_per_sundae_monday 
                                              num_sundaes_tuesday mms_per_sundae_tuesday 
                                              mms_per_pack) : 
  number_of_mms_packs num_sundaes_monday mms_per_sundae_monday 
                      num_sundaes_tuesday mms_per_sundae_tuesday 
                      mms_per_pack = 11 := by {
  -- Proof goes here
  sorry
}

end correct_mms_packs_used_l1975_197567


namespace required_run_rate_l1975_197583

theorem required_run_rate (target : ℝ) (initial_run_rate : ℝ) (initial_overs : ℕ) (remaining_overs : ℕ) :
  target = 282 → initial_run_rate = 3.8 → initial_overs = 10 → remaining_overs = 40 →
  (target - initial_run_rate * initial_overs) / remaining_overs = 6.1 :=
by
  intros
  sorry

end required_run_rate_l1975_197583


namespace A_is_guilty_l1975_197588

-- Define the conditions
variables (A B C : Prop)  -- A, B, C are the propositions that represent the guilt of the individuals A, B, and C
variable  (car : Prop)    -- car represents the fact that the crime involved a car
variable  (C_never_alone : C → A)  -- C never commits a crime without A

-- Facts:
variables (crime_committed : A ∨ B ∨ C) -- the crime was committed by A, B, or C (or a combination)
variable  (B_knows_drive : B → car)     -- B knows how to drive

-- The proof goal: Show that A is guilty.
theorem A_is_guilty : A :=
sorry

end A_is_guilty_l1975_197588


namespace hari_contribution_correct_l1975_197581

-- Translate the conditions into definitions
def praveen_investment : ℝ := 3360
def praveen_duration : ℝ := 12
def hari_duration : ℝ := 7
def profit_ratio_praveen : ℝ := 2
def profit_ratio_hari : ℝ := 3

-- The target Hari's contribution that we need to prove
def hari_contribution : ℝ := 2160

-- Problem statement: prove Hari's contribution given the conditions
theorem hari_contribution_correct :
  (praveen_investment * praveen_duration) / (hari_contribution * hari_duration) = profit_ratio_praveen / profit_ratio_hari :=
by {
  -- The statement is set up to prove equality of the ratios as given in the problem
  sorry
}

end hari_contribution_correct_l1975_197581


namespace complex_number_solution_l1975_197595

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (h_i : z * i = 2 + i) : z = 1 - 2 * i := by
  sorry

end complex_number_solution_l1975_197595


namespace find_expression_for_an_l1975_197552

-- Definitions for the problem conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def problem_conditions (a : ℕ → ℝ) (q : ℝ) :=
  geometric_sequence a q ∧
  a 1 + a 3 = 10 ∧
  a 2 + a 4 = 5

-- Statement of the problem
theorem find_expression_for_an (a : ℕ → ℝ) (q : ℝ) :
  problem_conditions a q → ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end find_expression_for_an_l1975_197552


namespace ellipse_tangency_construction_l1975_197566

theorem ellipse_tangency_construction
  (a : ℝ)
  (e1 e2 : ℝ → Prop)  -- Representing the parallel lines as propositions
  (F1 F2 : ℝ × ℝ)  -- Foci represented as points in the plane
  (d : ℝ)  -- Distance between the parallel lines
  (angle_condition : ℝ)
  (conditions : 2 * a > d ∧ angle_condition = 1 / 3) : 
  ∃ O : ℝ × ℝ,  -- Midpoint O
    ∃ (T1 T1' T2 T2' : ℝ × ℝ),  -- Points of tangency
      (∃ E1 E2 : ℝ, e1 E1 ∧ e2 E2) ∧  -- Intersection points on the lines
      (F1.1 * (T1.1 - F1.1) + F1.2 * (T1.2 - F1.2)) / 
      (F2.1 * (T2.1 - F2.1) + F2.2 * (T2.2 - F2.2)) = 1 / 3 :=
sorry

end ellipse_tangency_construction_l1975_197566


namespace range_of_a_l1975_197575

theorem range_of_a 
  (e : ℝ) (h_e_pos : 0 < e) 
  (a : ℝ) 
  (h_equation : ∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ (1 / e ^ x₁ - a / x₁ = 0) ∧ (1 / e ^ x₂ - a / x₂ = 0)) :
  0 < a ∧ a < 1 / e :=
by
  sorry

end range_of_a_l1975_197575


namespace distinct_integers_sum_l1975_197570

theorem distinct_integers_sum (m n p q : ℕ) (h1 : m ≠ n) (h2 : m ≠ p) (h3 : m ≠ q) (h4 : n ≠ p)
  (h5 : n ≠ q) (h6 : p ≠ q) (h71 : m > 0) (h72 : n > 0) (h73 : p > 0) (h74 : q > 0)
  (h_eq : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 := by
  sorry

end distinct_integers_sum_l1975_197570


namespace initial_bananas_l1975_197522

theorem initial_bananas (bananas_left: ℕ) (eaten: ℕ) (basket: ℕ) 
                        (h_left: bananas_left = 100) 
                        (h_eaten: eaten = 70) 
                        (h_basket: basket = 2 * eaten): 
  bananas_left + eaten + basket = 310 :=
by
  sorry

end initial_bananas_l1975_197522


namespace game_result_2013_game_result_2014_l1975_197569

inductive Player
| Barbara
| Jenna

def winning_player (n : ℕ) : Option Player :=
  if n % 5 = 3 then some Player.Jenna
  else if n % 5 = 4 then some Player.Barbara
  else none

theorem game_result_2013 : winning_player 2013 = some Player.Jenna := 
by sorry

theorem game_result_2014 : (winning_player 2014 = some Player.Barbara) ∨ (winning_player 2014 = some Player.Jenna) :=
by sorry

end game_result_2013_game_result_2014_l1975_197569


namespace triangle_area_is_correct_l1975_197532

-- Define the points
def point1 : (ℝ × ℝ) := (0, 3)
def point2 : (ℝ × ℝ) := (5, 0)
def point3 : (ℝ × ℝ) := (0, 6)
def point4 : (ℝ × ℝ) := (4, 0)

-- Define a function to calculate the area based on the intersection points
noncomputable def area_of_triangle (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  let slope1 := (p2.2 - p1.2) / (p2.1 - p1.1)
  let intercept1 := p1.2 - slope1 * p1.1
  let slope2 := (p4.2 - p3.2) / (p4.1 - p3.1)
  let intercept2 := p3.2 - slope2 * p3.1
  let x_intersect := (intercept2 - intercept1) / (slope1 - slope2)
  let y_intersect := slope1 * x_intersect + intercept1
  let base := x_intersect
  let height := y_intersect
  (1 / 2) * base * height

-- The proof problem statement in Lean
theorem triangle_area_is_correct :
  area_of_triangle point1 point2 point3 point4 = 5 / 3 :=
by
  sorry

end triangle_area_is_correct_l1975_197532


namespace white_balls_count_l1975_197501

-- Definitions for the conditions
variable (x y : ℕ) 

-- Lean statement representing the problem
theorem white_balls_count : 
  x < y ∧ y < 2 * x ∧ 2 * x + 3 * y = 60 → x = 9 := 
sorry

end white_balls_count_l1975_197501


namespace fractional_eq_k_l1975_197502

open Real

theorem fractional_eq_k (x k : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0) ↔ k ≠ -3 ∧ k ≠ 5 := 
sorry

end fractional_eq_k_l1975_197502


namespace sum_of_first_10_terms_l1975_197527

noncomputable def sum_first_n_terms (a_1 d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_10_terms (a : ℕ → ℕ) (a_2_a_4_sum : a 2 + a 4 = 4) (a_3_a_5_sum : a 3 + a 5 = 10) :
  sum_first_n_terms (a 1) (a 2 - a 1) 10 = 95 :=
  sorry

end sum_of_first_10_terms_l1975_197527


namespace largest_stamps_per_page_l1975_197558

theorem largest_stamps_per_page (a b c : ℕ) (h1 : a = 924) (h2 : b = 1260) (h3 : c = 1386) : 
  Nat.gcd (Nat.gcd a b) c = 42 := by
  sorry

end largest_stamps_per_page_l1975_197558


namespace initial_stock_of_coffee_l1975_197526

theorem initial_stock_of_coffee (x : ℝ) (h : x ≥ 0) 
  (h1 : 0.30 * x + 60 = 0.36 * (x + 100)) : x = 400 :=
by sorry

end initial_stock_of_coffee_l1975_197526


namespace root_exists_in_interval_l1975_197562

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_exists_in_interval : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := 
by
  sorry

end root_exists_in_interval_l1975_197562


namespace bowling_average_l1975_197572

theorem bowling_average (A : ℝ) (W : ℕ) (hW : W = 145) (hW7 : W + 7 ≠ 0)
  (h : ( A * W + 26 ) / ( W + 7 ) = A - 0.4) : A = 12.4 := 
by 
  sorry

end bowling_average_l1975_197572


namespace average_sitting_time_l1975_197516

theorem average_sitting_time (number_of_students : ℕ) (number_of_seats : ℕ) (total_travel_time : ℕ) 
  (h1 : number_of_students = 6) 
  (h2 : number_of_seats = 4) 
  (h3 : total_travel_time = 192) :
  (number_of_seats * total_travel_time) / number_of_students = 128 :=
by
  sorry

end average_sitting_time_l1975_197516


namespace line_tangent_to_parabola_l1975_197587

theorem line_tangent_to_parabola (d : ℝ) :
  (∀ x y: ℝ, y = 3 * x + d ↔ y^2 = 12 * x) → d = 1 :=
by
  sorry

end line_tangent_to_parabola_l1975_197587


namespace a_plus_2b_eq_21_l1975_197559

-- Definitions and conditions based on the problem statement
def a_log_250_2_plus_b_log_250_5_eq_3 (a b : ℤ) : Prop :=
  a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3

-- The theorem that needs to be proved
theorem a_plus_2b_eq_21 (a b : ℤ) (h : a_log_250_2_plus_b_log_250_5_eq_3 a b) : a + 2 * b = 21 := 
  sorry

end a_plus_2b_eq_21_l1975_197559


namespace inequality_proof_equality_condition_l1975_197514

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) :=
sorry

theorem equality_condition (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b ∧ a < 1) :=
sorry

end inequality_proof_equality_condition_l1975_197514


namespace additional_savings_is_297_l1975_197545

-- Define initial order amount
def initial_order_amount : ℝ := 12000

-- Define the first set of discounts
def discount_scheme_1 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.75
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 0.90
  final_price

-- Define the second set of discounts
def discount_scheme_2 (amount : ℝ) : ℝ :=
  let first_discount := amount * 0.70
  let second_discount := first_discount * 0.90
  let final_price := second_discount * 0.95
  final_price

-- Define the amount saved selecting the better discount scheme
def additional_savings : ℝ :=
  let final_price_1 := discount_scheme_1 initial_order_amount
  let final_price_2 := discount_scheme_2 initial_order_amount
  final_price_2 - final_price_1

-- Lean statement to prove the additional savings is $297
theorem additional_savings_is_297 : additional_savings = 297 := by
  sorry

end additional_savings_is_297_l1975_197545


namespace smallest_x_mod_conditions_l1975_197519

theorem smallest_x_mod_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 ∧ x = 209 := by
  sorry

end smallest_x_mod_conditions_l1975_197519


namespace sum_infinite_series_eq_half_l1975_197548

theorem sum_infinite_series_eq_half :
  (∑' n : ℕ, (n^5 + 2*n^3 + 5*n^2 + 20*n + 20) / (2^(n + 1) * (n^5 + 5))) = 1 / 2 := 
sorry

end sum_infinite_series_eq_half_l1975_197548


namespace sample_size_stratified_sampling_l1975_197573

theorem sample_size_stratified_sampling (n : ℕ) 
  (total_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (middle_aged_sample : ℕ)
  (stratified_sampling : n * middle_aged_employees = middle_aged_sample * total_employees)
  (total_employees_pos : total_employees = 750)
  (middle_aged_employees_pos : middle_aged_employees = 250) :
  n = 15 := 
by
  rw [total_employees_pos, middle_aged_employees_pos] at stratified_sampling
  sorry

end sample_size_stratified_sampling_l1975_197573


namespace solve_eq_64_16_pow_x_minus_1_l1975_197553

theorem solve_eq_64_16_pow_x_minus_1 (x : ℝ) (h : 64 = 4 * (16 : ℝ) ^ (x - 1)) : x = 2 :=
sorry

end solve_eq_64_16_pow_x_minus_1_l1975_197553


namespace largest_integer_value_neg_quadratic_l1975_197513

theorem largest_integer_value_neg_quadratic :
  ∃ m : ℤ, (4 < m ∧ m < 7) ∧ (m^2 - 11 * m + 28 < 0) ∧ ∀ n : ℤ, (4 < n ∧ n < 7 ∧ (n^2 - 11 * n + 28 < 0)) → n ≤ m :=
sorry

end largest_integer_value_neg_quadratic_l1975_197513


namespace find_x_l1975_197593

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) : hash x 7 = 63 → x = 3 :=
by
  sorry

end find_x_l1975_197593


namespace circles_disjoint_l1975_197549

theorem circles_disjoint :
  ∀ (x y u v : ℝ),
  (x^2 + y^2 = 1) →
  ((u-2)^2 + (v+2)^2 = 1) →
  (2^2 + (-2)^2) > (1 + 1)^2 :=
by sorry

end circles_disjoint_l1975_197549


namespace pentagon_stack_valid_sizes_l1975_197533

def valid_stack_size (n : ℕ) : Prop :=
  ¬ (n = 1) ∧ ¬ (n = 3)

theorem pentagon_stack_valid_sizes (n : ℕ) :
  valid_stack_size n :=
sorry

end pentagon_stack_valid_sizes_l1975_197533


namespace roots_of_modified_quadratic_l1975_197505

theorem roots_of_modified_quadratic 
  (k : ℝ) (hk : 0 < k) :
  (∃ z₁ z₂ : ℂ, (12 * z₁^2 - 4 * I * z₁ - k = 0) ∧ (12 * z₂^2 - 4 * I * z₂ - k = 0) ∧ (z₁ ≠ z₂) ∧ (z₁.im = 0) ∧ (z₂.im ≠ 0)) ↔ (k = 1/4) :=
by
  sorry

end roots_of_modified_quadratic_l1975_197505


namespace part1_union_part1_complement_part2_intersect_l1975_197541

namespace MathProof

open Set Real

def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }
def R : Set ℝ := univ  -- the set of all real numbers

theorem part1_union :
  A ∪ B = { x | 1 ≤ x ∧ x < 10 } :=
sorry

theorem part1_complement :
  R \ B = { x | x ≤ 2 ∨ x ≥ 10 } :=
sorry

theorem part2_intersect (a : ℝ) :
  (A ∩ C a ≠ ∅) → a > 1 :=
sorry

end MathProof

end part1_union_part1_complement_part2_intersect_l1975_197541


namespace widgets_per_shipping_box_l1975_197591

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l1975_197591


namespace count_primes_with_digit_three_l1975_197523

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_l1975_197523


namespace prime_p_and_cube_l1975_197521

noncomputable def p : ℕ := 307

theorem prime_p_and_cube (a : ℕ) (h : a^3 = 16 * p + 1) : 
  Nat.Prime p := by
  sorry

end prime_p_and_cube_l1975_197521


namespace remaining_bottles_l1975_197547

variable (s : ℕ) (b : ℕ) (ps : ℚ) (pb : ℚ)

theorem remaining_bottles (h1 : s = 6000) (h2 : b = 14000) (h3 : ps = 0.20) (h4 : pb = 0.23) : 
  s - Nat.floor (ps * s) + b - Nat.floor (pb * b) = 15580 :=
by
  sorry

end remaining_bottles_l1975_197547


namespace max_area_of_triangle_l1975_197510

theorem max_area_of_triangle (a b c : ℝ) (hC : C = 60) (h1 : 3 * a * b = 25 - c^2) :
  (∃ S : ℝ, S = (a * b * (Real.sqrt 3)) / 4 ∧ S = 25 * (Real.sqrt 3) / 16) :=
sorry

end max_area_of_triangle_l1975_197510


namespace nonnegative_solutions_eq1_l1975_197509

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l1975_197509


namespace num_friends_solved_problems_l1975_197507

theorem num_friends_solved_problems (x y n : ℕ) (h1 : 24 * x + 28 * y = 256) (h2 : n = x + y) : n = 10 :=
by
  -- Begin the placeholder proof
  sorry

end num_friends_solved_problems_l1975_197507


namespace Jessie_weight_loss_l1975_197550

theorem Jessie_weight_loss :
  let initial_weight := 74
  let current_weight := 67
  (initial_weight - current_weight) = 7 :=
by
  sorry

end Jessie_weight_loss_l1975_197550


namespace minimum_value_is_six_l1975_197564

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z)

theorem minimum_value_is_six
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + z = 9) (h2 : y = 2 * x) :
  minimum_value_expression x y z = 6 :=
by
  sorry

end minimum_value_is_six_l1975_197564


namespace solve_equation_l1975_197576

theorem solve_equation (x : ℝ) (h : x ≠ 2) : -x^2 = (4 * x + 2) / (x - 2) ↔ x = -2 :=
by sorry

end solve_equation_l1975_197576


namespace part1_part2_l1975_197540

-- Part 1: Showing x range for increasing actual processing fee
theorem part1 (x : ℝ) : (x ≤ 99.5) ↔ (∀ y, 0 < y → y ≤ x → (1/2) * Real.log (2 * y + 1) - y / 200 ≤ (1/2) * Real.log (2 * (y + 0.1) + 1) - (y + 0.1) / 200) :=
sorry

-- Part 2: Showing m range for no losses in processing production
theorem part2 (m x : ℝ) (hx : x ∈ Set.Icc 10 20) : 
  (m ≤ (Real.log 41 - 2) / 40) ↔ ((1/2) * Real.log (2 * x + 1) - m * x ≥ (1/20) * x) :=
sorry

end part1_part2_l1975_197540


namespace garden_roller_length_l1975_197582

/-- The length of a garden roller with diameter 1.4m,
covering 52.8m² in 6 revolutions, and using π = 22/7,
is 2 meters. -/
theorem garden_roller_length
  (diameter : ℝ)
  (total_area_covered : ℝ)
  (revolutions : ℕ)
  (approx_pi : ℝ)
  (circumference : ℝ := approx_pi * diameter)
  (area_per_revolution : ℝ := total_area_covered / (revolutions : ℝ))
  (length : ℝ := area_per_revolution / circumference) :
  diameter = 1.4 ∧ total_area_covered = 52.8 ∧ revolutions = 6 ∧ approx_pi = (22 / 7) → length = 2 :=
by
  sorry

end garden_roller_length_l1975_197582


namespace prove_expression_l1975_197557

-- Define the operation for real numbers
def op (a b c : ℝ) : ℝ := (a - b + c) ^ 2

-- Stating the theorem for the given expression
theorem prove_expression (x z : ℝ) :
  op ((x + z) ^ 2) ((z - x) ^ 2) ((x - z) ^ 2) = (x + z) ^ 4 := 
by  sorry

end prove_expression_l1975_197557


namespace nine_pow_2048_mod_50_l1975_197500

theorem nine_pow_2048_mod_50 : (9^2048) % 50 = 21 := sorry

end nine_pow_2048_mod_50_l1975_197500


namespace angle_in_third_quadrant_l1975_197524

theorem angle_in_third_quadrant
  (α : ℝ) (hα : 270 < α ∧ α < 360) : 90 < 180 - α ∧ 180 - α < 180 :=
by
  sorry

end angle_in_third_quadrant_l1975_197524


namespace exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l1975_197528

-- Define the notion of a balanced integer.
def isBalanced (N : ℕ) : Prop :=
  N = 1 ∨ ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ N = p ^ (2 * k)

-- Define the polynomial P(x) = (x + a)(x + b)
def P (a b x : ℕ) : ℕ := (x + a) * (x + b)

theorem exists_distinct_a_b_all_P_balanced :
  ∃ (a b : ℕ), a ≠ b ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → isBalanced (P a b n) :=
sorry

theorem P_balanced_implies_a_eq_b (a b : ℕ) :
  (∀ n : ℕ, isBalanced (P a b n)) → a = b :=
sorry

end exists_distinct_a_b_all_P_balanced_P_balanced_implies_a_eq_b_l1975_197528


namespace find_n_l1975_197536

theorem find_n (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ) (h : (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7
                      = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 29 - 7) : 7 = 7 :=
by
  sorry

end find_n_l1975_197536


namespace algebraic_expression_value_l1975_197577

-- Define the conditions given
variables {a b : ℝ}
axiom h1 : a ≠ b
axiom h2 : a^2 - 8 * a + 5 = 0
axiom h3 : b^2 - 8 * b + 5 = 0

-- Main theorem to prove the expression equals -20
theorem algebraic_expression_value:
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
sorry

end algebraic_expression_value_l1975_197577


namespace find_n_l1975_197530

theorem find_n (n : ℕ) (h_pos : n > 0) (h_ineq : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1) : n = 8 := by sorry

end find_n_l1975_197530


namespace sum_of_ages_is_50_l1975_197520

def youngest_child_age : ℕ := 4

def age_intervals : ℕ := 3

def ages_sum (n : ℕ) : ℕ :=
  youngest_child_age + (youngest_child_age + age_intervals) +
  (youngest_child_age + 2 * age_intervals) +
  (youngest_child_age + 3 * age_intervals) +
  (youngest_child_age + 4 * age_intervals)

theorem sum_of_ages_is_50 : ages_sum 5 = 50 :=
by
  sorry

end sum_of_ages_is_50_l1975_197520


namespace time_solution_l1975_197560

-- Define the condition as a hypothesis
theorem time_solution (x : ℝ) (h : x / 4 + (24 - x) / 2 = x) : x = 9.6 :=
by
  -- Proof skipped
  sorry

end time_solution_l1975_197560


namespace solve_for_x_l1975_197568

theorem solve_for_x (x : ℝ) (h : 3 / (x + 10) = 1 / (2 * x)) : x = 2 :=
sorry

end solve_for_x_l1975_197568


namespace parabola_standard_equation_l1975_197517

variable (a : ℝ) (h : a < 0)

theorem parabola_standard_equation :
  (∃ p : ℝ, y^2 = -2 * p * x ∧ p = -2 * a) → y^2 = 4 * a * x :=
by
  sorry

end parabola_standard_equation_l1975_197517


namespace special_hash_calculation_l1975_197589

-- Definition of the operation #
def special_hash (a b : ℤ) : ℚ := 2 * a + (a / b) + 3

-- Statement of the proof problem
theorem special_hash_calculation : special_hash 7 3 = 19 + 1/3 := 
by 
  sorry

end special_hash_calculation_l1975_197589
