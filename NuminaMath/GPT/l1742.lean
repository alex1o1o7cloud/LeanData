import Mathlib

namespace not_factorable_l1742_174207

-- Define the quartic polynomial P(x)
def P (x : ℤ) : ℤ := x^4 + 2 * x^2 + 2 * x + 2

-- Define the quadratic polynomials with integer coefficients
def Q₁ (a b x : ℤ) : ℤ := x^2 + a * x + b
def Q₂ (c d x : ℤ) : ℤ := x^2 + c * x + d

-- Define the condition for factorization, and the theorem to be proven
theorem not_factorable :
  ¬ ∃ (a b c d : ℤ), ∀ x : ℤ, P x = (Q₁ a b x) * (Q₂ c d x) := by
  sorry

end not_factorable_l1742_174207


namespace train_crossing_time_l1742_174283

def train_length : ℕ := 320
def train_speed_kmh : ℕ := 72
def kmh_to_ms (v : ℕ) : ℕ := v * 1000 / 3600
def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh
def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time : crossing_time train_length train_speed_ms = 16 := 
by {
  sorry
}

end train_crossing_time_l1742_174283


namespace M_gt_N_l1742_174240

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2 * (x + y - 1)

theorem M_gt_N : M x y > N x y := sorry

end M_gt_N_l1742_174240


namespace total_handshakes_l1742_174292

theorem total_handshakes (team1 team2 refs : ℕ) (players_per_team : ℕ) :
  team1 = 11 → team2 = 11 → refs = 3 → players_per_team = 11 →
  (players_per_team * players_per_team + (players_per_team * 2 * refs) = 187) :=
by
  intros h_team1 h_team2 h_refs h_players_per_team
  -- Now we want to prove that
  -- 11 * 11 + (11 * 2 * 3) = 187
  -- However, we can just add sorry here as the purpose is to write the statement
  sorry

end total_handshakes_l1742_174292


namespace prob_white_ball_is_0_25_l1742_174277

-- Let's define the conditions and the statement for the proof
variable (P_red P_white P_yellow : ℝ)

-- The given conditions 
def prob_red_or_white : Prop := P_red + P_white = 0.65
def prob_yellow_or_white : Prop := P_yellow + P_white = 0.6

-- The statement we want to prove
theorem prob_white_ball_is_0_25 (h1 : prob_red_or_white P_red P_white)
                               (h2 : prob_yellow_or_white P_yellow P_white) :
  P_white = 0.25 :=
sorry

end prob_white_ball_is_0_25_l1742_174277


namespace determine_p_l1742_174262

def is_tangent (circle_eq : ℝ → ℝ → Prop) (parabola_eq : ℝ → ℝ → Prop) (p : ℝ) : Prop :=
  ∃ x y : ℝ, parabola_eq x y ∧ circle_eq x y ∧ x = -p / 2 

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def parabola_eq (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

theorem determine_p (p : ℝ) (hpos : p > 0) :
  (is_tangent circle_eq (parabola_eq p) p) ↔ p = 2 := 
sorry

end determine_p_l1742_174262


namespace part_a_proof_part_b_proof_l1742_174222

-- Part (a) statement
def part_a_statement (n : ℕ) : Prop :=
  ∀ (m : ℕ), m = 9 → (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 12 ∨ n = 18)

theorem part_a_proof (n : ℕ) (m : ℕ) (h : m = 9) : part_a_statement n :=
  sorry

-- Part (b) statement
def part_b_statement (n m : ℕ) : Prop :=
  (n ≤ m) ∨ (n > m ∧ ∃ d : ℕ, d ∣ m ∧ n = m + d)

theorem part_b_proof (n m : ℕ) : part_b_statement n m :=
  sorry

end part_a_proof_part_b_proof_l1742_174222


namespace mean_of_set_l1742_174293

theorem mean_of_set (n : ℤ) (h_median : n + 7 = 14) : (n + (n + 4) + (n + 7) + (n + 10) + (n + 14)) / 5 = 14 := by
  sorry

end mean_of_set_l1742_174293


namespace union_of_M_and_N_l1742_174286

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x^2 - 4 * x < 0}
  let N := {x : ℝ | |x| ≤ 2}
  M ∪ N = {x : ℝ | -2 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_M_and_N_l1742_174286


namespace positive_difference_abs_eq_l1742_174227

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l1742_174227


namespace min_folds_exceed_12mm_l1742_174206

theorem min_folds_exceed_12mm : ∃ n : ℕ, 0.1 * (2: ℝ)^n > 12 ∧ ∀ m < n, 0.1 * (2: ℝ)^m ≤ 12 := 
by
  sorry

end min_folds_exceed_12mm_l1742_174206


namespace evaluate_expression_l1742_174282

theorem evaluate_expression : 3 - 5 * (6 - 2^3) / 2 = 8 :=
by
  sorry

end evaluate_expression_l1742_174282


namespace velocity_ratio_proof_l1742_174285

noncomputable def velocity_ratio (V U : ℝ) : ℝ := V / U

-- The conditions:
-- 1. A smooth horizontal surface.
-- 2. The speed of the ball is perpendicular to the face of the block.
-- 3. The mass of the ball is much smaller than the mass of the block.
-- 4. The collision is elastic.
-- 5. After the collision, the ball’s speed is halved and it moves in the opposite direction.

def ball_block_collision 
    (V U U_final : ℝ) 
    (smooth_surface : Prop) 
    (perpendicular_impact : Prop) 
    (ball_much_smaller : Prop) 
    (elastic_collision : Prop) 
    (speed_halved : Prop) : Prop :=
  U_final = U ∧ V / U = 4

theorem velocity_ratio_proof : 
  ∀ (V U U_final : ℝ)
    (smooth_surface : Prop)
    (perpendicular_impact : Prop)
    (ball_much_smaller : Prop)
    (elastic_collision : Prop)
    (speed_halved : Prop),
    ball_block_collision V U U_final smooth_surface perpendicular_impact ball_much_smaller elastic_collision speed_halved := 
sorry

end velocity_ratio_proof_l1742_174285


namespace ivan_total_pay_l1742_174219

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l1742_174219


namespace compare_y_values_l1742_174243

-- Define the quadratic function y = x^2 + 2x + c
def quadratic (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * x + c

-- Points A, B, and C on the quadratic function
variables 
  (c : ℝ) 
  (y1 y2 y3 : ℝ) 
  (hA : y1 = quadratic (-3) c) 
  (hB : y2 = quadratic (-2) c) 
  (hC : y3 = quadratic 2 c)

theorem compare_y_values :
  y3 > y1 ∧ y1 > y2 :=
by sorry

end compare_y_values_l1742_174243


namespace set_intersection_nonempty_implies_m_le_neg1_l1742_174299

theorem set_intersection_nonempty_implies_m_le_neg1
  (m : ℝ)
  (A : Set ℝ := {x | x^2 - 4 * m * x + 2 * m + 6 = 0})
  (B : Set ℝ := {x | x < 0}) :
  (A ∩ B).Nonempty → m ≤ -1 := 
sorry

end set_intersection_nonempty_implies_m_le_neg1_l1742_174299


namespace parallel_lines_slope_l1742_174225

theorem parallel_lines_slope (b : ℚ) :
  (∀ x y : ℚ, 3 * y + x - 1 = 0 → 2 * y + b * x - 4 = 0 ∨
    3 * y + x - 1 = 0 ∧ 2 * y + b * x - 4 = 0) →
  b = 2 / 3 :=
by
  intro h
  sorry

end parallel_lines_slope_l1742_174225


namespace tyrone_gave_25_marbles_l1742_174230

/-- Given that Tyrone initially had 97 marbles and Eric had 11 marbles, and after
    giving some marbles to Eric, Tyrone ended with twice as many marbles as Eric,
    we need to find the number of marbles Tyrone gave to Eric. -/
theorem tyrone_gave_25_marbles (x : ℕ) (t0 e0 : ℕ)
  (hT0 : t0 = 97)
  (hE0 : e0 = 11)
  (hT_end : (t0 - x) = 2 * (e0 + x)) :
  x = 25 := 
  sorry

end tyrone_gave_25_marbles_l1742_174230


namespace solve_cubic_equation_l1742_174246

theorem solve_cubic_equation : ∀ x : ℝ, (x^3 - 5*x^2 + 6*x - 2 = 0) → (x = 2) :=
by
  intro x
  intro h
  sorry

end solve_cubic_equation_l1742_174246


namespace gabriel_month_days_l1742_174257

theorem gabriel_month_days (forgot_days took_days : ℕ) (h_forgot : forgot_days = 3) (h_took : took_days = 28) : 
  forgot_days + took_days = 31 :=
by
  sorry

end gabriel_month_days_l1742_174257


namespace half_AB_equals_l1742_174229

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := (3, 2)
def vector_OB : ℝ × ℝ := (4, 7)

-- Prove that (1 / 2) * (OB - OA) = (1 / 2, 5 / 2)
theorem half_AB_equals :
  (1 / 2 : ℝ) • ((vector_OB.1 - vector_OA.1), (vector_OB.2 - vector_OA.2)) = (1 / 2, 5 / 2) := 
  sorry

end half_AB_equals_l1742_174229


namespace problem_statement_l1742_174254

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

noncomputable def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by {
  sorry
}

end problem_statement_l1742_174254


namespace carlton_school_earnings_l1742_174247

theorem carlton_school_earnings :
  let students_days_adams := 8 * 4
  let students_days_byron := 5 * 6
  let students_days_carlton := 6 * 10
  let total_wages := 1092
  students_days_adams + students_days_byron = 62 → 
  62 * (2 * x) + students_days_carlton * x = total_wages → 
  x = (total_wages : ℝ) / 184 → 
  (students_days_carlton : ℝ) * x = 356.09 := 
by
  intros _ _ _ 
  sorry

end carlton_school_earnings_l1742_174247


namespace largest_number_l1742_174253

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = 0) (h3 : c = 1) (h4 : d = -9) :
  max (max a b) (max c d) = c :=
by
  sorry

end largest_number_l1742_174253


namespace triangle_sides_proportional_l1742_174267

theorem triangle_sides_proportional (a b c r d : ℝ)
  (h1 : 2 * r < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : a = 2 * r + d)
  (h5 : b = 2 * r + 2 * d)
  (h6 : c = 2 * r + 3 * d)
  (hr_pos : r > 0)
  (hd_pos : d > 0) :
  ∃ k : ℝ, k > 0 ∧ a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end triangle_sides_proportional_l1742_174267


namespace solve_inequality_l1742_174237

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end solve_inequality_l1742_174237


namespace bud_age_uncle_age_relation_l1742_174218

variable (bud_age uncle_age : Nat)

theorem bud_age_uncle_age_relation (h : bud_age = 8) (h0 : bud_age = uncle_age / 3) : uncle_age = 24 := by
  sorry

end bud_age_uncle_age_relation_l1742_174218


namespace perpendicular_lines_foot_of_perpendicular_l1742_174216

theorem perpendicular_lines_foot_of_perpendicular 
  (m n p : ℝ) 
  (h1 : 2 * 2 + 3 * p - 1 = 0)
  (h2 : 3 * 2 - 2 * p + n = 0)
  (h3 : - (2 / m) * (3 / 2) = -1) 
  : p - m - n = 4 := 
by
  sorry

end perpendicular_lines_foot_of_perpendicular_l1742_174216


namespace probability_not_both_ends_l1742_174224

theorem probability_not_both_ends :
  let total_arrangements := 120
  let both_ends_arrangements := 12
  let favorable_arrangements := total_arrangements - both_ends_arrangements
  let probability := favorable_arrangements / total_arrangements
  total_arrangements = 120 ∧ both_ends_arrangements = 12 ∧ favorable_arrangements = 108 ∧ probability = 0.9 :=
by
  sorry

end probability_not_both_ends_l1742_174224


namespace car_speed_car_speed_correct_l1742_174235

theorem car_speed (d t s : ℝ) (hd : d = 810) (ht : t = 5) : s = d / t := 
by
  sorry

theorem car_speed_correct (d t : ℝ) (hd : d = 810) (ht : t = 5) : d / t = 162 :=
by
  sorry

end car_speed_car_speed_correct_l1742_174235


namespace panthers_score_l1742_174210

theorem panthers_score (P : ℕ) (wildcats_score : ℕ := 36) (score_difference : ℕ := 19) (h : wildcats_score = P + score_difference) : P = 17 := by
  sorry

end panthers_score_l1742_174210


namespace interval_contains_root_l1742_174220

theorem interval_contains_root :
  (∃ c, (0 < c ∧ c < 1) ∧ (2^c + c - 2 = 0) ∧ 
        (∀ x1 x2, x1 < x2 → 2^x1 + x1 - 2 < 2^x2 + x2 - 2) ∧ 
        (0 < 1) ∧ 
        ((2^0 + 0 - 2) = -1) ∧ 
        ((2^1 + 1 - 2) = 1)) := 
by 
  sorry

end interval_contains_root_l1742_174220


namespace complement_of_A_with_respect_to_U_l1742_174244

open Set

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}
def complement_U_A : Set ℕ := {4, 6}

theorem complement_of_A_with_respect_to_U :
  U \ A = complement_U_A := by
  sorry

end complement_of_A_with_respect_to_U_l1742_174244


namespace a_plus_c_eq_neg_300_l1742_174298

namespace Polynomials

variable {α : Type*} [LinearOrderedField α]

def f (a b x : α) := x^2 + a * x + b
def g (c d x : α) := x^2 + c * x + d

theorem a_plus_c_eq_neg_300 
  {a b c d : α}
  (h1 : ∀ x, f a b x ≥ -144) 
  (h2 : ∀ x, g c d x ≥ -144)
  (h3 : f a b 150 = -200) 
  (h4 : g c d 150 = -200)
  (h5 : ∃ x, (2*x + a = 0) ∧ g c d x = 0)
  (h6 : ∃ x, (2*x + c = 0) ∧ f a b x = 0) :
  a + c = -300 := 
sorry

end Polynomials

end a_plus_c_eq_neg_300_l1742_174298


namespace inequality_one_inequality_system_l1742_174234

-- Definition for the first problem
theorem inequality_one (x : ℝ) : 3 * x > 2 * (1 - x) ↔ x > 2 / 5 :=
by
  sorry

-- Definitions for the second problem
theorem inequality_system (x : ℝ) : 
  (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_one_inequality_system_l1742_174234


namespace cn_squared_eq_28_l1742_174276

theorem cn_squared_eq_28 (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end cn_squared_eq_28_l1742_174276


namespace larger_number_of_hcf_23_lcm_factors_13_15_l1742_174250

theorem larger_number_of_hcf_23_lcm_factors_13_15 :
  ∃ A B, (Nat.gcd A B = 23) ∧ (A * B = 23 * 13 * 15) ∧ (A = 345 ∨ B = 345) := sorry

end larger_number_of_hcf_23_lcm_factors_13_15_l1742_174250


namespace min_value_reciprocal_sum_l1742_174287

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 3) :
  (1 / x) + (1 / y) + (1 / z) ≥ 3 :=
sorry

end min_value_reciprocal_sum_l1742_174287


namespace pairs_divisible_by_4_l1742_174231

-- Define the set of valid pairs of digits from 00 to 99
def valid_pairs : List (Fin 100) := List.filter (λ n => n % 4 = 0) (List.range 100)

-- State the theorem
theorem pairs_divisible_by_4 : valid_pairs.length = 25 := by
  sorry

end pairs_divisible_by_4_l1742_174231


namespace sum_of_numbers_in_ratio_l1742_174201

theorem sum_of_numbers_in_ratio 
  (x : ℕ)
  (h : 5 * x = 560) : 
  2 * x + 3 * x + 4 * x + 5 * x = 1568 := 
by 
  sorry

end sum_of_numbers_in_ratio_l1742_174201


namespace solve_for_a_and_b_l1742_174204

noncomputable def A := {x : ℝ | (-2 < x ∧ x < -1) ∨ (x > 1)}
noncomputable def B (a b : ℝ) := {x : ℝ | a ≤ x ∧ x < b}

theorem solve_for_a_and_b (a b : ℝ) :
  (A ∪ B a b = {x : ℝ | x > -2}) ∧ (A ∩ B a b = {x : ℝ | 1 < x ∧ x < 3}) →
  a = -1 ∧ b = 3 :=
by
  sorry

end solve_for_a_and_b_l1742_174204


namespace neg_exists_equiv_forall_l1742_174223

theorem neg_exists_equiv_forall (p : ∃ n : ℕ, 2^n > 1000) :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ ∀ n : ℕ, 2^n ≤ 1000 := 
sorry

end neg_exists_equiv_forall_l1742_174223


namespace eval_expression_l1742_174232

theorem eval_expression : (Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end eval_expression_l1742_174232


namespace settle_debt_using_coins_l1742_174203

theorem settle_debt_using_coins :
  ∃ n m : ℕ, 49 * n - 99 * m = 1 :=
sorry

end settle_debt_using_coins_l1742_174203


namespace tangent_line_through_point_l1742_174252

theorem tangent_line_through_point (x y : ℝ) (tangent f : ℝ → ℝ) (M : ℝ × ℝ) :
  M = (1, 1) →
  f x = x^3 + 1 →
  tangent x = 3 * x^2 →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ ∀ x0 y0 : ℝ, (y0 = f x0) → (y - y0 = tangent x0 * (x - x0))) ∧
  (x, y) = M →
  (a = 0 ∧ b = 1 ∧ c = -1) ∨ (a = 27 ∧ b = -4 ∧ c = -23) :=
by
  sorry

end tangent_line_through_point_l1742_174252


namespace b5_plus_b9_l1742_174261

variable {a : ℕ → ℕ} -- Geometric sequence
variable {b : ℕ → ℕ} -- Arithmetic sequence

axiom geom_progression {r x y : ℕ} : a x = a 1 * r^(x - 1) ∧ a y = a 1 * r^(y - 1)
axiom arith_progression {d x y : ℕ} : b x = b 1 + d * (x - 1) ∧ b y = b 1 + d * (y - 1)

axiom a3a11_equals_4a7 : a 3 * a 11 = 4 * a 7
axiom a7_equals_b7 : a 7 = b 7

theorem b5_plus_b9 : b 5 + b 9 = 8 := by
  apply sorry

end b5_plus_b9_l1742_174261


namespace necessary_and_sufficient_for_perpendicular_l1742_174242

theorem necessary_and_sufficient_for_perpendicular (a : ℝ) :
  (a = -2) ↔ (∀ (x y : ℝ), x + 2 * y = 0 → ax + y = 1 → false) :=
by
  sorry

end necessary_and_sufficient_for_perpendicular_l1742_174242


namespace total_number_of_students_l1742_174212

/-- The total number of high school students in the school given sampling constraints. -/
theorem total_number_of_students (F1 F2 F3 : ℕ) (sample_size : ℕ) (consistency_ratio : ℕ) :
  F2 = 300 ∧ sample_size = 45 ∧ (F1 / F3) = 2 ∧ 
  (20 + 10 + (sample_size - 30)) = sample_size → F1 + F2 + F3 = 900 :=
by
  sorry

end total_number_of_students_l1742_174212


namespace domain_M_complement_domain_M_l1742_174270

noncomputable def f (x : ℝ) : ℝ :=
  1 / Real.sqrt (1 - x)

noncomputable def g (x : ℝ) : ℝ :=
  Real.log (1 + x)

def M : Set ℝ :=
  {x | 1 - x > 0}

def N : Set ℝ :=
  {x | 1 + x > 0}

def complement_M : Set ℝ :=
  {x | 1 - x ≤ 0}

theorem domain_M :
  M = {x | x < 1} := by
  sorry

theorem complement_domain_M :
  complement_M = {x | x ≥ 1} := by
  sorry

end domain_M_complement_domain_M_l1742_174270


namespace smallest_non_factor_l1742_174255

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end smallest_non_factor_l1742_174255


namespace line_does_not_intersect_circle_l1742_174211

theorem line_does_not_intersect_circle (a : ℝ) : 
  (a > 1 ∨ a < -1) → ¬ ∃ (x y : ℝ), (x + y = a) ∧ (x^2 + y^2 = 1) :=
by
  sorry

end line_does_not_intersect_circle_l1742_174211


namespace quadratic_inequality_false_iff_l1742_174241

theorem quadratic_inequality_false_iff (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_false_iff_l1742_174241


namespace problem_statement_l1742_174248

-- Define a multiple of 6 and a multiple of 9
variables (a b : ℤ)
variable (ha : ∃ k, a = 6 * k)
variable (hb : ∃ k, b = 9 * k)

-- Prove that a + b is a multiple of 3
theorem problem_statement : 
  (∃ k, a + b = 3 * k) ∧ 
  ¬((∀ m n, a = 6 * m ∧ b = 9 * n → (a + b = odd))) ∧ 
  ¬(∃ k, a + b = 6 * k) ∧ 
  ¬(∃ k, a + b = 9 * k) :=
by
  sorry

end problem_statement_l1742_174248


namespace exists_prime_q_l1742_174295

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) (h2 : 2 < p) : 
  ∃ q : ℕ, Nat.Prime q ∧ q < p ∧ ¬ (p ^ 2 ∣ q ^ (p - 1) - 1) := 
sorry

end exists_prime_q_l1742_174295


namespace train_length_is_correct_l1742_174259

noncomputable def length_of_train (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := (speed_train_kmph + speed_man_kmph)
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5 / 18)
  relative_speed_mps * (time_seconds : ℝ)

theorem train_length_is_correct :
  length_of_train 60 6 3 = 54.99 := 
by
  sorry

end train_length_is_correct_l1742_174259


namespace MarlySoupBags_l1742_174284

theorem MarlySoupBags :
  ∀ (milk chicken_stock vegetables bag_capacity total_soup total_bags : ℚ),
    milk = 6 ∧
    chicken_stock = 3 * milk ∧
    vegetables = 3 ∧
    bag_capacity = 2 ∧
    total_soup = milk + chicken_stock + vegetables ∧
    total_bags = total_soup / bag_capacity ∧
    total_bags.ceil = 14 :=
by
  intros
  sorry

end MarlySoupBags_l1742_174284


namespace evaluate_expression_x_eq_3_l1742_174266

theorem evaluate_expression_x_eq_3 : (3^5 - 5 * 3 + 7 * 3^3) = 417 := by
  sorry

end evaluate_expression_x_eq_3_l1742_174266


namespace evaluate_expression_l1742_174265

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end evaluate_expression_l1742_174265


namespace infinite_sum_problem_l1742_174258

theorem infinite_sum_problem :
  (∑' n : ℕ, if n = 0 then 0 else (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = (1 / 4) := 
by
  sorry

end infinite_sum_problem_l1742_174258


namespace part1_part2_part3_l1742_174272

/-- Proof for part (1): If the point P lies on the x-axis, then m = -1. -/
theorem part1 (m : ℝ) (hx : 3 * m + 3 = 0) : m = -1 := 
by {
  sorry
}

/-- Proof for part (2): If point P lies on a line passing through A(-5, 1) and parallel to the y-axis, 
then the coordinates of point P are (-5, -12). -/
theorem part2 (m : ℝ) (hy : 2 * m + 5 = -5) : (2 * m + 5, 3 * m + 3) = (-5, -12) := 
by {
  sorry
}

/-- Proof for part (3): If point P is moved 2 right and 3 up to point M, 
and point M lies in the third quadrant with a distance of 7 from the y-axis, then the coordinates of M are (-7, -15). -/
theorem part3 (m : ℝ) 
  (hc : 2 * m + 7 = -7)
  (config : 3 * m + 6 < 0) : (2 * m + 7, 3 * m + 6) = (-7, -15) := 
by {
  sorry
}

end part1_part2_part3_l1742_174272


namespace inlet_pipe_rate_16_liters_per_minute_l1742_174202

noncomputable def rate_of_inlet_pipe : ℝ :=
  let capacity := 21600 -- litres
  let outlet_time_alone := 10 -- hours
  let outlet_time_with_inlet := 18 -- hours
  let outlet_rate := capacity / outlet_time_alone
  let combined_rate := capacity / outlet_time_with_inlet
  let inlet_rate := outlet_rate - combined_rate
  inlet_rate / 60 -- converting litres/hour to litres/min

theorem inlet_pipe_rate_16_liters_per_minute : rate_of_inlet_pipe = 16 :=
by
  sorry

end inlet_pipe_rate_16_liters_per_minute_l1742_174202


namespace jana_distance_travel_in_20_minutes_l1742_174256

theorem jana_distance_travel_in_20_minutes :
  ∀ (usual_pace half_pace double_pace : ℚ)
    (first_15_minutes_distance second_5_minutes_distance total_distance : ℚ),
  usual_pace = 1 / 30 →
  half_pace = usual_pace / 2 →
  double_pace = usual_pace * 2 →
  first_15_minutes_distance = 15 * half_pace →
  second_5_minutes_distance = 5 * double_pace →
  total_distance = first_15_minutes_distance + second_5_minutes_distance →
  total_distance = 7 / 12 := 
by
  intros
  sorry

end jana_distance_travel_in_20_minutes_l1742_174256


namespace rabbit_carrots_l1742_174208

theorem rabbit_carrots (r f : ℕ) (hr : 3 * r = 5 * f) (hf : f = r - 6) : 3 * r = 45 :=
by
  sorry

end rabbit_carrots_l1742_174208


namespace remainder_proof_l1742_174288

-- Definitions and conditions
variables {x y u v : ℕ}
variables (hx : x = u * y + v)

-- Problem statement in Lean 4
theorem remainder_proof (hx : x = u * y + v) : ((x + 3 * u * y + y) % y) = v :=
sorry

end remainder_proof_l1742_174288


namespace bob_correct_answer_l1742_174245

theorem bob_correct_answer (y : ℕ) (h : (y - 7) / 5 = 47) : (y - 5) / 7 = 33 :=
by 
  -- assumption h and the statement to prove
  sorry

end bob_correct_answer_l1742_174245


namespace circumradius_of_triangle_l1742_174290

theorem circumradius_of_triangle (a b S : ℝ) (A : a = 2) (B : b = 3) (Area : S = 3 * Real.sqrt 15 / 4)
  (median_cond : ∃ c m, m = (a^2 + b^2 - c^2) / (2*a*b) ∧ m < c / 2) :
  ∃ R, R = 8 / Real.sqrt 15 :=
by
  sorry

end circumradius_of_triangle_l1742_174290


namespace hyperbola_eccentricity_squared_l1742_174200

/-- Given that F is the right focus of the hyperbola 
    \( C: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 \) with \( a > 0 \) and \( b > 0 \), 
    a line perpendicular to the x-axis is drawn through point F, 
    intersecting one asymptote of the hyperbola at point M. 
    If \( |FM| = 2a \), denote the eccentricity of the hyperbola as \( e \). 
    Prove that \( e^2 = \frac{1 + \sqrt{17}}{2} \).
 -/
theorem hyperbola_eccentricity_squared (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: c^2 = a^2 + b^2) (h4: b * c = 2 * a^2) : 
  (c / a)^2 = (1 + Real.sqrt 17) / 2 := 
sorry

end hyperbola_eccentricity_squared_l1742_174200


namespace number_of_friends_l1742_174239

def total_gold := 100
def lost_gold := 20
def gold_per_friend := 20

theorem number_of_friends :
  (total_gold - lost_gold) / gold_per_friend = 4 := by
  sorry

end number_of_friends_l1742_174239


namespace least_number_with_remainders_l1742_174238

theorem least_number_with_remainders :
  ∃ x, (x ≡ 4 [MOD 5]) ∧ (x ≡ 4 [MOD 6]) ∧ (x ≡ 4 [MOD 9]) ∧ (x ≡ 4 [MOD 18]) ∧ x = 94 := 
by 
  sorry

end least_number_with_remainders_l1742_174238


namespace tan_product_l1742_174209

open Real

theorem tan_product (x y : ℝ) 
(h1 : sin x * sin y = 24 / 65) 
(h2 : cos x * cos y = 48 / 65) :
tan x * tan y = 1 / 2 :=
by
  sorry

end tan_product_l1742_174209


namespace find_digit_sum_l1742_174273

theorem find_digit_sum (A B X D C Y : ℕ) :
  (A * 100 + B * 10 + X) + (C * 100 + D * 10 + Y) = Y * 1010 + X * 1010 →
  A + D = 6 :=
by
  sorry

end find_digit_sum_l1742_174273


namespace fraction_zero_l1742_174297

theorem fraction_zero (x : ℝ) (h : x ≠ -1) (h₀ : (x^2 - 1) / (x + 1) = 0) : x = 1 :=
by {
  sorry
}

end fraction_zero_l1742_174297


namespace bookcase_length_in_inches_l1742_174275

theorem bookcase_length_in_inches (feet_length : ℕ) (inches_per_foot : ℕ) (h1 : feet_length = 4) (h2 : inches_per_foot = 12) : (feet_length * inches_per_foot) = 48 :=
by
  sorry

end bookcase_length_in_inches_l1742_174275


namespace complex_i_power_l1742_174215

theorem complex_i_power (i : ℂ) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : i^2015 = -i := 
by
  sorry

end complex_i_power_l1742_174215


namespace range_of_k_is_l1742_174221

noncomputable def range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : Set ℝ :=
{k : ℝ | ∀ x : ℝ, a^x + 4 * a^(-x) - k > 0}

theorem range_of_k_is (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  range_of_k a h₁ h₂ = { k : ℝ | k < 4 ∧ k ≠ 3 } :=
sorry

end range_of_k_is_l1742_174221


namespace area_curve_is_correct_l1742_174274

-- Define the initial conditions
structure Rectangle :=
  (vertices : Fin 4 → ℝ × ℝ)
  (point : ℝ × ℝ)

-- Define the rotation transformation
def rotate_clockwise_90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  (cx + (py - cy), cy - (px - cx))

-- Given initial rectangle and the point to track
def initial_rectangle : Rectangle :=
  { vertices := ![(0, 0), (2, 0), (0, 3), (2, 3)],
    point := (1, 1) }

-- Perform the four specified rotations
def rotated_points : List (ℝ × ℝ) :=
  let r1 := rotate_clockwise_90 (2, 0) initial_rectangle.point
  let r2 := rotate_clockwise_90 (5, 0) r1
  let r3 := rotate_clockwise_90 (7, 0) r2
  let r4 := rotate_clockwise_90 (10, 0) r3
  [initial_rectangle.point, r1, r2, r3, r4]

-- Calculate the area below the curve and above the x-axis
noncomputable def area_below_curve : ℝ :=
  6 + (7 * Real.pi / 2)

-- The theorem statement
theorem area_curve_is_correct : 
  area_below_curve = 6 + (7 * Real.pi / 2) :=
  by trivial

end area_curve_is_correct_l1742_174274


namespace valid_base6_number_2015_l1742_174263

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015_l1742_174263


namespace shift_down_equation_l1742_174281

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f x - 3

theorem shift_down_equation : ∀ x : ℝ, g x = 2 * x := by
  sorry

end shift_down_equation_l1742_174281


namespace no_integer_n_satisfies_conditions_l1742_174233

theorem no_integer_n_satisfies_conditions :
  ¬ ∃ n : ℕ, 0 < n ∧ 1000 ≤ n / 5 ∧ n / 5 ≤ 9999 ∧ 1000 ≤ 5 * n ∧ 5 * n ≤ 9999 :=
by
  sorry

end no_integer_n_satisfies_conditions_l1742_174233


namespace bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l1742_174228

def bernardo (x : ℕ) : ℕ := 2 * x
def silvia (x : ℕ) : ℕ := x + 30

theorem bernardo_winning_N_initial (N : ℕ) :
  (∃ k : ℕ, (bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia N) = k
  ∧ 950 ≤ k ∧ k ≤ 999)
  → 34 ≤ N ∧ N ≤ 35 :=
by
  sorry

theorem bernardo_smallest_N (N : ℕ) (h : 34 ≤ N ∧ N ≤ 35) :
  (N = 34) :=
by
  sorry

theorem sum_of_digits_34 :
  (3 + 4 = 7) :=
by
  sorry

end bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l1742_174228


namespace parabola_tangent_midpoint_l1742_174294

theorem parabola_tangent_midpoint (p : ℝ) (h : p > 0) :
    (∃ M : ℝ × ℝ, M = (2, -2*p)) ∧ 
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
                      (∃ yA yB : ℝ, yA = (A.1^2)/(2*p) ∧ yB = (B.1^2)/(2*p)) ∧ 
                      (0.5 * (A.2 + B.2) = 6)) → p = 1 := by sorry

end parabola_tangent_midpoint_l1742_174294


namespace cashier_five_dollar_bills_l1742_174226

-- Define the conditions as a structure
structure CashierBills (x y : ℕ) : Prop :=
(total_bills : x + y = 126)
(total_value : 5 * x + 10 * y = 840)

-- State the theorem that we need to prove
theorem cashier_five_dollar_bills (x y : ℕ) (h : CashierBills x y) : x = 84 :=
sorry

end cashier_five_dollar_bills_l1742_174226


namespace probability_point_closer_to_7_than_0_l1742_174289

noncomputable def segment_length (a b : ℝ) : ℝ := b - a
noncomputable def closer_segment (a c b : ℝ) : ℝ := segment_length c b

theorem probability_point_closer_to_7_than_0 :
  let a := 0
  let b := 10
  let c := 7
  let midpoint := (a + c) / 2
  let total_length := b - a
  let closer_length := segment_length midpoint b
  (closer_length / total_length) = 0.7 :=
by
  sorry

end probability_point_closer_to_7_than_0_l1742_174289


namespace lyka_saving_per_week_l1742_174249

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l1742_174249


namespace savings_value_l1742_174279

def total_cost_individual (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let cost (n : ℕ) : ℝ := 
    let paid_windows := n - (n / 6) -- one free window per five
    cost_per_window * paid_windows
  let discount (amount : ℝ) : ℝ :=
    if s > 10 then 0.95 * amount else amount
  discount (cost g) + discount (cost s)

def total_cost_joint (g : ℕ) (s : ℕ) : ℝ :=
  let cost_per_window := 120
  let n := g + s
  let paid_windows := n - (n / 6) -- one free window per five
  let joint_cost := cost_per_window * paid_windows
  if n > 10 then 0.95 * joint_cost else joint_cost

def savings (g : ℕ) (s : ℕ) : ℝ :=
  total_cost_individual g s - total_cost_joint g s

theorem savings_value (g s : ℕ) (hg : g = 9) (hs : s = 13) : savings g s = 162 := 
by 
  simp [savings, total_cost_individual, total_cost_joint, hg, hs]
  -- Detailed calculation is omitted, since it's not required according to the instructions.
  sorry

end savings_value_l1742_174279


namespace find_parabola_equation_l1742_174291

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
    (y = a * x ^ 2 + b * x + c) ∧ 
    (y = (x - 3) ^ 2 - 2) ∧
    (a * (4 - 3) ^ 2 - 2 = 2)

theorem find_parabola_equation :
  ∃ (a b c : ℝ), parabola_equation a b c ∧ a = 4 ∧ b = -24 ∧ c = 34 :=
sorry

end find_parabola_equation_l1742_174291


namespace rest_stop_location_l1742_174269

theorem rest_stop_location (km_A km_B : ℕ) (fraction : ℚ) (difference := km_B - km_A) 
  (rest_stop_distance := fraction * difference) : 
  km_A = 30 → km_B = 210 → fraction = 4 / 5 → rest_stop_distance + km_A = 174 :=
by 
  intros h1 h2 h3
  sorry

end rest_stop_location_l1742_174269


namespace inequality_proof_l1742_174217

variable {x₁ x₂ x₃ x₄ : ℝ}

theorem inequality_proof
  (h₁ : x₁ ≥ x₂) (h₂ : x₂ ≥ x₃) (h₃ : x₃ ≥ x₄) (h₄ : x₄ ≥ 2)
  (h₅ : x₂ + x₃ + x₄ ≥ x₁) 
  : (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := 
by {
  sorry
}

end inequality_proof_l1742_174217


namespace sum_of_consecutive_integers_sqrt_28_l1742_174214

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end sum_of_consecutive_integers_sqrt_28_l1742_174214


namespace math_proof_statement_l1742_174296

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  let a := (cos x, sin x)
  let b := (sqrt 2, sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8 / 5) ∧ (π / 4 < x ∧ x < π / 2) ∧ 
  (cos (x - π / 4) = 4 / 5) ∧ (tan (x - π / 4) = 3 / 4) ∧ 
  (sin (2 * x) * (1 - tan x) / (1 + tan x) = -21 / 100)

theorem math_proof_statement (x : ℝ) : proof_problem x := 
by
  unfold proof_problem
  sorry

end math_proof_statement_l1742_174296


namespace redesigned_lock_additional_combinations_l1742_174251

-- Definitions for the problem conditions
def original_combinations : ℕ := Nat.choose 10 5
def total_new_combinations : ℕ := (Finset.range 10).sum (λ k => Nat.choose 10 (k + 1)) 
def additional_combinations := total_new_combinations - original_combinations - 2 -- subtract combinations for 0 and 10

-- Statement of the theorem
theorem redesigned_lock_additional_combinations : additional_combinations = 770 :=
by
  -- Proof omitted (insert 'sorry' to indicate incomplete proof state)
  sorry

end redesigned_lock_additional_combinations_l1742_174251


namespace georgia_total_carnation_cost_l1742_174271

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l1742_174271


namespace stratified_sampling_school_C_l1742_174236

theorem stratified_sampling_school_C 
  (teachers_A : ℕ) 
  (teachers_B : ℕ) 
  (teachers_C : ℕ) 
  (total_teachers : ℕ)
  (total_drawn : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 140)
  (hC : teachers_C = 160)
  (hTotal : total_teachers = teachers_A + teachers_B + teachers_C)
  (hDraw : total_drawn = 60) :
  (total_drawn * teachers_C / total_teachers) = 20 := 
by
  sorry

end stratified_sampling_school_C_l1742_174236


namespace new_point_in_fourth_quadrant_l1742_174213

-- Define the initial point P with coordinates (-3, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the move operation: 4 units to the right and 6 units down
def move (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 4, p.2 - 6)

-- Define the new point after the move operation
def P' : ℝ × ℝ := move P

-- Prove that the new point P' is in the fourth quadrant
theorem new_point_in_fourth_quadrant (x y : ℝ) (h : P' = (x, y)) : x > 0 ∧ y < 0 :=
by
  sorry

end new_point_in_fourth_quadrant_l1742_174213


namespace s9_s3_ratio_l1742_174280

variable {a_n : ℕ → ℝ}
variable {s_n : ℕ → ℝ}
variable {a : ℝ}

-- Conditions
axiom h_s6_s3_ratio : s_n 6 / s_n 3 = 1 / 2

-- Theorem to prove
theorem s9_s3_ratio (h : s_n 3 = a) : s_n 9 / s_n 3 = 3 / 4 := 
sorry

end s9_s3_ratio_l1742_174280


namespace find_C_l1742_174268

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : 
  C = 10 := 
by
  sorry

end find_C_l1742_174268


namespace find_x_l1742_174260

variable {a b x r : ℝ}
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (h₂ : r = (4 * a)^(2 * b))
variable (h₃ : r = (a^b * x^b)^2)
variable (h₄ : 0 < x)

theorem find_x : x = 4 := by
  sorry

end find_x_l1742_174260


namespace smallest_positive_integer_l1742_174264

theorem smallest_positive_integer (n : ℕ) : 3 * n ≡ 568 [MOD 34] → n = 18 := 
sorry

end smallest_positive_integer_l1742_174264


namespace points_per_draw_l1742_174205

-- Definitions based on conditions
def total_games : ℕ := 20
def wins : ℕ := 14
def losses : ℕ := 2
def total_points : ℕ := 46
def points_per_win : ℕ := 3
def points_per_loss : ℕ := 0

-- Calculation of the number of draws and points per draw
def draws : ℕ := total_games - wins - losses
def points_wins : ℕ := wins * points_per_win
def points_draws : ℕ := total_points - points_wins

-- Theorem statement
theorem points_per_draw : points_draws / draws = 1 := by
  sorry

end points_per_draw_l1742_174205


namespace find_t_l1742_174278

theorem find_t :
  ∃ (B : ℝ × ℝ) (t : ℝ), 
  B.1^2 + B.2^2 = 100 ∧ 
  B.1 - 2 * B.2 + 10 = 0 ∧ 
  B.1 > 0 ∧ B.2 > 0 ∧ 
  t = 20 ∧ 
  (∃ m : ℝ, 
    m = -2 ∧ 
    B.2 = m * B.1 + (8 + 2 * B.1 - m * B.1)) := 
by
  sorry

end find_t_l1742_174278
