import Mathlib

namespace NUMINAMATH_GPT_find_product_l290_29055

def a : ℕ := 4
def g : ℕ := 8
def d : ℕ := 10

theorem find_product (A B C D E F : ℕ) (hA : A % 2 = 0) (hB : B % 3 = 0) (hC : C % 4 = 0) 
  (hD : D % 5 = 0) (hE : E % 6 = 0) (hF : F % 7 = 0) :
  a * g * d = 320 :=
by
  sorry

end NUMINAMATH_GPT_find_product_l290_29055


namespace NUMINAMATH_GPT_find_k_l290_29032

noncomputable def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

def correct_k (k : ℝ) : Prop :=
  let slope1 := -k / 2
  let slope2 := -2
  slope1 * slope2 = -1

theorem find_k (k : ℝ) (h_perpendicular : correct_k k) : k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_l290_29032


namespace NUMINAMATH_GPT_sum_of_numbers_l290_29078

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 22 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l290_29078


namespace NUMINAMATH_GPT_difference_blue_yellow_l290_29005

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end NUMINAMATH_GPT_difference_blue_yellow_l290_29005


namespace NUMINAMATH_GPT_uma_fraction_part_l290_29093

theorem uma_fraction_part (r s t u : ℕ) 
  (hr : r = 6) 
  (hs : s = 5) 
  (ht : t = 7) 
  (hu : u = 8) 
  (shared_amount: ℕ)
  (hr_amount: shared_amount = r / 6)
  (hs_amount: shared_amount = s / 5)
  (ht_amount: shared_amount = t / 7)
  (hu_amount: shared_amount = u / 8) :
  ∃ total : ℕ, ∃ uma_total : ℕ, uma_total * 13 = 2 * total :=
sorry

end NUMINAMATH_GPT_uma_fraction_part_l290_29093


namespace NUMINAMATH_GPT_mixed_number_sum_l290_29031

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_sum_l290_29031


namespace NUMINAMATH_GPT_handshake_problem_l290_29040

theorem handshake_problem :
  let team_size := 6
  let teams := 2
  let referees := 3
  let handshakes_between_teams := team_size * team_size
  let handshakes_within_teams := teams * (team_size * (team_size - 1)) / 2
  let handshakes_with_referees := (teams * team_size) * referees
  handshakes_between_teams + handshakes_within_teams + handshakes_with_referees = 102 := by
  sorry

end NUMINAMATH_GPT_handshake_problem_l290_29040


namespace NUMINAMATH_GPT_min_dot_product_l290_29083

variable {α : Type}
variables {a b : α}

noncomputable def dot (x y : α) : ℝ := sorry

axiom condition (a b : α) : abs (3 * dot a b) ≤ 4

theorem min_dot_product : dot a b = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_dot_product_l290_29083


namespace NUMINAMATH_GPT_find_a2_b2_c2_l290_29095

theorem find_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) : 
  a^2 + b^2 + c^2 = 7 / 5 := 
sorry

end NUMINAMATH_GPT_find_a2_b2_c2_l290_29095


namespace NUMINAMATH_GPT_simplify_expr_l290_29074

noncomputable def expr : ℝ := (18 * 10^10) / (6 * 10^4) * 2

theorem simplify_expr : expr = 6 * 10^6 := sorry

end NUMINAMATH_GPT_simplify_expr_l290_29074


namespace NUMINAMATH_GPT_total_players_is_139_l290_29085

def num_kabadi := 60
def num_kho_kho := 90
def num_soccer := 40
def num_basketball := 70
def num_volleyball := 50
def num_badminton := 30

def num_k_kh := 25
def num_k_s := 15
def num_k_b := 13
def num_k_v := 20
def num_k_ba := 10
def num_kh_s := 35
def num_kh_b := 16
def num_kh_v := 30
def num_kh_ba := 12
def num_s_b := 20
def num_s_v := 18
def num_s_ba := 7
def num_b_v := 15
def num_b_ba := 8
def num_v_ba := 10

def num_k_kh_s := 5
def num_k_b_v := 4
def num_s_b_ba := 3
def num_v_ba_kh := 2

def num_all_sports := 1

noncomputable def total_players : Nat :=
  (num_kabadi + num_kho_kho + num_soccer + num_basketball + num_volleyball + num_badminton) 
  - (num_k_kh + num_k_s + num_k_b + num_k_v + num_k_ba + num_kh_s + num_kh_b + num_kh_v + num_kh_ba + num_s_b + num_s_v + num_s_ba + num_b_v + num_b_ba + num_v_ba)
  + (num_k_kh_s + num_k_b_v + num_s_b_ba + num_v_ba_kh)
  - num_all_sports

theorem total_players_is_139 : total_players = 139 := 
  by 
    sorry

end NUMINAMATH_GPT_total_players_is_139_l290_29085


namespace NUMINAMATH_GPT_proposition_B_proposition_C_l290_29028

variable (a b c d : ℝ)

-- Proposition B: If |a| > |b|, then a² > b²
theorem proposition_B (h : |a| > |b|) : a^2 > b^2 :=
sorry

-- Proposition C: If (a - b)c² > 0, then a > b
theorem proposition_C (h : (a - b) * c^2 > 0) : a > b :=
sorry

end NUMINAMATH_GPT_proposition_B_proposition_C_l290_29028


namespace NUMINAMATH_GPT_correct_total_count_l290_29099

variable (x : ℕ)

-- Define the miscalculation values
def value_of_quarter := 25
def value_of_dime := 10
def value_of_half_dollar := 50
def value_of_nickel := 5

-- Calculate the individual overestimations and underestimations
def overestimation_from_quarters := (value_of_quarter - value_of_dime) * (2 * x)
def underestimation_from_half_dollars := (value_of_half_dollar - value_of_nickel) * x

-- Calculate the net correction needed
def net_correction := overestimation_from_quarters - underestimation_from_half_dollars

theorem correct_total_count :
  net_correction x = 15 * x :=
by
  sorry

end NUMINAMATH_GPT_correct_total_count_l290_29099


namespace NUMINAMATH_GPT_segment_length_after_reflection_l290_29090

structure Point :=
(x : ℝ)
(y : ℝ)

def reflect_over_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

def distance (p1 p2 : Point) : ℝ :=
abs (p1.y - p2.y)

theorem segment_length_after_reflection :
  let C : Point := {x := -3, y := 2}
  let C' : Point := reflect_over_x_axis C
  distance C C' = 4 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_after_reflection_l290_29090


namespace NUMINAMATH_GPT_calculate_diff_of_squares_l290_29030

noncomputable def diff_of_squares (a b : ℕ) : ℕ :=
  a^2 - b^2

theorem calculate_diff_of_squares :
  diff_of_squares 601 597 = 4792 :=
by
  sorry

end NUMINAMATH_GPT_calculate_diff_of_squares_l290_29030


namespace NUMINAMATH_GPT_minimum_p_plus_q_l290_29017

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then 4 * Real.log x + 1 else 2 * x - 1

theorem minimum_p_plus_q (p q : ℝ) (hpq : p ≠ q) (hf : f p + f q = 2) :
  p + q = 3 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_GPT_minimum_p_plus_q_l290_29017


namespace NUMINAMATH_GPT_x_intercept_of_line_l290_29048

theorem x_intercept_of_line : ∃ x : ℝ, ∃ y : ℝ, 4 * x + 7 * y = 28 ∧ y = 0 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l290_29048


namespace NUMINAMATH_GPT_hardest_work_diff_l290_29007

theorem hardest_work_diff 
  (A B C D : ℕ) 
  (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x ∧ D = 4 * x)
  (h_total : A + B + C + D = 240) :
  (D - A) = 72 :=
by
  sorry

end NUMINAMATH_GPT_hardest_work_diff_l290_29007


namespace NUMINAMATH_GPT_geometric_body_view_circle_l290_29033

theorem geometric_body_view_circle (P : Type) (is_circle : P → Prop) (is_sphere : P → Prop)
  (is_cylinder : P → Prop) (is_cone : P → Prop) (is_rectangular_prism : P → Prop) :
  (∀ x, is_sphere x → is_circle x) →
  (∃ x, is_cylinder x ∧ is_circle x) →
  (∃ x, is_cone x ∧ is_circle x) →
  ¬ (∃ x, is_rectangular_prism x ∧ is_circle x) :=
by
  intros h_sphere h_cylinder h_cone h_rectangular_prism
  sorry

end NUMINAMATH_GPT_geometric_body_view_circle_l290_29033


namespace NUMINAMATH_GPT_fish_per_bowl_l290_29009

theorem fish_per_bowl (num_bowls num_fish : ℕ) (h1 : num_bowls = 261) (h2 : num_fish = 6003) :
  num_fish / num_bowls = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_fish_per_bowl_l290_29009


namespace NUMINAMATH_GPT_distance_to_y_axis_eq_reflection_across_x_axis_eq_l290_29013

-- Definitions based on the conditions provided
def point_P : ℝ × ℝ := (4, -2)

-- Statements we need to prove
theorem distance_to_y_axis_eq : (abs (point_P.1) = 4) := 
by
  sorry  -- Proof placeholder

theorem reflection_across_x_axis_eq : (point_P.1 = 4 ∧ -point_P.2 = 2) :=
by
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_distance_to_y_axis_eq_reflection_across_x_axis_eq_l290_29013


namespace NUMINAMATH_GPT_modular_expression_problem_l290_29058

theorem modular_expression_problem
  (m : ℕ)
  (hm : 0 ≤ m ∧ m < 29)
  (hmod : 4 * m % 29 = 1) :
  (5^m % 29)^4 - 3 % 29 = 13 % 29 :=
by
  sorry

end NUMINAMATH_GPT_modular_expression_problem_l290_29058


namespace NUMINAMATH_GPT_second_term_of_arithmetic_sequence_l290_29015

-- Define the statement of the problem
theorem second_term_of_arithmetic_sequence 
  (a d : ℝ) 
  (h : a + (a + 2 * d) = 10) : 
  a + d = 5 := 
by 
  sorry

end NUMINAMATH_GPT_second_term_of_arithmetic_sequence_l290_29015


namespace NUMINAMATH_GPT_kyler_wins_one_game_l290_29084

theorem kyler_wins_one_game :
  ∃ (Kyler_wins : ℕ),
    (Kyler_wins + 3 + 2 + 2 = 6 ∧
    Kyler_wins + 3 = 6 ∧
    Kyler_wins = 1) := by
  sorry

end NUMINAMATH_GPT_kyler_wins_one_game_l290_29084


namespace NUMINAMATH_GPT_sally_credit_card_balance_l290_29094

theorem sally_credit_card_balance (G P : ℝ) (X : ℝ)  
  (h1 : P = 2 * G)  
  (h2 : XP = X * P)  
  (h3 : G / 3 + XP = (5 / 12) * P) : 
  X = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sally_credit_card_balance_l290_29094


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l290_29019

theorem solve_equation_1 (x : Real) : 
  (1/3) * (x - 3)^2 = 12 → x = 9 ∨ x = -3 :=
by
  sorry

theorem solve_equation_2 (x : Real) : 
  (2 * x - 1)^2 = (1 - x)^2 → x = 0 ∨ x = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l290_29019


namespace NUMINAMATH_GPT_project_completion_time_l290_29059

theorem project_completion_time (x : ℕ) :
  (∀ (B_days : ℕ), B_days = 40 →
  (∀ (combined_work_days : ℕ), combined_work_days = 10 →
  (∀ (total_days : ℕ), total_days = 20 →
  10 * (1 / (x : ℚ) + 1 / 40) + 10 * (1 / 40) = 1))) →
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_project_completion_time_l290_29059


namespace NUMINAMATH_GPT_train_lengths_l290_29088

noncomputable def train_problem : Prop :=
  let speed_T1_mps := 54 * (5/18)
  let speed_T2_mps := 72 * (5/18)
  let L_T1 := speed_T1_mps * 20
  let L_p := (speed_T1_mps * 44) - L_T1
  let L_T2 := speed_T2_mps * 16
  (L_p = 360) ∧ (L_T1 = 300) ∧ (L_T2 = 320)

theorem train_lengths : train_problem := sorry

end NUMINAMATH_GPT_train_lengths_l290_29088


namespace NUMINAMATH_GPT_total_students_left_l290_29062

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end NUMINAMATH_GPT_total_students_left_l290_29062


namespace NUMINAMATH_GPT_equivalent_single_discount_l290_29097

theorem equivalent_single_discount (p : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount1) * (1 - discount2) * (1 - discount3) * p
  (1 - final_price / p) = 0.27325 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l290_29097


namespace NUMINAMATH_GPT_find_number_l290_29063

theorem find_number (x : ℝ) (h : ((x / 8) + 8 - 30) * 6 = 12) : x = 192 :=
sorry

end NUMINAMATH_GPT_find_number_l290_29063


namespace NUMINAMATH_GPT_initial_pokemon_cards_l290_29025

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_initial_pokemon_cards_l290_29025


namespace NUMINAMATH_GPT_gcd_of_105_1001_2436_l290_29024

noncomputable def gcd_problem : ℕ :=
  Nat.gcd (Nat.gcd 105 1001) 2436

theorem gcd_of_105_1001_2436 : gcd_problem = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_of_105_1001_2436_l290_29024


namespace NUMINAMATH_GPT_overall_average_speed_l290_29051

-- Define the conditions for Mark's travel
def time_cycling : ℝ := 1
def speed_cycling : ℝ := 20
def time_walking : ℝ := 2
def speed_walking : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ :=
  (time_cycling * speed_cycling) + (time_walking * speed_walking)

def total_time : ℝ :=
  time_cycling + time_walking

-- Define the proved statement for the average speed
theorem overall_average_speed : total_distance / total_time = 8.67 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_speed_l290_29051


namespace NUMINAMATH_GPT_difference_of_roots_of_quadratic_l290_29053

theorem difference_of_roots_of_quadratic :
  (∃ (r1 r2 : ℝ), 3 * r1 ^ 2 + 4 * r1 - 15 = 0 ∧
                  3 * r2 ^ 2 + 4 * r2 - 15 = 0 ∧
                  r1 + r2 = -4 / 3 ∧
                  r1 * r2 = -5 ∧
                  r1 - r2 = 14 / 3) :=
sorry

end NUMINAMATH_GPT_difference_of_roots_of_quadratic_l290_29053


namespace NUMINAMATH_GPT_find_chemistry_marks_l290_29004

theorem find_chemistry_marks 
    (marks_english : ℕ := 70)
    (marks_math : ℕ := 63)
    (marks_physics : ℕ := 80)
    (marks_biology : ℕ := 65)
    (average_marks : ℚ := 68.2) :
    ∃ (marks_chemistry : ℕ), 
      (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) = 5 * average_marks 
      → marks_chemistry = 63 :=
by
  sorry

end NUMINAMATH_GPT_find_chemistry_marks_l290_29004


namespace NUMINAMATH_GPT_total_revenue_correct_l290_29034

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l290_29034


namespace NUMINAMATH_GPT_correct_statement_b_l290_29068

open Set 

variables {Point Line Plane : Type}
variable (m n : Line)
variable (α : Plane)
variable (perpendicular_to_plane : Line → Plane → Prop) 
variable (parallel_to_plane : Line → Plane → Prop)
variable (is_subline_of_plane : Line → Plane → Prop)
variable (perpendicular_to_line : Line → Line → Prop)

theorem correct_statement_b (hm : perpendicular_to_plane m α) (hn : is_subline_of_plane n α) : perpendicular_to_line m n :=
sorry

end NUMINAMATH_GPT_correct_statement_b_l290_29068


namespace NUMINAMATH_GPT_intersection_M_N_l290_29080

def M : Set ℝ := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l290_29080


namespace NUMINAMATH_GPT_find_values_of_a_and_b_l290_29075

-- Define the problem
theorem find_values_of_a_and_b (a b : ℚ) (h1 : a + (a / 4) = 3) (h2 : b - 2 * a = 1) :
  a = 12 / 5 ∧ b = 29 / 5 := by
  sorry

end NUMINAMATH_GPT_find_values_of_a_and_b_l290_29075


namespace NUMINAMATH_GPT_track_length_l290_29056

theorem track_length (x : ℝ) (hb hs : ℝ) (h_opposite : hs = x / 2 - 120) (h_first_meet : hb = 120) (h_second_meet : hs + 180 = x / 2 + 60) : x = 600 := 
by
  sorry

end NUMINAMATH_GPT_track_length_l290_29056


namespace NUMINAMATH_GPT_find_a_l290_29054

-- Definition of * in terms of 2a - b^2
def custom_mul (a b : ℤ) := 2 * a - b^2

-- The proof statement
theorem find_a (a : ℤ) : custom_mul a 3 = 3 → a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l290_29054


namespace NUMINAMATH_GPT_max_f_eq_4_monotonic_increase_interval_l290_29027

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_f_eq_4 (x : ℝ) : ∀ x : ℝ, f x ≤ 4 := 
by
  sorry

theorem monotonic_increase_interval (k : ℤ) : ∀ x : ℝ, (k * Real.pi - Real.pi / 4 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) ↔ 
  (0 ≤ Real.sin (2 * x) ∧ Real.sin (2 * x) ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_max_f_eq_4_monotonic_increase_interval_l290_29027


namespace NUMINAMATH_GPT_eval_power_expression_l290_29073

theorem eval_power_expression : (3^3)^2 / 3^2 = 81 := by
  sorry -- Proof omitted as instructed

end NUMINAMATH_GPT_eval_power_expression_l290_29073


namespace NUMINAMATH_GPT_percentage_problem_l290_29037

theorem percentage_problem (P : ℕ) (n : ℕ) (h_n : n = 16)
  (h_condition : (40: ℚ) = 0.25 * n + 2) : P = 250 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l290_29037


namespace NUMINAMATH_GPT_determine_H_zero_l290_29008

theorem determine_H_zero (E F G H : ℕ) 
  (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (add_eq : 10 * E + F + 10 * G + E = 10 * H + E)
  (sub_eq : 10 * E + F - (10 * G + E) = E) : 
  H = 0 :=
sorry

end NUMINAMATH_GPT_determine_H_zero_l290_29008


namespace NUMINAMATH_GPT_clyde_picked_bushels_l290_29049

theorem clyde_picked_bushels (weight_per_bushel : ℕ) (weight_per_cob : ℕ) (cobs_picked : ℕ) :
  weight_per_bushel = 56 →
  weight_per_cob = 1 / 2 →
  cobs_picked = 224 →
  cobs_picked * weight_per_cob / weight_per_bushel = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_clyde_picked_bushels_l290_29049


namespace NUMINAMATH_GPT_gcd_2210_145_l290_29022

-- defining the constants a and b
def a : ℕ := 2210
def b : ℕ := 145

-- theorem stating that gcd(a, b) = 5
theorem gcd_2210_145 : Nat.gcd a b = 5 :=
sorry

end NUMINAMATH_GPT_gcd_2210_145_l290_29022


namespace NUMINAMATH_GPT_face_opposite_to_turquoise_is_pink_l290_29044

-- Declare the inductive type for the color of the face
inductive Color
| P -- Pink
| V -- Violet
| T -- Turquoise
| O -- Orange

open Color

-- Define the setup conditions of the problem
def cube_faces : List Color :=
  [P, P, P, V, V, T, O]

-- Define the positions of the faces for the particular folded cube configuration
-- Assuming the function cube_configuration gives the face opposite to a given face.
axiom cube_configuration : Color → Color

-- State the main theorem regarding the opposite face
theorem face_opposite_to_turquoise_is_pink : cube_configuration T = P :=
sorry

end NUMINAMATH_GPT_face_opposite_to_turquoise_is_pink_l290_29044


namespace NUMINAMATH_GPT_school_year_hours_per_week_l290_29076

-- Definitions based on the conditions of the problem
def summer_weeks : ℕ := 8
def summer_hours_per_week : ℕ := 40
def summer_earnings : ℕ := 3200

def school_year_weeks : ℕ := 24
def needed_school_year_earnings : ℕ := 6400

-- Question translated to a Lean statement
theorem school_year_hours_per_week :
  let hourly_rate := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := needed_school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks = (80 / 3) :=
by {
  -- The implementation of the proof goes here
  sorry
}

end NUMINAMATH_GPT_school_year_hours_per_week_l290_29076


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l290_29079

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l290_29079


namespace NUMINAMATH_GPT_problem_I_problem_II_l290_29000

-- Problem (I)
theorem problem_I (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) : 
  ∀ x, (f x < |x| + 1) → (0 < x ∧ x < 2) :=
by
  intro x hx
  have fx_def : f x = |2 * x - 1| := h x
  sorry

-- Problem (II)
theorem problem_II (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) :
  ∀ x y, (|x - y - 1| ≤ 1 / 3) → (|2 * y + 1| ≤ 1 / 6) → (f x ≤ 5 / 6) :=
by
  intro x y hx hy
  have fx_def : f x = |2 * x - 1| := h x
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l290_29000


namespace NUMINAMATH_GPT_ab2_plus_bc2_plus_ca2_le_27_div_8_l290_29072

theorem ab2_plus_bc2_plus_ca2_le_27_div_8 (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end NUMINAMATH_GPT_ab2_plus_bc2_plus_ca2_le_27_div_8_l290_29072


namespace NUMINAMATH_GPT_max_handshakes_l290_29001

theorem max_handshakes (n m : ℕ) (cond1 : n = 30) (cond2 : m = 5) 
                       (cond3 : ∀ (i : ℕ), i < 30 → ∀ (j : ℕ), j < 30 → i ≠ j → true)
                       (cond4 : ∀ (k : ℕ), k < 5 → ∃ (s : ℕ), s ≤ 10) : 
  ∃ (handshakes : ℕ), handshakes = 325 :=
by
  sorry

end NUMINAMATH_GPT_max_handshakes_l290_29001


namespace NUMINAMATH_GPT_michelle_initial_crayons_l290_29091

variable (m j : Nat)

axiom janet_crayons : j = 2
axiom michelle_has_after_gift : m + j = 4

theorem michelle_initial_crayons : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_michelle_initial_crayons_l290_29091


namespace NUMINAMATH_GPT_eval_expression_l290_29039

theorem eval_expression : (538 * 538) - (537 * 539) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l290_29039


namespace NUMINAMATH_GPT_trigonometric_expression_l290_29010

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 3) : 
  ((Real.cos (α - π / 2) + Real.cos (α + π)) / (2 * Real.sin α) = 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l290_29010


namespace NUMINAMATH_GPT_container_capacity_l290_29067

theorem container_capacity (C : ℝ) (h1 : C > 0) (h2 : 0.40 * C + 14 = 0.75 * C) : C = 40 := 
by 
  -- Would contain the proof here
  sorry

end NUMINAMATH_GPT_container_capacity_l290_29067


namespace NUMINAMATH_GPT_triangle_shape_statements_l290_29042

theorem triangle_shape_statements (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (h : a^2 + b^2 + c^2 = ab + bc + ca) :
  (a = b ∧ b = c ∧ a = c) :=
by
  sorry 

end NUMINAMATH_GPT_triangle_shape_statements_l290_29042


namespace NUMINAMATH_GPT_num_two_digit_numbers_with_digit_sum_10_l290_29012

theorem num_two_digit_numbers_with_digit_sum_10 : 
  ∃ n, n = 9 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 10 → ∃ m, 10 * a + b = m :=
sorry

end NUMINAMATH_GPT_num_two_digit_numbers_with_digit_sum_10_l290_29012


namespace NUMINAMATH_GPT_linear_equation_conditions_l290_29023

theorem linear_equation_conditions (m n : ℤ) :
  (∀ x y : ℝ, 4 * x^(m - n) - 5 * y^(m + n) = 6 → 
    m - n = 1 ∧ m + n = 1) →
  m = 1 ∧ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_conditions_l290_29023


namespace NUMINAMATH_GPT_find_t_from_integral_l290_29038

theorem find_t_from_integral :
  (∫ x in (1 : ℝ)..t, (-1 / x + 2 * x)) = (3 - Real.log 2) → t = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_t_from_integral_l290_29038


namespace NUMINAMATH_GPT_min_value_expression_l290_29098

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_expression : (1 + b / a) * (4 * a / b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l290_29098


namespace NUMINAMATH_GPT_total_cost_correct_l290_29041

def sandwich_cost : ℝ := 2.44
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_correct :
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 8.36 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l290_29041


namespace NUMINAMATH_GPT_oliver_total_money_l290_29096

-- Define the initial conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def cost_frisbee : ℕ := 4
def cost_puzzle : ℕ := 3
def birthday_money : ℕ := 8

-- Define the problem statement in Lean
theorem oliver_total_money : 
  (initial_amount + allowance_saved - (cost_frisbee + cost_puzzle) + birthday_money) = 15 := 
by 
  sorry

end NUMINAMATH_GPT_oliver_total_money_l290_29096


namespace NUMINAMATH_GPT_total_coin_value_l290_29077

theorem total_coin_value (total_coins : ℕ) (two_dollar_coins : ℕ) (one_dollar_value : ℕ)
  (two_dollar_value : ℕ) (h_total_coins : total_coins = 275)
  (h_two_dollar_coins : two_dollar_coins = 148)
  (h_one_dollar_value : one_dollar_value = 1)
  (h_two_dollar_value : two_dollar_value = 2) :
  total_coins - two_dollar_coins = 275 - 148
  ∧ ((total_coins - two_dollar_coins) * one_dollar_value + two_dollar_coins * two_dollar_value) = 423 :=
by
  sorry

end NUMINAMATH_GPT_total_coin_value_l290_29077


namespace NUMINAMATH_GPT_bianca_ate_candy_l290_29057

theorem bianca_ate_candy (original_candies : ℕ) (pieces_per_pile : ℕ) 
                         (number_of_piles : ℕ) 
                         (remaining_candies : ℕ) 
                         (h_original : original_candies = 78) 
                         (h_pieces_per_pile : pieces_per_pile = 8) 
                         (h_number_of_piles : number_of_piles = 6) 
                         (h_remaining : remaining_candies = pieces_per_pile * number_of_piles) :
  original_candies - remaining_candies = 30 := by
  subst_vars
  sorry

end NUMINAMATH_GPT_bianca_ate_candy_l290_29057


namespace NUMINAMATH_GPT_zero_function_l290_29050

variable (f : ℝ × ℝ × ℝ → ℝ)

theorem zero_function (h : ∀ x y z : ℝ, f (x, y, z) = 2 * f (z, x, y)) : ∀ x y z : ℝ, f (x, y, z) = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_zero_function_l290_29050


namespace NUMINAMATH_GPT_intersection_point_l290_29065

def satisfies_first_line (p : ℝ × ℝ) : Prop :=
  8 * p.1 - 5 * p.2 = 40

def satisfies_second_line (p : ℝ × ℝ) : Prop :=
  6 * p.1 + 2 * p.2 = 14

theorem intersection_point :
  satisfies_first_line (75 / 23, -64 / 23) ∧ satisfies_second_line (75 / 23, -64 / 23) :=
by 
  sorry

end NUMINAMATH_GPT_intersection_point_l290_29065


namespace NUMINAMATH_GPT_event_day_is_Sunday_l290_29036

def days_in_week := 7

def event_day := 1500

def start_day := "Friday"

def day_of_week_according_to_mod : Nat → String 
| 0 => "Friday"
| 1 => "Saturday"
| 2 => "Sunday"
| 3 => "Monday"
| 4 => "Tuesday"
| 5 => "Wednesday"
| 6 => "Thursday"
| _ => "Invalid"

theorem event_day_is_Sunday : day_of_week_according_to_mod (event_day % days_in_week) = "Sunday" :=
sorry

end NUMINAMATH_GPT_event_day_is_Sunday_l290_29036


namespace NUMINAMATH_GPT_circle_equation_condition_l290_29081

theorem circle_equation_condition (m : ℝ) : 
  (∃ h k r : ℝ, (r > 0) ∧ ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + y^2 - 2*x - 4*y + m = 0) ↔ m < 5 :=
sorry

end NUMINAMATH_GPT_circle_equation_condition_l290_29081


namespace NUMINAMATH_GPT_coefficient_x99_is_zero_l290_29014

open Polynomial

noncomputable def P (x : ℤ) : Polynomial ℤ := sorry
noncomputable def Q (x : ℤ) : Polynomial ℤ := sorry

theorem coefficient_x99_is_zero : 
    (P 0 = 1) → 
    ((P x)^2 = 1 + x + x^100 * Q x) → 
    (Polynomial.coeff ((P x + 1)^100) 99 = 0) :=
by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_coefficient_x99_is_zero_l290_29014


namespace NUMINAMATH_GPT_Razorback_shop_total_revenue_l290_29011

theorem Razorback_shop_total_revenue :
  let Tshirt_price := 62
  let Jersey_price := 99
  let Hat_price := 45
  let Keychain_price := 25
  let Tshirt_sold := 183
  let Jersey_sold := 31
  let Hat_sold := 142
  let Keychain_sold := 215
  let revenue := (Tshirt_price * Tshirt_sold) + (Jersey_price * Jersey_sold) + (Hat_price * Hat_sold) + (Keychain_price * Keychain_sold)
  revenue = 26180 :=
by
  sorry

end NUMINAMATH_GPT_Razorback_shop_total_revenue_l290_29011


namespace NUMINAMATH_GPT_complex_power_sum_eq_self_l290_29016

theorem complex_power_sum_eq_self (z : ℂ) (h : z^2 + z + 1 = 0) : z^100 + z^101 + z^102 + z^103 = z :=
sorry

end NUMINAMATH_GPT_complex_power_sum_eq_self_l290_29016


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l290_29006

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x
noncomputable def f' (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, x < 2 → f' x < 0 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l290_29006


namespace NUMINAMATH_GPT_ernie_income_ratio_l290_29061

-- Define constants and properties based on the conditions
def previous_income := 6000
def jack_income := 2 * previous_income
def combined_income := 16800

-- Lean proof statement that the ratio of Ernie's current income to his previous income is 2/3
theorem ernie_income_ratio (current_income : ℕ) (h1 : current_income + jack_income = combined_income) :
    current_income / previous_income = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ernie_income_ratio_l290_29061


namespace NUMINAMATH_GPT_p_plus_q_l290_29035

-- Define the problem conditions
def p (x : ℝ) : ℝ := 4 * (x - 2)
def q (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- Main theorem to prove the answer
theorem p_plus_q (x : ℝ) : p x + q x = x^2 + 4 * x - 12 := 
by
  sorry

end NUMINAMATH_GPT_p_plus_q_l290_29035


namespace NUMINAMATH_GPT_probability_five_digit_palindrome_divisible_by_11_l290_29082

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  n % 100 = 100*a + 10*b + c

def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem probability_five_digit_palindrome_divisible_by_11 :
  let count_palindromes := 9 * 10 * 10
  let count_divisible_by_11 := 165
  (count_divisible_by_11 : ℚ) / count_palindromes = 11 / 60 :=
by
  sorry

end NUMINAMATH_GPT_probability_five_digit_palindrome_divisible_by_11_l290_29082


namespace NUMINAMATH_GPT_largest_whole_number_l290_29021

theorem largest_whole_number (x : ℕ) (h1 : 9 * x < 150) : x ≤ 16 :=
by sorry

end NUMINAMATH_GPT_largest_whole_number_l290_29021


namespace NUMINAMATH_GPT_veronica_pits_cherries_in_2_hours_l290_29046

theorem veronica_pits_cherries_in_2_hours :
  ∀ (pounds_cherries : ℕ) (cherries_per_pound : ℕ)
    (time_first_pound : ℕ) (cherries_first_pound : ℕ)
    (time_second_pound : ℕ) (cherries_second_pound : ℕ)
    (time_third_pound : ℕ) (cherries_third_pound : ℕ)
    (minutes_per_hour : ℕ),
  pounds_cherries = 3 →
  cherries_per_pound = 80 →
  time_first_pound = 10 →
  cherries_first_pound = 20 →
  time_second_pound = 8 →
  cherries_second_pound = 20 →
  time_third_pound = 12 →
  cherries_third_pound = 20 →
  minutes_per_hour = 60 →
  ((time_first_pound / cherries_first_pound * cherries_per_pound) + 
   (time_second_pound / cherries_second_pound * cherries_per_pound) + 
   (time_third_pound / cherries_third_pound * cherries_per_pound)) / minutes_per_hour = 2 :=
by
  intros pounds_cherries cherries_per_pound
         time_first_pound cherries_first_pound
         time_second_pound cherries_second_pound
         time_third_pound cherries_third_pound
         minutes_per_hour
         pounds_eq cherries_eq
         time1_eq cherries1_eq
         time2_eq cherries2_eq
         time3_eq cherries3_eq
         mins_eq

  -- You would insert the proof here
  sorry

end NUMINAMATH_GPT_veronica_pits_cherries_in_2_hours_l290_29046


namespace NUMINAMATH_GPT_range_of_f_on_interval_l290_29047

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem range_of_f_on_interval :
  Set.Icc (-1 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x ∈ Set.Icc (0 : ℝ) (2 : ℝ), f x = y} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_on_interval_l290_29047


namespace NUMINAMATH_GPT_cos_of_pi_over_3_minus_alpha_l290_29045

theorem cos_of_pi_over_3_minus_alpha (α : Real) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) :
  Real.cos (Real.pi / 3 - α) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_pi_over_3_minus_alpha_l290_29045


namespace NUMINAMATH_GPT_min_value_expression_l290_29086

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 2) ^ 2 + (b + 2) ^ 2 = 25 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l290_29086


namespace NUMINAMATH_GPT_gold_copper_alloy_ratio_l290_29052

theorem gold_copper_alloy_ratio 
  (G C : ℝ) 
  (h_gold : G / weight_of_water = 19) 
  (h_copper : C / weight_of_water = 9)
  (weight_of_alloy : (G + C) / weight_of_water = 17) :
  G / C = 4 :=
sorry

end NUMINAMATH_GPT_gold_copper_alloy_ratio_l290_29052


namespace NUMINAMATH_GPT_total_cost_of_breakfast_l290_29026

-- Definitions based on conditions
def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def kiera_fruit_cup : ℕ := 1

-- The proof statement
theorem total_cost_of_breakfast : 
  muffin_cost * francis_muffins + 
  fruit_cup_cost * francis_fruit_cups + 
  muffin_cost * kiera_muffins + 
  fruit_cup_cost * kiera_fruit_cup = 17 := 
  by sorry

end NUMINAMATH_GPT_total_cost_of_breakfast_l290_29026


namespace NUMINAMATH_GPT_find_angle_beta_l290_29070

open Real

theorem find_angle_beta
  (α β : ℝ)
  (h1 : sin α = (sqrt 5) / 5)
  (h2 : sin (α - β) = - (sqrt 10) / 10)
  (hα_range : 0 < α ∧ α < π / 2)
  (hβ_range : 0 < β ∧ β < π / 2) :
  β = π / 4 :=
sorry

end NUMINAMATH_GPT_find_angle_beta_l290_29070


namespace NUMINAMATH_GPT_base_7_multiplication_addition_l290_29064

theorem base_7_multiplication_addition :
  (25 * 3 + 144) % 7^3 = 303 :=
by sorry

end NUMINAMATH_GPT_base_7_multiplication_addition_l290_29064


namespace NUMINAMATH_GPT_find_first_type_cookies_l290_29066

section CookiesProof

variable (x : ℕ)

-- Conditions
def box_first_type_cookies : ℕ := x
def box_second_type_cookies : ℕ := 20
def box_third_type_cookies : ℕ := 16
def boxes_first_type_sold : ℕ := 50
def boxes_second_type_sold : ℕ := 80
def boxes_third_type_sold : ℕ := 70
def total_cookies_sold : ℕ := 3320

-- Theorem to prove
theorem find_first_type_cookies 
  (h1 : 50 * x + 80 * box_second_type_cookies + 70 * box_third_type_cookies = total_cookies_sold) :
  x = 12 := by
    sorry

end CookiesProof

end NUMINAMATH_GPT_find_first_type_cookies_l290_29066


namespace NUMINAMATH_GPT_cos_double_angle_l290_29043

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l290_29043


namespace NUMINAMATH_GPT_searchlight_probability_l290_29092

theorem searchlight_probability (revolutions_per_minute : ℕ) (D : ℝ) (prob : ℝ)
  (h1 : revolutions_per_minute = 4)
  (h2 : prob = 0.6666666666666667) :
  D = (2 / 3) * (60 / revolutions_per_minute) :=
by
  -- To complete the proof, we will use the conditions given.
  sorry

end NUMINAMATH_GPT_searchlight_probability_l290_29092


namespace NUMINAMATH_GPT_christmas_trees_in_each_box_l290_29020

theorem christmas_trees_in_each_box
  (T : ℕ)
  (pieces_of_tinsel_in_each_box : ℕ := 4)
  (snow_globes_in_each_box : ℕ := 5)
  (total_boxes : ℕ := 12)
  (total_decorations : ℕ := 120)
  (decorations_per_box : ℕ := pieces_of_tinsel_in_each_box + T + snow_globes_in_each_box)
  (total_decorations_distributed : ℕ := total_boxes * decorations_per_box) :
  total_decorations_distributed = total_decorations → T = 1 := by
  sorry

end NUMINAMATH_GPT_christmas_trees_in_each_box_l290_29020


namespace NUMINAMATH_GPT_deck_card_count_l290_29060

theorem deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 :=
by
  sorry

end NUMINAMATH_GPT_deck_card_count_l290_29060


namespace NUMINAMATH_GPT_total_money_l290_29003

-- Define the problem with conditions and question transformed into proof statement
theorem total_money (A B : ℕ) (h1 : 2 * A / 3 = B / 2) (h2 : B = 484) : A + B = 847 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_total_money_l290_29003


namespace NUMINAMATH_GPT_total_sessions_l290_29002

theorem total_sessions (p1 p2 p3 p4 : ℕ) 
(h1 : p1 = 6) 
(h2 : p2 = p1 + 5) 
(h3 : p3 = 8) 
(h4 : p4 = 8) : 
p1 + p2 + p3 + p4 = 33 := 
by
  sorry

end NUMINAMATH_GPT_total_sessions_l290_29002


namespace NUMINAMATH_GPT_angles_with_same_terminal_side_pi_div_3_l290_29071

noncomputable def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * Real.pi

theorem angles_with_same_terminal_side_pi_div_3 :
  { α : ℝ | same_terminal_side α (Real.pi / 3) } =
  { α : ℝ | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 } :=
by
  sorry

end NUMINAMATH_GPT_angles_with_same_terminal_side_pi_div_3_l290_29071


namespace NUMINAMATH_GPT_perpendicular_lines_l290_29018

theorem perpendicular_lines (m : ℝ) : 
  (m = -2 → (2-m) * (-(m+3)/(2-m)) + m * (m-3) / (-(m+3)) = 0) → 
  (m = -2 ∨ m = 1) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_l290_29018


namespace NUMINAMATH_GPT_cost_of_traveling_roads_l290_29089

def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 40
def road_width : ℕ := 10
def cost_per_sqm : ℕ := 3

def area_road_parallel_length : ℕ := road_width * lawn_length
def area_road_parallel_breadth : ℕ := road_width * lawn_breadth
def area_intersection : ℕ := road_width * road_width

def total_area_roads : ℕ := area_road_parallel_length + area_road_parallel_breadth - area_intersection
def total_cost : ℕ := total_area_roads * cost_per_sqm

theorem cost_of_traveling_roads : total_cost = 3300 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_traveling_roads_l290_29089


namespace NUMINAMATH_GPT_calculate_expression_l290_29029

theorem calculate_expression : 
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l290_29029


namespace NUMINAMATH_GPT_probability_multiple_choice_and_essay_correct_l290_29069

noncomputable def probability_multiple_choice_and_essay (C : ℕ → ℕ → ℕ) : ℚ :=
    (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3)

theorem probability_multiple_choice_and_essay_correct (C : ℕ → ℕ → ℕ) :
    probability_multiple_choice_and_essay C = (C 12 1 * (C 6 1 * C 4 1 + C 6 2) + C 12 2 * C 6 1) / (C 22 3 - C 10 3) :=
by
  sorry

end NUMINAMATH_GPT_probability_multiple_choice_and_essay_correct_l290_29069


namespace NUMINAMATH_GPT_natasha_average_speed_l290_29087

theorem natasha_average_speed
  (time_up time_down : ℝ)
  (speed_up distance_up total_distance total_time average_speed : ℝ)
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 3)
  (h4 : distance_up = speed_up * time_up)
  (h5 : total_distance = distance_up + distance_up)
  (h6 : total_time = time_up + time_down)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 4 := by
  sorry

end NUMINAMATH_GPT_natasha_average_speed_l290_29087
