import Mathlib

namespace emerson_rowed_last_part_l1540_154069

-- Define the given conditions
def emerson_initial_distance: ℝ := 6
def emerson_continued_distance: ℝ := 15
def total_trip_distance: ℝ := 39

-- Define the distance Emerson covered before the last part
def distance_before_last_part := emerson_initial_distance + emerson_continued_distance

-- Define the distance Emerson rowed in the last part of his trip
def distance_last_part := total_trip_distance - distance_before_last_part

-- The theorem we need to prove
theorem emerson_rowed_last_part : distance_last_part = 18 := by
  sorry

end emerson_rowed_last_part_l1540_154069


namespace paul_money_left_l1540_154023

-- Conditions
def cost_of_bread : ℕ := 2
def cost_of_butter : ℕ := 3
def cost_of_juice : ℕ := 2 * cost_of_bread
def total_money : ℕ := 15

-- Definition of total cost
def total_cost := cost_of_bread + cost_of_butter + cost_of_juice

-- Statement of the theorem
theorem paul_money_left : total_money - total_cost = 6 := by
  -- Sorry, implementation skipped
  sorry

end paul_money_left_l1540_154023


namespace angles_equal_l1540_154041

variables {A B C M W L T : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace M] [MetricSpace W] [MetricSpace L] [MetricSpace T]

-- A, B, C are points of the triangle ABC with incircle k.
-- Line_segment AC is longer than line segment BC.
-- M is the intersection of median from C.
-- W is the intersection of angle bisector from C.
-- L is the intersection of altitude from C.
-- T is the point where the tangent from M to the incircle k, different from AB, touches k.
def triangle_ABC (A B C : Type*) : Prop := sorry
def incircle_k (A B C : Type*) (k : Type*) : Prop := sorry
def longer_AC (A B C : Type*) : Prop := sorry
def intersection_median_C (M C : Type*) : Prop := sorry
def intersection_angle_bisector_C (W C : Type*) : Prop := sorry
def intersection_altitude_C (L C : Type*) : Prop := sorry
def tangent_through_M (M T k : Type*) : Prop := sorry
def touches_k (T k : Type*) : Prop := sorry
def angle_eq (M T W L : Type*) : Prop := sorry

theorem angles_equal (A B C M W L T k : Type*)
  (h_triangle : triangle_ABC A B C)
  (h_incircle : incircle_k A B C k)
  (h_longer_AC : longer_AC A B C)
  (h_inter_median : intersection_median_C M C)
  (h_inter_bisector : intersection_angle_bisector_C W C)
  (h_inter_altitude : intersection_altitude_C L C)
  (h_tangent : tangent_through_M M T k)
  (h_touches : touches_k T k) :
  angle_eq M T W L := 
sorry


end angles_equal_l1540_154041


namespace simplify_and_evaluate_expression_l1540_154062

theorem simplify_and_evaluate_expression :
  ∀ x : ℤ, -1 ≤ x ∧ x ≤ 2 →
  (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2) →
  ( ( (x^2 - 1) / (x^2 - 2*x + 1) + ((x^2 - 2*x) / (x - 2)) / x ) = 1 ) :=
by
  intros x hx_constraints x_ne_criteria
  sorry

end simplify_and_evaluate_expression_l1540_154062


namespace equivalent_proof_problem_l1540_154075

variable {x : ℝ}

theorem equivalent_proof_problem (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5 * x^8 + 2 * x^6 = 1944 * Real.sqrt 7 * x - 2494 :=
sorry

end equivalent_proof_problem_l1540_154075


namespace intersection_S_T_l1540_154017

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l1540_154017


namespace simplify_expression_evaluate_expression_l1540_154094

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (a - 3 * a / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1)) = a / (a - 2) :=
by sorry

theorem evaluate_expression :
  (-2 - 3 * (-2) / (-2 + 1)) / (((-2)^2 - 4 * (-2) + 4) / (-2 + 1)) = 1 / 2 :=
by sorry

end simplify_expression_evaluate_expression_l1540_154094


namespace max_product_300_l1540_154050

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l1540_154050


namespace tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l1540_154034

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def tangent_line_p (x y : ℝ) : Prop :=
  2 * x - sqrt 5 * y - 9 = 0

noncomputable def line_q1 (x y : ℝ) : Prop :=
  x = 3

noncomputable def line_q2 (x y : ℝ) : Prop :=
  8 * x - 15 * y + 51 = 0

theorem tangent_line_through_P :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (2, -sqrt 5) →
    tangent_line_p x y := 
sorry

theorem tangent_line_through_Q1 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q1 x y := 
sorry

theorem tangent_line_through_Q2 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q2 x y := 
sorry

end tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l1540_154034


namespace find_other_number_l1540_154042

-- Define LCM and HCF conditions
def lcm_a_b := 2310
def hcf_a_b := 83
def number_a := 210

-- Define the problem to find the other number
def number_b : ℕ :=
  lcm_a_b * hcf_a_b / number_a

-- Statement: Prove that the other number is 913
theorem find_other_number : number_b = 913 := by
  -- Placeholder for proof
  sorry

end find_other_number_l1540_154042


namespace determine_min_k_l1540_154056

open Nat

theorem determine_min_k (n : ℕ) (h : n ≥ 3) 
  (a : Fin n → ℕ) (b : Fin (choose n 2) → ℕ) : 
  ∃ k, k = (n - 1) * (n - 2) / 2 + 1 := 
sorry

end determine_min_k_l1540_154056


namespace polynomial_expansion_l1540_154032

noncomputable def poly1 (z : ℝ) : ℝ := 3 * z ^ 3 + 2 * z ^ 2 - 4 * z + 1
noncomputable def poly2 (z : ℝ) : ℝ := 2 * z ^ 4 - 3 * z ^ 2 + z - 5
noncomputable def expanded_poly (z : ℝ) : ℝ := 6 * z ^ 7 + 4 * z ^ 6 - 4 * z ^ 5 - 9 * z ^ 3 + 7 * z ^ 2 + z - 5

theorem polynomial_expansion (z : ℝ) : poly1 z * poly2 z = expanded_poly z := by
  sorry

end polynomial_expansion_l1540_154032


namespace deborah_oranges_zero_l1540_154080

-- Definitions for given conditions.
def initial_oranges : Float := 55.0
def oranges_added_by_susan : Float := 35.0
def total_oranges_after : Float := 90.0

-- Defining Deborah's oranges in her bag.
def oranges_in_bag : Float := total_oranges_after - (initial_oranges + oranges_added_by_susan)

-- The theorem to be proved.
theorem deborah_oranges_zero : oranges_in_bag = 0 := by
  -- Placeholder for the proof.
  sorry

end deborah_oranges_zero_l1540_154080


namespace find_ending_number_of_range_l1540_154008

theorem find_ending_number_of_range :
  ∃ n : ℕ, (∀ avg_200_400 avg_100_n : ℕ,
    avg_200_400 = (200 + 400) / 2 ∧
    avg_100_n = (100 + n) / 2 ∧
    avg_100_n + 150 = avg_200_400) ∧
    n = 200 :=
sorry

end find_ending_number_of_range_l1540_154008


namespace part1_minimum_value_part2_zeros_inequality_l1540_154098

noncomputable def f (x a : ℝ) := x * Real.exp x - a * (Real.log x + x)

theorem part1_minimum_value (a : ℝ) :
  (∀ x > 0, f x a > 0) ∨ (∃ x > 0, f x a = a - a * Real.log a) :=
sorry

theorem part2_zeros_inequality (a x₁ x₂ : ℝ) (hx₁ : f x₁ a = 0) (hx₂ : f x₂ a = 0) :
  Real.exp (x₁ + x₂ - 2) > 1 / (x₁ * x₂) :=
sorry

end part1_minimum_value_part2_zeros_inequality_l1540_154098


namespace C_can_complete_work_in_100_days_l1540_154040

-- Definitions for conditions
def A_work_rate : ℚ := 1 / 20
def B_work_rate : ℚ := 1 / 15
def work_done_by_A_and_B : ℚ := 6 * (1 / 20 + 1 / 15)
def remaining_work : ℚ := 1 - work_done_by_A_and_B
def work_done_by_A_in_5_days : ℚ := 5 * (1 / 20)
def work_done_by_C_in_5_days : ℚ := remaining_work - work_done_by_A_in_5_days
def C_work_rate_in_5_days : ℚ := work_done_by_C_in_5_days / 5

-- Statement to prove
theorem C_can_complete_work_in_100_days : 
  work_done_by_C_in_5_days ≠ 0 → 1 / C_work_rate_in_5_days = 100 :=
by
  -- proof of the theorem
  sorry

end C_can_complete_work_in_100_days_l1540_154040


namespace dave_total_rides_l1540_154089

theorem dave_total_rides (rides_first_day rides_second_day : ℕ) (h1 : rides_first_day = 4) (h2 : rides_second_day = 3) :
  rides_first_day + rides_second_day = 7 :=
by
  sorry

end dave_total_rides_l1540_154089


namespace sum_of_four_squares_eq_20_l1540_154091

variable (x y : ℕ)

-- Conditions based on the provided problem
def condition1 := 2 * x + 2 * y = 16
def condition2 := 2 * x + 3 * y = 19

-- Theorem to be proven
theorem sum_of_four_squares_eq_20 (h1 : condition1 x y) (h2 : condition2 x y) : 4 * x = 20 :=
by
  sorry

end sum_of_four_squares_eq_20_l1540_154091


namespace product_of_three_numbers_l1540_154043

theorem product_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 210) (h2 : 5 * a = b - 11) (h3 : 5 * a = c + 11) : a * b * c = 168504 :=
  sorry

end product_of_three_numbers_l1540_154043


namespace sum_of_coefficients_is_256_l1540_154055

theorem sum_of_coefficients_is_256 :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  ((x : ℤ) - a)^8 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 → 
  a5 = 56 →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 256 :=
by
  intros
  sorry

end sum_of_coefficients_is_256_l1540_154055


namespace reflex_angle_at_G_correct_l1540_154003

noncomputable def reflex_angle_at_G
    (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80)
    : ℝ :=
  360 - (180 - (180 - angle_BAG) - (180 - angle_GEL))

theorem reflex_angle_at_G_correct :
    (∀ (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80),
    reflex_angle_at_G B A E L G on_line off_line angle_BAG angle_GEL h1 h2 = 340) := sorry

end reflex_angle_at_G_correct_l1540_154003


namespace dans_age_l1540_154085

variable {x : ℤ}

theorem dans_age (h : x + 20 = 7 * (x - 4)) : x = 8 := by
  sorry

end dans_age_l1540_154085


namespace yearly_production_target_l1540_154036

-- Definitions for the conditions
def p_current : ℕ := 100
def p_add : ℕ := 50

-- The theorem to be proven
theorem yearly_production_target : (p_current + p_add) * 12 = 1800 := by
  sorry  -- Proof is omitted

end yearly_production_target_l1540_154036


namespace smallest_n_congruent_l1540_154037

theorem smallest_n_congruent (n : ℕ) (h : 635 * n ≡ 1251 * n [MOD 30]) : n = 15 :=
sorry

end smallest_n_congruent_l1540_154037


namespace total_cost_of_stickers_l1540_154067

-- Definitions based on given conditions
def initial_funds_per_person := 9
def cost_of_deck_of_cards := 10
def Dora_packs_of_stickers := 2

-- Calculate the total amount of money collectively after buying the deck of cards
def remaining_funds := 2 * initial_funds_per_person - cost_of_deck_of_cards

-- Calculate the total packs of stickers if split evenly
def total_packs_of_stickers := 2 * Dora_packs_of_stickers

-- Prove the total cost of the boxes of stickers
theorem total_cost_of_stickers : remaining_funds = 8 := by
  -- Given initial funds per person, cost of deck of cards, and packs of stickers for Dora, the theorem should hold.
  sorry

end total_cost_of_stickers_l1540_154067


namespace find_linear_equation_l1540_154081

def is_linear_eq (eq : String) : Prop :=
  eq = "2x = 0"

theorem find_linear_equation :
  is_linear_eq "2x = 0" :=
by
  sorry

end find_linear_equation_l1540_154081


namespace tan_of_13pi_over_6_l1540_154007

theorem tan_of_13pi_over_6 : Real.tan (13 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_of_13pi_over_6_l1540_154007


namespace ratio_of_squares_l1540_154004

theorem ratio_of_squares (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + 2 * y + 3 * z = 0) :
    (x^2 + y^2 + z^2) / (x * y + y * z + z * x) = -4 := by
  sorry

end ratio_of_squares_l1540_154004


namespace range_of_a_minus_abs_b_l1540_154065

theorem range_of_a_minus_abs_b {a b : ℝ} (h1 : 1 < a ∧ a < 3) (h2 : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 :=
by
  sorry

end range_of_a_minus_abs_b_l1540_154065


namespace find_abc_l1540_154046

theorem find_abc : ∃ (a b c : ℝ), a + b + c = 1 ∧ 4 * a + 2 * b + c = 5 ∧ 9 * a + 3 * b + c = 13 ∧ a - b + c = 5 := by
  sorry

end find_abc_l1540_154046


namespace cube_split_includes_2015_l1540_154093

theorem cube_split_includes_2015 (m : ℕ) (h1 : m > 1) (h2 : ∃ (k : ℕ), 2 * k + 1 = 2015) : m = 45 :=
by
  sorry

end cube_split_includes_2015_l1540_154093


namespace derivative_sum_l1540_154070

theorem derivative_sum (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, deriv f x = f' x)
  (h : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 :=
sorry

end derivative_sum_l1540_154070


namespace expand_expression_l1540_154027

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 :=
by
  sorry

end expand_expression_l1540_154027


namespace total_cards_1750_l1540_154011

theorem total_cards_1750 (football_cards baseball_cards hockey_cards total_cards : ℕ)
  (h1 : baseball_cards = football_cards - 50)
  (h2 : football_cards = 4 * hockey_cards)
  (h3 : hockey_cards = 200)
  (h4 : total_cards = football_cards + baseball_cards + hockey_cards) :
  total_cards = 1750 :=
sorry

end total_cards_1750_l1540_154011


namespace inequality_proof_l1540_154006

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  1/a + 1/b + 1/c ≥ 2/(a + b) + 2/(b + c) + 2/(c + a) ∧ 2/(a + b) + 2/(b + c) + 2/(c + a) ≥ 9/(a + b + c) :=
sorry

end inequality_proof_l1540_154006


namespace find_b_l1540_154060

theorem find_b (a b : ℤ) (h1 : 0 ≤ a) (h2 : a < 2^2008) (h3 : 0 ≤ b) (h4 : b < 8) (h5 : 7 * (a + 2^2008 * b) % 2^2011 = 1) :
  b = 3 :=
sorry

end find_b_l1540_154060


namespace find_highest_score_l1540_154090

-- Define the conditions for the proof
section
  variable {runs_innings : ℕ → ℕ}

  -- Total runs scored in 46 innings
  def total_runs (average num_innings : ℕ) : ℕ := average * num_innings
  def total_runs_46_innings := total_runs 60 46
  def total_runs_excluding_H_L := total_runs 58 44

  -- Evaluated difference and sum of scores
  def diff_H_and_L : ℕ := 180
  def sum_H_and_L : ℕ := total_runs_46_innings - total_runs_excluding_H_L

  -- Define the proof goal
  theorem find_highest_score (H L : ℕ)
    (h1 : H - L = diff_H_and_L)
    (h2 : H + L = sum_H_and_L) :
    H = 194 :=
  by
    sorry

end

end find_highest_score_l1540_154090


namespace maximal_value_ratio_l1540_154014

theorem maximal_value_ratio (a b c h : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_altitude : h = (a * b) / c) :
  ∃ θ : ℝ, a = c * Real.cos θ ∧ b = c * Real.sin θ ∧ (1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2) ∧
  ( Real.cos θ * Real.sin θ = (1 + 2 * Real.cos θ * Real.sin θ - 1) / 2 ) → 
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end maximal_value_ratio_l1540_154014


namespace minimum_value_expression_l1540_154088

open Real

theorem minimum_value_expression (x y z : ℝ) (hxyz : x * y * z = 1 / 2) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * sqrt 6 :=
sorry

end minimum_value_expression_l1540_154088


namespace degree_g_greater_than_5_l1540_154028

-- Definitions according to the given conditions
variables {f g : Polynomial ℤ}
variables (h : Polynomial ℤ)
variables (r : Fin 81 → ℤ)

-- Condition 1: g(x) divides f(x), meaning there exists an h(x) such that f(x) = g(x) * h(x)
def divides (g f : Polynomial ℤ) := ∃ (h : Polynomial ℤ), f = g * h

-- Condition 2: f(x) - 2008 has at least 81 distinct integer roots
def has_81_distinct_roots (f : Polynomial ℤ) (roots : Fin 81 → ℤ) : Prop :=
  ∀ i : Fin 81, f.eval (roots i) = 2008 ∧ Function.Injective roots

-- The theorem to prove
theorem degree_g_greater_than_5 (nonconst_f : f.degree > 0) (nonconst_g : g.degree > 0) 
  (g_div_f : divides g f) (f_has_roots : has_81_distinct_roots (f - Polynomial.C 2008) r) :
  g.degree > 5 :=
sorry

end degree_g_greater_than_5_l1540_154028


namespace minimum_daily_production_to_avoid_losses_l1540_154025

theorem minimum_daily_production_to_avoid_losses (x : ℕ) :
  (∀ x, (10 * x) ≥ (5 * x + 4000)) → (x ≥ 800) :=
sorry

end minimum_daily_production_to_avoid_losses_l1540_154025


namespace min_value_of_function_l1540_154077

theorem min_value_of_function :
  ∀ x : ℝ, x > -1 → (y : ℝ) = (x^2 + 7*x + 10) / (x + 1) → y ≥ 9 :=
by
  intros x hx h
  sorry

end min_value_of_function_l1540_154077


namespace remainder_problem_l1540_154083

theorem remainder_problem
  (x : ℕ) (hx : x > 0) (h : 100 % x = 4) : 196 % x = 4 :=
by
  sorry

end remainder_problem_l1540_154083


namespace find_angle_C_find_max_perimeter_l1540_154078

-- Define the first part of the problem
theorem find_angle_C 
  (a b c A B C : ℝ) (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  C = (2 * Real.pi) / 3 :=
sorry

-- Define the second part of the problem
theorem find_max_perimeter 
  (a b A B : ℝ)
  (C : ℝ := (2 * Real.pi) / 3)
  (c : ℝ := Real.sqrt 3)
  (h1 : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) :
  (2 * Real.sqrt 3 < a + b + c) ∧ (a + b + c <= 2 + Real.sqrt 3) :=
sorry

end find_angle_C_find_max_perimeter_l1540_154078


namespace largest_angle_90_degrees_l1540_154009

def triangle_altitudes (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ 
  (9 * a = 12 * b) ∧ (9 * a = 18 * c)

theorem largest_angle_90_degrees (a b c : ℝ) 
  (h : triangle_altitudes a b c) : 
  exists (A B C : ℝ) (hApos : A > 0) (hBpos : B > 0) (hCpos : C > 0),
    (A^2 = B^2 + C^2) ∧ (B * C / 2 = 9 * a / 2 ∨ 
                         B * A / 2 = 12 * b / 2 ∨ 
                         C * A / 2 = 18 * c / 2) :=
sorry

end largest_angle_90_degrees_l1540_154009


namespace triangle_side_length_l1540_154031

theorem triangle_side_length (x : ℝ) (h1 : 6 < x) (h2 : x < 14) : x = 11 :=
by
  sorry

end triangle_side_length_l1540_154031


namespace ticTacToe_CarlWins_l1540_154072

def ticTacToeBoard := Fin 3 × Fin 3

noncomputable def countConfigurations : Nat := sorry

theorem ticTacToe_CarlWins :
  countConfigurations = 148 :=
sorry

end ticTacToe_CarlWins_l1540_154072


namespace lucy_fish_count_l1540_154059

theorem lucy_fish_count (initial_fish : ℕ) (additional_fish : ℕ) (final_fish : ℕ) : 
  initial_fish = 212 ∧ additional_fish = 68 → final_fish = 280 :=
by
  sorry

end lucy_fish_count_l1540_154059


namespace average_daily_production_correct_l1540_154049

noncomputable def average_daily_production : ℝ :=
  let jan_production := 3000
  let monthly_increase := 100
  let total_days := 365
  let total_production := jan_production + (11 * jan_production + (100 * (1 + 11))/2)
  (total_production / total_days : ℝ)

theorem average_daily_production_correct :
  average_daily_production = 121.1 :=
sorry

end average_daily_production_correct_l1540_154049


namespace polynomial_divisibility_l1540_154010

def P (a : ℤ) (x : ℤ) : ℤ := x^1000 + a*x^2 + 9

theorem polynomial_divisibility (a : ℤ) : (P a (-1) = 0) ↔ (a = -10) := by
  sorry

end polynomial_divisibility_l1540_154010


namespace inequality_problem_l1540_154054

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.logb 2 (1 / 3)
noncomputable def c : ℝ := Real.logb (1 / 2) (1 / 3)

theorem inequality_problem :
  c > a ∧ a > b := by
  sorry

end inequality_problem_l1540_154054


namespace find_t_l1540_154071

open Complex Real

theorem find_t (a b : ℂ) (t : ℝ) (h₁ : abs a = 3) (h₂ : abs b = 5) (h₃ : a * b = t - 3 * I) :
  t = 6 * Real.sqrt 6 := by
  sorry

end find_t_l1540_154071


namespace problem_solution_l1540_154095

noncomputable def negThreePower25 : Real := (-3) ^ 25
noncomputable def twoPowerExpression : Real := 2 ^ (4^2 + 5^2 - 7^2)
noncomputable def threeCubed : Real := 3^3

theorem problem_solution :
  negThreePower25 + twoPowerExpression + threeCubed = -3^25 + 27 + (1 / 256) :=
by
  -- proof omitted
  sorry

end problem_solution_l1540_154095


namespace sequence_a_general_term_sequence_b_sum_of_first_n_terms_l1540_154022

variable {n : ℕ}

def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℕ := 3^(n-1) + 2 * n

def T (n : ℕ) : ℕ := (3^n - 1) / 2 + n^2 + n

theorem sequence_a_general_term :
  (∀ n, a n = 2 * n) :=
by
  intro n
  sorry

theorem sequence_b_sum_of_first_n_terms :
  (∀ n, T n = (3^n - 1) / 2 + n^2 + n) :=
by
  intro n
  sorry

end sequence_a_general_term_sequence_b_sum_of_first_n_terms_l1540_154022


namespace turtles_remaining_l1540_154015

-- Define the initial number of turtles
def initial_turtles : ℕ := 9

-- Define the number of turtles that climbed onto the log
def climbed_turtles : ℕ := 3 * initial_turtles - 2

-- Define the total number of turtles on the log before any jump off
def total_turtles_before_jumping : ℕ := initial_turtles + climbed_turtles

-- Define the number of turtles remaining after half jump off
def remaining_turtles : ℕ := total_turtles_before_jumping / 2

theorem turtles_remaining : remaining_turtles = 17 :=
  by
  -- Placeholder for the proof
  sorry

end turtles_remaining_l1540_154015


namespace minimum_socks_to_guarantee_20_pairs_l1540_154092

-- Definitions and conditions
def red_socks := 120
def green_socks := 100
def blue_socks := 80
def black_socks := 50
def number_of_pairs := 20

-- Statement
theorem minimum_socks_to_guarantee_20_pairs 
  (red_socks green_socks blue_socks black_socks number_of_pairs: ℕ) 
  (h1: red_socks = 120) 
  (h2: green_socks = 100) 
  (h3: blue_socks = 80) 
  (h4: black_socks = 50) 
  (h5: number_of_pairs = 20) : 
  ∃ min_socks, min_socks = 43 := 
by 
  sorry

end minimum_socks_to_guarantee_20_pairs_l1540_154092


namespace starfish_arms_l1540_154001

variable (x : ℕ)

theorem starfish_arms :
  (7 * x + 14 = 49) → (x = 5) := by
  sorry

end starfish_arms_l1540_154001


namespace total_cost_meal_l1540_154048

-- Define the initial conditions
variables (x : ℝ) -- x represents the total cost of the meal

-- Initial number of friends
def initial_friends : ℝ := 4

-- New number of friends after additional friends join
def new_friends : ℝ := 7

-- The decrease in cost per friend
def cost_decrease : ℝ := 15

-- Lean statement to assert our proof
theorem total_cost_meal : x / initial_friends - x / new_friends = cost_decrease → x = 140 :=
by
  sorry

end total_cost_meal_l1540_154048


namespace math_equivalent_proof_l1540_154035

-- Define the probabilities given the conditions
def P_A1 := 3 / 4
def P_A2 := 2 / 3
def P_A3 := 1 / 2
def P_B1 := 3 / 5
def P_B2 := 2 / 5

-- Define events
def P_C : ℝ := (P_A1 * P_B1 * (1 - P_A2)) + (P_A1 * P_B1 * P_A2 * P_B2 * (1 - P_A3))

-- Probability distribution of X
def P_X_0 : ℝ := (1 - P_A1) + P_C
def P_X_600 : ℝ := P_A1 * (1 - P_B1)
def P_X_1500 : ℝ := P_A1 * P_B1 * P_A2 * (1 - P_B2)
def P_X_3000 : ℝ := P_A1 * P_B1 * P_A2 * P_B2 * P_A3

-- Expected value of X
def E_X : ℝ := 600 * P_X_600 + 1500 * P_X_1500 + 3000 * P_X_3000

-- Statement to prove P(C) and expected value E(X)
theorem math_equivalent_proof :
  P_C = 21 / 100 ∧ 
  P_X_0 = 23 / 50 ∧
  P_X_600 = 3 / 10 ∧
  P_X_1500 = 9 / 50 ∧
  P_X_3000 = 3 / 50 ∧ 
  E_X = 630 := 
by 
  sorry

end math_equivalent_proof_l1540_154035


namespace find_N_is_20_l1540_154021

theorem find_N_is_20 : ∃ (N : ℤ), ∃ (u v : ℤ), (N + 5 = u ^ 2) ∧ (N - 11 = v ^ 2) ∧ (N = 20) :=
by
  sorry

end find_N_is_20_l1540_154021


namespace find_common_ratio_l1540_154097

variable (a : ℕ → ℝ) -- represents the geometric sequence
variable (q : ℝ) -- represents the common ratio

-- conditions given in the problem
def a_3_condition : a 3 = 4 := sorry
def a_6_condition : a 6 = 1 / 2 := sorry

-- the general form of the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * q ^ n

-- the theorem we want to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 4) (h2 : a 6 = 1 / 2) 
  (hg : geometric_sequence a q) : q = 1 / 2 :=
sorry

end find_common_ratio_l1540_154097


namespace least_milk_l1540_154084

theorem least_milk (seokjin jungkook yoongi : ℚ) (h_seokjin : seokjin = 11 / 10)
  (h_jungkook : jungkook = 1.3) (h_yoongi : yoongi = 7 / 6) :
  seokjin < jungkook ∧ seokjin < yoongi :=
by
  sorry

end least_milk_l1540_154084


namespace baoh2_formation_l1540_154066

noncomputable def moles_of_baoh2_formed (moles_bao : ℕ) (moles_h2o : ℕ) : ℕ :=
  if moles_bao = moles_h2o then moles_bao else sorry

theorem baoh2_formation :
  moles_of_baoh2_formed 3 3 = 3 :=
by sorry

end baoh2_formation_l1540_154066


namespace div_by_7_of_sum_div_by_7_l1540_154033

theorem div_by_7_of_sum_div_by_7 (x y z : ℤ) (h : 7 ∣ x^3 + y^3 + z^3) : 7 ∣ x * y * z := by
  sorry

end div_by_7_of_sum_div_by_7_l1540_154033


namespace find_number_eq_150_l1540_154026

variable {x : ℝ}

theorem find_number_eq_150 (h : 0.60 * x - 40 = 50) : x = 150 :=
sorry

end find_number_eq_150_l1540_154026


namespace ratio_of_roots_l1540_154030

theorem ratio_of_roots (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) (h₃ : a * x₁^2 + b * x₁ + c = 0) (h₄ : a * x₂^2 + b * x₂ + c = 0) (h₅ : x₁ = 4 * x₂) : (b^2) / (a * c) = 25 / 4 :=
by
  sorry

end ratio_of_roots_l1540_154030


namespace parallel_lines_a_eq_2_l1540_154047

theorem parallel_lines_a_eq_2 {a : ℝ} :
  (∀ x y : ℝ, a * x + (a + 2) * y + 2 = 0 ∧ x + a * y - 2 = 0 → False) ↔ a = 2 :=
by
  sorry

end parallel_lines_a_eq_2_l1540_154047


namespace friend_spent_11_l1540_154079

-- Definitions of the conditions
def total_lunch_cost (you friend : ℝ) : Prop := you + friend = 19
def friend_spent_more (you friend : ℝ) : Prop := friend = you + 3

-- The theorem to prove
theorem friend_spent_11 (you friend : ℝ) 
  (h1 : total_lunch_cost you friend) 
  (h2 : friend_spent_more you friend) : 
  friend = 11 := 
by 
  sorry

end friend_spent_11_l1540_154079


namespace num_two_digit_primes_with_ones_digit_3_l1540_154082

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100
  
def ones_digit_is_3 (n : ℕ) : Prop :=
  n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_3 : 
  (∃ n1 n2 n3 n4 n5 n6 : ℕ, 
   two_digit_number n1 ∧ ones_digit_is_3 n1 ∧ is_prime n1 ∧ 
   two_digit_number n2 ∧ ones_digit_is_3 n2 ∧ is_prime n2 ∧ 
   two_digit_number n3 ∧ ones_digit_is_3 n3 ∧ is_prime n3 ∧ 
   two_digit_number n4 ∧ ones_digit_is_3 n4 ∧ is_prime n4 ∧ 
   two_digit_number n5 ∧ ones_digit_is_3 n5 ∧ is_prime n5 ∧ 
   two_digit_number n6 ∧ ones_digit_is_3 n6 ∧ is_prime n6) ∧
  (∀ n : ℕ, two_digit_number n → ones_digit_is_3 n → is_prime n → 
  n = n1 ∨ n = n2 ∨ n = n3 ∨ n = n4 ∨ n = n5 ∨ n = n6) :=
sorry

end num_two_digit_primes_with_ones_digit_3_l1540_154082


namespace original_solution_percentage_l1540_154086

theorem original_solution_percentage (P : ℝ) (h1 : 0.5 * P + 0.5 * 30 = 40) : P = 50 :=
by
  sorry

end original_solution_percentage_l1540_154086


namespace points_lie_on_line_l1540_154064

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t
  let y := (t - 1) / t
  x + y = 2 := by
  sorry

end points_lie_on_line_l1540_154064


namespace sides_of_regular_polygon_with_20_diagonals_l1540_154073

theorem sides_of_regular_polygon_with_20_diagonals :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end sides_of_regular_polygon_with_20_diagonals_l1540_154073


namespace sandy_distance_l1540_154057

theorem sandy_distance :
  ∃ d : ℝ, d = 18 * (1000 / 3600) * 99.9920006399488 := sorry

end sandy_distance_l1540_154057


namespace mixed_doubles_teams_l1540_154038

theorem mixed_doubles_teams (m n : ℕ) (h_m : m = 7) (h_n : n = 5) :
  (∃ (k : ℕ), k = 4) ∧ (m ≥ 2) ∧ (n ≥ 2) →
  ∃ (number_of_combinations : ℕ), number_of_combinations = 2 * Nat.choose 7 2 * Nat.choose 5 2 :=
by
  intros
  sorry

end mixed_doubles_teams_l1540_154038


namespace shorter_piece_length_l1540_154044

theorem shorter_piece_length (total_len : ℝ) (ratio : ℝ) (shorter_len : ℝ) (longer_len : ℝ) 
  (h1 : total_len = 49) (h2 : ratio = 2/5) (h3 : shorter_len = x) 
  (h4 : longer_len = (5/2) * x) (h5 : shorter_len + longer_len = total_len) : 
  shorter_len = 14 := 
by
  sorry

end shorter_piece_length_l1540_154044


namespace speed_of_stream_l1540_154087

variable (b s : ℝ)

-- Conditions:
def downstream_eq : Prop := 90 = (b + s) * 3
def upstream_eq : Prop := 72 = (b - s) * 3

-- Goal:
theorem speed_of_stream (h1 : downstream_eq b s) (h2 : upstream_eq b s) : s = 3 :=
by
  sorry

end speed_of_stream_l1540_154087


namespace find_valid_pair_l1540_154005

noncomputable def valid_angle (x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ x = 180 * (n - 2) / n

noncomputable def valid_pair (x k : ℕ) : Prop :=
  valid_angle x ∧ valid_angle (k * x) ∧ 1 < k ∧ k < 5

theorem find_valid_pair : valid_pair 60 2 :=
by
  sorry

end find_valid_pair_l1540_154005


namespace gym_guest_count_l1540_154016

theorem gym_guest_count (G : ℕ) (H1 : ∀ G, 0 < G → ∀ G, G * 5.7 = 285 ∧ G = 50) : G = 50 :=
by
  sorry

end gym_guest_count_l1540_154016


namespace people_in_room_l1540_154020

open Nat

theorem people_in_room (C : ℕ) (P : ℕ) (h1 : 1 / 4 * C = 6) (h2 : 3 / 4 * C = 2 / 3 * P) : P = 27 := by
  sorry

end people_in_room_l1540_154020


namespace younger_brother_height_l1540_154019

theorem younger_brother_height
  (O Y : ℕ)
  (h1 : O - Y = 12)
  (h2 : O + Y = 308) :
  Y = 148 :=
by
  sorry

end younger_brother_height_l1540_154019


namespace find_constants_l1540_154068

open Nat

variables {n : ℕ} (b c : ℤ)
def S (n : ℕ) := n^2 + b * n + c
def a (n : ℕ) := S n - S (n - 1)

theorem find_constants (a2a3_sum_eq_4 : a 2 + a 3 = 4) : 
  c = 0 ∧ b = -2 := 
by 
  sorry

end find_constants_l1540_154068


namespace find_sum_of_natural_numbers_l1540_154074

theorem find_sum_of_natural_numbers :
  ∃ (square triangle : ℕ), square^2 + 12 = triangle^2 ∧ square + triangle = 6 :=
by
  sorry

end find_sum_of_natural_numbers_l1540_154074


namespace min_cells_marked_l1540_154058

theorem min_cells_marked (grid_size : ℕ) (triomino_size : ℕ) (total_cells : ℕ) : 
  grid_size = 5 ∧ triomino_size = 3 ∧ total_cells = grid_size * grid_size → ∃ m, m = 9 :=
by
  intros h
  -- Placeholder for detailed proof steps
  sorry

end min_cells_marked_l1540_154058


namespace find_a_l1540_154051

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem find_a (a : ℝ) (h : binom_coeff 9 3 * (-a)^3 = -84) : a = 1 :=
by
  sorry

end find_a_l1540_154051


namespace range_of_a_l1540_154002

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - 2 * a) * (a * x - 1) < 0 → (x > 1 / a ∨ x < 2 * a)) → (a ≤ -Real.sqrt 2 / 2) :=
by
  intro h
  sorry

end range_of_a_l1540_154002


namespace solve_for_p_l1540_154053

theorem solve_for_p (a b c p t : ℝ) (h1 : a + b + c + p = 360) (h2 : t = 180 - c) : 
  p = 180 - a - b + t :=
by
  sorry

end solve_for_p_l1540_154053


namespace points_on_square_diagonal_l1540_154013

theorem points_on_square_diagonal (a : ℝ) (ha : a > 1) (Q : ℝ × ℝ) (hQ : Q = (a + 1, 4 * a + 1)) 
    (line : ℝ × ℝ → Prop) (hline : ∀ (x y : ℝ), line (x, y) ↔ y = a * x + 3) :
    ∃ (P R : ℝ × ℝ), line Q ∧ P = (6, 3) ∧ R = (-3, 6) :=
by
  sorry

end points_on_square_diagonal_l1540_154013


namespace fraction_of_4d_nails_l1540_154039

variables (fraction2d fraction2d_or_4d fraction4d : ℚ)

theorem fraction_of_4d_nails
  (h1 : fraction2d = 0.25)
  (h2 : fraction2d_or_4d = 0.75) :
  fraction4d = 0.50 :=
by
  sorry

end fraction_of_4d_nails_l1540_154039


namespace R_depends_on_a_d_n_l1540_154012

-- Definition of sum of an arithmetic progression
def sum_arithmetic_progression (n : ℕ) (a d : ℤ) : ℤ := 
  n * (2 * a + (n - 1) * d) / 2

-- Definitions for s1, s2, and s4
def s1 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression n a d
def s2 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (2 * n) a d
def s4 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (4 * n) a d

-- Definition of R
def R (n : ℕ) (a d : ℤ) : ℤ := s4 n a d - s2 n a d - s1 n a d

-- Theorem stating R depends on a, d, and n
theorem R_depends_on_a_d_n : 
  ∀ (n : ℕ) (a d : ℤ), ∃ (p q r : ℤ), R n a d = p * a + q * d + r := 
by
  sorry

end R_depends_on_a_d_n_l1540_154012


namespace woman_weaves_ten_day_units_l1540_154052

theorem woman_weaves_ten_day_units 
  (a₁ d : ℕ)
  (h₁ : 4 * a₁ + 6 * d = 24)
  (h₂ : a₁ + 6 * d = a₁ * (a₁ + d)) :
  a₁ + 9 * d = 21 := 
by
  sorry

end woman_weaves_ten_day_units_l1540_154052


namespace hyperbola_focus_coordinates_l1540_154024

theorem hyperbola_focus_coordinates :
  let a := 7
  let b := 11
  let h := 5
  let k := -3
  let c := Real.sqrt (a^2 + b^2)
  (∃ x y : ℝ, (x = h + c ∧ y = k) ∧ (∀ x' y', (x' = h + c ∧ y' = k) ↔ (x = x' ∧ y = y'))) :=
by
  sorry

end hyperbola_focus_coordinates_l1540_154024


namespace johns_subtraction_l1540_154096

theorem johns_subtraction : 
  ∀ (a : ℕ), 
  a = 40 → 
  (a - 1)^2 = a^2 - 79 := 
by 
  -- The proof is omitted as per instruction
  sorry

end johns_subtraction_l1540_154096


namespace Pablo_puzzle_completion_l1540_154063

theorem Pablo_puzzle_completion :
  let pieces_per_hour := 100
  let puzzles_400 := 15
  let pieces_per_puzzle_400 := 400
  let puzzles_700 := 10
  let pieces_per_puzzle_700 := 700
  let daily_work_hours := 6
  let daily_work_400_hours := 4
  let daily_work_700_hours := 2
  let break_every_hours := 2
  let break_time := 30 / 60   -- 30 minutes break in hours

  let total_pieces_400 := puzzles_400 * pieces_per_puzzle_400
  let total_pieces_700 := puzzles_700 * pieces_per_puzzle_700
  let total_pieces := total_pieces_400 + total_pieces_700

  let effective_daily_hours := daily_work_hours - (daily_work_hours / break_every_hours * break_time)
  let pieces_400_per_day := daily_work_400_hours * pieces_per_hour
  let pieces_700_per_day := (effective_daily_hours - daily_work_400_hours) * pieces_per_hour
  let total_pieces_per_day := pieces_400_per_day + pieces_700_per_day
  
  total_pieces / total_pieces_per_day = 26 := by
sorry

end Pablo_puzzle_completion_l1540_154063


namespace min_value_l1540_154018

open Real

theorem min_value (x y : ℝ) (h : x + y = 4) : x^2 + y^2 ≥ 8 := by
  sorry

end min_value_l1540_154018


namespace anya_takes_home_balloons_l1540_154045

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end anya_takes_home_balloons_l1540_154045


namespace sum_of_roots_of_quadratic_l1540_154076

theorem sum_of_roots_of_quadratic : 
  (∃ (x : ℝ), (x - 7) ^ 2 = 16 ∧ (x = 11 ∨ x = 3)) → 
  (11 + 3 = 14) :=
by 
  intro h
  sorry

end sum_of_roots_of_quadratic_l1540_154076


namespace syllogism_sequence_l1540_154061

theorem syllogism_sequence (P Q R : Prop)
  (h1 : R)
  (h2 : Q)
  (h3 : P) : 
  (Q ∧ R → P) → (R → P) ∧ (Q → (P ∧ R)) := 
by
  sorry

end syllogism_sequence_l1540_154061


namespace vector_AB_equality_l1540_154000

variable {V : Type*} [AddCommGroup V]

variables (a b : V)

theorem vector_AB_equality (BC CA : V) (hBC : BC = a) (hCA : CA = b) :
  CA - BC = b - a :=
by {
  sorry
}

end vector_AB_equality_l1540_154000


namespace ratio_of_speeds_l1540_154099

theorem ratio_of_speeds (v_A v_B : ℝ)
  (h₀ : 4 * v_A = abs (600 - 4 * v_B))
  (h₁ : 9 * v_A = abs (600 - 9 * v_B)) :
  v_A / v_B = 2 / 3 :=
sorry

end ratio_of_speeds_l1540_154099


namespace area_isosceles_right_triangle_l1540_154029

open Real

-- Define the condition that the hypotenuse of an isosceles right triangle is 4√2 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = (4 * sqrt 2)^2

-- State the theorem to prove the area of the triangle is 8 square units
theorem area_isosceles_right_triangle (a b : ℝ) (h : hypotenuse a b) : 
  a = b → 1/2 * a * b = 8 := 
by 
  intros
  -- Proof steps are not required, so we use 'sorry'
  sorry

end area_isosceles_right_triangle_l1540_154029
