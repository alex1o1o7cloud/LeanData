import Mathlib

namespace solve_floor_fractional_l430_43041

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem solve_floor_fractional (x : ℝ) :
  floor x * fractional_part x = 2019 * x ↔ x = 0 ∨ x = -1 / 2020 :=
by
  sorry

end solve_floor_fractional_l430_43041


namespace opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l430_43037

theorem opposite_number_of_2_eq_neg2 : -2 = -2 := by
  sorry

theorem abs_val_eq_2_iff_eq_2_or_neg2 (x : ℝ) : abs x = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l430_43037


namespace pooja_speed_l430_43061

theorem pooja_speed (v : ℝ) 
  (roja_speed : ℝ := 5)
  (distance : ℝ := 32)
  (time : ℝ := 4)
  (h : distance = (roja_speed + v) * time) : v = 3 :=
by
  sorry

end pooja_speed_l430_43061


namespace expression_value_l430_43018

theorem expression_value (a b c : ℕ) (h1 : 25^a * 5^(2*b) = 5^6) (h2 : 4^b / 4^c = 4) : a^2 + a * b + 3 * c = 6 := by
  sorry

end expression_value_l430_43018


namespace equation_of_parallel_line_passing_through_point_l430_43003

variable (x y : ℝ)

def is_point_on_line (x_val y_val : ℝ) (a b c : ℝ) : Prop := a * x_val + b * y_val + c = 0

def is_parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem equation_of_parallel_line_passing_through_point :
  (is_point_on_line (-1) 3 1 (-2) 7) ∧ (is_parallel (1 / 2) (1 / 2)) → (∀ x y, is_point_on_line x y 1 (-2) 7) :=
by
  sorry

end equation_of_parallel_line_passing_through_point_l430_43003


namespace family_members_before_baby_l430_43051

theorem family_members_before_baby 
  (n T : ℕ)
  (h1 : T = 17 * n)
  (h2 : (T + 3 * n + 2) / (n + 1) = 17)
  (h3 : 2 = 2) : n = 5 :=
sorry

end family_members_before_baby_l430_43051


namespace find_k_l430_43016

def f (x : ℤ) : ℤ := 3*x^2 - 2*x + 4
def g (x : ℤ) (k : ℤ) : ℤ := x^2 - k * x - 6

theorem find_k : 
  ∃ k : ℤ, f 10 - g 10 k = 10 ∧ k = -18 :=
by 
  sorry

end find_k_l430_43016


namespace final_price_set_l430_43057

variable (c ch s : ℕ)
variable (dc dtotal : ℚ)
variable (p_final : ℚ)

def coffee_price : ℕ := 6
def cheesecake_price : ℕ := 10
def sandwich_price : ℕ := 8
def coffee_discount : ℚ := 0.25 * 6
def final_discount : ℚ := 3

theorem final_price_set :
  p_final = (coffee_price - coffee_discount) + cheesecake_price + sandwich_price - final_discount :=
by
  sorry

end final_price_set_l430_43057


namespace man_l430_43073

theorem man's_speed_with_current (v c : ℝ) (h1 : c = 4.3) (h2 : v - c = 12.4) : v + c = 21 :=
by {
  sorry
}

end man_l430_43073


namespace rightmost_three_digits_of_3_pow_1987_l430_43010

theorem rightmost_three_digits_of_3_pow_1987 :
  3^1987 % 2000 = 187 :=
by sorry

end rightmost_three_digits_of_3_pow_1987_l430_43010


namespace initial_roses_l430_43005

theorem initial_roses (R : ℕ) (initial_orchids : ℕ) (current_orchids : ℕ) (current_roses : ℕ) (added_orchids : ℕ) (added_roses : ℕ) :
  initial_orchids = 84 →
  current_orchids = 91 →
  current_roses = 14 →
  added_orchids = current_orchids - initial_orchids →
  added_roses = added_orchids →
  (R + added_roses = current_roses) →
  R = 7 :=
by
  sorry

end initial_roses_l430_43005


namespace annual_decrease_rate_l430_43091

theorem annual_decrease_rate :
  ∀ (P₀ P₂ : ℕ) (t : ℕ) (rate : ℝ),
    P₀ = 20000 → P₂ = 12800 → t = 2 → P₂ = P₀ * (1 - rate) ^ t → rate = 0.2 :=
by
sorry

end annual_decrease_rate_l430_43091


namespace swim_ratio_l430_43083

theorem swim_ratio
  (V_m : ℝ) (h1 : V_m = 4.5)
  (V_s : ℝ) (h2 : V_s = 1.5)
  (V_u : ℝ) (h3 : V_u = V_m - V_s)
  (V_d : ℝ) (h4 : V_d = V_m + V_s)
  (T_u T_d : ℝ) (h5 : T_u / T_d = V_d / V_u) :
  T_u / T_d = 2 :=
by {
  sorry
}

end swim_ratio_l430_43083


namespace range_of_a_l430_43047
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 1| + |2 * x - a|

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x a ≥ (1 / 4) * a ^ 2 + 1) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_of_a_l430_43047


namespace remainder_when_divided_by_2_is_0_l430_43071

theorem remainder_when_divided_by_2_is_0 (n : ℕ)
  (h1 : ∃ r, n % 2 = r)
  (h2 : n % 7 = 5)
  (h3 : ∃ p, p = 5 ∧ (n + p) % 10 = 0) :
  n % 2 = 0 :=
by
  -- skipping the proof steps; hence adding sorry
  sorry

end remainder_when_divided_by_2_is_0_l430_43071


namespace revenue_from_full_price_tickets_l430_43040

theorem revenue_from_full_price_tickets (f h p : ℝ) (total_tickets : f + h = 160) (total_revenue : f * p + h * (p / 2) = 2400) :
  f * p = 960 :=
sorry

end revenue_from_full_price_tickets_l430_43040


namespace factor_by_resultant_is_three_l430_43076

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l430_43076


namespace find_p_l430_43098

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_configuration (p q s r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime s ∧ is_prime r ∧ 
  1 < p ∧ p < q ∧ q < s ∧ p + q + s = r

-- The theorem statement
theorem find_p (p q s r : ℕ) (h : is_valid_configuration p q s r) : p = 2 :=
by
  sorry

end find_p_l430_43098


namespace smallest_checkered_rectangle_area_l430_43093

def even (n: ℕ) : Prop := n % 2 = 0

-- Both figure types are present and areas of these types are 1 and 2 respectively
def isValidPieceComposition (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m * 1 + n * 2 = a * b

theorem smallest_checkered_rectangle_area :
  ∀ a b : ℕ, even a → even b → isValidPieceComposition a b → a * b ≥ 40 := 
by
  intro a b a_even b_even h_valid
  sorry

end smallest_checkered_rectangle_area_l430_43093


namespace factor_expression_l430_43030

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l430_43030


namespace sum_is_odd_square_expression_is_odd_l430_43033

theorem sum_is_odd_square_expression_is_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 :=
sorry

end sum_is_odd_square_expression_is_odd_l430_43033


namespace complement_intersection_l430_43085

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
sorry

end complement_intersection_l430_43085


namespace chess_tournament_games_l430_43032

def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games (n : ℕ) (h : n = 19) : games_played n = 171 :=
by
  rw [h]
  sorry

end chess_tournament_games_l430_43032


namespace largest_angle_in_pentagon_l430_43081

-- Define the angles of the pentagon
variables (C D E : ℝ) 

-- Given conditions
def is_pentagon (A B C D E : ℝ) : Prop :=
  A = 75 ∧ B = 95 ∧ D = C + 10 ∧ E = 2 * C + 20 ∧ A + B + C + D + E = 540

-- Prove that the measure of the largest angle is 190°
theorem largest_angle_in_pentagon (C D E : ℝ) : 
  is_pentagon 75 95 C D E → max 75 (max 95 (max C (max (C + 10) (2 * C + 20)))) = 190 :=
by 
  sorry

end largest_angle_in_pentagon_l430_43081


namespace compare_negatives_l430_43072

theorem compare_negatives : -3.3 < -3.14 :=
sorry

end compare_negatives_l430_43072


namespace find_a_b_l430_43025

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 7) :
  a + b = 17 / 3 :=
by
  sorry

end find_a_b_l430_43025


namespace root_at_neg_x0_l430_43001

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom x0_root : ∃ x0, f x0 = Real.exp x0

-- Theorem
theorem root_at_neg_x0 : 
  (∃ x0, (f (-x0) * Real.exp (-x0) + 1 = 0))
  → (∃ x0, (f x0 * Real.exp x0 + 1 = 0)) := 
sorry

end root_at_neg_x0_l430_43001


namespace inequality_greater_sqrt_two_l430_43006

theorem inequality_greater_sqrt_two (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
by 
  sorry

end inequality_greater_sqrt_two_l430_43006


namespace mutually_exclusive_scoring_l430_43044

-- Define conditions as types
def shoots_twice : Prop := true
def scoring_at_least_once : Prop :=
  ∃ (shot1 shot2 : Bool), shot1 || shot2
def not_scoring_both_times : Prop :=
  ∀ (shot1 shot2 : Bool), ¬(shot1 && shot2)

-- Statement of the problem: Prove the events are mutually exclusive.
theorem mutually_exclusive_scoring :
  shoots_twice → (scoring_at_least_once → not_scoring_both_times → false) :=
by
  intro h_shoots_twice
  intro h_scoring_at_least_once
  intro h_not_scoring_both_times
  sorry

end mutually_exclusive_scoring_l430_43044


namespace nonagon_diagonals_not_parallel_l430_43097

theorem nonagon_diagonals_not_parallel (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 18 ∧ 
    ∀ v₁ v₂, v₁ ≠ v₂ → (n : ℕ).choose 2 = 27 → 
    (v₂ - v₁) % n ≠ 4 ∧ (v₂ - v₁) % n ≠ n-4 :=
by
  sorry

end nonagon_diagonals_not_parallel_l430_43097


namespace part_1_part_2_l430_43080

-- Definitions for sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Proof problem 1: Prove that if M ∪ N = N, then m ≤ -2
theorem part_1 (m : ℝ) : (M ∪ N m = N m) → m ≤ -2 :=
by sorry

-- Proof problem 2: Prove that if M ∩ N = ∅, then m ≥ 3
theorem part_2 (m : ℝ) : (M ∩ N m = ∅) → m ≥ 3 :=
by sorry

end part_1_part_2_l430_43080


namespace initial_minutes_under_plan_A_l430_43079

theorem initial_minutes_under_plan_A (x : ℕ) (planA_initial : ℝ) (planA_rate : ℝ) (planB_rate : ℝ) (call_duration : ℕ) :
  planA_initial = 0.60 ∧ planA_rate = 0.06 ∧ planB_rate = 0.08 ∧ call_duration = 3 ∧
  (planA_initial + planA_rate * (call_duration - x) = planB_rate * call_duration) →
  x = 9 := 
by
  intros h
  obtain ⟨h1, h2, h3, h4, heq⟩ := h
  -- Skipping the proof
  sorry

end initial_minutes_under_plan_A_l430_43079


namespace find_y_l430_43048

theorem find_y (x y : ℝ) (h₁ : x = 51) (h₂ : x^3 * y - 2 * x^2 * y + x * y = 51000) : y = 2 / 5 := by
  sorry

end find_y_l430_43048


namespace union_A_B_complement_intersection_A_B_l430_43026

-- Define universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | -5 ≤ x ∧ x ≤ -1 }

-- Define set B
def B : Set ℝ := { x | x ≥ -4 }

-- Prove A ∪ B = [-5, +∞)
theorem union_A_B : A ∪ B = { x : ℝ | -5 ≤ x } :=
by {
  sorry
}

-- Prove complement of A ∩ B with respect to U = (-∞, -4) ∪ (-1, +∞)
theorem complement_intersection_A_B : U \ (A ∩ B) = { x : ℝ | x < -4 } ∪ { x : ℝ | x > -1 } :=
by {
  sorry
}

end union_A_B_complement_intersection_A_B_l430_43026


namespace repeating_block_length_five_sevenths_l430_43088

theorem repeating_block_length_five_sevenths : 
  ∃ n : ℕ, (∃ k : ℕ, (5 * 10^k - 5) % 7 = 0) ∧ n = 6 :=
sorry

end repeating_block_length_five_sevenths_l430_43088


namespace probability_kwoes_non_intersect_breads_l430_43054

-- Define the total number of ways to pick 3 points from 7
def total_combinations : ℕ := Nat.choose 7 3

-- Define the number of ways to pick 3 consecutive points from 7
def favorable_combinations : ℕ := 7

-- Define the probability of non-intersection
def non_intersection_probability : ℚ := favorable_combinations / total_combinations

-- Assert the final required probability
theorem probability_kwoes_non_intersect_breads :
  non_intersection_probability = 1 / 5 :=
by
  sorry

end probability_kwoes_non_intersect_breads_l430_43054


namespace johns_out_of_pocket_expense_l430_43017

-- Define the conditions given in the problem
def old_system_cost : ℤ := 250
def old_system_trade_in_value : ℤ := (80 * old_system_cost) / 100
def new_system_initial_cost : ℤ := 600
def new_system_discount : ℤ := (25 * new_system_initial_cost) / 100
def new_system_final_cost : ℤ := new_system_initial_cost - new_system_discount

-- Define the amount of money that came out of John's pocket
def out_of_pocket_expense : ℤ := new_system_final_cost - old_system_trade_in_value

-- State the theorem that needs to be proven
theorem johns_out_of_pocket_expense : out_of_pocket_expense = 250 := by
  sorry

end johns_out_of_pocket_expense_l430_43017


namespace positive_difference_of_complementary_angles_l430_43028

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end positive_difference_of_complementary_angles_l430_43028


namespace negation_of_p_l430_43031

open Real

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, exp x > log x

-- Theorem stating that the negation of p is as described
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, exp x ≤ log x :=
by
  sorry

end negation_of_p_l430_43031


namespace verify_calculations_l430_43050

theorem verify_calculations (m n x y a b : ℝ) :
  (2 * m - 3 * n) ^ 2 = 4 * m ^ 2 - 12 * m * n + 9 * n ^ 2 ∧
  (-x + y) ^ 2 = x ^ 2 - 2 * x * y + y ^ 2 ∧
  (a + 2 * b) * (a - 2 * b) = a ^ 2 - 4 * b ^ 2 ∧
  (-2 * x ^ 2 * y ^ 2) ^ 3 / (- x * y) ^ 3 ≠ -2 * x ^ 3 * y ^ 3 :=
by
  sorry

end verify_calculations_l430_43050


namespace intersection_of_function_and_inverse_l430_43039

theorem intersection_of_function_and_inverse (m : ℝ) :
  (∀ x y : ℝ, y = Real.sqrt (x - m) ↔ x = y^2 + m) →
  (∃ x : ℝ, Real.sqrt (x - m) = x) ↔ (m ≤ 1 / 4) :=
by
  sorry

end intersection_of_function_and_inverse_l430_43039


namespace surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l430_43075

-- First Problem:
theorem surface_area_cone_first_octant :
  ∃ (surface_area : ℝ), 
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ z^2 = 2*x*y) → surface_area = 16 :=
sorry

-- Second Problem:
theorem surface_area_sphere_inside_cylinder (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 ∧ x^2 + y^2 = R*x) → surface_area = 2 * R^2 * (π - 2) :=
sorry

-- Third Problem:
theorem surface_area_cylinder_inside_sphere (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 = R*x ∧ x^2 + y^2 + z^2 = R^2) → surface_area = 4 * R^2 :=
sorry

end surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l430_43075


namespace hardey_fitness_center_ratio_l430_43067

theorem hardey_fitness_center_ratio
  (f m : ℕ)
  (avg_female_weight : ℕ := 140)
  (avg_male_weight : ℕ := 180)
  (avg_overall_weight : ℕ := 160)
  (h1 : avg_female_weight * f + avg_male_weight * m = avg_overall_weight * (f + m)) :
  f = m :=
by
  sorry

end hardey_fitness_center_ratio_l430_43067


namespace union_sets_l430_43029

theorem union_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} :=
sorry

end union_sets_l430_43029


namespace surface_area_of_larger_prism_l430_43012

def volume_of_brick := 288
def number_of_bricks := 11
def target_surface_area := 1368

theorem surface_area_of_larger_prism
    (vol: ℕ := volume_of_brick)
    (num: ℕ := number_of_bricks)
    (target: ℕ := target_surface_area)
    (exists_a_b_h : ∃ (a b h : ℕ), a = 12 ∧ b = 8 ∧ h = 3)
    (large_prism_dimensions : ∃ (L W H : ℕ), L = 24 ∧ W = 12 ∧ H = 11):
    2 * (24 * 12 + 24 * 11 + 12 * 11) = target :=
by
  sorry

end surface_area_of_larger_prism_l430_43012


namespace find_theta_l430_43090

open Real

theorem find_theta (theta : ℝ) : sin theta = -1/3 ∧ -π < theta ∧ theta < -π / 2 ↔ theta = -π - arcsin (-1 / 3) :=
by
  sorry

end find_theta_l430_43090


namespace two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l430_43024

-- Define the number of boys and girls
def boys : ℕ := 2
def girls : ℕ := 3
def total_people : ℕ := boys + girls

-- Define assumptions about arrangements
def arrangements_in_two_rows : ℕ := sorry
def arrangements_with_person_A_not_head_tail : ℕ := sorry
def arrangements_with_girls_together : ℕ := sorry
def arrangements_with_boys_not_adjacent : ℕ := sorry

-- State the mathematical equivalence proof problems
theorem two_rows_arrangement : arrangements_in_two_rows = 60 := 
  sorry

theorem person_A_not_head_tail_arrangement : arrangements_with_person_A_not_head_tail = 72 := 
  sorry

theorem girls_together_arrangement : arrangements_with_girls_together = 36 := 
  sorry

theorem boys_not_adjacent_arrangement : arrangements_with_boys_not_adjacent = 72 := 
  sorry

end two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l430_43024


namespace power_function_increasing_m_eq_2_l430_43004

theorem power_function_increasing_m_eq_2 (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 :=
by
  sorry

end power_function_increasing_m_eq_2_l430_43004


namespace otimes_computation_l430_43082

-- Definition of ⊗ given m
def otimes (a b m : ℕ) : ℚ := (m * a + b) / (2 * a * b)

-- The main theorem we need to prove
theorem otimes_computation (m : ℕ) (h : otimes 1 4 m = otimes 2 3 m) :
  otimes 3 4 6 = 11 / 12 :=
sorry

end otimes_computation_l430_43082


namespace arithmetic_square_root_of_nine_l430_43038

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end arithmetic_square_root_of_nine_l430_43038


namespace ratio_boys_to_girls_l430_43066

theorem ratio_boys_to_girls (g b : ℕ) (h1 : g + b = 30) (h2 : b = g + 3) : 
  (b : ℚ) / g = 16 / 13 := 
by 
  sorry

end ratio_boys_to_girls_l430_43066


namespace paint_cans_used_l430_43063

theorem paint_cans_used (init_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (final_rooms : ℕ) :
  init_rooms = 50 → lost_cans = 5 → remaining_rooms = 40 → final_rooms = 40 → 
  remaining_rooms / (lost_cans / (init_rooms - remaining_rooms)) = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end paint_cans_used_l430_43063


namespace raft_time_l430_43002

-- Defining the conditions
def distance_between_villages := 1 -- unit: ed (arbitrary unit)

def steamboat_time := 1 -- unit: hours
def motorboat_time := 3 / 4 -- unit: hours
def motorboat_speed_ratio := 2 -- Motorboat speed is twice steamboat speed in still water

-- Speed equations with the current
def steamboat_speed_with_current (v_s v_c : ℝ) := v_s + v_c = 1 -- unit: ed/hr
def motorboat_speed_with_current (v_s v_c : ℝ) := 2 * v_s + v_c = 4 / 3 -- unit: ed/hr

-- Goal: Prove the time it takes for the raft to travel the same distance downstream
theorem raft_time : ∃ t : ℝ, t = 90 :=
by
  -- Definitions
  let v_s := 1 / 3 -- Speed of the steamboat in still water (derived)
  let v_c := 2 / 3 -- Speed of the current (derived)
  let raft_speed := v_c -- Raft speed equals the speed of the current
  
  -- Calculate the time for the raft to travel the distance
  let raft_time := distance_between_villages / raft_speed
  
  -- Convert time to minutes
  let raft_time_minutes := raft_time * 60
  
  -- Prove the raft time is 90 minutes
  existsi raft_time_minutes
  exact sorry

end raft_time_l430_43002


namespace find_z_value_l430_43068

theorem find_z_value (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (h1 : x = 2 + 1 / z)
  (h2 : z = 3 + 1 / x) : 
  z = (3 + Real.sqrt 15) / 2 :=
by
  sorry

end find_z_value_l430_43068


namespace total_animals_l430_43045

variable (rats chihuahuas : ℕ)
variable (h1 : rats = 60)
variable (h2 : rats = 6 * chihuahuas)

theorem total_animals (rats : ℕ) (chihuahuas : ℕ) (h1 : rats = 60) (h2 : rats = 6 * chihuahuas) : rats + chihuahuas = 70 := by
  sorry

end total_animals_l430_43045


namespace vicente_total_spent_l430_43034

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l430_43034


namespace length_of_one_string_l430_43008

theorem length_of_one_string (total_length : ℕ) (num_strings : ℕ) (h_total_length : total_length = 98) (h_num_strings : num_strings = 7) : total_length / num_strings = 14 := by
  sorry

end length_of_one_string_l430_43008


namespace roots_sum_and_product_l430_43053

theorem roots_sum_and_product (k p : ℝ) (hk : (k / 3) = 9) (hp : (p / 3) = 10) : k + p = 57 := by
  sorry

end roots_sum_and_product_l430_43053


namespace snail_max_distance_300_meters_l430_43056
-- Import required library

-- Define the problem statement
theorem snail_max_distance_300_meters 
  (n : ℕ) (left_turns : ℕ) (right_turns : ℕ) 
  (total_distance : ℕ)
  (h1 : n = 300)
  (h2 : left_turns = 99)
  (h3 : right_turns = 200)
  (h4 : total_distance = n) : 
  ∃ d : ℝ, d = 100 * Real.sqrt 2 :=
by
  sorry

end snail_max_distance_300_meters_l430_43056


namespace lower_limit_of_arun_weight_l430_43099

-- Given conditions for Arun's weight
variables (W : ℝ)
variables (avg_val : ℝ)

-- Define the conditions
def arun_weight_condition_1 := W < 72
def arun_weight_condition_2 := 60 < W ∧ W < 70
def arun_weight_condition_3 := W ≤ 67
def arun_weight_avg := avg_val = 66

-- The math proof problem statement
theorem lower_limit_of_arun_weight 
  (h1: arun_weight_condition_1 W) 
  (h2: arun_weight_condition_2 W) 
  (h3: arun_weight_condition_3 W) 
  (h4: arun_weight_avg avg_val) :
  ∃ (lower_limit : ℝ), lower_limit = 65 :=
sorry

end lower_limit_of_arun_weight_l430_43099


namespace find_monday_temperature_l430_43000

theorem find_monday_temperature
  (M T W Th F : ℤ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 35) :
  M = 43 :=
by
  sorry

end find_monday_temperature_l430_43000


namespace greater_number_l430_43046

theorem greater_number (x y : ℕ) (h1 : x + y = 22) (h2 : x - y = 4) : x = 13 := 
by sorry

end greater_number_l430_43046


namespace inequality_problem_l430_43020

theorem inequality_problem {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = a + b + c) :
  a^2 + b^2 + c^2 + 2 * a * b * c ≥ 5 :=
sorry

end inequality_problem_l430_43020


namespace range_of_a_l430_43060

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3 * x) - (3 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := (3 / (2 + 3 * x)) - 3 * x
noncomputable def valid_range (a : ℝ) : Prop := 
∀ x : ℝ, (1 / 6) ≤ x ∧ x ≤ (1 / 3) → |a - Real.log x| + Real.log (f' x + 3 * x) > 0

theorem range_of_a : { a : ℝ | valid_range a } = { a : ℝ | a ≠ Real.log (1 / 3) } := 
sorry

end range_of_a_l430_43060


namespace trajectory_no_intersection_distance_AB_l430_43014

variable (M : Type) [MetricSpace M]

-- Point M on the plane
variable (M : ℝ × ℝ)

-- Given conditions
def condition1 (M : ℝ × ℝ) : Prop := 
  (Real.sqrt ((M.1 - 8)^2 + M.2^2) = 2 * Real.sqrt ((M.1 - 2)^2 + M.2^2))

-- 1. Proving the trajectory C of M
theorem trajectory (M : ℝ × ℝ) (h : condition1 M) : M.1^2 + M.2^2 = 16 :=
by
  sorry

-- 2. Range of values for k such that y = kx - 5 does not intersect trajectory C
theorem no_intersection (k : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 16 → y ≠ k * x - 5) ↔ (-3 / 4 < k ∧ k < 3 / 4) :=
by
  sorry

-- 3. Distance between intersection points A and B of given circles
def intersection_condition (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16) ∧ (x^2 + y^2 - 8 * x - 8 * y + 16 = 0)

theorem distance_AB (A B : ℝ × ℝ) (hA : intersection_condition A.1 A.2) (hB : intersection_condition B.1 B.2) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

end trajectory_no_intersection_distance_AB_l430_43014


namespace percentage_of_women_employees_l430_43015

variable (E W M : ℝ)

-- Introduce conditions
def total_employees_are_married : Prop := 0.60 * E = (1 / 3) * M + 0.6842 * W
def total_employees_count : Prop := W + M = E
def percentage_of_women : Prop := W = 0.7601 * E

-- State the theorem to prove
theorem percentage_of_women_employees :
  total_employees_are_married E W M ∧ total_employees_count E W M → percentage_of_women E W :=
by sorry

end percentage_of_women_employees_l430_43015


namespace real_solutions_to_system_l430_43035

theorem real_solutions_to_system (x y : ℝ) (h1 : x^3 + y^3 = 1) (h2 : x^4 + y^4 = 1) :
  (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end real_solutions_to_system_l430_43035


namespace three_digit_integer_equal_sum_factorials_l430_43069

open Nat

theorem three_digit_integer_equal_sum_factorials :
  ∃ (a b c : ℕ), a = 1 ∧ b = 4 ∧ c = 5 ∧ 100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial :=
by
  use 1, 4, 5
  simp
  sorry

end three_digit_integer_equal_sum_factorials_l430_43069


namespace initial_books_eq_41_l430_43023

-- Definitions and conditions
def books_sold : ℕ := 33
def books_added : ℕ := 2
def books_remaining : ℕ := 10

-- Proof problem
theorem initial_books_eq_41 (B : ℕ) (h : B - books_sold + books_added = books_remaining) : B = 41 :=
by
  sorry

end initial_books_eq_41_l430_43023


namespace mod_remainder_1287_1499_l430_43036

theorem mod_remainder_1287_1499 : (1287 * 1499) % 300 = 213 := 
by 
  sorry

end mod_remainder_1287_1499_l430_43036


namespace monthly_profit_10000_daily_profit_15000_maximize_profit_l430_43094

noncomputable def price_increase (c p: ℕ) (x: ℕ) : ℕ := c + x - p
noncomputable def sales_volume (s d: ℕ) (x: ℕ) : ℕ := s - d * x
noncomputable def monthly_profit (price cost volume: ℕ) : ℕ := (price - cost) * volume
noncomputable def monthly_profit_equation (x: ℕ) : ℕ := (40 + x - 30) * (600 - 10 * x)

theorem monthly_profit_10000 (x: ℕ) : monthly_profit_equation x = 10000 ↔ x = 10 ∨ x = 40 :=
by sorry

theorem daily_profit_15000 (x: ℕ) : ¬∃ x, monthly_profit_equation x = 15000 :=
by sorry

theorem maximize_profit (x p y: ℕ) : (∀ x, monthly_profit (40 + x) 30 (600 - 10 * x) ≤ y) ∧ y = 12250 ∧ x = 65 :=
by sorry

end monthly_profit_10000_daily_profit_15000_maximize_profit_l430_43094


namespace length_of_yellow_line_l430_43007

theorem length_of_yellow_line
  (w1 w2 w3 w4 : ℝ) (path_width : ℝ) (middle_line_dist : ℝ)
  (h1 : w1 = 40) (h2 : w2 = 10) (h3 : w3 = 20) (h4 : w4 = 30) (h5 : path_width = 5) (h6 : middle_line_dist = 2.5) :
  w1 - path_width * middle_line_dist/2 + w2 + w3 + w4 - path_width * middle_line_dist/2 = 95 :=
by sorry

end length_of_yellow_line_l430_43007


namespace average_goals_per_game_l430_43019

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l430_43019


namespace change_received_l430_43078

def cost_per_banana_cents : ℕ := 30
def cost_per_banana_dollars : ℝ := 0.30
def number_of_bananas : ℕ := 5
def total_paid_dollars : ℝ := 10.00

def total_cost (cost_per_banana_dollars : ℝ) (number_of_bananas : ℕ) : ℝ :=
  cost_per_banana_dollars * number_of_bananas

theorem change_received :
  total_paid_dollars - total_cost cost_per_banana_dollars number_of_bananas = 8.50 :=
by
  sorry

end change_received_l430_43078


namespace attention_index_proof_l430_43042

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 10 then 100 * a ^ (x / 10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15 * x
  else 0

theorem attention_index_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 5 a = 140) :
  a = 4 ∧ f 5 4 > f 35 4 ∧ (5 ≤ (x : ℝ) ∧ x ≤ 100 / 3 → f x 4 ≥ 140) :=
by
  sorry

end attention_index_proof_l430_43042


namespace pat_initial_stickers_l430_43087

def initial_stickers (s : ℕ) : ℕ := s  -- Number of stickers Pat had on the first day of the week

def stickers_earned : ℕ := 22  -- Stickers earned during the week

def stickers_end_week (s : ℕ) : ℕ := initial_stickers s + stickers_earned  -- Stickers at the end of the week

theorem pat_initial_stickers (s : ℕ) (h : stickers_end_week s = 61) : s = 39 :=
by
  sorry

end pat_initial_stickers_l430_43087


namespace relationship_m_n_k_l_l430_43077

-- Definitions based on the conditions
variables (m n k l : ℕ)

-- Condition: Number of teachers (m), Number of students (n)
-- Each teacher teaches exactly k students
-- Any pair of students has exactly l common teachers

theorem relationship_m_n_k_l (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : 0 < l)
  (hk : k * (k - 1) / 2 = k * (k - 1) / 2) (hl : n * (n - 1) / 2 = n * (n - 1) / 2) 
  (h5 : m * (k * (k - 1)) = (n * (n - 1)) * l) :
  m * k * (k - 1) = n * (n - 1) * l :=
by 
  sorry

end relationship_m_n_k_l_l430_43077


namespace total_weight_of_peppers_l430_43022

theorem total_weight_of_peppers
  (green_peppers : ℝ) 
  (red_peppers : ℝ)
  (h_green : green_peppers = 0.33)
  (h_red : red_peppers = 0.33) :
  green_peppers + red_peppers = 0.66 := 
by
  sorry

end total_weight_of_peppers_l430_43022


namespace sara_initial_quarters_l430_43055

theorem sara_initial_quarters (borrowed quarters_current : ℕ) (q_initial : ℕ) :
  quarters_current = 512 ∧ quarters_borrowed = 271 → q_initial = 783 :=
by
  sorry

end sara_initial_quarters_l430_43055


namespace polynomial_coefficients_l430_43086

noncomputable def a : ℝ := 15
noncomputable def b : ℝ := -198
noncomputable def c : ℝ := 1

theorem polynomial_coefficients :
  (∀ x₁ x₂ x₃ : ℝ, 
    (x₁ + x₂ + x₃ = 0) ∧ 
    (x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -3) ∧ 
    (x₁ * x₂ * x₃ = -1) → 
    (a = 15) ∧ 
    (b = -198) ∧ 
    (c = 1)) := 
by sorry

end polynomial_coefficients_l430_43086


namespace evaluate_g_at_3_l430_43058

def g : ℝ → ℝ := fun x => x^2 - 3 * x + 2

theorem evaluate_g_at_3 : g 3 = 2 := by
  sorry

end evaluate_g_at_3_l430_43058


namespace cannot_eat_166_candies_l430_43096

-- Define parameters for sandwiches and candies equations
def sandwiches_eq (x y z : ℕ) := x + 2 * y + 3 * z = 100
def candies_eq (x y z : ℕ) := 3 * x + 4 * y + 5 * z = 166

theorem cannot_eat_166_candies (x y z : ℕ) : ¬ (sandwiches_eq x y z ∧ candies_eq x y z) :=
by {
  -- Proof will show impossibility of (x, y, z) as nonnegative integers solution
  sorry
}

end cannot_eat_166_candies_l430_43096


namespace framing_needed_l430_43021

def orig_width : ℕ := 5
def orig_height : ℕ := 7
def border_width : ℕ := 3
def doubling_factor : ℕ := 2
def inches_per_foot : ℕ := 12

-- Define the new dimensions after doubling
def new_width := orig_width * doubling_factor
def new_height := orig_height * doubling_factor

-- Define the dimensions after adding the border
def final_width := new_width + 2 * border_width
def final_height := new_height + 2 * border_width

-- Calculate the perimeter in inches
def perimeter := 2 * (final_width + final_height)

-- Convert perimeter to feet and round up if necessary
def framing_feet := (perimeter + inches_per_foot - 1) / inches_per_foot

theorem framing_needed : framing_feet = 6 := by
  sorry

end framing_needed_l430_43021


namespace calculate_expression_correct_l430_43070

theorem calculate_expression_correct :
  ( (6 + (7 / 8) - (2 + (1 / 2))) * (1 / 4) + (3 + (23 / 24) + 1 + (2 / 3)) / 4 ) / 2.5 = 1 := 
by 
  sorry

end calculate_expression_correct_l430_43070


namespace initial_cows_l430_43089

theorem initial_cows {D C : ℕ}
  (h1 : C = 2 * D)
  (h2 : 161 = (3 * C) / 4 + D / 4) :
  C = 184 :=
by
  sorry

end initial_cows_l430_43089


namespace choosing_officers_l430_43059

noncomputable def total_ways_to_choose_officers (members : List String) (boys : ℕ) (girls : ℕ) : ℕ :=
  let total_members := boys + girls
  let president_choices := total_members
  let vice_president_choices := boys - 1 + girls - 1
  let remaining_members := total_members - 2
  president_choices * vice_president_choices * remaining_members

theorem choosing_officers (members : List String) (boys : ℕ) (girls : ℕ) :
  boys = 15 → girls = 15 → members.length = 30 → total_ways_to_choose_officers members boys girls = 11760 :=
by
  intros hboys hgirls htotal
  rw [hboys, hgirls]
  sorry

end choosing_officers_l430_43059


namespace polygon_RS_ST_sum_l430_43064

theorem polygon_RS_ST_sum
  (PQ RS ST: ℝ)
  (PQ_eq : PQ = 10)
  (QR_eq : QR = 7)
  (TU_eq : TU = 6)
  (polygon_area : PQ * QR = 70)
  (PQRSTU_area : 70 = 70) :
  RS + ST = 80 :=
by
  sorry

end polygon_RS_ST_sum_l430_43064


namespace no_solution_ineq_positive_exponents_l430_43084

theorem no_solution_ineq (m : ℝ) (h : m < 6) : ¬∃ x : ℝ, |x + 1| + |x - 5| ≤ m := 
sorry

theorem positive_exponents (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) : a^a * b^b - a^b * b^a > 0 := 
sorry

end no_solution_ineq_positive_exponents_l430_43084


namespace problem1_problem2_l430_43027

variable {A B C a b c : ℝ}

-- Problem (1)
theorem problem1 (h : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B) : b = 2 * c := 
sorry

-- Problem (2)
theorem problem2 (a_eq : a = 1) (tanA_eq : Real.tan A = 2 * Real.sqrt 2) (b_eq_c : b = 2 * c): 
  Real.sqrt (c^2 * (1 - (Real.cos (A + B)))) = 2 * Real.sqrt 2 * b :=
sorry

end problem1_problem2_l430_43027


namespace cats_needed_to_catch_100_mice_in_time_l430_43013

-- Define the context and given conditions
def cats_mice_catch_time (cats mice minutes : ℕ) : Prop :=
  cats = 5 ∧ mice = 5 ∧ minutes = 5

-- Define the goal
theorem cats_needed_to_catch_100_mice_in_time :
  cats_mice_catch_time 5 5 5 → (∃ t : ℕ, t = 500) :=
by
  intro h
  sorry

end cats_needed_to_catch_100_mice_in_time_l430_43013


namespace range_of_a_l430_43095

open Real

noncomputable def C1 (t a : ℝ) : ℝ × ℝ := (2 * t + 2 * a, -t)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 2 + 2 * sin θ)

theorem range_of_a {a : ℝ} :
  (∃ (t θ : ℝ), C1 t a = C2 θ) ↔ 2 - sqrt 5 ≤ a ∧ a ≤ 2 + sqrt 5 :=
by 
  sorry

end range_of_a_l430_43095


namespace number_of_women_bathing_suits_correct_l430_43074

def men_bathing_suits : ℕ := 14797
def total_bathing_suits : ℕ := 19766

def women_bathing_suits : ℕ :=
  total_bathing_suits - men_bathing_suits

theorem number_of_women_bathing_suits_correct :
  women_bathing_suits = 19669 := by
  -- proof goes here
  sorry

end number_of_women_bathing_suits_correct_l430_43074


namespace find_interest_rate_l430_43052

theorem find_interest_rate (P r : ℝ) 
  (h1 : 460 = P * (1 + 3 * r)) 
  (h2 : 560 = P * (1 + 8 * r)) : 
  r = 0.05 :=
by
  sorry

end find_interest_rate_l430_43052


namespace false_disjunction_implies_both_false_l430_43043

theorem false_disjunction_implies_both_false (p q : Prop) (h : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
sorry

end false_disjunction_implies_both_false_l430_43043


namespace no_square_ends_with_four_identical_digits_except_0_l430_43065

theorem no_square_ends_with_four_identical_digits_except_0 (n : ℤ) :
  ¬ (∃ k : ℕ, (1 ≤ k ∧ k < 10) ∧ (n^2 % 10000 = k * 1111)) :=
by {
  sorry
}

end no_square_ends_with_four_identical_digits_except_0_l430_43065


namespace arithmetic_sequence_sum_l430_43092

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (d a1 : ℝ)
  (h_arith: ∀ n, a n = a1 + (n - 1) * d)
  (h_condition: a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
by {
  sorry
}

end arithmetic_sequence_sum_l430_43092


namespace condition_sufficient_but_not_necessary_l430_43049

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (|x| + |y| ≤ 1 → x^2 + y^2 ≤ 1) ∧ (x^2 + y^2 ≤ 1 → ¬ (|x| + |y| ≤ 1)) :=
sorry

end condition_sufficient_but_not_necessary_l430_43049


namespace root_in_interval_k_eq_2_l430_43062

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem root_in_interval_k_eq_2
  (k : ℤ)
  (h1 : 0 < f 2)
  (h2 : Real.log 2 + 2 * 2 - 5 < 0)
  (h3 : Real.log 3 + 2 * 3 - 5 > 0) 
  (h4 : f (k : ℝ) * f (k + 1 : ℝ) < 0) :
  k = 2 := 
sorry

end root_in_interval_k_eq_2_l430_43062


namespace min_value_of_squares_l430_43011

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧ (∃ (a' b' c' : ℝ), a' = b' ∧ b' = c' ∧ a' + b' + c' = t ∧ a'^2 + b'^2 + c'^2 = t^2 / 3) := 
by
  sorry

end min_value_of_squares_l430_43011


namespace men_in_first_group_l430_43009

theorem men_in_first_group (M : ℕ) (h1 : 20 * 30 * (480 / (20 * 30)) = 480) (h2 : M * 15 * (120 / (M * 15)) = 120) :
  M = 10 :=
by sorry

end men_in_first_group_l430_43009
