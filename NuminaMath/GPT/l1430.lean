import Mathlib

namespace dan_must_exceed_speed_to_arrive_before_cara_l1430_143000

noncomputable def minimum_speed_for_dan (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) : ℕ :=
  (distance / (distance / cara_speed - dan_delay)) + 1

theorem dan_must_exceed_speed_to_arrive_before_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  minimum_speed_for_dan distance cara_speed dan_delay > 36 :=
by
  sorry

end dan_must_exceed_speed_to_arrive_before_cara_l1430_143000


namespace num_positive_integers_l1430_143010

theorem num_positive_integers (N : ℕ) (h : N > 3) : (∃ (k : ℕ) (h_div : 48 % k = 0), k = N - 3) → (∃ (c : ℕ), c = 8) := sorry

end num_positive_integers_l1430_143010


namespace math_problem_l1430_143034

theorem math_problem :
  (10^2 + 6^2) / 2 = 68 :=
by
  sorry

end math_problem_l1430_143034


namespace greatest_y_value_l1430_143071

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_y_value_l1430_143071


namespace intersection_of_M_and_N_l1430_143087

def M : Set ℤ := {x : ℤ | -4 < x ∧ x < 2}
def N : Set ℤ := {x : ℤ | x^2 < 4}

theorem intersection_of_M_and_N : (M ∩ N) = { -1, 0, 1 } :=
by
  sorry

end intersection_of_M_and_N_l1430_143087


namespace find_a_l1430_143084

theorem find_a (f g : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, f x = 2 * x / 3 + 4) 
  (h₂ : ∀ x, g x = 5 - 2 * x) 
  (h₃ : f (g a) = 7) : 
  a = 1 / 4 := 
sorry

end find_a_l1430_143084


namespace age_ratio_l1430_143003

noncomputable def rahul_present_age (future_age : ℕ) (years_passed : ℕ) : ℕ := future_age - years_passed

theorem age_ratio (future_rahul_age : ℕ) (years_passed : ℕ) (deepak_age : ℕ) :
  future_rahul_age = 26 →
  years_passed = 6 →
  deepak_age = 15 →
  rahul_present_age future_rahul_age years_passed / deepak_age = 4 / 3 :=
by
  intros
  have h1 : rahul_present_age 26 6 = 20 := rfl
  sorry

end age_ratio_l1430_143003


namespace cookies_left_after_three_days_l1430_143039

theorem cookies_left_after_three_days
  (initial_cookies : ℕ)
  (first_day_fraction_eaten : ℚ)
  (second_day_fraction_eaten : ℚ)
  (initial_value : initial_cookies = 64)
  (first_day_fraction : first_day_fraction_eaten = 3/4)
  (second_day_fraction : second_day_fraction_eaten = 1/2) :
  initial_cookies - (first_day_fraction_eaten * 64) - (second_day_fraction_eaten * ((1 - first_day_fraction_eaten) * 64)) = 8 :=
by
  sorry

end cookies_left_after_three_days_l1430_143039


namespace cannot_fill_box_exactly_l1430_143078

def box_length : ℝ := 70
def box_width : ℝ := 40
def box_height : ℝ := 25
def cube_side : ℝ := 4.5

theorem cannot_fill_box_exactly : 
  ¬ (∃ n : ℕ, n * cube_side^3 = box_length * box_width * box_height ∧
               (∃ x y z : ℕ, x * cube_side = box_length ∧ 
                             y * cube_side = box_width ∧ 
                             z * cube_side = box_height)) :=
by sorry

end cannot_fill_box_exactly_l1430_143078


namespace number_whose_multiples_are_considered_for_calculating_the_average_l1430_143009

theorem number_whose_multiples_are_considered_for_calculating_the_average
  (x : ℕ)
  (n : ℕ)
  (a : ℕ)
  (b : ℕ)
  (h1 : n = 10)
  (h2 : a = (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7)
  (h3 : b = 2*n)
  (h4 : a^2 - b^2 = 0) :
  x = 5 := 
sorry

end number_whose_multiples_are_considered_for_calculating_the_average_l1430_143009


namespace nested_fraction_value_l1430_143029

theorem nested_fraction_value :
  1 + (1 / (1 + (1 / (2 + (2 / 3))))) = 19 / 11 :=
by sorry

end nested_fraction_value_l1430_143029


namespace inequality_transform_l1430_143045

theorem inequality_transform {a b : ℝ} (h : a < b) : -2 + 2 * a < -2 + 2 * b :=
sorry

end inequality_transform_l1430_143045


namespace find_f_l1430_143037

noncomputable def f (f'₁ : ℝ) (x : ℝ) : ℝ := f'₁ * Real.exp x - x ^ 2

theorem find_f'₁ (f'₁ : ℝ) (h : f f'₁ = λ x => f'₁ * Real.exp x - x ^ 2) :
  f'₁ = 2 * Real.exp 1 / (Real.exp 1 - 1) := by
  sorry

end find_f_l1430_143037


namespace find_m_l1430_143090

theorem find_m (m : ℝ) (A : Set ℝ) (hA : A = {0, m, m^2 - 3 * m + 2}) (h2 : 2 ∈ A) : m = 3 :=
  sorry

end find_m_l1430_143090


namespace functional_equation_solution_l1430_143008

theorem functional_equation_solution (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)) →
  ∀ x : ℝ, f x = a * x^2 + b * x :=
by
  intro h
  intro x
  have : ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y) := h
  sorry

end functional_equation_solution_l1430_143008


namespace six_digit_phone_number_count_l1430_143007

def six_digit_to_seven_digit_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) : ℕ :=
  let num_positions := 7
  let num_digits := 10
  num_positions * num_digits

theorem six_digit_phone_number_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) :
  six_digit_to_seven_digit_count six_digit h = 70 := by
  -- Proof goes here
  sorry

end six_digit_phone_number_count_l1430_143007


namespace l_shape_area_l1430_143031

theorem l_shape_area (large_length large_width small_length small_width : ℕ)
  (large_rect_area : large_length = 10 ∧ large_width = 7)
  (small_rect_area : small_length = 3 ∧ small_width = 2) :
  (large_length * large_width) - 2 * (small_length * small_width) = 58 :=
by 
  sorry

end l_shape_area_l1430_143031


namespace lines_parallel_l1430_143056

noncomputable def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
noncomputable def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, a^2-1)

def are_parallel (line1 line2 : ℝ × ℝ × ℝ) : Prop :=
  let ⟨a1, b1, _⟩ := line1
  let ⟨a2, b2, _⟩ := line2
  a1 * b2 = a2 * b1

theorem lines_parallel (a : ℝ) :
  are_parallel (line1 a) (line2 a) → a = -1 :=
sorry

end lines_parallel_l1430_143056


namespace total_wrappers_l1430_143061

theorem total_wrappers (a m : ℕ) (ha : a = 34) (hm : m = 15) : a + m = 49 :=
by
  sorry

end total_wrappers_l1430_143061


namespace giftWrapperPerDay_l1430_143083

variable (giftWrapperPerBox : ℕ)
variable (boxesPer3Days : ℕ)

def giftWrapperUsedIn3Days := giftWrapperPerBox * boxesPer3Days

theorem giftWrapperPerDay (h_giftWrapperPerBox : giftWrapperPerBox = 18)
  (h_boxesPer3Days : boxesPer3Days = 15) : giftWrapperUsedIn3Days giftWrapperPerBox boxesPer3Days / 3 = 90 :=
by
  sorry

end giftWrapperPerDay_l1430_143083


namespace boat_speed_still_water_l1430_143048

theorem boat_speed_still_water (V_b V_c : ℝ) (h1 : 45 / (V_b - V_c) = t) (h2 : V_b = 12)
(h3 : V_b + V_c = 15):
  V_b = 12 :=
by
  sorry

end boat_speed_still_water_l1430_143048


namespace stock_price_after_two_years_l1430_143019

theorem stock_price_after_two_years 
    (p0 : ℝ) (r1 r2 : ℝ) (p1 p2 : ℝ) 
    (h0 : p0 = 100) (h1 : r1 = 0.50) 
    (h2 : r2 = 0.30) 
    (h3 : p1 = p0 * (1 + r1)) 
    (h4 : p2 = p1 * (1 - r2)) : 
    p2 = 105 :=
by sorry

end stock_price_after_two_years_l1430_143019


namespace area_of_triangle_ABC_l1430_143088

-- Define the sides of the triangle
def AB : ℝ := 12
def BC : ℝ := 9

-- Define the expected area of the triangle
def expectedArea : ℝ := 54

-- Prove the area of the triangle using the given conditions
theorem area_of_triangle_ABC : (1/2) * AB * BC = expectedArea := 
by
  sorry

end area_of_triangle_ABC_l1430_143088


namespace b_days_to_complete_work_l1430_143026

theorem b_days_to_complete_work (x : ℕ) 
  (A : ℝ := 1 / 30) 
  (B : ℝ := 1 / x) 
  (C : ℝ := 1 / 40)
  (work_eq : 8 * (A + B + C) + 4 * (A + B) = 1) 
  (x_ne_0 : x ≠ 0) : 
  x = 30 := 
by
  sorry

end b_days_to_complete_work_l1430_143026


namespace like_terms_proof_l1430_143013

theorem like_terms_proof (m n : ℤ) 
  (h1 : m + 10 = 3 * n - m) 
  (h2 : 7 - n = n - m) :
  m^2 - 2 * m * n + n^2 = 9 := by
  sorry

end like_terms_proof_l1430_143013


namespace ribbon_cost_comparison_l1430_143016

theorem ribbon_cost_comparison 
  (A : Type)
  (yellow_ribbon_cost blue_ribbon_cost : ℕ)
  (h1 : yellow_ribbon_cost = 24)
  (h2 : blue_ribbon_cost = 36) :
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n < blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n > blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n = blue_ribbon_cost / n) :=
sorry

end ribbon_cost_comparison_l1430_143016


namespace percent_runs_by_running_between_wickets_l1430_143055

theorem percent_runs_by_running_between_wickets :
  (132 - (12 * 4 + 2 * 6)) / 132 * 100 = 54.54545454545455 :=
by
  sorry

end percent_runs_by_running_between_wickets_l1430_143055


namespace nancy_seeds_in_big_garden_l1430_143046

theorem nancy_seeds_in_big_garden :
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  seeds_in_big_garden = 28 := by
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  sorry

end nancy_seeds_in_big_garden_l1430_143046


namespace max_horizontal_distance_domino_l1430_143030

theorem max_horizontal_distance_domino (n : ℕ) : 
    (n > 0) → ∃ d, d = 2 * Real.log n := 
by {
    sorry
}

end max_horizontal_distance_domino_l1430_143030


namespace minimum_value_x2_minus_x1_range_of_a_l1430_143001

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) := a * x
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

-- Question (I)
theorem minimum_value_x2_minus_x1 : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ a = 1 / 3 ∧ f x₁ = g x₂ a → x₂ - x₁ = 3 := 
sorry

-- Question (II)
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, F x a ≥ F (-x) a) ↔ a ≤ 2 :=
sorry

end minimum_value_x2_minus_x1_range_of_a_l1430_143001


namespace power_set_card_greater_l1430_143082

open Set

variables {A : Type*} (α : ℕ) [Fintype A] (hA : Fintype.card A = α)

theorem power_set_card_greater (h : Fintype.card A = α) :
  2 ^ α > α :=
sorry

end power_set_card_greater_l1430_143082


namespace number_of_small_jars_l1430_143038

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end number_of_small_jars_l1430_143038


namespace angle_in_gradians_l1430_143052

noncomputable def gradians_in_full_circle : ℝ := 600
noncomputable def degrees_in_full_circle : ℝ := 360
noncomputable def angle_in_degrees : ℝ := 45

theorem angle_in_gradians :
  angle_in_degrees / degrees_in_full_circle * gradians_in_full_circle = 75 := 
by
  sorry

end angle_in_gradians_l1430_143052


namespace problem_statement_l1430_143002

theorem problem_statement (A B : ℝ) (hA : A = 10 * π / 180) (hB : B = 35 * π / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
  1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) + Real.tan A * (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) :=
by
  sorry

end problem_statement_l1430_143002


namespace evaluate_expression_l1430_143065

-- Definition of the given condition.
def sixty_four_eq_sixteen_squared : Prop := 64 = 16^2

-- The statement to prove that the given expression equals the answer.
theorem evaluate_expression (h : sixty_four_eq_sixteen_squared) : 
  (16^24) / (64^8) = 16^8 :=
by 
  -- h contains the condition that 64 = 16^2, but we provide a proof step later with sorry
  sorry

end evaluate_expression_l1430_143065


namespace colbert_planks_needed_to_buy_l1430_143069

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end colbert_planks_needed_to_buy_l1430_143069


namespace total_weight_of_remaining_macaroons_l1430_143049

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l1430_143049


namespace box_volume_increase_l1430_143091

theorem box_volume_increase (l w h : ℝ)
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
  sorry

end box_volume_increase_l1430_143091


namespace radius_of_semicircle_l1430_143093

theorem radius_of_semicircle (P : ℝ) (π_val : ℝ) (h1 : P = 162) (h2 : π_val = Real.pi) : 
  ∃ r : ℝ, r = 162 / (π + 2) :=
by
  use 162 / (Real.pi + 2)
  sorry

end radius_of_semicircle_l1430_143093


namespace solve_x2_y2_eq_3z2_in_integers_l1430_143022

theorem solve_x2_y2_eq_3z2_in_integers (x y z : ℤ) : x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_x2_y2_eq_3z2_in_integers_l1430_143022


namespace expression_simplification_l1430_143098

theorem expression_simplification (x y : ℝ) :
  20 * (x + y) - 19 * (x + y) = x + y :=
by
  sorry

end expression_simplification_l1430_143098


namespace min_value_SN64_by_aN_is_17_over_2_l1430_143077

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2_l1430_143077


namespace find_S20_l1430_143063

theorem find_S20 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n = 1 + 2 * a n)
  (h2 : a 1 = 2) : 
  S 20 = 2^19 + 1 := 
sorry

end find_S20_l1430_143063


namespace div_by_10_l1430_143068

theorem div_by_10 (n : ℕ) (hn : 10 ∣ (3^n + 1)) : 10 ∣ (3^(n+4) + 1) :=
by
  sorry

end div_by_10_l1430_143068


namespace student_age_is_24_l1430_143017

/-- A man is 26 years older than his student. In two years, his age will be twice the age of his student.
    Prove that the present age of the student is 24 years old. -/
theorem student_age_is_24 (S M : ℕ) (h1 : M = S + 26) (h2 : M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end student_age_is_24_l1430_143017


namespace present_ages_ratio_l1430_143027

noncomputable def ratio_of_ages (F S : ℕ) : ℚ :=
  F / S

theorem present_ages_ratio (F S : ℕ) (h1 : F + S = 220) (h2 : (F + 10) * 3 = (S + 10) * 5) :
  ratio_of_ages F S = 7 / 4 :=
by
  sorry

end present_ages_ratio_l1430_143027


namespace neg_of_p_l1430_143054

variable (x : ℝ)

def p : Prop := ∀ x ≥ 0, 2^x = 3

theorem neg_of_p : ¬p ↔ ∃ x ≥ 0, 2^x ≠ 3 :=
by
  sorry

end neg_of_p_l1430_143054


namespace sum_of_coeffs_l1430_143086

theorem sum_of_coeffs (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5, (2 - x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5)
  → (a_0 = 32 ∧ 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5)
  → a_1 + a_2 + a_3 + a_4 + a_5 = -31 :=
by
  sorry

end sum_of_coeffs_l1430_143086


namespace sum_of_faces_edges_vertices_l1430_143028

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l1430_143028


namespace symmetric_point_line_eq_l1430_143020

theorem symmetric_point_line_eq (A B : ℝ × ℝ) (l : ℝ → ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hSymmetric : ∀ x y, B = (2 * l x - A.1, 2 * l y - A.2)) :
  ∀ x y, l x = 3 * x - 5 ∧ l y = 3 * y + 6 :=
by
  sorry

end symmetric_point_line_eq_l1430_143020


namespace indistinguishable_distributions_l1430_143058

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end indistinguishable_distributions_l1430_143058


namespace minimum_value_expression_l1430_143033

theorem minimum_value_expression (a b c d e f : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
(h_sum : a + b + c + d + e + f = 7) : 
  ∃ min_val : ℝ, min_val = 63 ∧ 
  (∀ a b c d e f : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 0 < f → a + b + c + d + e + f = 7 → 
  (1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f) ≥ min_val) := 
sorry

end minimum_value_expression_l1430_143033


namespace find_a_of_exponential_passing_point_l1430_143025

theorem find_a_of_exponential_passing_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_point : a^2 = 4) : a = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a_of_exponential_passing_point_l1430_143025


namespace gold_coins_distribution_l1430_143067

theorem gold_coins_distribution (x y : ℝ) (h₁ : x + y = 25) (h₂ : x ≠ y)
  (h₃ : (x^2 - y^2) = k * (x - y)) : k = 25 :=
sorry

end gold_coins_distribution_l1430_143067


namespace profit_per_metre_l1430_143080

/-- 
Given:
1. A trader sells 85 meters of cloth for Rs. 8925.
2. The cost price of one metre of cloth is Rs. 95.

Prove:
The profit per metre of cloth is Rs. 10.
-/
theorem profit_per_metre 
  (SP : ℕ) (CP : ℕ)
  (total_SP : SP = 8925)
  (total_meters : ℕ := 85)
  (cost_per_meter : CP = 95) :
  (SP - total_meters * CP) / total_meters = 10 :=
by
  sorry

end profit_per_metre_l1430_143080


namespace sixth_grade_students_total_l1430_143064

noncomputable def total_students (x y : ℕ) : ℕ := x + y

theorem sixth_grade_students_total (x y : ℕ) 
(h1 : x + (1 / 3) * y = 105) 
(h2 : y + (1 / 2) * x = 105) 
: total_students x y = 147 := 
by
  sorry

end sixth_grade_students_total_l1430_143064


namespace intersection_point_parabola_l1430_143073

theorem intersection_point_parabola :
  ∃ k : ℝ, (∀ x : ℝ, (3 * (x - 4)^2 + k = 0 ↔ x = 2 ∨ x = 6)) :=
by
  sorry

end intersection_point_parabola_l1430_143073


namespace total_students_accommodated_l1430_143075

def num_columns : ℕ := 4
def num_rows : ℕ := 10
def num_buses : ℕ := 6

theorem total_students_accommodated : num_columns * num_rows * num_buses = 240 := by
  sorry

end total_students_accommodated_l1430_143075


namespace find_common_difference_l1430_143050

noncomputable def common_difference (a₁ d : ℤ) : Prop :=
  let a₂ := a₁ + d
  let a₃ := a₁ + 2 * d
  let S₅ := 5 * a₁ + 10 * d
  a₂ + a₃ = 8 ∧ S₅ = 25 → d = 2

-- Statement of the proof problem
theorem find_common_difference (a₁ d : ℤ) (h : common_difference a₁ d) : d = 2 :=
by sorry

end find_common_difference_l1430_143050


namespace polynomial_multiplication_l1430_143004

theorem polynomial_multiplication (x a : ℝ) : (x - a) * (x^2 + a * x + a^2) = x^3 - a^3 :=
by
  sorry

end polynomial_multiplication_l1430_143004


namespace cubic_roots_l1430_143053

theorem cubic_roots (a b x₃ : ℤ)
  (h1 : (2^3 + a * 2^2 + b * 2 + 6 = 0))
  (h2 : (3^3 + a * 3^2 + b * 3 + 6 = 0))
  (h3 : 2 * 3 * x₃ = -6) :
  a = -4 ∧ b = 1 ∧ x₃ = -1 :=
by {
  sorry
}

end cubic_roots_l1430_143053


namespace smallest_angle_in_triangle_l1430_143089

theorem smallest_angle_in_triangle (x : ℝ) 
  (h_ratio : 4 * x < 5 * x ∧ 5 * x < 9 * x) 
  (h_sum : 4 * x + 5 * x + 9 * x = 180) : 
  4 * x = 40 :=
by
  sorry

end smallest_angle_in_triangle_l1430_143089


namespace members_who_didnt_show_up_l1430_143095

theorem members_who_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) 
  (h1 : total_members = 5) (h2 : points_per_member = 6) (h3 : total_points = 18) : 
  total_members - total_points / points_per_member = 2 :=
by
  sorry

end members_who_didnt_show_up_l1430_143095


namespace probability_both_selected_l1430_143047

def probability_selection_ram : ℚ := 4 / 7
def probability_selection_ravi : ℚ := 1 / 5

theorem probability_both_selected : probability_selection_ram * probability_selection_ravi = 4 / 35 := 
by 
  -- Proof goes here
  sorry

end probability_both_selected_l1430_143047


namespace union_of_A_and_B_l1430_143059

open Set

-- Define the sets A and B based on given conditions
def A (x : ℤ) : Set ℤ := {y | y = x^2 ∨ y = 2 * x - 1 ∨ y = -4}
def B (x : ℤ) : Set ℤ := {y | y = x - 5 ∨ y = 1 - x ∨ y = 9}

-- Specific condition given in the problem
def A_intersect_B_condition (x : ℤ) : Prop :=
  A x ∩ B x = {9}

-- Prove problem statement that describes the union of A and B
theorem union_of_A_and_B (x : ℤ) (h : A_intersect_B_condition x) : A x ∪ B x = {-8, -7, -4, 4, 9} :=
sorry

end union_of_A_and_B_l1430_143059


namespace compute_xy_l1430_143018

theorem compute_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x ^ (Real.sqrt y) = 27) (h2 : (Real.sqrt x) ^ y = 9) :
  x * y = 12 * Real.sqrt 3 :=
sorry

end compute_xy_l1430_143018


namespace who_is_first_l1430_143076

def positions (A B C D : ℕ) : Prop :=
  A + B + D = 6 ∧ B + C = 6 ∧ B < A ∧ A + B + C + D = 10

theorem who_is_first (A B C D : ℕ) (h : positions A B C D) : D = 1 :=
sorry

end who_is_first_l1430_143076


namespace square_area_l1430_143040

theorem square_area (perimeter : ℝ) (h_perimeter : perimeter = 40) : 
  ∃ (area : ℝ), area = 100 := by
  sorry

end square_area_l1430_143040


namespace largest_whole_number_lt_150_l1430_143023

theorem largest_whole_number_lt_150 : ∃ (x : ℕ), (x <= 16 ∧ ∀ y : ℕ, y < 17 → 9 * y < 150) :=
by
  sorry

end largest_whole_number_lt_150_l1430_143023


namespace cube_root_of_x_sqrt_x_eq_x_half_l1430_143044

variable (x : ℝ) (h : 0 < x)

theorem cube_root_of_x_sqrt_x_eq_x_half : (x * Real.sqrt x) ^ (1/3) = x ^ (1/2) := by
  sorry

end cube_root_of_x_sqrt_x_eq_x_half_l1430_143044


namespace remainder_of_polynomial_division_l1430_143014

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 7 * x^4 - 16 * x^3 + 3 * x^2 - 5 * x - 20

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 2 * x - 4

-- The remainder theorem sets x to 2 and evaluates P(x)
theorem remainder_of_polynomial_division : P 2 = -34 :=
by
  -- We will substitute x=2 directly into P(x)
  sorry

end remainder_of_polynomial_division_l1430_143014


namespace equal_playing_time_l1430_143015

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l1430_143015


namespace martin_goldfish_count_l1430_143070

-- Define the initial number of goldfish
def initial_goldfish := 18

-- Define the number of goldfish that die each week
def goldfish_die_per_week := 5

-- Define the number of goldfish purchased each week
def goldfish_purchased_per_week := 3

-- Define the number of weeks
def weeks := 7

-- Calculate the expected number of goldfish after 7 weeks
noncomputable def final_goldfish := initial_goldfish - (goldfish_die_per_week * weeks) + (goldfish_purchased_per_week * weeks)

-- State the theorem and the proof target
theorem martin_goldfish_count : final_goldfish = 4 := 
sorry

end martin_goldfish_count_l1430_143070


namespace num_ways_books_distribution_l1430_143006

-- Given conditions
def num_copies_type1 : ℕ := 8
def num_copies_type2 : ℕ := 4
def min_books_in_library_type1 : ℕ := 1
def max_books_in_library_type1 : ℕ := 7
def min_books_in_library_type2 : ℕ := 1
def max_books_in_library_type2 : ℕ := 3

-- The proof problem statement
theorem num_ways_books_distribution : 
  (max_books_in_library_type1 - min_books_in_library_type1 + 1) * 
  (max_books_in_library_type2 - min_books_in_library_type2 + 1) = 21 := by
    sorry

end num_ways_books_distribution_l1430_143006


namespace find_a_l1430_143036

theorem find_a :
  let p1 := (⟨-3, 7⟩ : ℝ × ℝ)
  let p2 := (⟨2, -1⟩ : ℝ × ℝ)
  let direction := (5, -8)
  let target_direction := (a, -2)
  a = (direction.1 * -2) / (direction.2) := by
  sorry

end find_a_l1430_143036


namespace candy_mixture_cost_l1430_143096

/-- 
A club mixes 15 pounds of candy worth $8.00 per pound with 30 pounds of candy worth $5.00 per pound.
We need to find the cost per pound of the mixture.
-/
theorem candy_mixture_cost :
    (15 * 8 + 30 * 5) / (15 + 30) = 6 := 
by
  sorry

end candy_mixture_cost_l1430_143096


namespace frustum_slant_height_l1430_143062

theorem frustum_slant_height (r1 r2 V : ℝ) (h l : ℝ) 
    (H1 : r1 = 2) (H2 : r2 = 6) (H3 : V = 104 * π)
    (H4 : V = (1/3) * π * h * (r1^2 + r2^2 + r1 * r2)) 
    (H5 : h = 6)
    (H6 : l = Real.sqrt (h^2 + (r2 - r1)^2)) :
    l = 2 * Real.sqrt 13 :=
by sorry

end frustum_slant_height_l1430_143062


namespace solution_to_ball_problem_l1430_143011

noncomputable def probability_of_arithmetic_progression : Nat :=
  let p := 3
  let q := 9464
  p + q

theorem solution_to_ball_problem : probability_of_arithmetic_progression = 9467 := by
  sorry

end solution_to_ball_problem_l1430_143011


namespace irreducible_fraction_iff_not_congruent_mod_5_l1430_143072

theorem irreducible_fraction_iff_not_congruent_mod_5 (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := 
by 
  sorry

end irreducible_fraction_iff_not_congruent_mod_5_l1430_143072


namespace price_range_of_book_l1430_143094

variable (x : ℝ)

theorem price_range_of_book (h₁ : ¬(x ≥ 15)) (h₂ : ¬(x ≤ 12)) (h₃ : ¬(x ≤ 10)) : 12 < x ∧ x < 15 := 
by
  sorry

end price_range_of_book_l1430_143094


namespace tangent_line_equation_l1430_143005

theorem tangent_line_equation 
  (A : ℝ × ℝ)
  (hA : A = (-1, 2))
  (parabola : ℝ → ℝ)
  (h_parabola : ∀ x, parabola x = 2 * x ^ 2) 
  (tangent : ℝ × ℝ → ℝ)
  (h_tangent : ∀ P, tangent P = -4 * P.1 + 4 * (-1) + 2) : 
  tangent A = 4 * (-1) + 2 :=
by
  sorry

end tangent_line_equation_l1430_143005


namespace money_distribution_l1430_143066

theorem money_distribution :
  ∀ (A B C : ℕ), 
  A + B + C = 900 → 
  B + C = 750 → 
  C = 250 → 
  A + C = 400 := 
by
  intros A B C h1 h2 h3
  sorry

end money_distribution_l1430_143066


namespace area_sum_of_three_circles_l1430_143043

theorem area_sum_of_three_circles (R d : ℝ) (x y z : ℝ) 
    (hxyz : x^2 + y^2 + z^2 = d^2) :
    (π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2))) = π * (3 * R^2 - d^2) :=
by
  sorry

end area_sum_of_three_circles_l1430_143043


namespace minor_premise_is_wrong_l1430_143079

theorem minor_premise_is_wrong (a : ℝ) : ¬ (0 < a^2) := by
  sorry

end minor_premise_is_wrong_l1430_143079


namespace sum_mod_nine_l1430_143035

def a : ℕ := 1234
def b : ℕ := 1235
def c : ℕ := 1236
def d : ℕ := 1237
def e : ℕ := 1238
def modulus : ℕ := 9

theorem sum_mod_nine : (a + b + c + d + e) % modulus = 6 :=
by
  sorry

end sum_mod_nine_l1430_143035


namespace min_value_of_x_l1430_143051

-- Definitions for the conditions given in the problem
def men := 4
def women (x : ℕ) := x
def min_x := 594

-- Definition of the probability p
def C (n k : ℕ) : ℕ := sorry -- Define the binomial coefficient properly

def probability (x : ℕ) : ℚ :=
  (2 * (C (x+1) 2) + (x + 1)) /
  (C (x + 1) 3 + 3 * (C (x + 1) 2) + (x + 1))

-- The theorem statement to prove
theorem min_value_of_x (x : ℕ) : probability x ≤ 1 / 100 →  x = min_x := 
by
  sorry

end min_value_of_x_l1430_143051


namespace tan_eq_860_l1430_143085

theorem tan_eq_860 (n : ℤ) (hn : -180 < n ∧ n < 180) : 
  n = -40 ↔ (Real.tan (n * Real.pi / 180) = Real.tan (860 * Real.pi / 180)) := 
sorry

end tan_eq_860_l1430_143085


namespace total_shaded_area_l1430_143021

theorem total_shaded_area (S T : ℕ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 3) :
  (S * S) + 8 * (T * T) = 17 :=
by
  sorry

end total_shaded_area_l1430_143021


namespace find_x_l1430_143032

theorem find_x (x : ℝ) (h : 0.45 * x = (1 / 3) * x + 110) : x = 942.857 :=
by
  sorry

end find_x_l1430_143032


namespace player_B_questions_l1430_143060

theorem player_B_questions :
  ∀ (a b : ℕ → ℕ), (∀ i j, i ≠ j → a i + b j = a j + b i) →
  ∃ k, k = 11 := sorry

end player_B_questions_l1430_143060


namespace count_lattice_right_triangles_with_incenter_l1430_143012

def is_lattice_point (p : ℤ × ℤ) : Prop := ∃ x y : ℤ, p = (x, y)

def is_right_triangle (O A B : ℤ × ℤ) : Prop :=
  O = (0, 0) ∧ (O.1 = A.1 ∨ O.2 = A.2) ∧ (O.1 = B.1 ∨ O.2 = B.2) ∧
  (A.1 * B.2 - A.2 * B.1 ≠ 0) -- Ensure A and B are not collinear with O

def incenter (O A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1 - O.1) / 2, (A.2 + B.2 - O.2) / 2)

theorem count_lattice_right_triangles_with_incenter :
  let I := (2015, 7 * 2015)
  ∃ (O A B : ℤ × ℤ), is_right_triangle O A B ∧ incenter O A B = I :=
sorry

end count_lattice_right_triangles_with_incenter_l1430_143012


namespace find_set_A_l1430_143024

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5})
variable (h1 : (U \ A) ∩ B = {0, 4})
variable (h2 : (U \ A) ∩ (U \ B) = {3, 5})

theorem find_set_A :
  A = {1, 2} :=
by
  sorry

end find_set_A_l1430_143024


namespace point_B_is_4_l1430_143042

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end point_B_is_4_l1430_143042


namespace horse_food_calculation_l1430_143099

theorem horse_food_calculation
  (num_sheep : ℕ)
  (ratio_sheep_horses : ℕ)
  (total_horse_food : ℕ)
  (H : ℕ)
  (num_sheep_eq : num_sheep = 56)
  (ratio_eq : ratio_sheep_horses = 7)
  (total_food_eq : total_horse_food = 12880)
  (num_horses : H = num_sheep * 1 / ratio_sheep_horses)
  : num_sheep = ratio_sheep_horses → total_horse_food / H = 230 :=
by
  sorry

end horse_food_calculation_l1430_143099


namespace faye_age_l1430_143081

variables (C D E F G : ℕ)
variables (h1 : D = E - 2)
variables (h2 : E = C + 6)
variables (h3 : F = C + 4)
variables (h4 : G = C - 5)
variables (h5 : D = 16)

theorem faye_age : F = 16 :=
by
  -- Proof will be placed here
  sorry

end faye_age_l1430_143081


namespace value_of_expression_l1430_143074

theorem value_of_expression : (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) :=
by
  sorry

end value_of_expression_l1430_143074


namespace min_k_plus_p_is_19199_l1430_143097

noncomputable def find_min_k_plus_p : ℕ :=
  let D := 1007
  let domain_len := 1 / D
  let min_k : ℕ := 19  -- Minimum k value for which domain length condition holds, found via problem conditions
  let p_for_k (k : ℕ) : ℕ := (D * (k^2 - 1)) / k
  let k_plus_p (k : ℕ) : ℕ := k + p_for_k k
  k_plus_p min_k

theorem min_k_plus_p_is_19199 : find_min_k_plus_p = 19199 :=
  sorry

end min_k_plus_p_is_19199_l1430_143097


namespace total_gallons_in_tanks_l1430_143041

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l1430_143041


namespace total_capacity_of_schools_l1430_143092

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end total_capacity_of_schools_l1430_143092


namespace divisible_by_12_l1430_143057

theorem divisible_by_12 (n : ℕ) (h1 : (5140 + n) % 4 = 0) (h2 : (5 + 1 + 4 + n) % 3 = 0) : n = 8 :=
by
  sorry

end divisible_by_12_l1430_143057
