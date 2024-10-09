import Mathlib

namespace fewer_cubes_needed_l869_86998

variable (cubeVolume : ℕ) (length : ℕ) (width : ℕ) (depth : ℕ) (TVolume : ℕ)

theorem fewer_cubes_needed : 
  cubeVolume = 5 → 
  length = 7 → 
  width = 7 → 
  depth = 6 → 
  TVolume = 3 → 
  (length * width * depth - TVolume = 291) :=
by
  intros hc hl hw hd ht
  sorry

end fewer_cubes_needed_l869_86998


namespace surface_area_of_large_cube_is_486_cm_squared_l869_86973

noncomputable def surfaceAreaLargeCube : ℕ :=
  let small_box_count := 27
  let edge_small_box := 3
  let edge_large_cube := (small_box_count^(1/3)) * edge_small_box
  6 * edge_large_cube^2

theorem surface_area_of_large_cube_is_486_cm_squared :
  surfaceAreaLargeCube = 486 := 
sorry

end surface_area_of_large_cube_is_486_cm_squared_l869_86973


namespace sufficient_not_necessary_condition_l869_86929

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iic (-2) → (x^2 + 2 * a * x - 2) ≤ ((x - 1)^2 + 2 * a * (x - 1) - 2)) ↔ a ≤ 2 := by
  sorry

end sufficient_not_necessary_condition_l869_86929


namespace area_union_of_reflected_triangles_l869_86924

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 7)
def C : ℝ × ℝ := (6, 2)
def A' : ℝ × ℝ := (3, 2)
def B' : ℝ × ℝ := (7, 5)
def C' : ℝ × ℝ := (2, 6)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem area_union_of_reflected_triangles :
  let area_ABC := triangle_area A B C
  let area_A'B'C' := triangle_area A' B' C'
  area_ABC + area_A'B'C' = 19 := by
  sorry

end area_union_of_reflected_triangles_l869_86924


namespace perc_freshmen_in_SLA_l869_86939

variables (T : ℕ) (P : ℝ)

-- 60% of students are freshmen
def freshmen (T : ℕ) : ℝ := 0.60 * T

-- 4.8% of students are freshmen psychology majors in the school of liberal arts
def freshmen_psych_majors (T : ℕ) : ℝ := 0.048 * T

-- 20% of freshmen in the school of liberal arts are psychology majors
def perc_fresh_psych (F_LA : ℝ) : ℝ := 0.20 * F_LA

-- Number of freshmen in the school of liberal arts as a percentage P of the total number of freshmen
def fresh_in_SLA_as_perc (T : ℕ) (P : ℝ) : ℝ := P * (0.60 * T)

theorem perc_freshmen_in_SLA (T : ℕ) (P : ℝ) :
  (0.20 * (P * (0.60 * T)) = 0.048 * T) → P = 0.4 :=
sorry

end perc_freshmen_in_SLA_l869_86939


namespace min_value_expr_l869_86987

theorem min_value_expr (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, m > 0 ∧ (forall (n : ℕ), 0 < n → (n/2 + 50/n : ℝ) ≥ 10) ∧ 
           (n = 10) → (n/2 + 50/n : ℝ) = 10 :=
by
  sorry

end min_value_expr_l869_86987


namespace beavers_help_l869_86938

theorem beavers_help (initial final : ℝ) (h_initial : initial = 2.0) (h_final : final = 3) : final - initial = 1 :=
  by
    sorry

end beavers_help_l869_86938


namespace number_of_green_balls_l869_86984

theorem number_of_green_balls
  (total_balls white_balls yellow_balls red_balls purple_balls : ℕ)
  (prob : ℚ)
  (H_total : total_balls = 100)
  (H_white : white_balls = 50)
  (H_yellow : yellow_balls = 10)
  (H_red : red_balls = 7)
  (H_purple : purple_balls = 3)
  (H_prob : prob = 0.9) :
  ∃ (green_balls : ℕ), 
    (white_balls + green_balls + yellow_balls) / total_balls = prob ∧ green_balls = 30 := by
  sorry

end number_of_green_balls_l869_86984


namespace janet_stuffies_l869_86974

theorem janet_stuffies (total_stuffies kept_stuffies given_away_stuffies janet_stuffies : ℕ) 
 (h1 : total_stuffies = 60)
 (h2 : kept_stuffies = total_stuffies / 3)
 (h3 : given_away_stuffies = total_stuffies - kept_stuffies)
 (h4 : janet_stuffies = given_away_stuffies / 4) : 
 janet_stuffies = 10 := 
sorry

end janet_stuffies_l869_86974


namespace trigonometric_identity_l869_86951

open Real

variable (α : ℝ)
variable (h1 : π < α)
variable (h2 : α < 2 * π)
variable (h3 : cos (α - 7 * π) = -3 / 5)

theorem trigonometric_identity :
  sin (3 * π + α) * tan (α - 7 * π / 2) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l869_86951


namespace k_is_square_l869_86982

theorem k_is_square (a b : ℕ) (h_a : a > 0) (h_b : b > 0) (k : ℕ) (h_k : k > 0)
    (h : (a^2 + b^2) = k * (a * b + 1)) : ∃ (n : ℕ), n^2 = k :=
sorry

end k_is_square_l869_86982


namespace cost_of_each_box_of_cereal_l869_86903

theorem cost_of_each_box_of_cereal
  (total_groceries_cost : ℝ)
  (gallon_of_milk_cost : ℝ)
  (number_of_cereal_boxes : ℕ)
  (banana_cost_each : ℝ)
  (number_of_bananas : ℕ)
  (apple_cost_each : ℝ)
  (number_of_apples : ℕ)
  (cookie_cost_multiplier : ℝ)
  (number_of_cookie_boxes : ℕ) :
  total_groceries_cost = 25 →
  gallon_of_milk_cost = 3 →
  number_of_cereal_boxes = 2 →
  banana_cost_each = 0.25 →
  number_of_bananas = 4 →
  apple_cost_each = 0.5 →
  number_of_apples = 4 →
  cookie_cost_multiplier = 2 →
  number_of_cookie_boxes = 2 →
  (total_groceries_cost - (gallon_of_milk_cost + (banana_cost_each * number_of_bananas) + 
                           (apple_cost_each * number_of_apples) + 
                           (number_of_cookie_boxes * (cookie_cost_multiplier * gallon_of_milk_cost)))) / 
  number_of_cereal_boxes = 3.5 := 
sorry

end cost_of_each_box_of_cereal_l869_86903


namespace linear_equation_must_be_neg2_l869_86957

theorem linear_equation_must_be_neg2 {m : ℝ} (h1 : |m| - 1 = 1) (h2 : m ≠ 2) : m = -2 :=
sorry

end linear_equation_must_be_neg2_l869_86957


namespace sufficient_not_necessary_l869_86906

theorem sufficient_not_necessary (x : ℝ) :
  (|x - 1| < 2 → x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end sufficient_not_necessary_l869_86906


namespace reduced_price_per_kg_l869_86928

-- Definitions
variables {P R Q : ℝ}

-- Conditions
axiom reduction_price : R = P * 0.82
axiom original_quantity : Q * P = 1080
axiom reduced_quantity : (Q + 8) * R = 1080

-- Proof statement
theorem reduced_price_per_kg : R = 24.30 :=
by {
  sorry
}

end reduced_price_per_kg_l869_86928


namespace cistern_fill_time_l869_86900

-- Let F be the rate at which the first tap fills the cistern (cisterns per hour)
def F : ℚ := 1 / 4

-- Let E be the rate at which the second tap empties the cistern (cisterns per hour)
def E : ℚ := 1 / 5

-- Prove that the time it takes to fill the cistern is 20 hours given the rates F and E
theorem cistern_fill_time : (1 / (F - E)) = 20 := 
by
  -- Insert necessary proofs here
  sorry

end cistern_fill_time_l869_86900


namespace find_certain_number_l869_86997

theorem find_certain_number (N : ℝ) 
  (h : 3.6 * N * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001)
  : N = 0.48 :=
sorry

end find_certain_number_l869_86997


namespace find_number_l869_86948

theorem find_number (N : ℝ) (h : 0.6 * (3 / 5) * N = 36) : N = 100 :=
by sorry

end find_number_l869_86948


namespace part2_x_values_part3_no_real_x_for_2000_l869_86958

noncomputable def average_daily_sales (x : ℝ) : ℝ :=
  24 + 4 * x

noncomputable def profit_per_unit (x : ℝ) : ℝ :=
  60 - 5 * x

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  (60 - 5 * x) * (24 + 4 * x)

theorem part2_x_values : 
  {x : ℝ | daily_sales_profit x = 1540} = {1, 5} := sorry

theorem part3_no_real_x_for_2000 : 
  ∀ x : ℝ, daily_sales_profit x ≠ 2000 := sorry

end part2_x_values_part3_no_real_x_for_2000_l869_86958


namespace abs_sum_lt_abs_diff_l869_86989

theorem abs_sum_lt_abs_diff (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end abs_sum_lt_abs_diff_l869_86989


namespace num_5_letter_words_with_at_least_one_A_l869_86994

theorem num_5_letter_words_with_at_least_one_A :
  let total := 6 ^ 5
  let without_A := 5 ^ 5
  total - without_A = 4651 := by
sorry

end num_5_letter_words_with_at_least_one_A_l869_86994


namespace cost_of_six_dozen_l869_86996

variable (cost_of_four_dozen : ℕ)
variable (dozens_to_purchase : ℕ)

theorem cost_of_six_dozen :
  cost_of_four_dozen = 24 →
  dozens_to_purchase = 6 →
  (dozens_to_purchase * (cost_of_four_dozen / 4)) = 36 :=
by
  intros h1 h2
  sorry

end cost_of_six_dozen_l869_86996


namespace minimum_value_inequality_l869_86967

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 1) * (y^2 + 5 * y + 1) * (z^2 + 5 * y + 1) / (x * y * z) ≥ 343 :=
by sorry

end minimum_value_inequality_l869_86967


namespace problem1_problem2_l869_86968

-- Problem 1
theorem problem1 (α : ℝ) (h : Real.tan α = -3/4) :
  (Real.cos ((π / 2) + α) * Real.sin (-π - α)) /
  (Real.cos ((11 * π) / 2 - α) * Real.sin ((9 * π) / 2 + α)) = -3 / 4 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ π) (h2 : Real.sin α + Real.cos α = 1 / 5) :
  Real.cos (2 * α - π / 4) = -31 * Real.sqrt 2 / 50 :=
by sorry

end problem1_problem2_l869_86968


namespace distance_between_neg5_and_neg1_l869_86970

theorem distance_between_neg5_and_neg1 : 
  dist (-5 : ℝ) (-1) = 4 := by
sorry

end distance_between_neg5_and_neg1_l869_86970


namespace candies_in_box_more_than_pockets_l869_86956

theorem candies_in_box_more_than_pockets (x : ℕ) : 
  let initial_pockets := 2 * x
  let pockets_after_return := 2 * (x - 6)
  let candies_returned_to_box := 12
  let total_candies_after_return := initial_pockets + candies_returned_to_box
  (total_candies_after_return - pockets_after_return) = 24 :=
by
  sorry

end candies_in_box_more_than_pockets_l869_86956


namespace maximum_area_of_rectangular_playground_l869_86964

theorem maximum_area_of_rectangular_playground (P : ℕ) (A : ℕ) (h : P = 150) :
  ∃ (x y : ℕ), x + y = 75 ∧ A ≤ x * y ∧ A = 1406 :=
sorry

end maximum_area_of_rectangular_playground_l869_86964


namespace square_area_is_correct_l869_86952

-- Define the condition: the side length of the square field
def side_length : ℝ := 7

-- Define the theorem to prove the area of the square field with given side length
theorem square_area_is_correct : side_length * side_length = 49 := by
  -- Proof goes here
  sorry

end square_area_is_correct_l869_86952


namespace tip_percentage_correct_l869_86913

def lunch_cost := 50.20
def total_spent := 60.24
def tip_percentage := ((total_spent - lunch_cost) / lunch_cost) * 100

theorem tip_percentage_correct : tip_percentage = 19.96 := 
by
  sorry

end tip_percentage_correct_l869_86913


namespace buns_left_l869_86955

theorem buns_left (buns_initial : ℕ) (h1 : buns_initial = 15)
                  (x : ℕ) (h2 : 13 * x ≤ buns_initial)
                  (buns_taken_by_bimbo : ℕ) (h3 : buns_taken_by_bimbo = x)
                  (buns_taken_by_little_boy : ℕ) (h4 : buns_taken_by_little_boy = 3 * x)
                  (buns_taken_by_karlsson : ℕ) (h5 : buns_taken_by_karlsson = 9 * x)
                  :
                  buns_initial - (buns_taken_by_bimbo + buns_taken_by_little_boy + buns_taken_by_karlsson) = 2 :=
by
  sorry

end buns_left_l869_86955


namespace pureGalaTrees_l869_86962

theorem pureGalaTrees {T F C : ℕ} (h1 : F + C = 204) (h2 : F = (3 / 4 : ℝ) * T) (h3 : C = (1 / 10 : ℝ) * T) : (0.15 * T : ℝ) = 36 :=
by
  sorry

end pureGalaTrees_l869_86962


namespace sum_of_first_10_terms_l869_86976

def general_term (n : ℕ) : ℕ := 2 * n + 1

def sequence_sum (n : ℕ) : ℕ := n / 2 * (general_term 1 + general_term n)

theorem sum_of_first_10_terms : sequence_sum 10 = 120 := by
  sorry

end sum_of_first_10_terms_l869_86976


namespace scientific_notation_l869_86988

theorem scientific_notation (x : ℝ) (h : x = 70819) : x = 7.0819 * 10^4 :=
by 
  -- Proof goes here
  sorry

end scientific_notation_l869_86988


namespace necessary_not_sufficient_l869_86905

variable (a b : ℝ)

theorem necessary_not_sufficient : 
  (a > b) -> ¬ (a > b+1) ∨ (a > b+1 ∧ a > b) :=
by
  intro h
  have h1 : ¬ (a > b+1) := sorry
  have h2 : (a > b+1 -> a > b) := sorry
  exact Or.inl h1

end necessary_not_sufficient_l869_86905


namespace kenneth_past_finish_line_l869_86921

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end kenneth_past_finish_line_l869_86921


namespace subtraction_equals_eleven_l869_86927

theorem subtraction_equals_eleven (K A N G R O : ℕ) (h1: K ≠ A) (h2: K ≠ N) (h3: K ≠ G) (h4: K ≠ R) (h5: K ≠ O) (h6: A ≠ N) (h7: A ≠ G) (h8: A ≠ R) (h9: A ≠ O) (h10: N ≠ G) (h11: N ≠ R) (h12: N ≠ O) (h13: G ≠ R) (h14: G ≠ O) (h15: R ≠ O) (sum_eq : 100 * K + 10 * A + N + 10 * G + A = 100 * R + 10 * O + O) : 
  (10 * R + N) - (10 * K + G) = 11 := 
by 
  sorry

end subtraction_equals_eleven_l869_86927


namespace logistics_center_correct_l869_86915

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-6, 9)
def C : ℝ × ℝ := (-3, -8)

def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_correct : 
  ∀ L : ℝ × ℝ, 
  (rectilinear_distance L A = rectilinear_distance L B) ∧ 
  (rectilinear_distance L B = rectilinear_distance L C) ∧
  (rectilinear_distance L A = rectilinear_distance L C) → 
  L = logistics_center := sorry

end logistics_center_correct_l869_86915


namespace calculation_proof_l869_86934

theorem calculation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end calculation_proof_l869_86934


namespace hypotenuse_length_l869_86925

noncomputable def hypotenuse_of_30_60_90_triangle (r : ℝ) : ℝ :=
  let a := (r * 3) / Real.sqrt 3
  2 * a

theorem hypotenuse_length (r : ℝ) (h : r = 3) : hypotenuse_of_30_60_90_triangle r = 6 * Real.sqrt 3 :=
  by sorry

end hypotenuse_length_l869_86925


namespace white_balls_count_l869_86991

theorem white_balls_count (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) (W : ℕ)
    (h_total : total_balls = 100)
    (h_green : green_balls = 30)
    (h_yellow : yellow_balls = 10)
    (h_red : red_balls = 37)
    (h_purple : purple_balls = 3)
    (h_prob : prob_neither_red_nor_purple = 0.6)
    (h_computation : W = total_balls * prob_neither_red_nor_purple - (green_balls + yellow_balls)) :
    W = 20 := 
sorry

end white_balls_count_l869_86991


namespace flowchart_output_value_l869_86950

theorem flowchart_output_value :
  ∃ n : ℕ, S = n * (n + 1) / 2 ∧ n = 10 → S = 55 :=
by
  sorry

end flowchart_output_value_l869_86950


namespace sum_of_edges_l869_86931

theorem sum_of_edges (a r : ℝ) 
  (h_vol : (a / r) * a * (a * r) = 432) 
  (h_surf_area : 2 * ((a * a) / r + (a * a) * r + a * a) = 384) 
  (h_geom_prog : r ≠ 1) :
  4 * ((6 * Real.sqrt 2) / r + 6 * Real.sqrt 2 + (6 * Real.sqrt 2) * r) = 72 * (Real.sqrt 2) := 
sorry

end sum_of_edges_l869_86931


namespace simplify_fraction_l869_86926

theorem simplify_fraction (num denom : ℚ) (h_num: num = (3/7 + 5/8)) (h_denom: denom = (5/12 + 2/3)) :
  (num / denom) = (177/182) := 
  sorry

end simplify_fraction_l869_86926


namespace set_intersection_subset_condition_l869_86959

-- Define the sets A and B
def A (x : ℝ) : Prop := 1 < x - 1 ∧ x - 1 ≤ 4
def B (a : ℝ) (x : ℝ) : Prop := x < a

-- First proof problem: A ∩ B = {x | 2 < x < 3}
theorem set_intersection (a : ℝ) (x : ℝ) (h_a : a = 3) :
  A x ∧ B a x ↔ 2 < x ∧ x < 3 :=
by
  sorry

-- Second proof problem: a > 5 given A ⊆ B
theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ a > 5 :=
by
  sorry

end set_intersection_subset_condition_l869_86959


namespace vector_addition_proof_l869_86933

def vector_add (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 + b.1, a.2 + b.2)

theorem vector_addition_proof :
  let a := (2, 0)
  let b := (-1, -2)
  vector_add a b = (1, -2) :=
by
  sorry

end vector_addition_proof_l869_86933


namespace avg_speed_train_l869_86935

theorem avg_speed_train {D V : ℝ} (h1 : D = 20 * (90 / 60)) (h2 : 360 = 6 * 60) : 
  V = D / (360 / 60) :=
  by sorry

end avg_speed_train_l869_86935


namespace swap_correct_l869_86995

variable (a b c : ℕ)

noncomputable def swap_and_verify (a : ℕ) (b : ℕ) : Prop :=
  let c := b
  let b := a
  let a := c
  a = 2012 ∧ b = 2011

theorem swap_correct :
  ∀ a b : ℕ, a = 2011 → b = 2012 → swap_and_verify a b :=
by
  intros a b ha hb
  sorry

end swap_correct_l869_86995


namespace intersection_A_B_l869_86942

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection (S T : Set ℝ) : Set ℝ := {x | x ∈ S ∧ x ∈ T}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_A_B_l869_86942


namespace first_candidate_percentage_l869_86986

theorem first_candidate_percentage (P : ℝ) 
    (total_votes : ℝ) (votes_second : ℝ)
    (h_total_votes : total_votes = 1200)
    (h_votes_second : votes_second = 480) :
    (P / 100) * total_votes + votes_second = total_votes → P = 60 := 
by
  intro h
  rw [h_total_votes, h_votes_second] at h
  sorry

end first_candidate_percentage_l869_86986


namespace find_M_M_superset_N_M_intersection_N_l869_86993

-- Define the set M as per the given condition
def M : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

-- Define the set N based on parameters a and b
def N (a b : ℝ) : Set ℝ := { x : ℝ | a < x ∧ x < b }

-- Prove that M = (-1, 2)
theorem find_M : M = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Prove that if M ⊇ N, then a ≥ -1
theorem M_superset_N (a b : ℝ) (h : M ⊇ N a b) : -1 ≤ a :=
sorry

-- Prove that if M ∩ N = M, then b ≥ 2
theorem M_intersection_N (a b : ℝ) (h : M ∩ (N a b) = M) : 2 ≤ b :=
sorry

end find_M_M_superset_N_M_intersection_N_l869_86993


namespace quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l869_86992

-- Condition for the quadratic equation having two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)) ↔ (k ≥ 3 / 2) :=
sorry

-- Condition linking the roots of the equation and the properties of the rectangle
theorem roots_form_rectangle_with_diagonal (k : ℝ) 
  (h : k ≥ 3 / 2) :
  (∃ x1 x2 : ℝ, (x1 + x2 = k + 1) ∧ (x1 * x2 = 1/4 * k^2 + 1)
  ∧ (x1^2 + x2^2 = 5)) ↔ (k = 2) :=
sorry

end quadratic_has_two_real_roots_roots_form_rectangle_with_diagonal_l869_86992


namespace stephanie_falls_l869_86965

theorem stephanie_falls 
  (steven_falls : ℕ := 3)
  (sonya_falls : ℕ := 6)
  (h1 : sonya_falls = 6)
  (h2 : ∃ S : ℕ, sonya_falls = (S / 2) - 2 ∧ S > steven_falls) :
  ∃ S : ℕ, S - steven_falls = 13 :=
by
  sorry

end stephanie_falls_l869_86965


namespace part_one_part_two_l869_86907

def discriminant (a b c : ℝ) := b^2 - 4*a*c

theorem part_one (a : ℝ) (h : 0 < a) : 
  (∃ x : ℝ, ax^2 - 3*x + 2 < 0) ↔ 0 < a ∧ a < 9/8 := 
by 
  sorry

theorem part_two (a x : ℝ) : 
  (ax^2 - 3*x + 2 > ax - 1) ↔ 
  (a = 0 ∧ x < 1) ∨ 
  (a < 0 ∧ 3/a < x ∧ x < 1) ∨ 
  (0 < a ∧ (a > 3 ∧ (x < 3/a ∨ x > 1)) ∨ (a = 3 ∧ x ≠ 1) ∨ (0 < a ∧ a < 3 ∧ (x < 1 ∨ x > 3/a))) :=
by 
  sorry

end part_one_part_two_l869_86907


namespace max_y_value_l869_86953

theorem max_y_value (x y : Int) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
by sorry

end max_y_value_l869_86953


namespace susan_age_indeterminate_l869_86917

-- Definitions and conditions
def james_age_in_15_years : ℕ := 37
def current_james_age : ℕ := james_age_in_15_years - 15
def james_age_8_years_ago : ℕ := current_james_age - 8
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def current_janet_age : ℕ := janet_age_8_years_ago + 8

-- Problem: Prove that without Janet's age when Susan was born, we cannot determine Susan's age in 5 years.
theorem susan_age_indeterminate (susan_current_age : ℕ) : 
  (∃ janet_age_when_susan_born : ℕ, susan_current_age = current_janet_age - janet_age_when_susan_born) → 
  ¬ (∃ susan_age_in_5_years : ℕ, susan_age_in_5_years = susan_current_age + 5) := 
by
  sorry

end susan_age_indeterminate_l869_86917


namespace circumference_of_smaller_circle_l869_86932

variable (R : ℝ)
variable (A_shaded : ℝ)

theorem circumference_of_smaller_circle :
  (A_shaded = (32 / π) ∧ 3 * (π * R ^ 2) - π * R ^ 2 = A_shaded) → 
  2 * π * R = 4 :=
by
  sorry

end circumference_of_smaller_circle_l869_86932


namespace find_f_neg2_l869_86930

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem find_f_neg2 : f (-2) = -8 := by
  sorry

end find_f_neg2_l869_86930


namespace find_N_l869_86961

theorem find_N (a b c N : ℚ) (h_sum : a + b + c = 84)
    (h_a : a - 7 = N) (h_b : b + 7 = N) (h_c : c / 7 = N) : 
    N = 28 / 3 :=
sorry

end find_N_l869_86961


namespace largest_two_digit_integer_l869_86914

theorem largest_two_digit_integer
  (a b : ℕ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 3 * (10 * a + b) = 10 * b + a + 5) :
  10 * a + b = 13 :=
by {
  -- Sorry is placed here to indicate that the proof is not provided
  sorry
}

end largest_two_digit_integer_l869_86914


namespace brenda_skittles_l869_86919

theorem brenda_skittles (initial additional : ℕ) (h1 : initial = 7) (h2 : additional = 8) :
  initial + additional = 15 :=
by {
  -- Proof would go here
  sorry
}

end brenda_skittles_l869_86919


namespace max_min_value_of_f_l869_86901

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end max_min_value_of_f_l869_86901


namespace larger_number_l869_86922

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end larger_number_l869_86922


namespace rhombus_diagonal_solution_l869_86936

variable (d1 : ℝ) (A : ℝ)

def rhombus_other_diagonal (d1 d2 A : ℝ) : Prop :=
  A = (d1 * d2) / 2

theorem rhombus_diagonal_solution (h1 : d1 = 16) (h2 : A = 80) : rhombus_other_diagonal d1 10 A :=
by
  rw [h1, h2]
  sorry

end rhombus_diagonal_solution_l869_86936


namespace ratio_of_radii_l869_86941

namespace CylinderAndSphere

variable (r R : ℝ)
variable (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2)

theorem ratio_of_radii (r R : ℝ) (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2) :
    R / r = Real.sqrt 2 :=
by
  sorry

end CylinderAndSphere

end ratio_of_radii_l869_86941


namespace cos_pi_minus_2alpha_l869_86975

theorem cos_pi_minus_2alpha {α : ℝ} (h : Real.sin α = 2 / 3) : Real.cos (π - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l869_86975


namespace number_of_possible_scenarios_l869_86902

theorem number_of_possible_scenarios 
  (subjects : ℕ) 
  (students : ℕ) 
  (h_subjects : subjects = 4) 
  (h_students : students = 3) : 
  (subjects ^ students) = 64 := 
by
  -- Provide proof here
  sorry

end number_of_possible_scenarios_l869_86902


namespace KatieMarbles_l869_86920

variable {O P : ℕ}

theorem KatieMarbles :
  13 + O + P = 33 → P = 4 * O → 13 - O = 9 :=
by
  sorry

end KatieMarbles_l869_86920


namespace bucket_full_weight_l869_86978

theorem bucket_full_weight (x y c d : ℝ) 
  (h1 : x + (3/4) * y = c)
  (h2 : x + (3/5) * y = d) :
  x + y = (5/3) * c - (5/3) * d :=
by
  sorry

end bucket_full_weight_l869_86978


namespace fraction_to_decimal_l869_86980

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l869_86980


namespace f_1991_eq_1988_l869_86937

def f (n : ℕ) : ℕ := sorry

theorem f_1991_eq_1988 : f 1991 = 1988 :=
by sorry

end f_1991_eq_1988_l869_86937


namespace part1_part2_l869_86918

variable (x : ℝ)

def A : Set ℝ := { x | 2 * x + 1 < 5 }
def B : Set ℝ := { x | x^2 - x - 2 < 0 }

theorem part1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem part2 : A ∪ { x | x ≤ -1 ∨ x ≥ 2 } = Set.univ :=
sorry

end part1_part2_l869_86918


namespace distance_AB_l869_86979

def A : ℝ := -1
def B : ℝ := 2023

theorem distance_AB : |B - A| = 2024 := by
  sorry

end distance_AB_l869_86979


namespace nat_prime_p_and_5p_plus_1_is_prime_l869_86983

theorem nat_prime_p_and_5p_plus_1_is_prime (p : ℕ) (hp : Nat.Prime p) (h5p1 : Nat.Prime (5 * p + 1)) : p = 2 := 
by 
  -- Sorry is added to skip the proof
  sorry 

end nat_prime_p_and_5p_plus_1_is_prime_l869_86983


namespace union_M_N_l869_86966

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | x ≥ 1 }

theorem union_M_N : M ∪ N = { x | x > -1 } := 
by sorry

end union_M_N_l869_86966


namespace quadratic_inequality_solution_l869_86923

theorem quadratic_inequality_solution (a m : ℝ) (h : a < 0) :
  (∀ x : ℝ, ax^2 + 6*x - a^2 < 0 ↔ (x < 1 ∨ x > m)) → m = 2 :=
by
  sorry

end quadratic_inequality_solution_l869_86923


namespace time_for_trains_to_cross_l869_86972

def length_train1 := 500 -- 500 meters
def length_train2 := 750 -- 750 meters
def speed_train1 := 60 * 1000 / 3600 -- 60 km/hr to m/s
def speed_train2 := 40 * 1000 / 3600 -- 40 km/hr to m/s
def relative_speed := speed_train1 + speed_train2 -- relative speed in m/s
def combined_length := length_train1 + length_train2 -- sum of lengths of both trains

theorem time_for_trains_to_cross :
  (combined_length / relative_speed) = 45 := 
by
  sorry

end time_for_trains_to_cross_l869_86972


namespace acrobat_count_l869_86912

theorem acrobat_count (a e c : ℕ) (h1 : 2 * a + 4 * e + 2 * c = 88) (h2 : a + e + c = 30) : a = 2 :=
by
  sorry

end acrobat_count_l869_86912


namespace div_result_l869_86949

theorem div_result : 2.4 / 0.06 = 40 := 
sorry

end div_result_l869_86949


namespace train_speed_l869_86960

theorem train_speed (length : ℝ) (time_seconds : ℝ) (speed : ℝ) :
  length = 320 → time_seconds = 16 → speed = 72 :=
by 
  sorry

end train_speed_l869_86960


namespace unattainable_y_l869_86911

theorem unattainable_y (x : ℝ) (h : x ≠ -(5 / 4)) :
    (∀ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3 / 4) :=
by
  -- Placeholder for the proof
  sorry

end unattainable_y_l869_86911


namespace number_of_balls_greater_l869_86916

theorem number_of_balls_greater (n x : ℤ) (h1 : n = 25) (h2 : n - x = 30 - n) : x = 20 := by
  sorry

end number_of_balls_greater_l869_86916


namespace find_x_l869_86910

theorem find_x :
  ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 :=
by sorry

end find_x_l869_86910


namespace option_A_correct_l869_86954

theorem option_A_correct (y x : ℝ) : y * x - 2 * (x * y) = - (x * y) :=
by
  sorry

end option_A_correct_l869_86954


namespace quadratic_roots_l869_86990

theorem quadratic_roots (a b c : ℝ) :
  ∃ x y : ℝ, (x ≠ y ∧ (x^2 - (a + b) * x + (ab - c^2) = 0) ∧ (y^2 - (a + b) * y + (ab - c^2) = 0)) ∧
  (x = y ↔ a = b ∧ c = 0) := sorry

end quadratic_roots_l869_86990


namespace total_amount_after_5_months_l869_86969

-- Definitions from the conditions
def initial_deposit : ℝ := 100
def monthly_interest_rate : ℝ := 0.0036  -- 0.36% expressed as a decimal

-- Definition of the function relationship y with respect to x
def total_amount (x : ℕ) : ℝ := initial_deposit + initial_deposit * monthly_interest_rate * x

-- Prove the total amount after 5 months is 101.8
theorem total_amount_after_5_months : total_amount 5 = 101.8 :=
by
  sorry

end total_amount_after_5_months_l869_86969


namespace solve_fraction_equation_l869_86946

theorem solve_fraction_equation :
  ∀ (x : ℚ), (5 * x + 3) / (7 * x - 4) = 4128 / 4386 → x = 115 / 27 := by
  sorry

end solve_fraction_equation_l869_86946


namespace Mr_Caiden_payment_l869_86940

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end Mr_Caiden_payment_l869_86940


namespace no_real_roots_of_quadratic_eq_l869_86944

theorem no_real_roots_of_quadratic_eq (k : ℝ) (h : k < -1) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - k = 0 :=
by
  sorry

end no_real_roots_of_quadratic_eq_l869_86944


namespace simplify_fraction_l869_86908

variable {R : Type*} [Field R]
variables (x y z : R)

theorem simplify_fraction : (6 * x * y / (5 * z ^ 2)) * (10 * z ^ 3 / (9 * x * y)) = (4 * z) / 3 := by
  sorry

end simplify_fraction_l869_86908


namespace probability_of_three_cards_l869_86963

-- Conditions
def deck_size : ℕ := 52
def spades : ℕ := 13
def spades_face_cards : ℕ := 3
def face_cards : ℕ := 12
def diamonds : ℕ := 13

-- Probability of drawing specific cards
def prob_first_spade_non_face : ℚ := 10 / 52
def prob_second_face_given_first_spade_non_face : ℚ := 12 / 51
def prob_third_diamond_given_first_two : ℚ := 13 / 50

def prob_first_spade_face : ℚ := 3 / 52
def prob_second_face_given_first_spade_face : ℚ := 9 / 51

-- Final probability
def final_probability := 
  (prob_first_spade_non_face * prob_second_face_given_first_spade_non_face * prob_third_diamond_given_first_two) +
  (prob_first_spade_face * prob_second_face_given_first_spade_face * prob_third_diamond_given_first_two)

theorem probability_of_three_cards :
  final_probability = 1911 / 132600 := 
by
  sorry

end probability_of_three_cards_l869_86963


namespace students_left_during_year_l869_86977

theorem students_left_during_year (initial_students : ℕ) (new_students : ℕ) (final_students : ℕ) (students_left : ℕ) :
  initial_students = 4 →
  new_students = 42 →
  final_students = 43 →
  students_left = initial_students + new_students - final_students →
  students_left = 3 :=
by
  intro h_initial h_new h_final h_students_left
  rw [h_initial, h_new, h_final] at h_students_left
  exact h_students_left

end students_left_during_year_l869_86977


namespace proof_n_eq_neg2_l869_86904

theorem proof_n_eq_neg2 (n : ℤ) (h : |n + 6| = 2 - n) : n = -2 := 
by
  sorry

end proof_n_eq_neg2_l869_86904


namespace sufficient_condition_for_reciprocal_inequality_l869_86999

variable (a b : ℝ)

theorem sufficient_condition_for_reciprocal_inequality 
  (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 :=
sorry

end sufficient_condition_for_reciprocal_inequality_l869_86999


namespace four_digit_number_count_l869_86971

theorem four_digit_number_count :
  (∃ (n : ℕ), n ≥ 1000 ∧ n < 10000 ∧ 
    ((n / 1000 < 5 ∧ (n / 100) % 10 < 5) ∨ (n / 1000 > 5 ∧ (n / 100) % 10 > 5)) ∧ 
    (((n % 100) / 10 < 5 ∧ n % 10 < 5) ∨ ((n % 100) / 10 > 5 ∧ n % 10 > 5))) →
    ∃ (count : ℕ), count = 1681 :=
by
  sorry

end four_digit_number_count_l869_86971


namespace impossible_to_transport_stones_l869_86981

-- Define the conditions of the problem
def stones : List ℕ := List.range' 370 (468 - 370 + 2 + 1) 2
def truck_capacity : ℕ := 3000
def number_of_trucks : ℕ := 7
def number_of_stones : ℕ := 50

-- Prove that it is impossible to transport the stones using the given trucks
theorem impossible_to_transport_stones :
  stones.length = number_of_stones →
  (∀ weights ∈ stones.sublists, (weights.sum ≤ truck_capacity → List.length weights ≤ number_of_trucks)) → 
  false :=
by
  sorry

end impossible_to_transport_stones_l869_86981


namespace sum_of_numbers_l869_86943

theorem sum_of_numbers (x y : ℕ) (h1 : x = 18) (h2 : y = 2 * x - 3) : x + y = 51 :=
by
  sorry

end sum_of_numbers_l869_86943


namespace no_n_makes_g_multiple_of_5_and_7_l869_86909

def g (n : ℕ) : ℕ := 4 + 2 * n + 3 * n^2 + n^3 + 4 * n^4 + 3 * n^5

theorem no_n_makes_g_multiple_of_5_and_7 :
  ¬ ∃ n, (2 ≤ n ∧ n ≤ 100) ∧ (g n % 5 = 0 ∧ g n % 7 = 0) :=
by
  -- Proof goes here
  sorry

end no_n_makes_g_multiple_of_5_and_7_l869_86909


namespace sum_of_squares_eight_l869_86985

theorem sum_of_squares_eight (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := 
  sorry

end sum_of_squares_eight_l869_86985


namespace generalized_inequality_combinatorial_inequality_l869_86947

-- Part 1: Generalized Inequality
theorem generalized_inequality (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (Finset.univ.sum (fun i => (b i)^2 / (a i))) ≥
  ((Finset.univ.sum (fun i => b i))^2 / (Finset.univ.sum (fun i => a i))) :=
sorry

-- Part 2: Combinatorial Inequality
theorem combinatorial_inequality (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => (2 * k + 1) / (Nat.choose n k)) ≥
  ((n + 1)^3 / (2^n : ℝ)) :=
sorry

end generalized_inequality_combinatorial_inequality_l869_86947


namespace factorize_expr_l869_86945

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l869_86945
