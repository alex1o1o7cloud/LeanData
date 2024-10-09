import Mathlib

namespace f_at_47_l1211_121167

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_equation : ∀ x : ℝ, f (x - 1) + f (x + 1) = 0
axiom f_interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem f_at_47 : f 47 = -1 := by
  sorry

end f_at_47_l1211_121167


namespace determine_a_l1211_121174

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 - 2 * a * x + 1 

theorem determine_a (a : ℝ) (h : ¬ (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0)) : a > 1 :=
sorry

end determine_a_l1211_121174


namespace sum_of_fractions_eq_five_fourteen_l1211_121127

theorem sum_of_fractions_eq_five_fourteen :
  (1 : ℚ) / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) = 5 / 14 := 
by
  sorry

end sum_of_fractions_eq_five_fourteen_l1211_121127


namespace felix_brother_lifting_capacity_is_600_l1211_121176

-- Define the conditions
def felix_lifting_capacity (felix_weight : ℝ) : ℝ := 1.5 * felix_weight
def felix_brother_weight (felix_weight : ℝ) : ℝ := 2 * felix_weight
def felix_brother_lifting_capacity (brother_weight : ℝ) : ℝ := 3 * brother_weight
def felix_actual_lifting_capacity : ℝ := 150

-- Define the proof problem
theorem felix_brother_lifting_capacity_is_600 :
  ∃ felix_weight : ℝ,
    felix_lifting_capacity felix_weight = felix_actual_lifting_capacity ∧
    felix_brother_lifting_capacity (felix_brother_weight felix_weight) = 600 :=
by
  sorry

end felix_brother_lifting_capacity_is_600_l1211_121176


namespace volume_tetrahedron_375sqrt2_l1211_121131

noncomputable def tetrahedronVolume (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  let h_BCD := (2 * area_BCD) / BC
  let h_D_ABD := h_BCD * Real.sin angle_ABC_BCD
  (1 / 3) * area_ABC * h_D_ABD

theorem volume_tetrahedron_375sqrt2 :
  tetrahedronVolume 150 90 12 (Real.pi / 4) = 375 * Real.sqrt 2 := by
  sorry

end volume_tetrahedron_375sqrt2_l1211_121131


namespace buoy_min_force_l1211_121170

-- Define the problem in Lean
variables (M : ℝ) (ax : ℝ) (T_star : ℝ) (a : ℝ) (F_current : ℝ)
-- Conditions
variables (h_horizontal_component : T_star * Real.sin a = F_current)
          (h_zero_net_force : M * ax = 0)

theorem buoy_min_force (h_horizontal_component : T_star * Real.sin a = F_current) : 
  F_current = 400 := 
sorry

end buoy_min_force_l1211_121170


namespace winner_percentage_l1211_121124

theorem winner_percentage (votes_winner : ℕ) (votes_difference : ℕ) (total_votes : ℕ) 
  (h1 : votes_winner = 1044) 
  (h2 : votes_difference = 288) 
  (h3 : total_votes = votes_winner + (votes_winner - votes_difference)) :
  (votes_winner * 100) / total_votes = 58 :=
by
  sorry

end winner_percentage_l1211_121124


namespace planned_daily_catch_l1211_121181

theorem planned_daily_catch (x y : ℝ) 
  (h1 : x * y = 1800)
  (h2 : (x / 3) * (y - 20) + ((2 * x / 3) - 1) * (y + 20) = 1800) :
  y = 100 :=
by
  sorry

end planned_daily_catch_l1211_121181


namespace some_number_is_l1211_121138

theorem some_number_is (x some_number : ℤ) (h1 : x = 4) (h2 : 5 * x + 3 = 10 * x - some_number) : some_number = 17 := by
  sorry

end some_number_is_l1211_121138


namespace value_of_k_l1211_121177

noncomputable def find_k (x1 x2 : ℝ) (k : ℝ) : Prop :=
  (2 * x1^2 + k * x1 - 2 = 0) ∧ (2 * x2^2 + k * x2 - 2 = 0) ∧ ((x1 - 2) * (x2 - 2) = 10)

theorem value_of_k (x1 x2 : ℝ) (k : ℝ) (h : find_k x1 x2 k) : k = 7 :=
sorry

end value_of_k_l1211_121177


namespace radius_of_circle_l1211_121144

theorem radius_of_circle (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → (∃ r, r = 4) :=
by
  intro h
  sorry

end radius_of_circle_l1211_121144


namespace combinations_sum_l1211_121119
open Nat

theorem combinations_sum : 
  let d := [1, 2, 3, 4]
  let count_combinations (n : Nat) := factorial n
  count_combinations 1 + count_combinations 2 + count_combinations 3 + count_combinations 4 = 64 :=
  by
    sorry

end combinations_sum_l1211_121119


namespace ellipse_eq_l1211_121156

theorem ellipse_eq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 - b^2 = 4)
  (h4 : ∃ (line_eq : ℝ → ℝ), ∀ (x : ℝ), line_eq x = 3 * x + 7)
  (h5 : ∃ (mid_y : ℝ), mid_y = 1 ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    ((y1 = 3 * x1 + 7) ∧ (y2 = 3 * x2 + 7)) ∧ 
    (y1 + y2) / 2 = mid_y): 
  (∀ x y : ℝ, (y^2 / (a^2 - 4) + x^2 / b^2 = 1) ↔ 
  (x^2 / 8 + y^2 / 12 = 1)) :=
by { sorry }

end ellipse_eq_l1211_121156


namespace find_a_plus_b_l1211_121133

theorem find_a_plus_b (a b : ℝ) (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 0)
                      (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 16) : a + b = -16 :=
by sorry

end find_a_plus_b_l1211_121133


namespace cost_price_of_computer_table_l1211_121158

theorem cost_price_of_computer_table (S : ℝ) (C : ℝ) (h1 : 1.80 * C = S) (h2 : S = 3500) : C = 1944.44 :=
by
  sorry

end cost_price_of_computer_table_l1211_121158


namespace cube_difference_l1211_121115

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 :=
sorry

end cube_difference_l1211_121115


namespace equal_expense_sharing_l1211_121103

variables (O L B : ℝ)

theorem equal_expense_sharing (h1 : O < L) (h2 : O < B) : 
    (L + B - 2 * O) / 6 = (O + L + B) / 3 - O :=
by
    sorry

end equal_expense_sharing_l1211_121103


namespace find_sum_A_B_l1211_121195

-- Define ω as a root of the polynomial x^2 + x + 1
noncomputable def ω : ℂ := sorry

-- Define the polynomial P
noncomputable def P (x : ℂ) (A B : ℂ) : ℂ := x^101 + A * x + B

-- State the main theorem
theorem find_sum_A_B (A B : ℂ) : 
  (∀ x : ℂ, (x^2 + x + 1 = 0) → P x A B = 0) → A + B = 2 :=
by
  intros Divisibility
  -- Here, you would provide the steps to prove the theorem if necessary
  sorry

end find_sum_A_B_l1211_121195


namespace average_mark_of_first_class_is_40_l1211_121185

open Classical

noncomputable def average_mark_first_class (n1 n2 : ℕ) (m2 : ℕ) (a : ℚ) : ℚ :=
  let x := (a * (n1 + n2) - n2 * m2) / n1
  x

theorem average_mark_of_first_class_is_40 : average_mark_first_class 30 50 90 71.25 = 40 := by
  sorry

end average_mark_of_first_class_is_40_l1211_121185


namespace doughnut_cost_l1211_121180

theorem doughnut_cost:
  ∃ (D C : ℝ), 
    3 * D + 4 * C = 4.91 ∧ 
    5 * D + 6 * C = 7.59 ∧ 
    D = 0.45 :=
by
  sorry

end doughnut_cost_l1211_121180


namespace matrix_det_is_neg16_l1211_121118

def matrix := Matrix (Fin 2) (Fin 2) ℤ
def given_matrix : matrix := ![![ -7, 5], ![6, -2]]

theorem matrix_det_is_neg16 : Matrix.det given_matrix = -16 := 
by
  sorry

end matrix_det_is_neg16_l1211_121118


namespace superhero_speed_l1211_121100

def convert_speed (speed_mph : ℕ) (mile_to_km : ℚ) : ℚ :=
  let speed_kmh := (speed_mph : ℚ) * (1 / mile_to_km)
  speed_kmh / 60

theorem superhero_speed :
  convert_speed 36000 (6 / 10) = 1000 :=
by sorry

end superhero_speed_l1211_121100


namespace quadratic_discriminant_correct_l1211_121183

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant_correct :
  discriminant 5 (5 + 1/2) (-1/2) = 161 / 4 :=
by
  -- let's prove the equality directly
  sorry

end quadratic_discriminant_correct_l1211_121183


namespace smallest_b_theorem_l1211_121162

open Real

noncomputable def smallest_b (a b c: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) : Prop :=
  b = 5

theorem smallest_b_theorem (a b c: ℝ) (r: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) :
  smallest_b a b c h1 h2 h3 h4 :=
by {
  sorry
}

end smallest_b_theorem_l1211_121162


namespace tank_capacity_l1211_121154

theorem tank_capacity (T : ℕ) (h1 : T > 0) 
    (h2 : (2 * T) / 5 + 15 + 20 = T - 25) : 
    T = 100 := 
  by 
    sorry

end tank_capacity_l1211_121154


namespace domain_of_function_l1211_121194

theorem domain_of_function :
  {x : ℝ | -3 < x ∧ x < 2 ∧ x ≠ 1} = {x : ℝ | (2 - x > 0) ∧ (12 + x - x^2 ≥ 0) ∧ (x ≠ 1)} :=
by
  sorry

end domain_of_function_l1211_121194


namespace rational_solutions_exist_l1211_121108

theorem rational_solutions_exist (x p q : ℚ) (h : p^2 - x * q^2 = 1) :
  ∃ (a b : ℤ), p = (a^2 + x * b^2) / (a^2 - x * b^2) ∧ q = (2 * a * b) / (a^2 - x * b^2) :=
by
  sorry

end rational_solutions_exist_l1211_121108


namespace car_distance_l1211_121160

noncomputable def distance_covered (S : ℝ) (T : ℝ) (new_speed : ℝ) : ℝ :=
  S * T

theorem car_distance (S : ℝ) (T : ℝ) (new_time : ℝ) (new_speed : ℝ)
  (h1 : T = 12)
  (h2 : new_time = (3/4) * T)
  (h3 : new_speed = 60)
  (h4 : distance_covered new_speed new_time = 540) :
    distance_covered S T = 540 :=
by
  sorry

end car_distance_l1211_121160


namespace interval_contains_integer_l1211_121172

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, (3 * a < n) ∧ (n < 5 * a - 2)) ↔ (1.2 < a ∧ a < 4 / 3) ∨ (7 / 5 < a) :=
by sorry

end interval_contains_integer_l1211_121172


namespace bananas_proof_l1211_121148

noncomputable def number_of_bananas (total_oranges : ℕ) (total_fruits_percent_good : ℝ) 
  (percent_rotten_oranges : ℝ) (percent_rotten_bananas : ℝ) : ℕ := 448

theorem bananas_proof :
  let total_oranges := 600
  let percent_rotten_oranges := 0.15
  let percent_rotten_bananas := 0.08
  let total_fruits_percent_good := 0.878
  
  number_of_bananas total_oranges total_fruits_percent_good percent_rotten_oranges percent_rotten_bananas = 448 :=
by
  sorry

end bananas_proof_l1211_121148


namespace john_money_left_l1211_121163

-- Given definitions
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := q
def large_pizza_cost (q : ℝ) := 4 * q
def initial_amount := 50

-- Problem statement
theorem john_money_left (q : ℝ) : initial_amount - (4 * drink_cost q + 2 * small_pizza_cost q + large_pizza_cost q) = 50 - 10 * q :=
by
  sorry

end john_money_left_l1211_121163


namespace problem_part_I_problem_part_II_l1211_121159

-- Define the problem and the proof requirements in Lean 4
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (sinB_nonneg : 0 ≤ Real.sin B) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) 
(h_a : a = 2) (h_b : b = 2) : 
Real.cos B = 1/4 :=
sorry

theorem problem_part_II (a b c : ℝ) (A B C : ℝ) (h_B : B = π / 2) 
(h_a : a = Real.sqrt 2) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) :
1/2 * a * c = 1 :=
sorry

end problem_part_I_problem_part_II_l1211_121159


namespace divisible_by_pow3_l1211_121142

-- Define the digit sequence function
def num_with_digits (a n : Nat) : Nat :=
  a * ((10 ^ (3 ^ n) - 1) / 9)

-- Main theorem statement
theorem divisible_by_pow3 (a n : Nat) (h_pos : 0 < n) : (num_with_digits a n) % (3 ^ n) = 0 := 
by
  sorry

end divisible_by_pow3_l1211_121142


namespace calculate_1307_squared_l1211_121190

theorem calculate_1307_squared : 1307 * 1307 = 1709849 := sorry

end calculate_1307_squared_l1211_121190


namespace quadratic_has_two_equal_real_roots_l1211_121132

theorem quadratic_has_two_equal_real_roots : ∃ c : ℝ, ∀ x : ℝ, (x^2 - 6*x + c = 0 ↔ (x = 3)) :=
by
  sorry

end quadratic_has_two_equal_real_roots_l1211_121132


namespace circle_area_l1211_121178

theorem circle_area (r : ℝ) (h : 5 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 5 := 
by
  sorry -- Proof is not required, placeholder for the actual proof

end circle_area_l1211_121178


namespace walking_speed_is_correct_l1211_121145

-- Define the conditions
def time_in_minutes : ℝ := 10
def distance_in_meters : ℝ := 1666.6666666666665
def speed_in_km_per_hr : ℝ := 2.777777777777775

-- Define the theorem to prove
theorem walking_speed_is_correct :
  (distance_in_meters / time_in_minutes) * 60 / 1000 = speed_in_km_per_hr :=
sorry

end walking_speed_is_correct_l1211_121145


namespace fraction_simplify_l1211_121136

theorem fraction_simplify :
  (3 + 9 - 27 + 81 - 243 + 729) / (9 + 27 - 81 + 243 - 729 + 2187) = 1 / 3 :=
by
  sorry

end fraction_simplify_l1211_121136


namespace find_k_l1211_121120

theorem find_k (k : ℝ) (h_line : ∀ x y : ℝ, 3 * x + 5 * y + k = 0)
    (h_sum_intercepts : - (k / 3) - (k / 5) = 16) : k = -30 := by
  sorry

end find_k_l1211_121120


namespace polynomial_value_l1211_121192
variable {x y : ℝ}
theorem polynomial_value (h : 3 * x^2 + 4 * y + 9 = 8) : 9 * x^2 + 12 * y + 8 = 5 :=
by
   sorry

end polynomial_value_l1211_121192


namespace additional_grassy_area_l1211_121147

theorem additional_grassy_area (r1 r2 : ℝ) (r1_pos : r1 = 10) (r2_pos : r2 = 35) : 
  let A1 := π * r1^2
  let A2 := π * r2^2
  (A2 - A1) = 1125 * π :=
by 
  sorry

end additional_grassy_area_l1211_121147


namespace actual_distance_between_towns_l1211_121197

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns_l1211_121197


namespace count_invitations_l1211_121161

theorem count_invitations (teachers : Finset ℕ) (A B : ℕ) (hA : A ∈ teachers) (hB : B ∈ teachers) (h_size : teachers.card = 10):
  ∃ (ways : ℕ), ways = 140 ∧ ∀ (S : Finset ℕ), S.card = 6 → ((A ∈ S ∧ B ∉ S) ∨ (A ∉ S ∧ B ∈ S) ∨ (A ∉ S ∧ B ∉ S)) ↔ ways = 140 := 
sorry

end count_invitations_l1211_121161


namespace exists_cube_number_divisible_by_six_in_range_l1211_121165

theorem exists_cube_number_divisible_by_six_in_range :
  ∃ (y : ℕ), y > 50 ∧ y < 350 ∧ (∃ (n : ℕ), y = n^3) ∧ y % 6 = 0 :=
by 
  use 216
  sorry

end exists_cube_number_divisible_by_six_in_range_l1211_121165


namespace intersection_S_T_eq_T_l1211_121143

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l1211_121143


namespace team_incorrect_answers_l1211_121129

theorem team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) :
  total_questions = 35 → riley_mistakes = 3 → 
  ofelia_correct = ((total_questions - riley_mistakes) / 2 + 5) → 
  riley_mistakes + (total_questions - ofelia_correct) = 17 :=
by
  intro h1 h2 h3
  sorry

end team_incorrect_answers_l1211_121129


namespace cookie_distribution_l1211_121188

theorem cookie_distribution : 
  ∀ (n c T : ℕ), n = 6 → c = 4 → T = n * c → T = 24 :=
by 
  intros n c T h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cookie_distribution_l1211_121188


namespace simplify_evaluate_expr_l1211_121106

theorem simplify_evaluate_expr (x y : ℚ) (h₁ : x = -1) (h₂ : y = -1 / 2) :
  (4 * x * y + (2 * x^2 + 5 * x * y - y^2) - 2 * (x^2 + 3 * x * y)) = 5 / 4 :=
by
  rw [h₁, h₂]
  -- Here we would include the specific algebra steps to convert the LHS to 5/4.
  sorry

end simplify_evaluate_expr_l1211_121106


namespace hemisphere_containers_needed_l1211_121111

theorem hemisphere_containers_needed 
  (total_volume : ℕ) (volume_per_hemisphere : ℕ) 
  (h₁ : total_volume = 11780) 
  (h₂ : volume_per_hemisphere = 4) : 
  total_volume / volume_per_hemisphere = 2945 := 
by
  sorry

end hemisphere_containers_needed_l1211_121111


namespace sum_a_b_eq_34_over_3_l1211_121134

theorem sum_a_b_eq_34_over_3 (a b: ℚ)
  (h1 : 2 * a + 5 * b = 43)
  (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 :=
sorry

end sum_a_b_eq_34_over_3_l1211_121134


namespace set_intersection_eq_l1211_121196

theorem set_intersection_eq (M N : Set ℝ) (hM : M = { x : ℝ | 0 < x ∧ x < 1 }) (hN : N = { x : ℝ | -2 < x ∧ x < 2 }) :
  M ∩ N = M :=
sorry

end set_intersection_eq_l1211_121196


namespace relationship_a_b_l1211_121146

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem relationship_a_b (a b : ℝ) (h_pos : a > 0) :
  (∀ x : ℝ, a * x + b = g x) → (2 * a < b ∧ b < (a + 1)^2 / 4 + 2 ∧ 0 < a ∧ a < 3) :=
sorry

end relationship_a_b_l1211_121146


namespace trig_identity_proof_l1211_121149

theorem trig_identity_proof
  (h1: Float.sin 50 = Float.cos 40)
  (h2: Float.tan 45 = 1)
  (h3: Float.tan 10 = Float.sin 10 / Float.cos 10)
  (h4: Float.sin 80 = Float.cos 10) :
  Float.sin 50 * (Float.tan 45 + Float.sqrt 3 * Float.tan 10) = 1 :=
by
  sorry

end trig_identity_proof_l1211_121149


namespace total_area_of_room_l1211_121171

theorem total_area_of_room : 
  let length_rect := 8 
  let width_rect := 6 
  let base_triangle := 6 
  let height_triangle := 3 
  let area_rect := length_rect * width_rect 
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle 
  let total_area := area_rect + area_triangle 
  total_area = 57 := 
by 
  sorry

end total_area_of_room_l1211_121171


namespace correct_average_is_19_l1211_121140

-- Definitions
def incorrect_avg : ℕ := 16
def num_values : ℕ := 10
def incorrect_reading : ℕ := 25
def correct_reading : ℕ := 55

-- Theorem to prove
theorem correct_average_is_19 :
  ((incorrect_avg * num_values - incorrect_reading + correct_reading) / num_values) = 19 :=
by
  sorry

end correct_average_is_19_l1211_121140


namespace Ron_needs_to_drink_80_percent_l1211_121116

theorem Ron_needs_to_drink_80_percent 
  (volume_each : ℕ)
  (volume_intelligence : ℕ)
  (volume_beauty : ℕ)
  (volume_strength : ℕ)
  (volume_second_pitcher : ℕ)
  (effective_volume : ℕ)
  (volume_intelligence_left : ℕ)
  (volume_beauty_left : ℕ)
  (volume_strength_left : ℕ)
  (total_volume : ℕ)
  (Ron_needs : ℕ)
  (intelligence_condition : effective_volume = 30)
  (initial_volumes : volume_each = 300)
  (first_drink : volume_intelligence = volume_each / 2)
  (mix_before_second_drink : volume_second_pitcher = volume_intelligence + volume_beauty)
  (Hermione_drink : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left)
  (Harry_drink : volume_strength_left = volume_each / 2)
  (second_mix : volume_second_pitcher = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (final_mix : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (Ron_needs_condition : Ron_needs = effective_volume / volume_intelligence_left * 100)
  : Ron_needs = 80 := sorry

end Ron_needs_to_drink_80_percent_l1211_121116


namespace find_a22_l1211_121155

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l1211_121155


namespace positive_integer_with_four_smallest_divisors_is_130_l1211_121110

theorem positive_integer_with_four_smallest_divisors_is_130:
  ∃ n : ℕ, ∀ p1 p2 p3 p4 : ℕ, 
    n = p1^2 + p2^2 + p3^2 + p4^2 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
    ∀ p : ℕ, p ∣ n → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4) → 
    n = 130 :=
  by
  sorry

end positive_integer_with_four_smallest_divisors_is_130_l1211_121110


namespace acute_triangle_inequality_l1211_121164

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C) ≤
    Real.pi * (1 / A + 1 / B + 1 / C) :=
sorry

end acute_triangle_inequality_l1211_121164


namespace intersection_of_circle_and_line_l1211_121125

theorem intersection_of_circle_and_line 
  (α : ℝ) 
  (x y : ℝ)
  (h1 : x = Real.cos α) 
  (h2 : y = 1 + Real.sin α) 
  (h3 : y = 1) :
  (x, y) = (1, 1) :=
by
  sorry

end intersection_of_circle_and_line_l1211_121125


namespace smallest_number_groups_l1211_121187

theorem smallest_number_groups (x : ℕ) (h₁ : x % 18 = 0) (h₂ : x % 45 = 0) : x = 90 :=
sorry

end smallest_number_groups_l1211_121187


namespace max_value_2ab_plus_2ac_sqrt3_l1211_121189

variable (a b c : ℝ)
variable (h1 : a^2 + b^2 + c^2 = 1)
variable (h2 : 0 ≤ a)
variable (h3 : 0 ≤ b)
variable (h4 : 0 ≤ c)

theorem max_value_2ab_plus_2ac_sqrt3 : 2 * a * b + 2 * a * c * Real.sqrt 3 ≤ 1 := by
  sorry

end max_value_2ab_plus_2ac_sqrt3_l1211_121189


namespace problem_solution_l1211_121130

variables (p q : Prop)

theorem problem_solution (h1 : ¬ (p ∧ q)) (h2 : p ∨ q) : ¬ p ∨ ¬ q := by
  sorry

end problem_solution_l1211_121130


namespace range_of_a_l1211_121105

def P (x : ℝ) : Prop := x^2 ≤ 1

def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : ∀ x, (P x ∨ x = a) ↔ P x) : P a :=
by
  sorry

end range_of_a_l1211_121105


namespace greg_rolls_more_ones_than_fives_l1211_121152

def probability_more_ones_than_fives (n : ℕ) : ℚ :=
  if n = 6 then 695 / 1944 else 0

theorem greg_rolls_more_ones_than_fives :
  probability_more_ones_than_fives 6 = 695 / 1944 :=
by sorry

end greg_rolls_more_ones_than_fives_l1211_121152


namespace resulting_curve_eq_l1211_121112

def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

def transformed_curve (x y: ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), 
    is_on_circle x0 y0 ∧ 
    x = x0 ∧ 
    y = 4 * y0

theorem resulting_curve_eq : ∀ (x y : ℝ), transformed_curve x y → (x^2 / 9 + y^2 / 144 = 1) :=
by
  intros x y h
  sorry

end resulting_curve_eq_l1211_121112


namespace greatest_n_4022_l1211_121193

noncomputable def arithmetic_sequence_greatest_n 
  (a : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (cond1 : a 2011 + a 2012 > 0)
  (cond2 : a 2011 * a 2012 < 0) : ℕ :=
  4022

theorem greatest_n_4022 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h1 : a 1 > 0)
  (h2 : a 2011 + a 2012 > 0)
  (h3 : a 2011 * a 2012 < 0):
  arithmetic_sequence_greatest_n a h1 h2 h3 = 4022 :=
sorry

end greatest_n_4022_l1211_121193


namespace probability_ace_king_queen_same_suit_l1211_121104

theorem probability_ace_king_queen_same_suit :
  let total_probability := (1 : ℝ) / 52 * (1 : ℝ) / 51 * (1 : ℝ) / 50
  total_probability = (1 : ℝ) / 132600 :=
by
  sorry

end probability_ace_king_queen_same_suit_l1211_121104


namespace opposite_numbers_abs_l1211_121182

theorem opposite_numbers_abs (a b : ℤ) (h : a + b = 0) : |a - 2014 + b| = 2014 :=
by
  -- proof here
  sorry

end opposite_numbers_abs_l1211_121182


namespace division_problem_solution_l1211_121109

theorem division_problem_solution (x : ℝ) (h : (2.25 / x) * 12 = 9) : x = 3 :=
sorry

end division_problem_solution_l1211_121109


namespace ratio_blue_gill_to_bass_l1211_121107

theorem ratio_blue_gill_to_bass (bass trout blue_gill : ℕ) 
  (h1 : bass = 32)
  (h2 : trout = bass / 4)
  (h3 : bass + trout + blue_gill = 104) 
: blue_gill / bass = 2 := 
sorry

end ratio_blue_gill_to_bass_l1211_121107


namespace Lizzie_group_number_l1211_121169

theorem Lizzie_group_number (x : ℕ) (h1 : x + (x + 17) = 91) : x + 17 = 54 :=
by
  sorry

end Lizzie_group_number_l1211_121169


namespace solution_l1211_121186

-- Define the conditions
def equation (x : ℝ) : Prop :=
  (x / 15) = (15 / x)

theorem solution (x : ℝ) : equation x → x = 15 ∨ x = -15 :=
by
  intros h
  -- The proof would go here.
  sorry

end solution_l1211_121186


namespace possible_values_of_sum_l1211_121121

theorem possible_values_of_sum
  (p q r : ℝ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_system : q = p * (4 - p) ∧ r = q * (4 - q) ∧ p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end possible_values_of_sum_l1211_121121


namespace base9_num_digits_2500_l1211_121150

theorem base9_num_digits_2500 : 
  ∀ (n : ℕ), (9^1 = 9) → (9^2 = 81) → (9^3 = 729) → (9^4 = 6561) → n = 4 := by
  sorry

end base9_num_digits_2500_l1211_121150


namespace negation_equivalent_statement_l1211_121117

theorem negation_equivalent_statement (x y : ℝ) :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
sorry

end negation_equivalent_statement_l1211_121117


namespace entrance_ticket_cost_l1211_121199

theorem entrance_ticket_cost
  (students teachers : ℕ)
  (total_cost : ℕ)
  (students_count : students = 20)
  (teachers_count : teachers = 3)
  (cost : total_cost = 115) :
  total_cost / (students + teachers) = 5 := by
  sorry

end entrance_ticket_cost_l1211_121199


namespace area_of_square_l1211_121139

theorem area_of_square 
  (a : ℝ)
  (h : 4 * a = 28) :
  a^2 = 49 :=
sorry

end area_of_square_l1211_121139


namespace prove_ineq_l1211_121123

-- Define the quadratic equation
def quadratic_eqn (a b x : ℝ) : Prop :=
  3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0

-- Define the root relation
def root_relation (x1 x2 : ℝ) : Prop :=
  x1 * (x1 + 1) + x2 * (x2 + 1) = (x1 + 1) * (x2 + 1)

-- State the theorem
theorem prove_ineq (a b : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eqn a b x1 ∧ quadratic_eqn a b x2 ∧ root_relation x1 x2) →
  (a + b)^2 ≤ 4 :=
by
  sorry

end prove_ineq_l1211_121123


namespace henri_total_miles_l1211_121157

noncomputable def g_total : ℕ := 315 * 3
noncomputable def h_total : ℕ := g_total + 305

theorem henri_total_miles : h_total = 1250 :=
by
  -- proof goes here
  sorry

end henri_total_miles_l1211_121157


namespace company_ordered_weight_of_stone_l1211_121114

theorem company_ordered_weight_of_stone :
  let weight_concrete := 0.16666666666666666
  let weight_bricks := 0.16666666666666666
  let total_material := 0.8333333333333334
  let weight_stone := total_material - (weight_concrete + weight_bricks)
  weight_stone = 0.5 :=
by
  sorry

end company_ordered_weight_of_stone_l1211_121114


namespace youngest_sibling_age_l1211_121175

theorem youngest_sibling_age
  (Y : ℕ)
  (h1 : Y + (Y + 3) + (Y + 6) + (Y + 7) = 120) :
  Y = 26 :=
by
  -- proof steps would be here 
  sorry

end youngest_sibling_age_l1211_121175


namespace find_triples_l1211_121128

theorem find_triples (a b c : ℝ) : 
  a + b + c = 14 ∧ a^2 + b^2 + c^2 = 84 ∧ a^3 + b^3 + c^3 = 584 ↔ (a = 4 ∧ b = 2 ∧ c = 8) ∨ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 4) :=
by
  sorry

end find_triples_l1211_121128


namespace domain_of_f_2x_minus_1_l1211_121153

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  ∀ x, (1 / 2) ≤ x ∧ x ≤ (3 / 2) → ∃ y, f y = (2 * x - 1) :=
by
  intros h x hx
  sorry

end domain_of_f_2x_minus_1_l1211_121153


namespace two_abs_inequality_l1211_121184

theorem two_abs_inequality (x y : ℝ) :
  2 * abs (x + y) ≤ abs x + abs y ↔ 
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -x / 3) ∨ 
  (x < 0 ∧ -x / 3 ≤ y ∧ y ≤ -3 * x) :=
by
  sorry

end two_abs_inequality_l1211_121184


namespace bins_of_vegetables_l1211_121166

-- Define the conditions
def total_bins : ℝ := 0.75
def bins_of_soup : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

-- Define the statement to be proved
theorem bins_of_vegetables :
  total_bins = bins_of_soup + (0.13) + bins_of_pasta := 
sorry

end bins_of_vegetables_l1211_121166


namespace uncolored_area_of_rectangle_l1211_121135

theorem uncolored_area_of_rectangle :
  let width := 30
  let length := 50
  let radius := width / 4
  let rectangle_area := width * length
  let circle_area := π * (radius ^ 2)
  let total_circles_area := 4 * circle_area
  rectangle_area - total_circles_area = 1500 - 225 * π := by
  sorry

end uncolored_area_of_rectangle_l1211_121135


namespace perpendicular_vectors_x_l1211_121191

theorem perpendicular_vectors_x 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, -2))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : 
  x = 4 := 
  by 
  sorry

end perpendicular_vectors_x_l1211_121191


namespace inequality_division_l1211_121113

variable (m n : ℝ)

theorem inequality_division (h : m > n) : (m / 4) > (n / 4) :=
sorry

end inequality_division_l1211_121113


namespace oranges_left_to_sell_today_l1211_121151

theorem oranges_left_to_sell_today (initial_dozen : Nat)
    (reserved_fraction1 reserved_fraction2 sold_fraction eaten_fraction : ℚ)
    (rotten_oranges : Nat) 
    (h1 : initial_dozen = 7)
    (h2 : reserved_fraction1 = 1/4)
    (h3 : reserved_fraction2 = 1/6)
    (h4 : sold_fraction = 3/7)
    (h5 : eaten_fraction = 1/10)
    (h6 : rotten_oranges = 4) : 
    let total_oranges := initial_dozen * 12
    let reserved1 := total_oranges * reserved_fraction1
    let reserved2 := total_oranges * reserved_fraction2
    let remaining_after_reservation := total_oranges - reserved1 - reserved2
    let sold_yesterday := remaining_after_reservation * sold_fraction
    let remaining_after_sale := remaining_after_reservation - sold_yesterday
    let eaten_by_birds := remaining_after_sale * eaten_fraction
    let remaining_after_birds := remaining_after_sale - eaten_by_birds
    let final_remaining := remaining_after_birds - rotten_oranges
    final_remaining = 22 :=
by
    sorry

end oranges_left_to_sell_today_l1211_121151


namespace find_a_and_tangent_point_l1211_121126

noncomputable def tangent_line_and_curve (a : ℚ) (P : ℚ × ℚ) : Prop :=
  ∃ (x₀ : ℚ), (P = (x₀, x₀ + a)) ∧ (P = (x₀, x₀^3 - x₀^2 + 1)) ∧ (3*x₀^2 - 2*x₀ = 1)

theorem find_a_and_tangent_point :
  ∃ (a : ℚ) (P : ℚ × ℚ), tangent_line_and_curve a P ∧ a = 32/27 ∧ P = (-1/3, 23/27) :=
sorry

end find_a_and_tangent_point_l1211_121126


namespace area_of_T_l1211_121168

open Complex Real

noncomputable def omega := -1 / 2 + (1 / 2) * Complex.I * Real.sqrt 3
noncomputable def omega2 := -1 / 2 - (1 / 2) * Complex.I * Real.sqrt 3

def inT (z : ℂ) (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 ∧
  0 ≤ b ∧ b ≤ 1 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  z = a + b * omega + c * omega2

theorem area_of_T : ∃ A : ℝ, A = 2 * Real.sqrt 3 :=
sorry

end area_of_T_l1211_121168


namespace symmetric_points_l1211_121141

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l1211_121141


namespace probability_at_least_one_blue_l1211_121198

-- Definitions of the setup
def red_balls := 2
def blue_balls := 2
def total_balls := red_balls + blue_balls
def total_outcomes := (total_balls * (total_balls - 1)) / 2  -- choose 2 out of total
def favorable_outcomes := 10  -- by counting outcomes with at least one blue ball

-- Definition of the proof problem
theorem probability_at_least_one_blue (a b : ℕ) (h1: a = red_balls) (h2: b = blue_balls) :
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry  

end probability_at_least_one_blue_l1211_121198


namespace lucky_larry_l1211_121102

theorem lucky_larry (a b c d e k : ℤ) 
    (h1 : a = 2) 
    (h2 : b = 3) 
    (h3 : c = 4) 
    (h4 : d = 5)
    (h5 : a - b - c - d + e = 2 - (b - (c - (d + e)))) 
    (h6 : k * 2 = e) : 
    k = 2 := by
  sorry

end lucky_larry_l1211_121102


namespace mul_exponent_property_l1211_121179

variable (m : ℕ)  -- Assuming m is a natural number for simplicity

theorem mul_exponent_property : m^2 * m^3 = m^5 := 
by {
  sorry
}

end mul_exponent_property_l1211_121179


namespace find_xyz_l1211_121101

variables (A B C B₁ A₁ C₁ : Type)
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B] [AddCommGroup C] [Module ℝ C]

def AC1 (AB BC CC₁ : A) (x y z : ℝ) : A :=
  x • AB + 2 • y • BC + 3 • z • CC₁

theorem find_xyz (AB BC CC₁ AC1 : A)
  (h1 : AC1 = AB + BC + CC₁)
  (h2 : AC1 = x • AB + 2 • y • BC + 3 • z • CC₁) :
  x + y + z = 11 / 6 :=
sorry

end find_xyz_l1211_121101


namespace park_area_calculation_l1211_121137

noncomputable def width_of_park := Real.sqrt (9000000 / 65)
noncomputable def length_of_park := 8 * width_of_park

def actual_area_of_park (w l : ℝ) : ℝ := w * l

theorem park_area_calculation :
  let w := width_of_park
  let l := length_of_park
  actual_area_of_park w l = 1107746.48 :=
by
  -- Calculations from solution are provided here directly as conditions and definitions
  sorry

end park_area_calculation_l1211_121137


namespace mean_height_basketball_team_l1211_121173

def heights : List ℕ :=
  [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

def mean_height (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_height_basketball_team :
  mean_height heights = 70 := by
  sorry

end mean_height_basketball_team_l1211_121173


namespace mean_temperature_l1211_121122

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_temperature (temps : List ℝ) (length_temps_10 : temps.length = 10)
    (temps_vals : temps = [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]) : 
    mean temps = 88.2 := by
  sorry

end mean_temperature_l1211_121122
