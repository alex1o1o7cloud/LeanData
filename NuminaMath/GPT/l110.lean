import Mathlib

namespace intersection_complement_eq_three_l110_11067

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_eq_three : N ∩ (U \ M) = {3} := by
  sorry

end intersection_complement_eq_three_l110_11067


namespace first_person_days_l110_11034

theorem first_person_days (x : ℝ) (hp : 30 ≥ 0) (ht : 10 ≥ 0) (h_work : 1/x + 1/30 = 1/10) : x = 15 :=
by
  -- Begin by acknowledging the assumptions: hp, ht, and h_work
  sorry

end first_person_days_l110_11034


namespace problem_statement_l110_11075

theorem problem_statement : 2017 - (1 / 2017) = (2018 * 2016) / 2017 :=
by
  sorry

end problem_statement_l110_11075


namespace units_digit_of_expression_l110_11007

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_expression : units_digit (7 * 18 * 1978 - 7^4) = 7 := by
  sorry

end units_digit_of_expression_l110_11007


namespace find_width_of_metallic_sheet_l110_11015

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l110_11015


namespace four_digit_number_l110_11019

-- Defining the cards and their holders
def cards : List ℕ := [2, 0, 1, 5]
def A : ℕ := 5
def B : ℕ := 1
def C : ℕ := 2
def D : ℕ := 0

-- Conditions based on statements
def A_statement (a b c d : ℕ) : Prop := 
  ¬ ((b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1))

def B_statement (a b c d : ℕ) : Prop := 
  (b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1)

def C_statement (c : ℕ) : Prop := ¬ (c = 1 ∨ c = 2 ∨ c = 5)
def D_statement (d : ℕ) : Prop := d ≠ 0

-- Truth conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def tells_truth (n : ℕ) : Prop := is_odd n
def lies (n : ℕ) : Prop := is_even n

-- Proof statement
theorem four_digit_number (a b c d : ℕ) 
  (ha : a ∈ cards) (hb : b ∈ cards) (hc : c ∈ cards) (hd : d ∈ cards) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (truth_A : tells_truth a → A_statement a b c d)
  (lie_A : lies a → ¬ A_statement a b c d)
  (truth_B : tells_truth b → B_statement a b c d)
  (lie_B : lies b → ¬ B_statement a b c d)
  (truth_C : tells_truth c → C_statement c)
  (lie_C : lies c → ¬ C_statement c)
  (truth_D : tells_truth d → D_statement d)
  (lie_D : lies d → ¬ D_statement d) :
  a * 1000 + b * 100 + c * 10 + d = 5120 := 
  by
    sorry

end four_digit_number_l110_11019


namespace height_of_smaller_cone_removed_l110_11081

noncomputable def frustum_area_lower_base : ℝ := 196 * Real.pi
noncomputable def frustum_area_upper_base : ℝ := 16 * Real.pi
def frustum_height : ℝ := 30

theorem height_of_smaller_cone_removed (r1 r2 H : ℝ)
  (h1 : r1 = Real.sqrt (frustum_area_lower_base / Real.pi))
  (h2 : r2 = Real.sqrt (frustum_area_upper_base / Real.pi))
  (h3 : r2 / r1 = 2 / 7)
  (h4 : frustum_height = (5 / 7) * H) :
  H - frustum_height = 12 :=
by 
  sorry

end height_of_smaller_cone_removed_l110_11081


namespace extracurricular_books_counts_l110_11096

theorem extracurricular_books_counts 
  (a b c d : ℕ)
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by
  sorry

end extracurricular_books_counts_l110_11096


namespace min_value_of_expression_l110_11016

theorem min_value_of_expression (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ min_value, min_value = 9 / 2 ∧ ∀ z, z = (1 / (x + 1) + 4 / y) → z ≥ min_value :=
sorry

end min_value_of_expression_l110_11016


namespace tom_helicopter_hours_l110_11095

theorem tom_helicopter_hours (total_cost : ℤ) (cost_per_hour : ℤ) (days : ℤ) (h : total_cost = 450) (c : cost_per_hour = 75) (d : days = 3) :
  total_cost / cost_per_hour / days = 2 := by
  -- Proof goes here
  sorry

end tom_helicopter_hours_l110_11095


namespace possible_values_of_expr_l110_11003

-- Define conditions
variables (x y : ℝ)
axiom h1 : x + y = 2
axiom h2 : y > 0
axiom h3 : x ≠ 0

-- Define the expression we're investigating
noncomputable def expr : ℝ := (1 / (abs x)) + (abs x / (y + 2))

-- The statement of the problem
theorem possible_values_of_expr :
  expr x y = 3 / 4 ∨ expr x y = 5 / 4 :=
sorry

end possible_values_of_expr_l110_11003


namespace perfect_squares_50_to_200_l110_11083

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end perfect_squares_50_to_200_l110_11083


namespace triangle_with_angle_ratios_l110_11032

theorem triangle_with_angle_ratios {α β γ : ℝ} (h : α + β + γ = 180 ∧ (α / 2 = β / 3) ∧ (α / 2 = γ / 5)) : (α = 90 ∨ β = 90 ∨ γ = 90) :=
by
  sorry

end triangle_with_angle_ratios_l110_11032


namespace invitations_per_package_l110_11068

theorem invitations_per_package (total_friends : ℕ) (total_packs : ℕ) (invitations_per_pack : ℕ) 
  (h1 : total_friends = 10) (h2 : total_packs = 5)
  (h3 : invitations_per_pack * total_packs = total_friends) : 
  invitations_per_pack = 2 :=
by
  sorry

end invitations_per_package_l110_11068


namespace min_value_frac_sum_l110_11022

open Real

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
    ∃ (z : ℝ), z = 1 + (sqrt 3) / 2 ∧ 
    (∀ t, (t > 0 → ∃ (u : ℝ), u > 0 ∧ t + u = 4 → ∀ t' (h : t' = (1 / t) + (3 / u)), t' ≥ z)) :=
by sorry

end min_value_frac_sum_l110_11022


namespace race_distance_l110_11026

theorem race_distance (D : ℝ)
  (A_time : D / 36 * 45 = D + 20) : 
  D = 80 :=
by
  sorry

end race_distance_l110_11026


namespace beads_taken_out_l110_11001

/--
There is 1 green bead, 2 brown beads, and 3 red beads in a container.
Tom took some beads out of the container and left 4 in.
Prove that Tom took out 2 beads.
-/
theorem beads_taken_out : 
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  initial_beads - beads_left = 2 :=
by
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  show initial_beads - beads_left = 2
  sorry

end beads_taken_out_l110_11001


namespace smallest_other_divisor_of_40_l110_11023

theorem smallest_other_divisor_of_40 (n : ℕ) (h₁ : n > 1) (h₂ : 40 % n = 0) (h₃ : n ≠ 8) :
  (∀ m : ℕ, m > 1 → 40 % m = 0 → m ≠ 8 → n ≤ m) → n = 5 :=
by 
  sorry

end smallest_other_divisor_of_40_l110_11023


namespace red_to_blue_ratio_l110_11046

theorem red_to_blue_ratio
    (total_balls : ℕ)
    (num_white_balls : ℕ)
    (num_blue_balls : ℕ)
    (num_red_balls : ℕ) :
    total_balls = 100 →
    num_white_balls = 16 →
    num_blue_balls = num_white_balls + 12 →
    num_red_balls = total_balls - (num_white_balls + num_blue_balls) →
    (num_red_balls / num_blue_balls : ℚ) = 2 :=
by
  intro h1 h2 h3 h4
  -- Proof is omitted
  sorry

end red_to_blue_ratio_l110_11046


namespace fraction_of_yard_occupied_l110_11056

/-
Proof Problem: Given a rectangular yard that measures 30 meters by 8 meters and contains
an isosceles trapezoid-shaped flower bed with parallel sides measuring 14 meters and 24 meters,
and a height of 6 meters, prove that the fraction of the yard occupied by the flower bed is 19/40.
-/

theorem fraction_of_yard_occupied (length_yard width_yard b1 b2 h area_trapezoid area_yard : ℝ) 
  (h_length_yard : length_yard = 30) 
  (h_width_yard : width_yard = 8) 
  (h_b1 : b1 = 14) 
  (h_b2 : b2 = 24) 
  (h_height_trapezoid : h = 6) 
  (h_area_trapezoid : area_trapezoid = (1/2) * (b1 + b2) * h) 
  (h_area_yard : area_yard = length_yard * width_yard) : 
  area_trapezoid / area_yard = 19 / 40 := 
by {
  -- Follow-up steps to prove the statement would go here
  sorry
}

end fraction_of_yard_occupied_l110_11056


namespace test_end_time_l110_11078

def start_time := 12 * 60 + 35  -- 12 hours 35 minutes in minutes
def duration := 4 * 60 + 50     -- 4 hours 50 minutes in minutes

theorem test_end_time : (start_time + duration) = 17 * 60 + 25 := by
  sorry

end test_end_time_l110_11078


namespace country_of_second_se_asian_fields_medal_recipient_l110_11009

-- Given conditions as definitions
def is_highest_recognition (award : String) : Prop :=
  award = "Fields Medal"

def fields_medal_freq (years : Nat) : Prop :=
  years = 4 -- Fields Medal is awarded every four years

def second_se_asian_recipient (name : String) : Prop :=
  name = "Ngo Bao Chau"

-- The main theorem to prove
theorem country_of_second_se_asian_fields_medal_recipient :
  ∀ (award : String) (years : Nat) (name : String),
    is_highest_recognition award ∧ fields_medal_freq years ∧ second_se_asian_recipient name →
    (name = "Ngo Bao Chau" → ∃ (country : String), country = "Vietnam") :=
by
  intros award years name h
  sorry

end country_of_second_se_asian_fields_medal_recipient_l110_11009


namespace largest_integer_solution_l110_11062

theorem largest_integer_solution :
  ∀ (x : ℤ), x - 5 > 3 * x - 1 → x ≤ -3 := by
  sorry

end largest_integer_solution_l110_11062


namespace find_divisor_l110_11052

variable {N : ℤ} (k q : ℤ) {D : ℤ}

theorem find_divisor (h1 : N = 158 * k + 50) (h2 : N = D * q + 13) (h3 : D > 13) (h4 : D < 158) :
  D = 37 :=
by 
  sorry

end find_divisor_l110_11052


namespace prime_expression_integer_value_l110_11029

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_expression_integer_value (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ n, (p * q + p^p + q^q) % (p + q) = 0 → n = 3 :=
by
  sorry

end prime_expression_integer_value_l110_11029


namespace rhombus_area_l110_11061

theorem rhombus_area (side d1 : ℝ) (h_side : side = 28) (h_d1 : d1 = 12) : 
  (side = 28 ∧ d1 = 12) →
  ∃ area : ℝ, area = 328.32 := 
by 
  sorry

end rhombus_area_l110_11061


namespace matching_shoes_probability_is_one_ninth_l110_11020

def total_shoes : ℕ := 10
def pairs_of_shoes : ℕ := 5
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2
def matching_combinations : ℕ := pairs_of_shoes

def matching_shoes_probability : ℚ := matching_combinations / total_combinations

theorem matching_shoes_probability_is_one_ninth :
  matching_shoes_probability = 1 / 9 :=
by
  sorry

end matching_shoes_probability_is_one_ninth_l110_11020


namespace task_completion_time_l110_11027

variable (x : Real) (y : Real)

theorem task_completion_time :
  (1 / 16) * y + (1 / 12) * x = 1 ∧ y + 5 = 8 → x = 3 ∧ y = 3 :=
  by {
    sorry 
  }

end task_completion_time_l110_11027


namespace arithmetic_mean_q_r_l110_11005

theorem arithmetic_mean_q_r (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) 
  (h3 : r - p = 24) : 
  (q + r) / 2 = 22 := 
by
  sorry

end arithmetic_mean_q_r_l110_11005


namespace sum_of_areas_of_tangent_circles_l110_11010

theorem sum_of_areas_of_tangent_circles :
  ∀ (a b c : ℝ), 
    a + b = 5 →
    a + c = 12 →
    b + c = 13 →
    π * (a^2 + b^2 + c^2) = 113 * π :=
by
  intros a b c h₁ h₂ h₃
  sorry

end sum_of_areas_of_tangent_circles_l110_11010


namespace max_possible_player_salary_l110_11021

theorem max_possible_player_salary (n : ℕ) (min_salary total_salary : ℕ) (num_players : ℕ) 
  (h1 : num_players = 24) 
  (h2 : min_salary = 20000) 
  (h3 : total_salary = 960000)
  (h4 : n = 23 * min_salary + 500000) 
  (h5 : 23 * min_salary + 500000 ≤ total_salary) 
  : n = total_salary :=
by {
  -- The proof will replace this sorry.
  sorry
}

end max_possible_player_salary_l110_11021


namespace range_of_a_l110_11038

variable {x a : ℝ}

def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := |x| > a

theorem range_of_a (h : ¬p x → ¬q x a) : a ≤ 1 :=
sorry

end range_of_a_l110_11038


namespace board_game_cost_l110_11011

theorem board_game_cost
  (v h : ℝ)
  (h1 : 3 * v = h + 490)
  (h2 : 5 * v = 2 * h + 540) :
  h = 830 := by
  sorry

end board_game_cost_l110_11011


namespace min_a2_b2_l110_11004

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + 2 * x^2 + b * x + 1 = 0) : a^2 + b^2 ≥ 8 :=
sorry

end min_a2_b2_l110_11004


namespace joey_route_length_l110_11045

-- Definitions
def time_one_way : ℝ := 1
def avg_speed : ℝ := 8
def return_speed : ℝ := 12

-- Theorem to prove
theorem joey_route_length : (∃ D : ℝ, D = 6 ∧ (D / avg_speed = time_one_way + D / return_speed)) :=
sorry

end joey_route_length_l110_11045


namespace linear_regression_solution_l110_11050

theorem linear_regression_solution :
  let barx := 5
  let bary := 50
  let sum_xi_squared := 145
  let sum_xiyi := 1380
  let n := 5
  let b := (sum_xiyi - barx * bary) / (sum_xi_squared - n * barx^2)
  let a := bary - b * barx
  let predicted_y := 6.5 * 10 + 17.5
  b = 6.5 ∧ a = 17.5 ∧ predicted_y = 82.5 := 
by
  intros
  sorry

end linear_regression_solution_l110_11050


namespace brass_weight_l110_11030

theorem brass_weight (copper zinc brass : ℝ) (h_ratio : copper / zinc = 3 / 7) (h_zinc : zinc = 70) : brass = 100 :=
by
  sorry

end brass_weight_l110_11030


namespace prime_power_implies_one_l110_11080

theorem prime_power_implies_one (p : ℕ) (a : ℤ) (n : ℕ) (h_prime : Nat.Prime p) (h_eq : 2^p + 3^p = a^n) :
  n = 1 :=
sorry

end prime_power_implies_one_l110_11080


namespace necessary_condition_l110_11028

theorem necessary_condition (m : ℝ) : 
  (∀ x > 0, (x / 2) + (1 / (2 * x)) - (3 / 2) > m) → (m ≤ -1 / 2) :=
by
  -- Proof omitted
  sorry

end necessary_condition_l110_11028


namespace cubic_sum_l110_11042

theorem cubic_sum (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 :=
by 
  sorry

end cubic_sum_l110_11042


namespace length_of_field_l110_11089

theorem length_of_field (width : ℕ) (distance_covered : ℕ) (n : ℕ) (L : ℕ) 
  (h1 : width = 15) 
  (h2 : distance_covered = 540) 
  (h3 : n = 3) 
  (h4 : 2 * (L + width) = perimeter)
  (h5 : n * perimeter = distance_covered) : 
  L = 75 :=
by 
  sorry

end length_of_field_l110_11089


namespace min_value_quadratic_function_l110_11055

def f (a b c x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_value_quadratic_function :
  ∃ a b c : ℝ, 
    (1 ≤ a ∧ a < 10) ∧
    (1 ≤ b ∧ b < 10) ∧
    (1 ≤ c ∧ c < 10) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (∀ x : ℝ, f a b c x ≥ -128) :=
sorry

end min_value_quadratic_function_l110_11055


namespace find_coeff_sum_l110_11094

def parabola_eq (a b c : ℚ) (y : ℚ) : ℚ := a*y^2 + b*y + c

theorem find_coeff_sum 
  (a b c : ℚ)
  (h_eq : ∀ y, parabola_eq a b c y = - ((y + 6)^2) / 3 + 7)
  (h_pass : parabola_eq a b c 0 = 5) :
  a + b + c = -32 / 3 :=
by
  sorry

end find_coeff_sum_l110_11094


namespace length_of_other_train_is_correct_l110_11033

noncomputable def length_of_other_train
  (l1 : ℝ) -- length of the first train in meters
  (s1 : ℝ) -- speed of the first train in km/hr
  (s2 : ℝ) -- speed of the second train in km/hr
  (t : ℝ)  -- time in seconds
  (h1 : l1 = 500)
  (h2 : s1 = 240)
  (h3 : s2 = 180)
  (h4 : t = 12) :
  ℝ :=
  let s1_m_s := s1 * 1000 / 3600
  let s2_m_s := s2 * 1000 / 3600
  let relative_speed := s1_m_s + s2_m_s
  let total_distance := relative_speed * t
  total_distance - l1

theorem length_of_other_train_is_correct :
  length_of_other_train 500 240 180 12 rfl rfl rfl rfl = 900 := sorry

end length_of_other_train_is_correct_l110_11033


namespace exists_fixed_point_subset_l110_11082

-- Definitions of set and function f with the required properties
variable {α : Type} [DecidableEq α]
variable (H : Finset α)
variable (f : Finset α → Finset α)

-- Conditions
axiom increasing_mapping (X Y : Finset α) : X ⊆ Y → f X ⊆ f Y
axiom range_in_H (X : Finset α) : f X ⊆ H

-- Statement to prove
theorem exists_fixed_point_subset : ∃ H₀ ⊆ H, f H₀ = H₀ :=
sorry

end exists_fixed_point_subset_l110_11082


namespace average_percent_score_l110_11031

def num_students : ℕ := 180

def score_distrib : List (ℕ × ℕ) :=
[(95, 12), (85, 30), (75, 50), (65, 45), (55, 30), (45, 13)]

noncomputable def total_score : ℕ :=
(95 * 12) + (85 * 30) + (75 * 50) + (65 * 45) + (55 * 30) + (45 * 13)

noncomputable def average_score : ℕ :=
total_score / num_students

theorem average_percent_score : average_score = 70 :=
by 
  -- Here you would provide the proof, but for now we will leave it as:
  sorry

end average_percent_score_l110_11031


namespace Conor_can_chop_116_vegetables_in_a_week_l110_11058

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l110_11058


namespace max_cone_cross_section_area_l110_11059

theorem max_cone_cross_section_area
  (V A B : Type)
  (E : Type)
  (l : ℝ)
  (α : ℝ) :
  0 < l ∧ 0 < α ∧ α < 180 → 
  ∃ (area : ℝ), area = (1 / 2) * l^2 :=
by
  sorry

end max_cone_cross_section_area_l110_11059


namespace total_pieces_eq_21_l110_11018

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end total_pieces_eq_21_l110_11018


namespace correct_eq_count_l110_11041

-- Define the correctness of each expression
def eq1 := (∀ x : ℤ, (-2 * x)^3 = 2 * x^3 = false)
def eq2 := (∀ a : ℤ, a^2 * a^3 = a^3 = false)
def eq3 := (∀ x : ℤ, (-x)^9 / (-x)^3 = x^6 = true)
def eq4 := (∀ a : ℤ, (-3 * a^2)^3 = -9 * a^6 = false)

-- Define the condition that there are exactly one correct equation
def num_correct_eqs := (1 = 1)

-- The theorem statement, proving the count of correct equations is 1
theorem correct_eq_count : eq1 → eq2 → eq3 → eq4 → num_correct_eqs :=
  by intros; sorry

end correct_eq_count_l110_11041


namespace dasha_strip_problem_l110_11085

theorem dasha_strip_problem (a b c : ℕ) (h : a * (2 * b + 2 * c - a) = 43) :
  a = 1 ∧ b + c = 22 :=
by {
  sorry
}

end dasha_strip_problem_l110_11085


namespace problem_inequality_l110_11087

variable {α : Type*} [LinearOrder α]

def M (x y : α) : α := max x y
def m (x y : α) : α := min x y

theorem problem_inequality (a b c d e : α) (h : a < b) (h1 : b < c) (h2 : c < d) (h3 : d < e) : 
  M (M a (m b c)) (m d (m a e)) = b := sorry

end problem_inequality_l110_11087


namespace remainder_div2_l110_11064

   theorem remainder_div2 :
     ∀ z x : ℕ, (∃ k : ℕ, z = 4 * k) → (∃ n : ℕ, x = 2 * n) → (z + x + 4 + z + 3) % 2 = 1 :=
   by
     intros z x h1 h2
     sorry
   
end remainder_div2_l110_11064


namespace segments_either_disjoint_or_common_point_l110_11076

theorem segments_either_disjoint_or_common_point (n : ℕ) (segments : List (ℝ × ℝ)) 
  (h_len : segments.length = n^2 + 1) : 
  (∃ (disjoint_segments : List (ℝ × ℝ)), disjoint_segments.length ≥ n + 1 ∧ 
    (∀ (s1 s2 : (ℝ × ℝ)), s1 ∈ disjoint_segments → s2 ∈ disjoint_segments 
    → s1 ≠ s2 → ¬ (s1.1 ≤ s2.2 ∧ s2.1 ≤ s1.2))) 
  ∨ 
  (∃ (common_point_segments : List (ℝ × ℝ)), common_point_segments.length ≥ n + 1 ∧ 
    (∃ (p : ℝ), ∀ (s : (ℝ × ℝ)), s ∈ common_point_segments → s.1 ≤ p ∧ p ≤ s.2)) :=
sorry

end segments_either_disjoint_or_common_point_l110_11076


namespace sequence_formula_sequence_inequality_l110_11013

open Nat

-- Definition of the sequence based on the given conditions
noncomputable def a : ℕ → ℚ
| 0     => 1                -- 0-indexed for Lean handling convenience, a_1 = 1 is a(0) in Lean
| (n+1) => 2 - 1 / (a n)    -- recurrence relation

-- Proof for part (I) that a_n = (n + 1) / n
theorem sequence_formula (n : ℕ) : a (n + 1) = (n + 2) / (n + 1) := sorry

-- Proof for part (II)
theorem sequence_inequality (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  (1 + a (n + 1)) / a (k + 1) < 2 ∨ (1 + a (k + 1)) / a (n + 1) < 2 := sorry

end sequence_formula_sequence_inequality_l110_11013


namespace geometric_sequence_eighth_term_l110_11000

theorem geometric_sequence_eighth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : r = 16 / 4) :
  a * r^(7) = 65536 :=
by
  sorry

end geometric_sequence_eighth_term_l110_11000


namespace ratio_two_to_three_nights_ago_l110_11084

def question (x : ℕ) (k : ℕ) : (ℕ × ℕ) := (x, k)

def pages_three_nights_ago := 15
def additional_pages_last_night (x : ℕ) := x + 5
def total_pages := 100
def pages_tonight := 20

theorem ratio_two_to_three_nights_ago :
  ∃ (x : ℕ), 
    (x + additional_pages_last_night x = total_pages - (pages_three_nights_ago + pages_tonight)) 
    ∧ (x / pages_three_nights_ago = 2 / 1) :=
by
  sorry

end ratio_two_to_three_nights_ago_l110_11084


namespace volleyball_team_selection_l110_11036

noncomputable def numberOfWaysToChooseStarters : ℕ :=
  (Nat.choose 13 4 * 3) + (Nat.choose 14 4 * 1)

theorem volleyball_team_selection :
  numberOfWaysToChooseStarters = 3146 := by
  sorry

end volleyball_team_selection_l110_11036


namespace probability_at_least_60_cents_l110_11090

theorem probability_at_least_60_cents :
  let num_total_outcomes := Nat.choose 16 8
  let num_successful_outcomes := 
    (Nat.choose 4 2) * (Nat.choose 5 1) * (Nat.choose 7 5) +
    1 -- only one way to choose all 8 dimes
  num_successful_outcomes / num_total_outcomes = 631 / 12870 := by
  sorry

end probability_at_least_60_cents_l110_11090


namespace sum_of_smallest_two_consecutive_numbers_l110_11074

theorem sum_of_smallest_two_consecutive_numbers (n : ℕ) (h : n * (n + 1) * (n + 2) = 210) : n + (n + 1) = 11 :=
sorry

end sum_of_smallest_two_consecutive_numbers_l110_11074


namespace function_relationship_l110_11017

variable {A B : Type} [Nonempty A] [Nonempty B]
variable (f : A → B) 

def domain (f : A → B) : Set A := {a | ∃ b, f a = b}
def range (f : A → B) : Set B := {b | ∃ a, f a = b}

theorem function_relationship (M : Set A) (N : Set B) (hM : M = Set.univ)
                              (hN : N = range f) : M = Set.univ ∧ N ⊆ Set.univ :=
  sorry

end function_relationship_l110_11017


namespace sum_of_reversed_integers_l110_11073

-- Definitions of properties and conditions
def reverse_digits (m n : ℕ) : Prop :=
  let to_digits (x : ℕ) : List ℕ := x.digits 10
  to_digits m = (to_digits n).reverse

-- The main theorem statement
theorem sum_of_reversed_integers
  (m n : ℕ)
  (h_rev: reverse_digits m n)
  (h_prod: m * n = 1446921630) :
  m + n = 79497 :=
sorry

end sum_of_reversed_integers_l110_11073


namespace range_of_a_l110_11069

def A := {x : ℝ | |x| >= 3}
def B (a : ℝ) := {x : ℝ | x >= a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a <= -3 :=
sorry

end range_of_a_l110_11069


namespace find_blue_balloons_l110_11060

theorem find_blue_balloons (purple_balloons : ℕ) (left_balloons : ℕ) (total_balloons : ℕ) (blue_balloons : ℕ) :
  purple_balloons = 453 →
  left_balloons = 378 →
  total_balloons = left_balloons * 2 →
  total_balloons = purple_balloons + blue_balloons →
  blue_balloons = 303 := by
  intros h1 h2 h3 h4
  sorry

end find_blue_balloons_l110_11060


namespace rationalize_denominator_l110_11014

noncomputable def sqrt_12 := Real.sqrt 12
noncomputable def sqrt_5 := Real.sqrt 5
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_15 := Real.sqrt 15

theorem rationalize_denominator :
  (sqrt_12 + sqrt_5) / (sqrt_3 + sqrt_5) = (-1 / 2) + (sqrt_15 / 2) :=
sorry

end rationalize_denominator_l110_11014


namespace poly_constant_or_sum_constant_l110_11008

-- definitions of the polynomials as real-coefficient polynomials
variables (P Q R : Polynomial ℝ)

-- conditions
#check ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ) -- Considering 'constant' as 1 for simplicity

-- target
theorem poly_constant_or_sum_constant 
  (h : ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ)) :
  (∃ c : ℝ, ∀ x, P.eval x = c) ∨ (∃ c : ℝ, ∀ x, Q.eval x + R.eval x = c) :=
sorry

end poly_constant_or_sum_constant_l110_11008


namespace sin_angle_calculation_l110_11057

theorem sin_angle_calculation (α : ℝ) (h : α = 240) : Real.sin (150 - α) = -1 :=
by
  rw [h]
  norm_num
  sorry

end sin_angle_calculation_l110_11057


namespace find_value_of_expression_l110_11040

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry

def condition1 : Prop := x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 + 11 * x6 = 2
def condition2 : Prop := 3 * x1 + 5 * x2 + 7 * x3 + 9 * x4 + 11 * x5 + 13 * x6 = 15
def condition3 : Prop := 5 * x1 + 7 * x2 + 9 * x3 + 11 * x4 + 13 * x5 + 15 * x6 = 52

theorem find_value_of_expression : condition1 → condition2 → condition3 → (7 * x1 + 9 * x2 + 11 * x3 + 13 * x4 + 15 * x5 + 17 * x6 = 65) :=
by
  intros h1 h2 h3
  sorry

end find_value_of_expression_l110_11040


namespace initial_money_l110_11049

theorem initial_money (cost_of_candy_bar : ℕ) (change_received : ℕ) (initial_money : ℕ) 
  (h_cost : cost_of_candy_bar = 45) (h_change : change_received = 5) :
  initial_money = cost_of_candy_bar + change_received :=
by
  -- here is the place for the proof which is not needed
  sorry

end initial_money_l110_11049


namespace prob_first_3_heads_last_5_tails_eq_l110_11091

-- Define the conditions
def prob_heads : ℚ := 3/5
def prob_tails : ℚ := 1 - prob_heads
def heads_flips (n : ℕ) : ℚ := prob_heads ^ n
def tails_flips (n : ℕ) : ℚ := prob_tails ^ n
def first_3_heads_last_5_tails (first_n : ℕ) (last_m : ℕ) : ℚ := (heads_flips first_n) * (tails_flips last_m)

-- Specify the problem
theorem prob_first_3_heads_last_5_tails_eq :
  first_3_heads_last_5_tails 3 5 = 864/390625 := 
by
  -- conditions and calculation here
  sorry

end prob_first_3_heads_last_5_tails_eq_l110_11091


namespace count_expressible_integers_l110_11093

theorem count_expressible_integers :
  ∃ (count : ℕ), count = 1138 ∧ (∀ n, (n ≤ 2000) → (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)) :=
sorry

end count_expressible_integers_l110_11093


namespace circumference_of_base_of_cone_l110_11063

theorem circumference_of_base_of_cone (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) 
  (h1 : V = 24 * Real.pi) (h2 : h = 6) (h3 : V = (1/3) * Real.pi * r^2 * h) 
  (h4 : r = Real.sqrt 12) : C = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end circumference_of_base_of_cone_l110_11063


namespace sum_of_reciprocals_l110_11025

theorem sum_of_reciprocals (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 31 / 21) :=
sorry

end sum_of_reciprocals_l110_11025


namespace steve_halfway_time_longer_l110_11044

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer_l110_11044


namespace fathers_age_after_further_8_years_l110_11006

variable (R F : ℕ)

def age_relation_1 : Prop := F = 4 * R
def age_relation_2 : Prop := F + 8 = (5 * (R + 8)) / 2

theorem fathers_age_after_further_8_years (h1 : age_relation_1 R F) (h2 : age_relation_2 R F) : (F + 16) = 2 * (R + 16) :=
by 
  sorry

end fathers_age_after_further_8_years_l110_11006


namespace melted_mixture_weight_l110_11051

theorem melted_mixture_weight
    (Z C : ℝ)
    (ratio_eq : Z / C = 9 / 11)
    (zinc_weight : Z = 33.3) :
    Z + C = 74 :=
by
  sorry

end melted_mixture_weight_l110_11051


namespace simplify_fraction_l110_11079

theorem simplify_fraction :
  (30 / 35) * (21 / 45) * (70 / 63) - (2 / 3) = - (8 / 15) :=
by
  sorry

end simplify_fraction_l110_11079


namespace no_valid_pair_for_tangential_quadrilateral_l110_11077

theorem no_valid_pair_for_tangential_quadrilateral (a d : ℝ) (h : d > 0) :
  ¬((∃ a d, a + (a + 2 * d) = (a + d) + (a + 3 * d))) :=
by
  sorry

end no_valid_pair_for_tangential_quadrilateral_l110_11077


namespace find_a_l110_11039

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l110_11039


namespace trajectory_midpoint_l110_11053

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

-- Define the condition that a line passes through the point (0, 1)
def line_through_fixed_point (k x y : ℝ) := y = k * x + 1

-- Define the theorem to prove the trajectory of the midpoint of the chord
theorem trajectory_midpoint (x y k : ℝ) (h : ∃ x y, hyperbola x y ∧ line_through_fixed_point k x y) : 
    4 * x^2 - y^2 + y = 0 := 
sorry

end trajectory_midpoint_l110_11053


namespace total_cost_l110_11047

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 5

theorem total_cost : (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 31 := by
  sorry

end total_cost_l110_11047


namespace avg_age_diff_l110_11098

noncomputable def avg_age_team : ℕ := 28
noncomputable def num_players : ℕ := 11
noncomputable def wicket_keeper_age : ℕ := avg_age_team + 3
noncomputable def total_age_team : ℕ := avg_age_team * num_players
noncomputable def age_captain : ℕ := avg_age_team

noncomputable def total_age_remaining_players : ℕ := total_age_team - age_captain - wicket_keeper_age
noncomputable def num_remaining_players : ℕ := num_players - 2
noncomputable def avg_age_remaining_players : ℕ := total_age_remaining_players / num_remaining_players

theorem avg_age_diff :
  avg_age_team - avg_age_remaining_players = 3 :=
by
  sorry

end avg_age_diff_l110_11098


namespace cubic_polynomial_has_three_real_roots_l110_11024

open Polynomial

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry

axiom P_degree : degree P = 2
axiom Q_degree : degree Q = 3
axiom R_degree : degree R = 3
axiom PQR_relationship : ∀ x : ℝ, P.eval x ^ 2 + Q.eval x ^ 2 = R.eval x ^ 2

theorem cubic_polynomial_has_three_real_roots : 
  (∃ x : ℝ, Q.eval x = 0 ∧ ∃ y : ℝ, Q.eval y = 0 ∧ ∃ z : ℝ, Q.eval z = 0) ∨
  (∃ x : ℝ, R.eval x = 0 ∧ ∃ y : ℝ, R.eval y = 0 ∧ ∃ z : ℝ, R.eval z = 0) :=
sorry

end cubic_polynomial_has_three_real_roots_l110_11024


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l110_11097

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l110_11097


namespace parabola_relationship_l110_11035

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem parabola_relationship (a b m n t : ℝ) (ha : a ≠ 0)
  (h1 : 3 * a + b > 0) (h2 : a + b < 0)
  (hm : parabola a b (-3) = m)
  (hn : parabola a b 2 = n)
  (ht : parabola a b 4 = t) :
  n < t ∧ t < m :=
by
  sorry

end parabola_relationship_l110_11035


namespace total_colored_hangers_l110_11002

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end total_colored_hangers_l110_11002


namespace simplify_fraction_l110_11048

variables {x y : ℝ}

theorem simplify_fraction (h : x / y = 2 / 5) : (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 :=
by
  sorry

end simplify_fraction_l110_11048


namespace total_handshakes_at_convention_l110_11072

theorem total_handshakes_at_convention :
  let gremlins := 25
  let imps := 18
  let specific_gremlins := 5
  let friendly_gremlins := gremlins - specific_gremlins
  let handshakes_among_gremlins := (friendly_gremlins * (friendly_gremlins - 1)) / 2
  let handshakes_between_imps_and_gremlins := imps * gremlins
  handshakes_among_gremlins + handshakes_between_imps_and_gremlins = 640 := by
  sorry

end total_handshakes_at_convention_l110_11072


namespace greatest_possible_gcd_value_l110_11037

noncomputable def sn (n : ℕ) := n ^ 2
noncomputable def expression (n : ℕ) := 2 * sn n + 10 * n
noncomputable def gcd_value (a b : ℕ) := Nat.gcd a b 

theorem greatest_possible_gcd_value :
  ∃ n : ℕ, gcd_value (expression n) (n - 3) = 42 :=
sorry

end greatest_possible_gcd_value_l110_11037


namespace shirts_per_minute_l110_11088

theorem shirts_per_minute (shirts_in_6_minutes : ℕ) (time_minutes : ℕ) (h1 : shirts_in_6_minutes = 36) (h2 : time_minutes = 6) : 
  ((shirts_in_6_minutes / time_minutes) = 6) :=
by
  sorry

end shirts_per_minute_l110_11088


namespace problem_l110_11099

noncomputable def K : ℕ := 36
noncomputable def L : ℕ := 147
noncomputable def M : ℕ := 56

theorem problem (h1 : 4 / 7 = K / 63) (h2 : 4 / 7 = 84 / L) (h3 : 4 / 7 = M / 98) :
  (K + L + M) = 239 :=
by
  sorry

end problem_l110_11099


namespace smallest_possible_value_l110_11071

theorem smallest_possible_value (x : ℝ) (hx : 11 = x^2 + 1 / x^2) :
  x + 1 / x = -Real.sqrt 13 :=
by
  sorry

end smallest_possible_value_l110_11071


namespace divisible_by_n_sequence_l110_11054

theorem divisible_by_n_sequence (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ n ∣ (2^k - 1) :=
by {
  sorry
}

end divisible_by_n_sequence_l110_11054


namespace sum_of_coefficients_l110_11065

def P (x : ℝ) : ℝ := 3 * (x^8 - 2 * x^5 + x^3 - 7) - 5 * (x^6 + 3 * x^2 - 6) + 2 * (x^4 - 5)

theorem sum_of_coefficients : P 1 = -19 := by
  sorry

end sum_of_coefficients_l110_11065


namespace people_in_rooms_l110_11070

theorem people_in_rooms (x y : ℕ) (h1 : x + y = 76) (h2 : x - 30 = y - 40) : x = 33 ∧ y = 43 := by
  sorry

end people_in_rooms_l110_11070


namespace alberto_more_than_bjorn_and_charlie_l110_11043

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie_l110_11043


namespace valid_k_for_triangle_l110_11086

theorem valid_k_for_triangle (k : ℕ) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ b + c > a ∧ c + a > b)) → k ≥ 6 :=
by
  sorry

end valid_k_for_triangle_l110_11086


namespace prime_power_value_l110_11066

theorem prime_power_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h1 : Nat.Prime (7 * p + q)) (h2 : Nat.Prime (p * q + 11)) : 
  p ^ q = 8 ∨ p ^ q = 9 := 
sorry

end prime_power_value_l110_11066


namespace no_y_satisfies_both_inequalities_l110_11092

variable (y : ℝ)

theorem no_y_satisfies_both_inequalities :
  ¬ (3 * y^2 - 4 * y - 5 < (y + 1)^2 ∧ (y + 1)^2 < 4 * y^2 - y - 1) :=
by
  sorry

end no_y_satisfies_both_inequalities_l110_11092


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l110_11012

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l110_11012
